"""Train our own guards (PLAN.md section 6): the TF-IDF + logistic regression floor, then Narrow (ModernBERT-base).

Training data: PromptShield train minus the dev benign slice, plus the 3,000-row LLMail train slice (drawn by dedup
group, disjoint from dev and test). PromptShield attacks are subsampled to 1,500 so attacks are about 1:1.5 to benign.
Nothing from NotInject, AgentDojo, tier 5/6 or any test tier. Dev = the dev split (thresholds and model selection).

    python train_narrow.py tfidf
    python train_narrow.py narrow
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import tiers

HERE = Path(__file__).parent
MODELS = HERE / "models"
HARNESS = HERE / "harness"
PROMPTSHIELD_ATTACKS = 1_500

NARROW_CONFIG = {"base": "answerdotai/ModernBERT-base", "base_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
                 "max_length": 1024, "batch_size": 16, "learning_rate": 2e-5, "epochs": 3, "bf16": True,
                 "warmup_ratio": 0.06, "weight_decay": 0.01, "seed": tiers.SEED, "selection": "best dev AUROC per epoch"}
TFIDF_CONFIG = {"char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3, "max_features": 300_000, "sublinear_tf": True},
                "word": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 2, "max_features": 200_000, "sublinear_tf": True},
                "C_grid": [0.25, 1.0, 4.0], "selection": "best dev AUROC", "seed": tiers.SEED}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def training_data():
    items, _ = tiers.build_items()
    train = items[items["split"] == "train"]
    promptshield_attacks = train[(train["label"] == 1) & train["item_id"].str.startswith("promptshield")]
    keep = promptshield_attacks.sample(PROMPTSHIELD_ATTACKS, random_state=tiers.SEED).index
    train = train[~train.index.isin(promptshield_attacks.index) | train.index.isin(keep)]
    dev = items[items["split"] == "dev"]
    assert not set(train["group"]) & set(items.loc[items["split"] != "train", "group"]), "train group leaks into dev/test"
    return train.sample(frac=1, random_state=tiers.SEED), dev


def train_tfidf():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    train, dev = training_data()
    best = None
    for C in TFIDF_CONFIG["C_grid"]:
        features = FeatureUnion([(name, TfidfVectorizer(**{k: tuple(v) if k == "ngram_range" else v for k, v in TFIDF_CONFIG[name].items()}))
                                 for name in ("char", "word")])
        model = make_pipeline(features, LogisticRegression(C=C, max_iter=2_000, random_state=tiers.SEED))
        model.fit(train["text"], train["label"])
        auroc = roc_auc_score(dev["label"], model.predict_proba(dev["text"])[:, 1])
        print(f"C={C}: dev AUROC {auroc:.4f}")
        if best is None or auroc > best[0]:
            best = (auroc, C, model)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(best[2], MODELS / "tfidf-v1.joblib")
    record = {"config": TFIDF_CONFIG, "chosen_C": best[1], "dev_auroc": best[0], "train_rows": len(train),
              "train_attacks": int(train["label"].sum()), "train_hash": sha(sorted(train["text_hash"]))}
    record["config_fingerprint"] = sha({k: record[k] for k in ("config", "chosen_C", "train_hash")})
    HARNESS.mkdir(exist_ok=True)
    (HARNESS / "tfidf-v1.json").write_text(json.dumps(record, indent=2))
    print(record)


def train_narrow():
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)

    set_seed(tiers.SEED)
    train, dev = training_data()
    config = NARROW_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(config["base"], revision=config["base_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base"], revision=config["base_revision"], num_labels=2,
        id2label={0: "benign", 1: "attack"}, label2id={"benign": 0, "attack": 1})

    def encode(frame):
        data = Dataset.from_dict({"text": frame["text"].tolist(), "label": frame["label"].astype(int).tolist()})
        return data.map(lambda b: tokenizer(b["text"], truncation=True, max_length=config["max_length"]), batched=True,
                        remove_columns=["text"])

    def metrics(prediction):
        logits, labels = prediction
        probabilities = torch.softmax(torch.tensor(logits).float(), dim=-1)[:, 1].numpy()
        return {"auroc": roc_auc_score(labels, probabilities)}

    output = MODELS / "narrow-v1-checkpoints"
    arguments = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=32,
        learning_rate=config["learning_rate"], num_train_epochs=config["epochs"], bf16=config["bf16"],
        warmup_steps=config["warmup_ratio"], weight_decay=config["weight_decay"], eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="auroc", greater_is_better=True,
        save_total_limit=1, logging_steps=50, seed=tiers.SEED, report_to="none", train_sampling_strategy="group_by_length",
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=arguments, train_dataset=encode(train), eval_dataset=encode(dev),
                      data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer, compute_metrics=metrics)
    trainer.train()
    final = trainer.evaluate()
    trainer.save_model(str(MODELS / "narrow-v1"))
    tokenizer.save_pretrained(str(MODELS / "narrow-v1"))
    history = [entry for entry in trainer.state.log_history if "eval_auroc" in entry]
    record = {"config": config, "dev_auroc": final["eval_auroc"], "epochs": history, "train_rows": len(train),
              "train_attacks": int(train["label"].sum()), "train_hash": sha(sorted(train["text_hash"]))}
    record["config_fingerprint"] = sha({k: record[k] for k in ("config", "train_hash")} | {"dev_auroc": round(final["eval_auroc"], 6)})
    HARNESS.mkdir(exist_ok=True)
    (HARNESS / "narrow-v1.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    {"tfidf": train_tfidf, "narrow": train_narrow}[sys.argv[1]]()
