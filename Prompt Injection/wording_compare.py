"""Dev-only comparison of the two question wordings (adapters.QUESTION_WORDINGS) for the API arms.

Rule, fixed before looking: freeze the wording with the higher mean dev AUROC across the API arms that finished both.
Writes harness/wording_choice.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import adapters
import metrics
import runner
import tiers

HERE = Path(__file__).parent


def compare(limit=200):
    # Sampled from every tier-"dev" row, including the 3 later purged as near-copies of test items, so the comparison
    # stays the one that was run. It only picks the API arms' question; none of them trains on anything here.
    items = runner.dev_sample(tiers.build_items()[0].query("tier == 'dev'"), limit)
    rows = []
    for arm_name in adapters.API_ARMS:
        for wording in adapters.QUESTION_WORDINGS:
            arm = adapters.make_arm(arm_name, wording)
            cached = runner.matching(arm, items)
            if len(cached) < len(items):
                continue
            joined = items.join(cached[["score", "valid"]], on="item_id")
            scores, labels = joined["score"].astype(float).to_numpy(), joined["label"].to_numpy()
            rows.append({"arm": arm_name, "wording": wording, "n": len(joined),
                         "dev AUROC": metrics.auroc(scores[labels == 1], scores[labels == 0]),
                         "recall @0.5": float(np.mean(scores[labels == 1] >= 0.5)),
                         "FPR @0.5": float(np.mean(scores[labels == 0] >= 0.5)),
                         "invalid": int((~joined["valid"].astype(bool)).sum())})
    table = pd.DataFrame(rows)
    both = table.groupby("arm")["wording"].nunique()
    complete = table[table["arm"].isin(both[both == len(adapters.QUESTION_WORDINGS)].index)]
    means = complete.groupby("wording")["dev AUROC"].mean()
    choice = {"rule": __doc__.split("Rule, fixed before looking: ")[1].split("\n")[0], "mean_dev_auroc": means.to_dict(),
              "chosen": means.idxmax(), "arms": sorted(complete["arm"].unique()), "table": table.to_dict("records")}
    (HERE / "harness").mkdir(exist_ok=True)
    (HERE / "harness" / "wording_choice.json").write_text(json.dumps(choice, indent=2))
    return table, choice


if __name__ == "__main__":
    table, choice = compare()
    print(table.round(4).to_string(index=False))
    print("chosen:", choice["chosen"], choice["mean_dev_auroc"])
