"""Fingerprinted, append-only score cache and the run loop (PLAN.md section 3).

cache/{arm}/{tier}.parquet: item_id, text_hash, score, valid, latency_ms, input_tokens, output_tokens, n_chunks, raw,
error, config_fp, ts. A run skips items whose (text_hash, config_fp) already has an error-free row.

    python runner.py "Jev 1.14" --wording v2            # every scored tier
    python runner.py "Jev 1.14" --wording v2 --smoke    # 3 items, print raw outputs, stop on any failure
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import adapters
import tiers

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
COLUMNS = ["item_id", "text_hash", "score", "valid", "latency_ms", "input_tokens", "output_tokens", "n_chunks", "raw",
           "error", "config_fp", "ts"]
SCORED_SPLITS = ("dev", "test")   # "train" rows are Narrow's training data, never scored


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def cache_path(arm_name, tier):
    return CACHE_DIR / slug(arm_name) / f"tier-{tier}.parquet"


def read_cache(arm_name, tier):
    path = cache_path(arm_name, tier)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


class FileLock:
    """Lock file created exclusively: two processes writing one arm's cache can't drop each other's rows."""
    def __init__(self, path):
        self.path = path.with_suffix(".lock")

    def __enter__(self):
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - self.path.stat().st_mtime > 120:   # stale lock from a killed process
                    self.path.unlink(missing_ok=True)
                time.sleep(0.05)

    def __exit__(self, *exc):
        os.close(self.fd)
        self.path.unlink(missing_ok=True)


def append_cache(arm_name, tier, rows):
    if not rows:
        return
    path = cache_path(arm_name, tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path):
        frame = pd.concat([read_cache(arm_name, tier), pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)


def matching(arm, items):
    """Latest error-free cached row per item for the CURRENT config and the CURRENT text. Index = item_id."""
    parts = []
    fp = arm.config_fingerprint()
    for tier, group in items.groupby("tier"):
        cached = read_cache(arm.name, tier)
        if cached.empty:
            continue
        cached = cached[(cached["config_fp"] == fp) & (cached["error"] == "")]
        expected = dict(zip(group["item_id"], group["text_hash"]))
        cached = cached[cached["text_hash"] == cached["item_id"].map(expected)]
        parts.append(cached.sort_values("ts", kind="stable").drop_duplicates("item_id", keep="last"))   # latest by timestamp
    if not parts:
        return pd.DataFrame(columns=COLUMNS).set_index("item_id")
    return pd.concat(parts).set_index("item_id")


def run(arm, items, chunk=200, max_attempts=3, verbose=True):
    """Score every item not yet cached for this arm's config. Every attempt is cached, including failures."""
    fp = arm.config_fingerprint()
    for attempt in range(1, max_attempts + 1):
        done = set(matching(arm, items).index)
        todo = items[~items["item_id"].isin(done)]
        if verbose:
            print(f"{arm.name} [fp {fp}] attempt {attempt}: {len(items) - len(todo)} cached, {len(todo)} to score", flush=True)
        if todo.empty:
            return
        failures, started = 0, time.time()
        for tier, group in todo.groupby("tier", sort=False):
            for start in range(0, len(group), chunk):
                part = group.iloc[start:start + chunk]
                outputs = arm.predict(part["text"].tolist())
                now = datetime.now(timezone.utc).isoformat()
                rows = [{"item_id": item_id, "text_hash": text_hash, "config_fp": fp, "ts": now, **out}
                        for item_id, text_hash, out in zip(part["item_id"], part["text_hash"], outputs)]
                append_cache(arm.name, tier, rows)
                failures += sum(bool(r["error"]) for r in rows)
                if verbose:
                    last = next((r["error"] for r in reversed(rows) if r["error"]), "")
                    print(f"  tier {tier}: {min(start + chunk, len(group))}/{len(group)}, failures so far {failures}"
                          f"{', last: ' + last[:120] if last else ''} ({time.time() - started:.0f}s)", flush=True)
        if not failures:
            return
        time.sleep(15 * attempt)


def latency_pass(arm, items, n=200):
    """Batch size 1, sequential, the same 200 items for every arm. Stored as tier 'latency'."""
    sample = items[items["tier"].isin(["1", "3"])].sample(n, random_state=tiers.SEED)
    path = cache_path(arm.name, "latency")
    fp = arm.config_fingerprint()
    if path.exists():
        cached = pd.read_parquet(path)
        if (cached["config_fp"] == fp).sum() >= n:
            return
    rows = []
    if hasattr(arm, "load"):
        arm.load()
        arm.predict(["warm-up"])   # keep load and kernel compilation out of the timing
    for _, item in sample.iterrows():
        out = arm.predict([item["text"]])[0] if not isinstance(arm, adapters.APIArm) else arm.one(item["text"])
        rows.append({"item_id": item["item_id"], "text_hash": item["text_hash"], "config_fp": fp,
                     "ts": datetime.now(timezone.utc).isoformat(), **out})
    append_cache(arm.name, "latency", rows)


def scored_items(items):
    return items[items["split"].isin(SCORED_SPLITS)]


def dev_sample(items, n):
    """A fixed random n per (tier, label): the dev subset used to compare question wordings."""
    return pd.concat([group.sample(min(n, len(group)), random_state=tiers.SEED) for _, group in items.groupby(["tier", "label"])])


def smoke(arm, items):
    sample = scored_items(items).groupby("label").head(1)
    sample = pd.concat([sample, items[items["tier"] == "4"].head(1)])
    outputs = arm.predict(sample["text"].tolist())
    for (_, item), out in zip(sample.iterrows(), outputs):
        print(f"--- {item['item_id']} label={item['label']} -> score={out['score']} valid={out['valid']} "
              f"in={out['input_tokens']} out={out['output_tokens']} {out['latency_ms']:.0f} ms")
        print("    raw:", (out["raw"] or "")[:300].replace("\n", " "))
        if out["error"]:
            raise SystemExit(f"smoke test failed: {out['error']}")
    print(f"{arm.name}: smoke test OK, config fingerprint {arm.config_fingerprint()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--wording", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tiers", default=None, help="comma-separated tiers, default all scored tiers")
    parser.add_argument("--latency", action="store_true", help="also run the batch-size-1 latency pass")
    parser.add_argument("--limit", type=int, default=None, help="first N items per tier (dev wording comparison)")
    args = parser.parse_args()
    items, _ = tiers.build_items()
    items = scored_items(items)
    if args.tiers:
        items = items[items["tier"].isin(args.tiers.split(","))]
    if args.limit:
        items = dev_sample(items, args.limit)
    arm = adapters.make_arm(args.arm, args.wording)
    if args.smoke:
        smoke(arm, items)
        sys.exit()
    run(arm, items)
    if args.latency:
        latency_pass(arm, tiers.build_items()[0])
    print("done:", json.dumps({"arm": arm.name, "fp": arm.config_fingerprint(), "scored": len(matching(arm, items))}))
