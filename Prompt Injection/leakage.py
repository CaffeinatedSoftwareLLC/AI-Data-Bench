"""Cross-split near-copy search with exact verification (added after the independent review, see VERIFY.md).

The first near-dedupe (tiers.near_dedupe) used LSH tuned for a 0.8 threshold: 9 bands x 13 rows, which proposes a
true 0.8-overlap pair only ~40% of the time. So near-copies could sit in different splits. This module:
1. proposes candidate pairs with a loose LSH (threshold 0.5, 256 permutations: ~100% recall at overlap >= 0.8),
2. keeps pairs whose EXACT Jaccard overlap of normalised word 5-shingles is >= 0.8,
3. texts with fewer than 5 normalised words are compared only by exact match of their lowercased text WITH digits
   (normalise() strips digits, so "Multiply 887 and 4" and "Multiply 4 and 7" would otherwise collide).

Only pairs in different splits matter: train-dev, train-test, dev-test.
"""
import re

import numpy as np
import pandas as pd

import tiers

THRESHOLD = 0.8
SHINGLE = 5


def shingles(text):
    words = tiers.normalise(text).split()
    if len(words) < SHINGLE:
        return None
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def short_key(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def cross_split_pairs(items):
    """All verified near-copy pairs between different splits. Columns: a, b, a_split, b_split, jaccard."""
    from datasketch import MinHash, MinHashLSH

    frame = items[items["split"].isin(["train", "dev", "test"])].reset_index(drop=True)
    sets = [shingles(text) for text in frame["text"]]
    pairs = []

    # Short texts: exact lowercase match only
    short = frame[[s is None for s in sets]].assign(key=lambda f: f["text"].map(short_key))
    for _, group in short.groupby("key"):
        if group["split"].nunique() > 1:
            rows = group.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if rows[i]["split"] != rows[j]["split"]:
                        pairs.append((rows[i]["item_id"], rows[j]["item_id"], rows[i]["split"], rows[j]["split"], 1.0))

    long_index = [i for i, s in enumerate(sets) if s is not None]
    hashes = MinHash.bulk([[g.encode("utf-8") for g in sets[i]] for i in long_index], num_perm=256, seed=7)
    lsh = MinHashLSH(threshold=0.5, num_perm=256)
    for position, minhash in zip(long_index, hashes):
        lsh.insert(position, minhash)
    split = frame["split"].to_numpy()
    seen = set()
    for position, minhash in zip(long_index, hashes):
        for other in lsh.query(minhash):
            if other <= position or split[other] == split[position] or (position, other) in seen:
                continue
            seen.add((position, other))
            a, b = sets[position], sets[other]
            jaccard = len(a & b) / len(a | b)
            if jaccard >= THRESHOLD:
                pairs.append((frame.at[position, "item_id"], frame.at[other, "item_id"], split[position], split[other], jaccard))
    return pd.DataFrame(pairs, columns=["a", "b", "a_split", "b_split", "jaccard"])


def purge_list(pairs):
    """Which rows to drop so no near-copy crosses splits, keeping the test set whole:
    a train row near any dev/test row goes; a dev row near any test row goes."""
    drop = {}
    for row in pairs.itertuples():
        ends = {row.a_split: row.a, row.b_split: row.b}
        if "train" in ends:
            drop[ends["train"]] = f"near-copy of {row.b_split if row.a_split == 'train' else row.a_split} item " \
                                  f"{ends.get('test', ends.get('dev'))} (J={row.jaccard:.3f})"
        elif "dev" in ends and "test" in ends:
            drop[ends["dev"]] = f"near-copy of test item {ends['test']} (J={row.jaccard:.3f})"
    return drop


def residual_duplication(sample_n=2_000, seed=tiers.SEED):
    """Share of the LLMail near-dedupe survivors that still have a near-copy (exact Jaccard >= 0.8) among the other
    survivors, estimated on a random sample. Measures how much the first dedupe's LSH missed."""
    from datasketch import MinHash, MinHashLSH

    survivors, _ = tiers.load_llmail()
    sets = [shingles(text) or {short_key(text)} for text in survivors["text"]]
    hashes = MinHash.bulk([[g.encode("utf-8") for g in s] for s in sets], num_perm=256, seed=7)
    lsh = MinHashLSH(threshold=0.5, num_perm=256)
    for i, minhash in enumerate(hashes):
        lsh.insert(i, minhash)
    sample = np.random.default_rng(seed).choice(len(sets), sample_n, replace=False)
    hits = 0
    for i in sample:
        a = sets[i]
        hits += any(j != i and len(a & sets[j]) / len(a | sets[j]) >= THRESHOLD for j in lsh.query(hashes[i]))
    share = hits / sample_n
    half_width = 1.96 * np.sqrt(share * (1 - share) / sample_n)
    return {"survivors": len(sets), "sample": sample_n, "share_with_near_copy": share, "ci95": [share - half_width, share + half_width]}


if __name__ == "__main__":
    items, _ = tiers.build_items(purge=False)
    pairs = cross_split_pairs(items)
    print(pairs.groupby(["a_split", "b_split"]).size())
    print(len(purge_list(pairs)), "rows to purge")
