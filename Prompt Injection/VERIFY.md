# Independent verification: DataBench 3 (prompt-injection detection)

Paste everything below the line into a fresh session of another model, with this folder (`Prompt Injection/`)
available to it.

---

You are reviewing a benchmark that scores prompt-injection guards (classifiers, decision models and LLMs) on a
fixed set of texts. The code was written with AI help. Your job is to find mistakes that would change a reported
number. Do not trust the write-up: work from the code and the saved data. Report problems; don't fix the code.

## What to read

- `PLAN.md`: the intended method (data tiers, dev/test rule, metrics). The notebook's first cell lists the
  deviations from it.
- `tiers.py`: builds every item (text, label 1 = attack / 0 = benign, tier, split dev/test/train, dedup group).
- `adapters.py`: one class per guard; each returns a score in [0, 1].
- `runner.py`: runs a guard and caches scores; `matching()` decides which cached rows count.
- `wording_compare.py`: chooses the question wording for the API guards.
- `train_narrow.py`: trains the two in-house guards (TF-IDF + logistic regression, and a fine-tuned ModernBERT).
- `metrics.py`: thresholds, AUROC, recall, false-positive rate, bootstrap.
- `audit.py`: the authors' own checks. Read it, but do your own checks too.
- Data you can use without any API key:
  - `data/items_manifest.csv.gz`: one row per item: `item_id, tier, split, label, group, text_hash, n_tokens,
    truncated`. No text.
  - `cache/<guard>/tier-<tier>.parquet`: every scoring attempt, columns `item_id, text_hash, score, valid,
    latency_ms, input_tokens, output_tokens, n_chunks, raw, error, config_fp, ts`. It is append-only, so it also
    holds superseded rows (other configurations, failed attempts).
  - `harness/round3_harness.json`: the current configuration fingerprint (`config_fp`) of each guard, and the
    exact prompts.
  - `harness/wording_choice.json`, `harness/narrow-v1.json`, `harness/tfidf-v1.json`.
  - `results/summary_full.csv` (one row per guard, every metric), `results/bootstrap.csv`.
- The item texts are not committed. `python -c "import tiers; tiers.build_items()"` rebuilds them from pinned
  Hugging Face commits (a few minutes; it needs `datasets datasketch tiktoken agentdojo`). You only need this for
  check 1.

## The four checks

### 1. Leakage between training, dev and test
Can any test item, or a near-copy of one, appear in the dev split or in the in-house guards' training data?
- Read how LLMail attacks are deduplicated and split (`tiers.load_llmail`, `near_dedupe`, `llmail_items`), how the
  PromptShield splits are used (`promptshield_items`), and what `train_narrow.training_data` trains on.
- With the rebuilt items, test it directly: exact text overlap between `split == "train"` and dev/test, overlap
  after the code's own `normalise()`, and any dedup `group` that appears in more than one split.
- Look for subtler leaks too: an evasion item (tiers 5 / 5b) built from a train or dev item, or AgentDojo / custom
  items that share text with training data.

### 2. Is anything chosen using test data?
The rule: anything tuned (question wording, thresholds, the in-house models' settings and epoch) is chosen on dev
only, and test is scored once.
- `wording_compare.py` and `harness/wording_choice.json`: which items were used to pick the wording?
- `metrics.thresholds()` and every call to it in `metrics.py` and the notebook: which scores set each threshold?
  One column is deliberately chosen on test (labelled "oracle"). Confirm it is the only one, and that nothing
  else uses it.
- `train_narrow.py`: which data picks the TF-IDF `C` and the ModernBERT checkpoint?

### 3. Can a stale or wrong cached score be used?
- In `runner.matching()`: a cached score should count only if its `config_fp` equals the guard's current
  fingerprint, its `text_hash` equals the item's current hash, and `error` is empty, keeping the latest such row
  per item.
- Does the fingerprint (`Arm.config()` in `adapters.py`) cover everything that changes a score: model id and
  revision, the full prompt or question, the schema, chunking? Is anything that changes a score left out?
- Check the data: for each guard, are there cached rows under other `config_fp` values, and could any of them
  reach the results?
- How are invalid answers (refusals, unparseable output) scored, and is that consistent with `PLAN.md`?

### 4. Recompute headline numbers from the saved scores, with your own code
Don't import the project's modules for this check. Use pandas and sklearn on the manifest and the cache files only.
For each guard in `results/summary_full.csv` (Prompt Guard 2 was not run):
- Keep cache rows with the guard's fingerprint from `harness/round3_harness.json`, empty `error`, and `text_hash`
  equal to the manifest's; take the latest `ts` per `item_id`. Confirm every dev and test item has exactly one score.
- Recompute and compare with `results/summary_full.csv`:
  - `AUROC: LLMail vs LLMail benign emails`: test items, tier `1` (label 1) against tier `3e` (label 0).
  - `AUROC: AgentDojo attack vs its benign`: tier `4`, label 1 against label 0.
  - `t_1pct` from the 3,000 dev benign scores (`split == "dev"`, `label == 0`), per the definition in
    `metrics.thresholds`; then `recall LLMail @t_1pct` (tier 1) and `FPR NotInject @t_1pct` (tier 2), where an
    item is flagged when its score is above the threshold.
- Say whether the threshold definition matches PLAN.md's "the score at which 1% of dev benign is flagged",
  especially when many scores are tied (the LLM scores are whole numbers out of 100).

## What to report

For each check, give:
1. **Verdict**: no problem found / problem found / can't tell (and what you'd need to tell).
2. **Evidence**: file and line, plus the numbers you computed (for check 4, a table of yours vs the CSV for every
   guard and metric).
3. **Impact**: for any problem, which reported numbers it would change, and roughly by how much.

Then list anything else you noticed that would change a headline number, most serious first. Leave out style
comments and refactoring suggestions. If everything passes, say so plainly; don't invent problems to have
something to report.
