# DataBench 2: one fixed question, the newest models

Written 2026-10-19. Merges pitches 2 and 3 from `NEXT_BENCHMARKS.md` into one run: redefine the identity question,
then score every current model on the same 1,000 test chats.

Round one (`02_benchmark.ipynb`, `call_log.csv`) stays frozen as published. Round two gets its own notebook and log.

## Where we stopped (2026-10-20)

1. **Dev is complete:** 8 API models x 4 wordings x 300 chats. **v3 is frozen.** The rule change is disclosed in
   section 10 and the Caveats.
2. **Next: the test run.** 1,000 chats x 8 API models with v3, then the probe and repeat control, then the
   secondary arms (v0, v2) and the evidence arm (3 LLMs). Roughly 21,000 calls, about $8-10. Cloudflare's pace of
   3.1 s per call makes Clef the slowest, about 1.5 h per arm.
3. **Not yet:** the local Clef-flash. It needs a pinned revision, a read of `joint_schema_model.py` and a
   quantization setting.
4. Nothing is committed: `04_round_two.ipynb`, `DATABENCH_2.md` and `call_log_round2.csv` (10,378 dev rows, including the logged 429 failures).

## Reddit feedback: what round two tests (added 2026-10-20)

Two commenters on the round-one post. Each suggestion becomes a test, so both replies can point at numbers.

| # | Suggestion | Who | What round two does |
|---|---|---|---|
| 1 | Reword to "did the agent confirm the details matched the account" | avisangle | **v2 wording**, tested on dev as written |
| 2 | Add one example of a request that is not verification | avisangle | folded into v2's "no" description (one example) |
| 3 | "Prompt fix or model limit?" | avisangle | ceiling analysis on train (below): partly both |
| 4 | Post before/after on identity per model | avisangle | already built: section 13 + figure 11 |
| 5 | Make the model quote the exact line it bases "verified" on; check the quote against the transcript | xapep | **evidence arm**, separate from the wording test (see below) |
| 6 | False-yes rate per dollar as a column | xapep | add "false yeses per 1,000 chats" next to "cost per 1,000 chats" |
| 7 | Caching under 4k tokens drives cost | xapep | Haiku 4.5 retired; every cheap model now caches (Haiku 5.5 hit 89%) |
| 8 | Pin the harness: prompts, max tokens, temperature, system prompt | xapep | fingerprints cover it; also export each version's rendered prompt to a committed `harness/` folder. Temperature is never set (provider default), and that is recorded |
| 9 | Nerf-tracking tracks, dataset licences | xapep | future benches (`NEXT_BENCHMARKS.md`). ABCD is MIT-licensed at the pinned commit, so commercial use is OK |

**Ceiling (train split only, 8,034 chats), the answer to #3:**
- The store's own policy, knowing the true request type, gets F2 0.853 (163 false alarms, 441 misses). F2 is a 0-1
  score where a miss costs 4x a false alarm.
- The best possible rule on request type alone gets 0.859, so about 7% of chats break policy and can't be called
  from request type.
- Clues in the agent's lines (verified vs not): says "verify" 31% vs 4%, asks zip code 20% vs 1%, asks phone 21% vs
  2%, asks account ID 76% vs 24%, asks order ID 77% vs 29%.
- So it's not rigged, but part of it is a model limit: the label is mostly store policy plus some visible clues, and
  no zero-shot model is told the policy.

**Wordings left (two rewrites):**
- **v2, avisangle's:** "Did the agent confirm that the customer's details matched the account?"
  - yes: the agent confirmed the details matched the account.
  - no: e.g. the agent asked "May I have your full name?" and then answered a general question; asking for details
    is not verification.
- **v3, policy + clues (from `kb.json` and the train clues above, never dev or test):** agents verify identity before
  account changes, subscription and billing questions, order status or changes, price disputes and username
  recovery, usually asking for zip code, phone, or account ID plus order ID. For refunds, returns, missing items,
  promo codes and password resets they only look the account up by name, which is not verification.
- Run both on the same 300 dev chats. Freeze the best mean F2 (and show false alarms and misses).

**Evidence arm (xapep's check), run after the wording is frozen:**
- The LLMs get one extra output field: the exact transcript line their identity answer rests on.
- A "yes" whose quote isn't found verbatim in the transcript is overridden to "no".
- Measured: false yeses removed vs true yeses lost, per model, on test, against the same models without the field.
- Decision models can't quote text, so they're the comparison: a different way to get the same reliability.
- Prediction to check, not assume: round one's false yeses came from real lines ("May I have your full name?"), so a
  verbatim match alone may remove few of them. The test settles it.

## 1. The question change: identity verification

**Why.** Round one's biggest error was definitional. The truth is the agent's `verify-identity` action, but the
question asked whether the agent "verified the customer's identity using their details". Models said yes whenever
the agent asked for a name or an order ID.

**What the log actually distinguishes** (checked on the pinned dataset, 2026-10-19):

| Action | Values entered | What it is |
|---|---|---|
| `verify-identity` | always **3**: full name plus two of account ID, order ID, zip code, phone | the truth for this question |
| `pull-up-account` | **1**: a name or an account ID | account lookup, not verification |
| `validate-purchase` | username, email, order ID | checks a purchase, not the person |

**366 of the 1,000 test chats** have a lookup or purchase validation but no `verify-identity`. That matches round
one's 378-471 false alarms per LLM.

**Draft v1 wording** (dev only until frozen):

```python
"identity_verified": {"type": "noul",
    "instructions": "Did the agent run the store's formal identity verification, checking three of the customer's "
                    "details together: their full name plus two of account ID, order ID, zip code or phone number?",
    "criteria": {"true": "Formal identity verification was done: three details checked together",
                 "false": "No formal verification. Pulling up the account from only a name or account ID does not "
                          "count, and neither does validating a purchase with username, email and order ID"}},
```

**Rules for the change.**
- Only `identity_verified` changes. The other six questions keep round one's exact wording.
- Iterate on a fixed 300-chat **dev** sample (seed 42), at most 3 versions. v2 and v3 are written only after
  reading v1's dev errors. Record every version and its dev scores.
- Freeze the winner (best mean F2 across all models, not one model's best), then score **test** once.
- All questions are asked in one call, so the new wording could move other answers. Check that too (section 4).

### Dev results, v0 vs v1 (2026-10-19, 300 dev chats)

| Model | F2 v0 | F2 v1 | False alarms v0 → v1 | Misses v0 → v1 |
|---|---|---|---|---|
| Claude Haiku 5.5 | 0.835 | 0.827 | 101 → 6 | 3 → 23 |
| GPT-6 Luna | 0.832 | 0.865 | 103 → 10 | 3 → 17 |
| Gemini 3.5 Flash-Lite | 0.833 | 0.712 | 102 → 2 | 3 → 39 |
| Jev 1.13 | 0.769 | 0.032 | 60 → 0 | 21 → 114 |
| Jev 1.14 | 0.846 | 0.165 | 82 → 0 | 5 → 101 |
| Metis (GreenPT) | 0.843 | 0.594 | 94 → 7 | 3 → 53 |
| Clef | 0.819 | (v1 incomplete) | 99 → ? | 6 → ? |

**v1 overcorrected.** Mean F2 fell from 0.825 to 0.533: false alarms nearly vanished, but misses rose about 9x. The other
six questions didn't move (mean headline 0.866 under both).

**Why: "three details" isn't in the transcript.** The three values in `verify-identity` are filled in from the
agent's system. In many verified chats the customer gives only a name (dev convos 964, 503, 930). The wording
described the action's inputs, not anything the models can see.

**What actually predicts the label (train split only):** the request type. ABCD's action policy (`kb.json`) lists
`verify-identity` for 22 intents: account changes, subscription and billing status, order status and changes, price
disputes, username recovery. It lists only `pull-up-account` / `validate-purchase` for 21 others: refunds, returns,
missing items, promo codes, password and 2FA resets, service status. Predicting the label from the request type alone
is right on 92.6% of train chats. So `identity_verified` is mostly a policy step, not a visible event.

### Dev results, v2 and v3 (2026-10-20, same 300 dev chats, all 8 API models)

F2 = 0-1 score for yes/no labels where a miss costs 4x a false alarm. Counts are out of 300 dev chats.

| Model | F2 v0 / v1 / v2 / v3 | False alarms v0 / v1 / v2 / v3 | Misses v0 / v1 / v2 / v3 |
|---|---|---|---|
| Claude Haiku 5.5 | 0.835 / 0.827 / 0.407 / **0.844** | 101 / 6 / 29 / 34 | 3 / 23 / 73 / 15 |
| GPT-6 Luna | 0.832 / 0.865 / 0.577 / **0.885** | 103 / 10 / 30 / 20 | 3 / 17 / 52 / 12 |
| Gemini 3.5 Flash-Lite | 0.833 / 0.712 / 0.811 / **0.864** | 102 / 2 / 74 / 30 | 3 / 39 / 12 / 13 |
| Jev 1.13 | **0.769** / 0.032 / 0.274 / 0.712 | 60 / 0 / 15 / 8 | 21 / 114 / 89 / 38 |
| Jev 1.14 | **0.846** / 0.165 / 0.281 / 0.778 | 82 / 0 / 19 / 4 | 5 / 101 / 88 / 30 |
| Metis (GreenPT) | 0.843 / 0.594 / 0.450 / **0.858** | 94 / 7 / 28 / 15 | 3 / 53 / 68 / 17 |
| Clef | 0.819 / 0.847 / 0.551 / **0.869** | 99 / 42 / 33 / 17 | 6 / 13 / 55 / 15 |
| Clef-flash (Cloudflare) | **0.838** / 0.494 / 0.435 / 0.830 | 93 / 6 / 25 / 14 | 4 / 65 / 70 / 21 |
| **Mean, 8 models** | 0.827 / 0.567 / 0.473 / **0.830** | 91.8 / 9.1 / 31.6 / 17.8 | 6.0 / 53.1 / 63.4 / 20.1 |

- **v2 (avisangle's wording) backfired too:** mean F2 0.467. "Confirm the details matched the account" made
  models wait for an explicit confirmation that agents rarely say.
- **v3 (policy + clues) ties v0 on mean F2** (0.824 vs 0.825) but makes **60% fewer errors**: about 39 per model
  vs 98. It trades about 73 false alarms for about 15 extra misses, and F2 counts each miss 4x. Every LLM and
  Metis score higher on v3; both Jevs score lower (they miss more).
- The other six questions didn't move under any version (mean headline 0.864-0.866).
- **What's still wrong under v3 is mostly the ceiling:** chats where the customer gives only a name and the log
  still says verified (dev 503, 930, 964, 3477), and one where the agent says "your data has been verified" but the
  log says not verified (3651). No wording can fix those.
- **v3's error types (6 models, 300 dev chats, 117 really verified):** per model, about 18.5 false alarms
  (precision errors) and 20.8 misses (recall errors), where v0 had about 92 and 6. The LLMs lean to false alarms
  (Haiku 34, Flash-Lite 30, Luna 20); the Jevs lean to misses (38 and 30).
  - False alarms: 91% are chats where policy says no verification. In about half of them the text has a strong clue
    ("verified", zip, phone, or account ID plus order ID), e.g. purchase validation phrased as "your data has been
    verified".
  - Misses: 59% are chats where policy says verify and the model didn't apply it. 41% are chats where the agent
    verified against policy; that part is mostly unknowable.
  - Reference: the policy rule alone, given the true request type, makes 2 false alarms and 16 misses on these 300
    (F2 0.884), better than every model. The models lose the rest by misreading the request type or following a
    misleading clue.
- **Frozen: v3** (2026-10-20, before any test call). With 6 models, the rule as written ("best mean F2") picked v0
  by 0.001, so a tie-break was added after seeing those results: within 0.005 is a tie, and fewer total errors
  wins. This is disclosed in the notebook's section 10 and Caveats. With all 8 models, v3 has the best mean F2
  outright (0.830 vs 0.827), so both rules pick it. v3: about 38 errors per model, vs 98 for v0.
- **Declared secondary test arms:** v0 (same-day control) and v2 (avisangle's wording), shown next to the
  headline in section 13a.

## 2. The models

| Model | Status | Model ID | Price $/1M (in / out) | Notes |
|---|---|---|---|---|
| Claude Haiku 5.5 | kept | `claude-haiku-5-5` | 0.10 / 0.50 | still Anthropic's newest budget model |
| GPT-6 Luna | kept | `gpt-6-luna` | 0.10 / 0.50 | still OpenAI's cheapest (launched 2026-09-22) |
| Gemini 3.5 Flash-Lite | kept | `gemini-3.5-flash-lite` | 0.30 / 2.50 | still the newest Flash-Lite (2026-07-21) |
| Jev 1.13 | re-run | `jev-1.13.0` | 0.042 / 0 | re-run on the new wording so 1.13 vs 1.14 is a clean comparison |
| **Jev 1.14** | new | `jev-1.14.0` (pin it; `jev-latest` moves to it ~2026-10-21) | 0.042 / 0 (unchanged, per MarkTechPost) | released 2026-10-14; TypeSafe says the biggest gain is multilingual; leave the experimental "danger zone" headers (images, box, long context) off |
| **Clef** | new | `@cf/cloudflare/clef` (27B) | 0.24 / 0 (Cloudflare page, 2026-10-19) | Cloudflare Workers AI, Jev-compatible API, released 2026-10-01 |
| **Clef-flash** | new | `@cf/cloudflare/clef-flash` (9B) | 0.038 / 0 (Cloudflare page, 2026-10-19) | the price cut may have shortened hosted context to 24,576 tokens; still fits (~2.7k per call) |
| **Metis** | new | `metis` | €0.12 / 0 ≈ $0.135 (GreenPT API + ECB 1.1225, 2026-10-19) | GreenPT. **Clef-flash at full precision**, EU-only, renewable-powered, reports energy per call |
| **Clef-flash (local)** | new | `Cloudflare/clef-flash`, pinned revision | $0 | same weights on the 5070 Ti, quantized (see section 2b) |
| Claude Haiku 4.5 | retired | | | 20x the cost of Haiku 5.5 |
| Laya | dropped | | | near chance untuned, by its own docs; return only fine-tuned (home run) |

New-model prices confirmed 2026-10-19: Clef and Clef-flash on Cloudflare's Workers AI pricing page; Metis via
GreenPT's `GET /v1/pricing/metis` (`"promptToken": 0.12`; GreenPT's docs quote API prices in EUR per 1M tokens).
Per 1,000 chats at ~2.7k tokens: Jev ~$0.10, Clef-flash ~$0.10, Metis ~$0.36, Clef ~$0.64. Metis costs ~3.5x
Clef-flash on Cloudflare for the same weights.

**Clef wiring.** POST `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef`,
bearer token, body `{"state": ..., "questions": ...}` (Jev's System One shape). Reuse `SYSTEM_ONE_QUESTIONS` and
`parse_system_one_answers`. New env vars: `CLEF_KEY`, `CF_ACCOUNT_ID`. The request example is from a blog post that
says it was not run: the smoke test decides.

**Metis wiring** (from [GreenPT's System One docs](https://docs.greenpt.ai/systemone)):
- `POST https://api.eu.greenpt.ai/v1/systemone` (also listed: `api.greenpt.ai`), `Authorization: Bearer ...`,
  body `{"model": "metis", "state": ..., "questions": ...}`. Same question shape as Jev. New env var: `METIS_KEY`.
- **EU-only:** the US endpoint returns `403 endpoint_not_available_in_location`. Smoke test from home first; if
  blocked, run Metis from the cloud environment in an EU region or skip it.
- Response: `answers` (Jev shape), `usage.input_tokens` (output always 0), and `impact`: `inferenceTime`,
  `energy`, `emissions`, `version`. Log all four as new columns.
- Limits: 32 questions, 32,768 tokens, 600 requests per 5 minutes (sequential at ~1/s fits).
- TypeSafe's SDK works if both base URL and default model are set; set the timeout to 60 s. Plain `requests` is
  the fallback.
- **GreenPT says the model behind `metis` can change without the ID changing.** Log `impact.version` and the
  response `model` on every call; if either changes mid-run, stop and re-run that model.

### 2b. Same weights, three hosts

Clef-flash is Apache 2.0, so the exact same 9B model runs three ways. That is its own question: **does where you
run a model change its answers?**

| Host | Precision | Cost | Energy |
|---|---|---|---|
| Cloudflare Workers AI | not stated (ask or infer from agreement with Metis) | $0.09/1M (or $0.038) | not reported |
| GreenPT Metis | full precision (BF16), per GreenPT | GreenPT pricing | reported per call |
| Local, 5070 Ti | **quantized**: BF16 needs ~18-21 GB; 13 GB free today | $0 | measure: sample `nvidia-smi` power during the run, GPU only |

Local options, in order:
1. The model card's release loader (`load_release_model` + `systemone()` from `joint_schema_model.py`, pinned
   revision), loaded in 8-bit if it accepts a quantization config. Read `joint_schema_model.py` before running it.
2. SGLang with `/v1/systemone` (the card's main path) under WSL/Docker, with a quantized checkpoint.
3. Community INT8 (`Aikimi/clef-flash-int8`): not an official release; last resort, pinned and inspected first.

**Metrics:** per-label agreement between each pair of hosts on the 1,000 test chats, mean absolute difference in
P(yes), accuracy differences (permutation tests), latency, cost, energy per 1,000 chats. Expect Cloudflare and
Metis to agree almost exactly if both run BF16; local differences measure quantization, not the host.

## 3. The run

| Step | What | Calls | Est. cost |
|---|---|---|---|
| 1 | Smoke test: 1 dev chat per model | 9 | ~$0 |
| 2 | Dev iterations: 300 dev chats x up to 3 wordings x 9 models | up to 8,100 | ~$3 |
| 3 | Test: 1,000 chats x 9 models, frozen wording | 9,000 | ~$3 |
| 4 | Memorization probe: 200 perturbed chats x 9 models | 1,800 | ~$1 |
| 5 | Repeat control: the same 200 originals again x 9 models (finally splits probe flips from noise) | 1,800 | ~$1 |

**Total: about $8-10**, about 21,000 calls. At ~1-2 s a call, one API model is about 45-75 minutes; run providers
in parallel (one model per command), commit `call_log_round2.csv` after each model. The local run happens at home,
not in the cloud.

Spend caps per key (one-day keys, as in round one): Anthropic $3, OpenAI $3, Google $5, TypeSafe $3, Cloudflare $3,
GreenPT $3 until its price is known.

## 4. Scoring

- Same scorers, permutation tests (exact when 16 or fewer answers differ), Bonferroni.
- **Headline 1, the fix:** identity F2, false alarms and misses per model, round one wording vs round two, paired
  on the same 1,000 chats.
- **Headline 2, the leaderboard:** every question, every model, round-two wording.
- **Collateral check:** for the six unchanged questions, paired test of round one vs round two answers per model.
  Anything significant gets reported, not explained away.
- **Version check:** Jev 1.13 vs 1.14 on identical inputs.
- **Thresholds for decision models chosen on dev** (fixes audit finding 6), then applied to test once.
- Audit pass (`audit_results.py` extended to the new log) before anything is published.

## 5. Build checklist

Built (2026-10-19): `04_round_two.ipynb`, with 02 left untouched. It has the identity wordings `v0`/`v1`,
`call_log_round2.csv`, `figures/round2/`, the 9 models with per-provider fingerprints, the Cloudflare / Metis /
local callers, the dev section, before/after, three hosts, the repeat control and dev-chosen thresholds. Tested end
to end with fake models.

**Smoke test, 2026-10-19 (real calls, one dev chat each, not logged):** Clef, Clef-flash (Cloudflare) and Metis all
answered from home, and all parsed with 0 invalid answers. Metis's EU endpoint works from the US. Each reported 2,684
input tokens. Latency was ~1.0-1.1 s per call; Metis's own `inferenceTime` was 203 ms, so most of that is network.
All three gave the same labels (and all missed the truth on identity and resolution for that one chat).
- Cloudflare rounds probabilities to 4 decimals; GreenPT doesn't.
- Metis impact: `{"energy": {"total": 47467, "unit": "Wms"}}` = 0.013 Wh per call, plus `emissions` in µgCO2e.
  The notebook now converts energy to Wh.
- Served model names: `clef`, `clef-flash`, `metis`; impact version `20250922`.

Before the first real run:
1. Keys: done (`CLEF_KEY`, `METIS_KEY`, `CF_ACCOUNT_ID`, plus round one's four).
2. Prices: done (Clef $0.24, Clef-flash $0.038, Metis €0.12 ≈ $0.135 per 1M input tokens).
3. Local model: pin `PROVIDER_SETTINGS["clef_local"]["revision"]`, read `joint_schema_model.py` at that revision,
   and set a quantization option in `load_kwargs` (BF16 doesn't fit in 16 GB).
4. Smoke test: all 8 API models pass (2026-10-19), 0 invalid answers. Jev 1.14 took 0.33 s and gave the same
   labels as 1.13. Haiku 5.5 and Luna took ~4 s on this first, cache-writing call. On this one chat (truth:
   identity verified, resolution "changed something"), only Luna said "verified" under v1; every model said
   "information only". One chat proves the wiring, not the wording. Local model after item 3.
5. Dev (`RUN_DEV = True`), then read v1's errors. Write v2 only if needed. Set `FROZEN_IDENTITY_VERSION`.
6. Test, probe, repeat control. Commit `call_log_round2.csv` after each model.
7. Extend `audit_results.py` to the round-two log before publishing.

## Open decisions

- Codename Metis until the reveal, as planned for "the EU model"? The "it's secretly Clef-flash" reveal is a
  natural video beat.
- Run the API models locally or in the cloud environment (`NEXT_BENCHMARKS.md` checklist)? Metis's EU-only rule
  may decide this.

Sources for model facts: [Jev 1.14 on OpenRouter](https://openrouter.ai/typesafe/jev-1.14),
[Clef on Developers Digest](https://www.developersdigest.tech/blog/cloudflare-clef-decision-models-2026),
[Clef-flash price cut (techenclave)](https://techenclave.com/news/items/cloudflare-adds-clef-omni-and-cuts-clef-flash-pricing-with-a-shorter-hosted-context),
[Metis launch (Dutch IT Channel)](https://www.dutchitchannel.nl/news/762301/greenpt-lanceert-europese-ai-beslismodel-metis),
[GPT-6 Luna launch](https://www.silicon.co.uk/ai-2/openai-sol-luna-gpt-6-631684),
[Gemini 3.5 Flash-Lite](https://geotoolbox.ai/blog/gemini-3-6-flash-vs-3-5-flash-lite).
