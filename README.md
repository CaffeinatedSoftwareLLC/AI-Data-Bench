# AI Data Bench

Business-relatable AI benchmarks with real ground truth, from Caffeinated Software LLC. Each benchmark asks a
question a company would actually ask (who's about to quit? what happened on this support call?) and scores cheap
LLMs, decision models and plain old ML on the same data, for accuracy, cost and speed.

Every model call is logged and fingerprinted, and every notebook re-scores from the logs without calling an API.
Anyone can check the numbers.

Not sponsored by any of the model providers tested.

---

## The benchmarks

| Folder | Question | Contenders | Notebooks |
|---|---|---|---|
| `IBM Attrition/` | Which employees are about to leave? | Random Forest vs Claude Sonnet 5, GPT-6 Sol, Gemini 3.8 Flash vs Jev | `attrition_bench.ipynb` |
| `ABCD Call Labeling/`, round one | Label 1,000 support chats: why they called, what the agent did, how far it went | Claude Haiku 4.5 and 5.5, GPT-6 Luna, Gemini 3.5 Flash-Lite, Jev 1.13, Laya, a trained TF-IDF model, a regex | `01_explore`, `02_benchmark`, `03_haiku_comparison` |
| `ABCD Call Labeling/`, round two | Same chats, one question fixed, newer decision models | Haiku 5.5, Luna, Flash-Lite, Jev 1.13 and 1.14, Cloudflare's Clef and Clef-flash, GreenPT's Metis | `04_round_two` (plan and results in `DATABENCH_2.md`) |
| `Prompt Injection/` | Does this email, tool output or message contain instructions aimed at the AI? | ProtectAI v2, PIGuard, Jev 1.14, Metis, Haiku 5.5, Luna, our fine-tuned ModernBERT ("Narrow"), TF-IDF | `01_injection_bench` (plan in `PLAN.md`) |

## Round two at a glance

Round one's biggest error was a definition. LLMs read "may I have your name?" as identity verification, about 390
false alarms each. Round two rewrote that one question, chose the wording on held-out dev chats only, then scored the
1,000 test chats once.

Identity F2 is a 0-1 score where a miss counts 4x a false alarm.

| Model | Identity F2, round one → two | False alarms | Misses | Cost per 1,000 chats |
|---|---|---|---|---|
| GPT-6 Luna | 0.807 → 0.908 | 389 → 53 | 4 → 27 | $0.13 |
| Claude Haiku 5.5 | 0.803 → 0.904 | 394 → 96 | 5 → 19 | $0.17 |
| Metis (GreenPT) | new: 0.900 | 26 | 36 | $0.35 |
| Clef (Cloudflare) | new: 0.899 | 42 | 33 | $0.63 |
| Clef-flash (Cloudflare) | new: 0.893 | 30 | 38 | $0.10 |
| Gemini 3.5 Flash-Lite | 0.810 → 0.849 | 381 → 94 | 4 → 43 | $0.68 |
| Jev 1.14 | new: 0.760 | 6 | 96 | $0.10 |
| Jev 1.13 | 0.775 → 0.694 | 198 → 6 | 55 → 121 | $0.10 |

- The fix worked for the LLMs and the Clef family. Jev became too cautious at its default cut-off, but a threshold
  chosen on dev recovers it (95% precision).
- A model trained on 8,006 hand-labelled chats still leads (0.922). Part of this label isn't in the text at all: it
  follows store policy.
- Two reader suggestions from Reddit were tested too: a rewording, and a "quote your evidence" check. Neither helped
  much; the notebook has the numbers.
- Metis is Cloudflare's Clef-flash weights hosted by GreenPT. The two hosts agree on 95.7-100% of labels.

The full write-up, including what was changed along the way and why, is in `ABCD Call Labeling/DATABENCH_2.md` and
the notebook's Caveats.

## Prompt injection at a glance

24,705 items per guard: real LLMail-Inject attack emails, LLMail's own benign emails, PromptShield, NotInject
(benign prompts full of words like "ignore"), AgentDojo tool outputs, five evasion tricks, and 40 items of our own.
Thresholds chosen on dev, test scored once. AUROC is 0.5 for a coin flip and 1.0 for a perfect split.

| Guard | Attack emails vs benign emails (AUROC) | Attacked vs clean tool outputs (AUROC) | Benign "ignore"-style prompts flagged | Recall lost to base64 / rot13 | $ per 1,000 |
|---|---|---|---|---|---|
| Jev 1.14 | 0.999 | 1.000 | 12.7% | none* | $0.025 |
| Metis (GreenPT) | 0.998 | 1.000 | 19.8% | 74 / 70 points | $0.057 |
| ProtectAI v2 (local) | 0.994 | 0.735 | 48% | none* | $0 |
| PIGuard (local) | 0.990 | 0.931 | 10.3% | none* | $0 |
| Claude Haiku 5.5 | 0.964 | 1.000 | 8.6% | 47 / 94 points (API refusals) | $0.078 |
| GPT-6 Luna | 0.950 | 0.999 | 5.3% | none* | $0.087 |

\* "None" partly by flagging encoded text in general: ProtectAI and PIGuard flag 100% of *benign* base64, Jev 22%,
Luna 26%.

- The plan's own question ("does it try to give instructions to an AI?") flagged up to 69% of benign prompts. A
  wording that says what isn't an injection, chosen on dev, fixed it for every model.
- Our fine-tuned model scored 0.99999 on dev and then flagged every benign email: it learned the format, not the
  attack.
- Checked by a second model working only from the code and saved scores (`Prompt Injection/VERIFY.md`, report in
  `review/`). It reproduced every number it recomputed and found near-copies across the training/dev/test split;
  those rows were removed from training and dev (test unchanged) and our two in-house guards retrained.
- Prompt Guard 2 joins when Meta approves the licence. Write-up: the notebook's Findings and Caveats.

## How every benchmark is run

- **Same input for everyone:** one conversation (or employee) per call, identical wording and option descriptions,
  structured outputs. Nothing is tuned per model.
- **Out of the box:** no tools, no temperature or system prompt set, and reasoning left at each provider's default
  unless a notebook says otherwise.
- **Tune on dev, score test once.** Anything chosen (wordings, thresholds) is chosen on a dev split. Rule changes are
  disclosed in the notebook.
- **Honest statistics:** invalid answers count as wrong. Paired permutation tests (exact when 16 or fewer answers
  differ), with a Bonferroni correction across each family of comparisons.
- **Fingerprinted logs:** every call row carries a hash of the full request configuration and of the exact input.
  Notebooks only score rows that match, so a changed prompt can't silently reuse old answers. The exact prompts for
  round two are in `ABCD Call Labeling/harness/`.

## Reproduce it

1. Python 3.13. The first code cell of each notebook installs what it needs.
2. Open a notebook and run it top to bottom. Every `RUN_*` flag is `False`, so it re-scores from the committed logs
   (`call_log.csv`, `call_log_round2.csv.gz`, which pandas reads directly) **without calling any API**.
3. To make new calls, set the API keys as environment variables (`CLAUDE_KEY`, `GPT_KEY`, `GEM_KEY`, `JEV_KEY`,
   `CLEF_KEY` + `CF_ACCOUNT_ID`, `METIS_KEY`), turn on the smoke test first, then the run you want. Runs resume where
   they stopped.

Datasets download on first run and are cached outside the repo:
- **ABCD** (Action-Based Conversations Dataset, ASAPP, NAACL 2021): pinned to one commit of
  `asappresearch/abcd`, MIT licence.
- **IBM HR attrition data:** from Kaggle (`pavansubhasht/ibm-hr-analytics-attrition-dataset`) through `kagglehub`,
  never committed. No model ever sees a real employee: the committed rows in `IBM Attrition/synthetic_data/` come
  from a generator trained on it. Check the dataset's Kaggle page for its terms.

## What's next

`NEXT_BENCHMARKS.md` lists the pitches: prompt-injection security, bigger models and more thinking, consistency
across runs, few-shot vs hand labels, a second dataset, and calibration.

## Licence

Code and notebooks: MIT (`LICENSE`). Datasets keep their own licences (above). Model names are their owners'
trademarks.
