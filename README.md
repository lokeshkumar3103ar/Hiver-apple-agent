# AppleSupport AI agent (Lokesh Kumar A R)

Prototype for the Hiver SDE Intern take-home: classify an AppleSupport tweet, retrieve similar historical resolutions, draft a public reply, and decide whether a person should handle it.

## Submission status

The pipeline is runnable and the 209-row golden set has been manually reviewed. Because a complete LLM run over all 209 rows would require roughly 1,000 API calls, the final end-to-end comparison is a deterministic, stratified 10-row pilot. The small evaluation is reported transparently and its confidence intervals are wide.

| Check | Current state |
|---|---|
| Golden evaluation set | 209/209 rows manually reviewed by Lokesh Kumar A R |
| Two leakage-safe baselines | Evaluated on the same 10-row pilot |
| Full agent and LLM judge | 10/10 rows completed; zero recorded errors |
| Judge/human agreement | 10/10 replies scored by a human |
| Submission validation | `python validate_project.py --submission` |

## Reproduce the headline pilot

Tested with Python 3.11. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the Agent Platform express-mode key:

```env
GOOGLE_API_KEY=your_key_here
```

Build the local 15,000-row retrieval index, run the offline tests, and reproduce the same deterministic sample selection:

```bash
python -m src.rag_retriever --build --source sample
python -m unittest discover -v
python run_eval.py --sample 10 --human-review-size 10 --require-human-labels
streamlit run app.py
```

The evaluation makes approximately 50-60 logical model calls. Model outputs can vary between runs even though row selection is fixed. The complete 79,241-row retrieval index remains optional (`--source full`).

## Measured results

These results come from `data/processed/evaluation_summary.sample-10.json`. The pilot contains one example from each of the ten intents present after human relabeling.

| System | Intent accuracy (95% CI) | Escalation accuracy | LLM reply score |
|---|---:|---:|---:|
| Constant baseline | 10.0% (1.8-40.4) | 60.0% | 3.1/5 |
| TF-IDF nearest neighbor | 40.0% (16.8-68.7) | 60.0% | 2.5/5 |
| Full agent | 90.0% (59.6-98.2) | 70.0% | 4.5/5 |

All three systems were evaluated on identical rows. All 209 golden IDs were excluded from TF-IDF training, and the current row was excluded from semantic retrieval. There were no predictor or judge errors. These numbers are a cost-bounded pilot, not a precise estimate of performance across the 209-row golden set or real AppleSupport traffic.

## Judge/human agreement

A human manually assigned an overall 1-5 score to the ten full-agent replies. Agreement with the LLM judge, recorded in `data/processed/judge_human_agreement.sample-10.json`, was:

| Metric | Result |
|---|---:|
| Exact agreement | 0.800 |
| Within one point | 1.000 |
| Mean absolute error | 0.200 |
| Quadratic weighted kappa | 0.846 |
| Spearman rho | 0.849 |

The sample is too small to establish judge validity conclusively, and the judge uses the same model family as the generation agent. It is evidence that the rubric operated sensibly on this pilot, not proof of general agreement.

To recompute agreement after filling `human_score` in the exported CSV:

```powershell
python run_eval.py --agreement data/processed/human_reply_review.sample-10.csv |
    Tee-Object data/processed/judge_human_agreement.sample-10.json
```

## Evaluation safeguards

- The 209 golden candidates originate in `labeled_subset.csv`; all their IDs are removed from TF-IDF training.
- The current evaluation ID is excluded from RAG results, preventing exact self-retrieval.
- Agent and judge failures are stored and counted; a failed judge never receives an invented neutral score.
- Sample runs use suffixed files and cannot silently overwrite a full evaluation.
- Accuracy includes failed predictions and includes a 95% Wilson interval.
- Escalation precision, recall, and F1 are reported because accuracy alone can hide class imbalance.

## Architecture

```text
customer tweet
  -> Gemini intent classifier (validated JSON and fixed taxonomy)
  -> MiniLM/Chroma retrieval (same-intent historical cases)
  -> Gemini reply draft (retrieved text treated as untrusted data)
  -> conservative rules, then Gemini for ambiguous escalation cases
  -> {intent, confidence, draft, escalate, reason, evidence}
```

Public replies are capped at 280 characters. Account security, data-loss/service-recovery categories, explicit safety/legal signals, and classifier confidence below 0.55 are routed to a person.

## Data and rebuilding

Download `twcs.csv` from the [Customer Support on Twitter Kaggle dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it at `data/raw/twcs.csv`.

```bash
python -m src.data_prep
python -m src.intent_discovery
python create_golden_set.py --overwrite
python review_golden_set.py --reviewer "your-name"
python -m src.rag_retriever --build --source sample
```

Intent discovery searches k=6 through k=13. The best recorded silhouette score is only 0.0316 at k=11, so the selected taxonomy is a routing prototype, not evidence of eleven natural classes. Gemini named whole clusters; the 209 row-level labels and escalation decisions were subsequently reviewed manually.

## Important files

```text
src/settings.py           shared paths, models, and canonical intent taxonomy
src/llm_client.py         validated Agent Platform REST client
src/agent.py              classify -> retrieve -> draft -> escalate
src/rag_retriever.py      index build and leakage-safe query
run_eval.py               evaluation and judge-agreement implementation
baselines.py              constant and TF-IDF baselines
review_golden_set.py      checkpointed human annotation workflow
validate_project.py       artifact and submission-readiness checks
tests/test_core.py        offline unit tests
REPORT.md                 findings, limitations, and next work
DECISION_LOG.md           non-obvious decisions and tradeoffs
```

`.streamlit/config.toml` is intentionally versioned. It disables usage telemetry and a file watcher that conflicts with some PyTorch installations; it is configuration, not generated cache.

## Scope and privacy

This is an offline prototype over public 2017 tweets, not an Apple product and not evidence of current Apple policy. It does not access accounts or send replies. Do not enter real personal information in the demo.

## References

- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- [Sentence Transformers: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Chroma documentation](https://docs.trychroma.com/)
- [Gemini Enterprise Agent Platform express-mode REST API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/express-mode/api-reference)
- [Zheng et al., 2023, Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
