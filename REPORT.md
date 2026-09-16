# AppleSupport AI agent - evaluation report

## 1. Problem framing

The prototype handles one public customer tweet at a time. "Good" means assigning a useful routing intent, drafting a concise response grounded in similar historical AppleSupport interactions, and sending uncertain or sensitive cases to a person with a concrete reason.

Safety matters more than maximizing automation. The agent must not request public account details, promise data recovery, invent policy, or expose internal errors. It does not send messages, access accounts, reconstruct full threads, verify current Apple policy, or claim production readiness.

## 2. Data, taxonomy, and labels

The source is the public Customer Support on Twitter dataset. Preparation finds direct customer tweets answered by `AppleSupport`, minimally cleans text, applies a coarse ASCII-ratio language filter, and deduplicates normalized Apple replies. The resulting artifact contains 79,241 pairs.

Intent discovery embeds a deterministic 15,000-row sample and searches k=6 through k=13 with K-Means. The maximum silhouette score is 0.0316 at k=11. This is weak separation: k=11 is only the best tested candidate, not proof of eleven natural classes. Gemini proposed initial names for entire clusters.

The candidate golden set sampled 19 rows from each machine cluster, producing 209 rows. Lokesh Kumar A R manually reviewed all 209 customer messages, corrected the row-level intent when necessary, and independently labeled whether the case required a human. The final labels occupy ten of the eleven allowed intents; no reviewed row retained `all_apps_crashing`. The golden set is balanced by initial cluster, not by final human label and not by real traffic prevalence.

## 3. Evaluation design

### Systems

1. **Constant baseline:** always predicts `recurring_device_issues`, returns a canned reply, and never escalates.
2. **TF-IDF baseline:** retrieves the nearest labeled customer message using word/bigram TF-IDF and copies its intent and historical reply; a small sensitive-keyword rule controls escalation.
3. **Full agent:** Gemini classification, MiniLM/Chroma retrieval, Gemini drafting, then conservative rules plus Gemini reasoning for ambiguous escalation cases.

### Cost-bounded sample

The final end-to-end run uses a deterministic stratified sample of 10/209 golden rows, one from each final intent represented in the data. A full run would require roughly 1,000 model calls. The smaller run makes the headline protocol reproducible within the assignment's time and API-cost constraints, but it produces wide uncertainty intervals and does not support fine-grained per-intent conclusions.

### Leakage controls and metrics

The golden candidates came from the labeled pool and also exist in the RAG corpus. The harness excludes all 209 golden IDs from TF-IDF training and excludes the current row from each RAG query.

Metrics include intent accuracy and macro F1; escalation accuracy, precision, recall, and F1; 95% Wilson intervals; reply-length compliance; API/error counts; and a five-point reply-quality judge covering relevance, safety, actionability, and style. Judge errors remain missing and are counted instead of being converted to a neutral score.

## 4. Results

Source: `data/processed/evaluation_summary.sample-10.json`.

| System | Intent accuracy (95% CI) | Macro F1 | Escalation accuracy | Escalation F1 | Reply judge |
|---|---:|---:|---:|---:|---:|
| Constant | 10.0% (1.8-40.4) | 0.017 | 60.0% | 0.000 | 3.1/5 |
| TF-IDF, leakage controlled | 40.0% (16.8-68.7) | 0.303 | 60.0% | 0.000 | 2.5/5 |
| Full agent | 90.0% (59.6-98.2) | 0.788 | 70.0% | 0.571 | 4.5/5 |

All systems completed all ten rows, every reply was at most 280 characters, and both predictor and judge error counts were zero. The full agent was substantially better on this pilot, but n=10 is too small to claim that the observed gaps will persist across the full golden set.

### Judge agreement

The same ten full-agent replies received a manual human overall score from 1 to 5. The LLM and human matched exactly on 8/10 and were within one point on 10/10. Mean absolute error was 0.200, quadratic weighted kappa was 0.846, and Spearman rho was 0.849.

This is positive pilot evidence, not conclusive validation: n=10 is small, and the judge and generation model are from the same family.

## 5. Top failure modes

1. **Missing parent context makes a message under-specified.** Tweet `1046603` says only, "That doesn't make much sense. Why not?" The full agent assigned the reviewed intent but unnecessarily escalated it, and its reply received the lowest judge score (2/5). Adding the parent tweet should improve both routing and reply specificity.

2. **Adjacent update intents overlap.** Tweet `1512168` was reviewed as `macos_update_issues`, while the agent predicted `software_update_issues`. The message says only that an OS update caused the issue and does not name the device. The taxonomy asks the model to infer a distinction absent from the text.

3. **Escalation rules miss some frustration language.** Tweet `459544` says prior help failed and the customer is "very angry," but the agent did not escalate. The current frustration list covers words such as "furious" and "unacceptable" but not this phrasing.

4. **A confident routine classification can suppress escalation.** Tweet `1808046` reports several functions broken after an update. The intent was correct, but the agent auto-handled a row reviewed as escalation-worthy. Multi-symptom severity should be considered separately from classification confidence.

5. **Lexical retrieval is brittle and can copy an unsuitable response.** On security-sensitive tweet `53537`, TF-IDF found the correct broad account intent but did not escalate and its copied reply scored 1/5. Intent match alone does not establish that a historical resolution is safe for the current case.

## 6. What is misleading about my headline number?

"90% intent accuracy" sounds much more certain than it is. It is 9 correct predictions from a stratified sample of only 10 rows; its 95% Wilson interval is 59.6%-98.2%. The sample is balanced across observed intents rather than weighted like real traffic, comes from 2017 public tweets, omits parent context, and excludes one allowed intent because no reviewed golden row retained that label. The 4.5/5 judge score also comes from the same model family that drafted the replies. These results demonstrate a working, promising pipeline, not production readiness or 90% real-world accuracy.

## 7. What I would do with one more week

1. Run the full 209-row evaluation and publish per-intent confusion matrices and bootstrap comparisons.
2. Add parent-tweet context and compare single-turn with two-turn performance on a fixed split.
3. Refine overlapping update intents using human disagreement patterns, or merge categories the text cannot support.
4. Expand blinded human scoring and use a different model family as a second judge.
5. Tune escalation against manually defined risk and severity labels, emphasizing recall for security, data-loss, and repeated-failure cases.
