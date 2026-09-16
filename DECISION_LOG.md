# Decision log

1. **Use AppleSupport.** It has enough direct response pairs for retrieval and a recognizable public-support style. The artifact contains 79,241 cleaned, reply-deduplicated pairs.

2. **Model direct customer-to-brand pairs, not full threads.** This keeps the prototype small, but it is a known limitation: 52/209 evaluation candidates reply to an earlier tweet.

3. **Discover a routing taxonomy from a 15,000-row sample.** This makes the initial categories data-informed and keeps clustering practical on a laptop.

4. **Keep k=11 only as the best searched candidate.** Its 0.0316 silhouette score is weak. The project records that limitation instead of describing 11 as proven natural intents.

5. **Treat Gemini cluster names as proposals.** Whole-cluster naming is not row-level annotation. `review_golden_set.py` creates the required human audit trail.

6. **Start with a balanced 209-row diagnostic set.** Nineteen rows per machine cluster made the initial audit broad. Human relabeling changed the class distribution and left one proposed intent empty, so the final set is not described as class-balanced.

7. **Exclude evaluation IDs from retrieval and training.** Golden rows originate in the same sampled corpus, so exact-row exclusion is mandatory for a fair nearest-neighbor comparison.

8. **Use a constant and a TF-IDF baseline.** The constant baseline exposes class balance; TF-IDF tests whether semantic embeddings and generation add value beyond a simple lexical neighbor.

9. **Use local MiniLM embeddings and Chroma.** They avoid embedding API cost and keep retrieval data local. The default reviewer build uses 15,000 rows; `--source full` is an optional higher-cost run.

10. **Centralize models, paths, and intents.** `src/settings.py` prevents taxonomy drift and makes commands independent of the caller's current directory.

11. **Constrain generation and validate structured output.** Customer/retrieved text is explicitly untrusted, JSON responses are type/range checked, and public drafts are capped at 280 characters.

12. **Escalate conservatively before asking an LLM.** Explicit security/safety signals, high-risk intents, and confidence below 0.55 route to a person. The LLM handles only ambiguous middle cases.

13. **Never turn evaluation errors into average scores.** Missing agent or judge outputs remain missing, are counted, and are visible per row.

14. **Use a rubric judge and measure human agreement.** The judge scores relevance, safety, actionability, and style. A human scored all ten pilot replies; exact agreement was 0.80 and weighted kappa was 0.846. The small sample and same-family bias are disclosed.

15. **Version `.streamlit/config.toml`.** Streamlit does not need to generate it. The file intentionally avoids a PyTorch watcher conflict and disables telemetry; generated Streamlit cache remains ignored.
