# Golden-set sampling and labeling notes

## Source and sampling

- Source: Customer Support on Twitter, filtered to direct customer messages answered by `AppleSupport`.
- Prepared corpus: 79,241 deduplicated customer/reply pairs.
- Candidate pool: the 15,000-row clustering sample in `data/processed/labeled_subset.csv`.
- Initial sample: 19 rows from each of 11 machine clusters (209 rows), after requiring customer text longer than 30 characters and reply text longer than 20 characters.
- Random seed: 42.

The initial sample is balanced by machine cluster. After row-level relabeling, the final human-label distribution is not balanced and should not be interpreted as production traffic prevalence.

## Label construction

K-Means provided each candidate's initial cluster. Gemini suggested one name per whole cluster, so the initial `intent` was only a machine proposal. The initial `gold_escalate` value was a proxy based on whether the historical public response requested a DM.

Lokesh Kumar A R subsequently reviewed all 209 rows. For each row, the reviewer read the customer message and historical response, selected the most appropriate allowed intent, and independently decided whether the message should be handled by a person. The CSV records `human_verified`, `reviewer`, and `reviewed_at`. All 209 rows are marked verified.

The final intent counts are:

| Intent | Rows |
|---|---:|
| `software_update_issues` | 45 |
| `recurring_device_issues` | 38 |
| `keyboard_autocorrect_bug` | 27 |
| `account_and_store_issues` | 20 |
| `macos_update_issues` | 18 |
| `ios_update_issues` | 14 |
| `battery_drain_after_update` | 14 |
| `customer_service_complaint` | 12 |
| `apple_music_issue` | 11 |
| `missing_photos` | 10 |
| `all_apps_crashing` | 0 |

The weak best silhouette score (0.0316 at k=11) and the empty final class reinforce that the taxonomy is a prototype routing scheme rather than a discovered natural ground truth.

## Final evaluation sample

The end-to-end comparison uses `--sample 10`, which deterministically selects one row from every final intent represented in the golden set. This cost-bounded pilot evaluates 10/209 rows with all three systems and the LLM judge. It is suitable for checking the complete pipeline but too small for precise performance claims.

## Leakage control

All candidates came from the 15,000-row labeled pool and also exist in the historical retrieval corpus. `run_eval.py` therefore:

- removes every golden `reply_tweet_id` from TF-IDF training; and
- excludes the current row's `reply_tweet_id` from semantic retrieval.

Without these controls, nearest-neighbor systems could retrieve the evaluated row itself and inflate results.

## Known scope limits

- The data is from 2017 and is not evidence of current Apple policy.
- The ASCII-ratio language filter is only a coarse heuristic.
- A single public tweet may omit earlier thread context.
- The final evaluation covers only 10 of 209 reviewed rows.
- The human escalation target is a judgment call, not a known internal Apple handoff.
