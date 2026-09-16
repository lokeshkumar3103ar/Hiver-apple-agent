"""Fast integrity checks for generated artifacts and submission readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.settings import GOLDEN_SET_PATH, INTENTS, LABELED_PATH, PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", action="store_true", help="Fail if human review/evidence is incomplete")
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=PROCESSED_DIR / "evaluation_summary.sample-10.json",
        help="Evaluation summary used as submission evidence",
    )
    parser.add_argument(
        "--agreement",
        type=Path,
        default=PROCESSED_DIR / "judge_human_agreement.sample-10.json",
        help="Saved LLM-judge/human agreement metrics",
    )
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    golden = pd.read_csv(GOLDEN_SET_PATH)
    labeled = pd.read_csv(LABELED_PATH)
    if not 150 <= len(golden) <= 250:
        errors.append(f"Golden set has {len(golden)} rows; required range is 150-250")
    if golden["reply_tweet_id"].duplicated().any():
        errors.append("Golden set has duplicate reply_tweet_id values")
    unknown = set(golden["intent"]) - set(INTENTS)
    if unknown:
        errors.append(f"Unknown golden-set intents: {sorted(unknown)}")
    missing_values = golden[["customer_message", "apple_reply", "intent", "gold_escalate"]].isna().sum().sum()
    if missing_values:
        errors.append(f"Golden set has {missing_values} missing required values")

    verified = golden.get("human_verified", pd.Series(False, index=golden.index)).astype(str).str.lower()
    verified_count = int(verified.isin({"true", "1", "yes"}).sum())
    reviewers = golden.get("reviewer", pd.Series("", index=golden.index)).fillna("").astype(str).str.strip()
    reviewed_at = golden.get("reviewed_at", pd.Series("", index=golden.index)).fillna("").astype(str).str.strip()
    if verified_count != len(golden):
        message = f"Only {verified_count}/{len(golden)} golden rows have recorded human verification"
        (errors if args.submission else warnings).append(message)
    if (reviewers == "").any():
        message = "At least one golden row has no reviewer attribution"
        (errors if args.submission else warnings).append(message)
    if (reviewed_at == "").any():
        message = "At least one golden row has no review timestamp"
        (errors if args.submission else warnings).append(message)

    overlap = int(golden["reply_tweet_id"].isin(set(labeled["reply_tweet_id"])).sum())
    if overlap:
        warnings.append(
            f"{overlap}/{len(golden)} golden rows originate in labeled_subset.csv; run_eval.py excludes "
            "all golden IDs from the TF-IDF training pool"
        )

    silhouette_path = PROCESSED_DIR / "silhouette_scores.json"
    if silhouette_path.exists():
        scores = json.loads(silhouette_path.read_text(encoding="utf-8"))
        best_k, best_score = max(scores.items(), key=lambda item: item[1])
        if float(best_score) < 0.1:
            warnings.append(f"Weak cluster separation: best silhouette={best_score:.4f} at k={best_k}")

    summary_path = args.evaluation_summary
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("metadata", {}).get("all_labels_human_verified", False):
            message = "Evaluation summary says its labels were not all human-verified"
            (errors if args.submission else warnings).append(message)
        systems = summary.get("systems", {})
        missing_systems = {"trivial", "tfidf", "full_agent"}.difference(systems)
        if missing_systems:
            message = f"Evaluation summary is missing systems: {sorted(missing_systems)}"
            (errors if args.submission else warnings).append(message)
        sample_size = int(summary.get("metadata", {}).get("sample_size", 0))
        if sample_size < len(golden):
            warnings.append(
                f"Submission evidence is a cost-bounded sample ({sample_size}/{len(golden)} golden rows)"
            )
        for name, metrics in systems.items():
            if metrics.get("errors", 0):
                warnings.append(f"{name} recorded {metrics['errors']} prediction errors")
            if metrics.get("judge_errors", 0):
                warnings.append(f"{name} recorded {metrics['judge_errors']} judge errors")
    else:
        message = f"Evaluation summary is missing: {summary_path}"
        (errors if args.submission else warnings).append(message)

    if args.agreement.exists():
        agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
        required_agreement = {
            "n",
            "exact_agreement",
            "within_one_point",
            "mean_absolute_error",
            "quadratic_weighted_kappa",
            "spearman_rho",
        }
        missing_agreement = required_agreement.difference(agreement)
        if missing_agreement:
            message = f"Agreement artifact is missing fields: {sorted(missing_agreement)}"
            (errors if args.submission else warnings).append(message)
        elif int(agreement["n"]) < 2:
            message = "Agreement evidence needs at least two scored rows"
            (errors if args.submission else warnings).append(message)
        elif int(agreement["n"]) < 30:
            warnings.append(f"Judge/human agreement is based on only {agreement['n']} rows")
    else:
        message = f"Agreement artifact is missing: {args.agreement}"
        (errors if args.submission else warnings).append(message)

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    print(f"Evaluation evidence: {summary_path}")
    print(f"Agreement evidence: {args.agreement}")
    print(f"Checked {len(golden)} golden rows and {len(labeled)} labeled rows: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
