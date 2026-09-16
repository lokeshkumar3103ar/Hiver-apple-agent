"""Create a stratified *candidate* evaluation set for subsequent human review."""

from __future__ import annotations

import argparse

import pandas as pd

from src.settings import GOLDEN_DIR, GOLDEN_SET_PATH, LABELED_PATH


SAMPLES_PER_INTENT = 19
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing golden_set.csv")
    args = parser.parse_args()
    if GOLDEN_SET_PATH.exists() and not args.overwrite:
        raise FileExistsError(f"{GOLDEN_SET_PATH} exists; pass --overwrite to replace it")

    df = pd.read_csv(LABELED_PATH)
    required = {"reply_tweet_id", "customer_message", "apple_reply", "intent", "created_at"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Labeled subset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["intent", "customer_message", "apple_reply"])
    df = df[(df["customer_message"].str.len() > 30) & (df["apple_reply"].str.len() > 20)]

    parts = [
        group.sample(n=min(SAMPLES_PER_INTENT, len(group)), random_state=RANDOM_SEED)
        for _, group in df.groupby("intent", sort=True)
    ]
    golden = pd.concat(parts).reset_index(drop=True)
    golden = golden[["reply_tweet_id", "customer_message", "apple_reply", "intent", "created_at"]].copy()
    golden.insert(3, "cluster_intent", golden["intent"])
    dm_proxy = golden["apple_reply"].str.contains(
        r"\bdm\b|\bdirect message\b|\bprivate message\b", case=False, regex=True
    )
    golden["gold_escalate"] = dm_proxy
    golden["human_verified"] = False
    golden["reviewer"] = ""
    golden["reviewed_at"] = ""
    golden["review_notes"] = ""

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(golden)} candidate rows to {GOLDEN_SET_PATH}")
    print("These are cluster/proxy labels, not human ground truth. Run: python review_golden_set.py")


if __name__ == "__main__":
    main()
