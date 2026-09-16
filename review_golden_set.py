"""Checkpointed terminal workflow for genuinely human-reviewing the golden set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.settings import GOLDEN_SET_PATH, INTENTS


def parse_answer(prompt: str, default: bool) -> bool:
    value = input(prompt).strip().lower()
    if not value:
        return default
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Enter y or n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True, help="Your name or initials for the audit trail")
    args = parser.parse_args()
    df = pd.read_csv(GOLDEN_SET_PATH, keep_default_na=False)
    for column, default in {
        "cluster_intent": df.get("intent", ""),
        "human_verified": False,
        "reviewer": "",
        "reviewed_at": "",
        "review_notes": "",
    }.items():
        if column not in df:
            df[column] = default

    names = list(INTENTS)
    pending = df.index[~df["human_verified"].astype(str).str.lower().isin({"true", "1", "yes"})]
    print(f"{len(pending)} rows remain. Enter q at the intent prompt to save and quit.")
    for position, index in enumerate(pending, 1):
        row = df.loc[index]
        print(f"\n[{position}/{len(pending)}] Customer: {row['customer_message']}")
        print(f"Historical reply: {row['apple_reply']}")
        for number, name in enumerate(names, 1):
            marker = " *" if name == row["intent"] else ""
            print(f"  {number:2}. {name}{marker}")
        choice = input("Intent number (Enter accepts *, q quits): ").strip().lower()
        if choice == "q":
            break
        if choice:
            selected = int(choice)
            if not 1 <= selected <= len(names):
                raise ValueError("Intent number is out of range")
            df.at[index, "intent"] = names[selected - 1]
        current_escalation = str(row["gold_escalate"]).lower() in {"true", "1", "yes"}
        df.at[index, "gold_escalate"] = parse_answer(
            f"Should a human handle this? [y/n, Enter={('y' if current_escalation else 'n')}]: ",
            current_escalation,
        )
        df.at[index, "review_notes"] = input("Optional note: ").strip()
        df.at[index, "human_verified"] = True
        df.at[index, "reviewer"] = args.reviewer
        df.at[index, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        df.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
    complete = df["human_verified"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
    print(f"Saved. Human-verified rows: {complete}/{len(df)}")


if __name__ == "__main__":
    main()
