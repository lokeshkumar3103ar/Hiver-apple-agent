"""
data_prep.py — Phase 1: Data Preparation for AppleSupport AI Agent

What this script does:
1. Reads the full 516MB Twitter dataset (twcs.csv)
2. Filters for AppleSupport brand conversations
3. Reconstructs customer → brand conversation threads
4. Cleans tweet text (removes @mentions, URLs, boilerplate)
5. Filters to English-only tweets
6. Saves clean conversation pairs to data/processed/

LESSON: Data prep is the foundation of any AI pipeline.
        Bad data = bad agent, no matter how good your model is.
"""

import pandas as pd
import re
import json
import html
from tqdm import tqdm

from src.settings import CONVERSATIONS_PATH, PROCESSED_DIR, RAW_DATA_PATH

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BRAND = "AppleSupport"
MAX_ROWS = None  # Set to e.g. 500000 to limit memory use. None = load all.


def clean_text(text: str) -> str:
    """
    Clean a tweet's text for use as input/context to the LLM.

    LESSON: We do minimal cleaning here — we remove @mentions (noise)
    and URLs (not useful in text form) but preserve the actual content.
    Over-cleaning can destroy signal!
    """
    if not isinstance(text, str):
        return ""
    # Remove @mentions (e.g., @115712, @AppleSupport)
    text = html.unescape(text)
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove agent initials like "^MC" or "/AR" at the end (common in brand replies)
    text = re.sub(r"\s[\^/][A-Z]{1,3}\s*$", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def is_english(text: str) -> bool:
    """
    Coarse ASCII-ratio language filter. This is fast but not a language model;
    its limitations are documented in the report.
    """
    if not text or len(text) < 5:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= 0.85


def load_and_filter_dataset(path: str, brand: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the full dataset and filter to brand-relevant rows only.

    The source currently fits in memory. A production version should use
    chunked reads; this implementation intentionally does not claim to.
    """
    print(f"Loading dataset from: {path}")
    print("(This may take 30-60 seconds for 516MB...)")

    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "inbound": "bool",
            "text": "str",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "str",
        },
        nrows=MAX_ROWS,
    )
    print(f"  Loaded {len(df):,} total rows")

    # Get all AppleSupport reply tweet IDs (outbound = brand replies)
    apple_replies = df[df["author_id"] == brand].copy()
    print(f"  Found {len(apple_replies):,} {brand} replies")

    # Get the tweet IDs that AppleSupport replied TO (these are customer tweets)
    apple_reply_targets = set(apple_replies["in_response_to_tweet_id"].dropna().astype(str))

    # Get those customer tweets
    customer_tweets = df[
        df["inbound"] &
        (df["tweet_id"].astype(str).isin(apple_reply_targets))
    ].copy()
    print(f"  Found {len(customer_tweets):,} corresponding customer tweets")

    return apple_replies, customer_tweets


def reconstruct_threads(apple_replies: pd.DataFrame, customer_tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each customer tweet with the Apple Support reply to it.

    This creates our core unit of data: a (customer_message, brand_reply) pair.

    LESSON: Real Twitter conversations are multi-turn trees, not simple pairs.
    For this project, we simplify to single pairs (customer message → first reply).
    This is a deliberate scoping decision — worth noting in your decision log!
    """
    print("\nReconstructing conversation pairs...")

    # Index customer tweets by tweet_id for fast lookup
    customer_index = customer_tweets.set_index("tweet_id")[["author_id", "text", "created_at"]]
    customer_index.index = customer_index.index.astype(str)

    pairs = []
    for _, reply in tqdm(apple_replies.iterrows(), total=len(apple_replies), desc="Building pairs"):
        target_id = str(reply.get("in_response_to_tweet_id", ""))
        if target_id not in customer_index.index:
            continue

        customer_row = customer_index.loc[target_id]
        customer_text = clean_text(customer_row["text"])
        apple_text = clean_text(reply["text"])

        # Skip very short messages (noise)
        if len(customer_text) < 10 or len(apple_text) < 10:
            continue

        # English-only filter
        if not is_english(customer_text):
            continue

        pairs.append({
            "reply_tweet_id": reply["tweet_id"],
            "customer_tweet_id": target_id,
            "customer_author": customer_row["author_id"],
            "customer_message": customer_text,
            "apple_reply": apple_text,
            "created_at": reply["created_at"],
        })

    df_pairs = pd.DataFrame(pairs)
    print(f"  Built {len(df_pairs):,} clean conversation pairs")
    return df_pairs


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate Apple replies (templated/copy-paste responses).

    LESSON: Duplicate replies in RAG = misleading retrieval. If 500 conversations
    all retrieve the same "Please DM us" reply, the LLM will over-rely on it.
    We deduplicate on the Apple reply text (normalized).
    """
    print("\nDeduplicating...")
    before = len(df)

    # Normalize for dedup comparison (lowercase, no punctuation)
    df["_reply_norm"] = df["apple_reply"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_reply_norm"], keep="first")
    df = df.drop(columns=["_reply_norm"])

    after = len(df)
    print(f"  Removed {before - after:,} duplicate replies ({before:,} -> {after:,})")
    return df.reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONVERSATIONS_PATH

    # Step 1: Load & filter
    apple_replies, customer_tweets = load_and_filter_dataset(RAW_DATA_PATH, BRAND)

    # Step 2: Reconstruct threads
    df_pairs = reconstruct_threads(apple_replies, customer_tweets)

    # Step 3: Deduplicate
    df_pairs = deduplicate(df_pairs)

    # Step 4: Save
    df_pairs.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df_pairs):,} conversation pairs to: {output_path}")

    # Print a few examples so we can verify quality
    print("\n=== SAMPLE CONVERSATION PAIRS ===")
    for _, row in df_pairs.sample(min(3, len(df_pairs)), random_state=42).iterrows():
        print(f"\nCustomer: {row['customer_message']}")
        print(f"Apple:    {row['apple_reply']}")
        print("-" * 60)

    # Save summary stats
    stats = {
        "total_pairs": len(df_pairs),
        "brand": BRAND,
        "avg_customer_msg_len": round(df_pairs["customer_message"].str.len().mean(), 1),
        "avg_apple_reply_len": round(df_pairs["apple_reply"].str.len().mean(), 1),
    }
    with open(PROCESSED_DIR / "prep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {PROCESSED_DIR / 'prep_stats.json'}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
