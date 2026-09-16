"""Leakage-safe trivial and TF-IDF baselines."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.settings import LABELED_PATH


TRIVIAL_REPLY = "We're sorry you're having trouble. Please send us a DM so we can help."


def trivial_baseline(customer_message: str) -> dict[str, Any]:
    del customer_message
    return {
        "intent": "recurring_device_issues",
        "draft_reply": TRIVIAL_REPLY,
        "escalate": False,
        "escalation_reason": "Always predicts the same class and action.",
    }


class TfidfBaseline:
    """Nearest-neighbor intent/reply baseline trained outside the evaluation set."""

    def __init__(self, exclude_reply_ids: Collection[int | str] | None = None) -> None:
        df = pd.read_csv(LABELED_PATH)
        required = {"reply_tweet_id", "customer_message", "apple_reply", "intent"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Labeled data is missing columns: {sorted(missing)}")
        if exclude_reply_ids:
            excluded = {str(value) for value in exclude_reply_ids}
            df = df[~df["reply_tweet_id"].astype(str).isin(excluded)]
        df = df.dropna(subset=["customer_message", "apple_reply", "intent"])
        df = df[df["customer_message"].str.len() > 20].reset_index(drop=True)
        if df.empty:
            raise ValueError("No TF-IDF training rows remain after exclusions")

        self.df = df
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(df["customer_message"])

    def predict(self, customer_message: str) -> dict[str, Any]:
        query = self.vectorizer.transform([customer_message])
        similarities = linear_kernel(query, self.matrix).ravel()
        best_idx = int(similarities.argmax())
        row = self.df.iloc[best_idx]
        message = customer_message.lower()
        escalate = any(
            phrase in message
            for phrase in ("stolen", "hacked", "fraud", "lawsuit", "data breach", "lost everything")
        )
        return {
            "intent": str(row["intent"]),
            "draft_reply": str(row["apple_reply"]),
            "escalate": escalate,
            "escalation_reason": "Sensitive keyword match." if escalate else "No sensitive keyword match.",
            "retrieval_similarity": float(similarities[best_idx]),
            "retrieved_reply_tweet_id": int(row["reply_tweet_id"]),
        }


_default_tfidf: TfidfBaseline | None = None


def tfidf_baseline(customer_message: str) -> dict[str, Any]:
    """Convenience wrapper for demos; evaluations should instantiate with exclusions."""
    global _default_tfidf
    if _default_tfidf is None:
        _default_tfidf = TfidfBaseline()
    return _default_tfidf.predict(customer_message)


if __name__ == "__main__":
    sample = "My iPhone battery dies quickly after the update"
    print(trivial_baseline(sample))
    print(tfidf_baseline(sample))
