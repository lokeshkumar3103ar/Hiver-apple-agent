"""Build and query the local semantic retrieval index."""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.artifact_utils import dataframe_fingerprint, load_embedding_cache, save_embedding_cache
from src.settings import CHROMA_DIR, CONVERSATIONS_PATH, EMBED_MODEL, LABELED_PATH, PROCESSED_DIR


COLLECTION_NAME = "apple_support_rag"
EMBED_BATCH_SIZE = 256


def _load_embedding_model(model_name: str) -> SentenceTransformer:
    """Prefer an installed cache; download only when the model is absent."""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def load_data_with_intents(source_path: Path = LABELED_PATH) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"RAG source not found: {source_path}")
    df = pd.read_csv(source_path)
    required = {"reply_tweet_id", "customer_message", "apple_reply"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"RAG source is missing columns: {sorted(missing)}")
    if "intent" not in df:
        labeled = pd.read_csv(LABELED_PATH, usecols=["reply_tweet_id", "intent"])
        df = df.merge(labeled, on="reply_tweet_id", how="left")
    df["intent"] = df["intent"].fillna("unknown")
    if df["reply_tweet_id"].duplicated().any():
        raise ValueError("RAG source contains duplicate reply_tweet_id values")
    return df.reset_index(drop=True)


def embed_all(df: pd.DataFrame, model_name: str, cache_path: Path) -> np.ndarray:
    """Embed customer messages, reusing only a source-matched cache."""
    fingerprint = dataframe_fingerprint(df, ["reply_tweet_id", "customer_message"], model_name)
    cached = load_embedding_cache(cache_path, (len(df), 384), fingerprint)
    if cached is not None:
        return cached
    if cache_path.exists():
        print(f"Ignoring unverified or stale embedding cache: {cache_path}")

    model = _load_embedding_model(model_name)
    embeddings = model.encode(
        df["customer_message"].fillna("").astype(str).tolist(),
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    save_embedding_cache(cache_path, embeddings, fingerprint, model_name)
    return embeddings


def build_chroma_index(df: pd.DataFrame, embeddings: np.ndarray) -> chromadb.Collection:
    if len(df) != len(embeddings):
        raise ValueError(f"Row/embedding mismatch: {len(df)} rows vs {len(embeddings)} vectors")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in tqdm(range(0, len(df), 2000), desc="Indexing RAG corpus"):
        batch = df.iloc[start : start + 2000]
        collection.add(
            ids=batch["reply_tweet_id"].astype(str).tolist(),
            embeddings=embeddings[start : start + len(batch)].tolist(),
            documents=batch["customer_message"].fillna("").astype(str).tolist(),
            metadatas=[
                {
                    "apple_reply": str(row.apple_reply)[:1000],
                    "intent": str(row.intent),
                    "reply_tweet_id": int(row.reply_tweet_id),
                }
                for row in batch.itertuples(index=False)
            ],
        )
    return collection


def get_retriever(embed_model_name: str = EMBED_MODEL):
    model = _load_embedding_model(embed_model_name)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("RAG index is missing. Run: python -m src.rag_retriever --build") from exc

    def retrieve(
        query: str,
        k: int = 5,
        intent_filter: str | None = None,
        exclude_reply_ids: Collection[int | str] | None = None,
    ) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        excluded = {str(value) for value in (exclude_reply_ids or ())}
        available = collection.count()
        if available == 0:
            return []
        # Over-fetch so evaluation can remove the exact gold row without leakage.
        requested = min(available, k + len(excluded) + 20)
        where = {"intent": intent_filter} if intent_filter else None
        results = collection.query(
            query_embeddings=model.encode([query], normalize_embeddings=True).tolist(),
            n_results=requested,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[dict[str, Any]] = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if str(meta["reply_tweet_id"]) in excluded:
                continue
            retrieved.append(
                {
                    "reply_tweet_id": meta["reply_tweet_id"],
                    "customer_message": doc,
                    "apple_reply": meta["apple_reply"],
                    "intent": meta["intent"],
                    "similarity": round(1 - float(distance), 4),
                }
            )
            if len(retrieved) == k:
                break
        return retrieved

    return retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG index.")
    parser.add_argument("--build", action="store_true", help="Accepted for README compatibility")
    parser.add_argument(
        "--source",
        choices=("sample", "full"),
        default="sample",
        help="sample=15k included rows (fast); full=all prepared Apple conversations",
    )
    args = parser.parse_args()

    source_path = LABELED_PATH if args.source == "sample" else CONVERSATIONS_PATH
    cache_name = "embeddings_sample.npy" if args.source == "sample" else "embeddings_all.npy"
    df = load_data_with_intents(source_path)
    embeddings = embed_all(df, EMBED_MODEL, PROCESSED_DIR / cache_name)
    collection = build_chroma_index(df, embeddings)
    stats = {
        "total_indexed": collection.count(),
        "source": source_path.relative_to(source_path.parents[2]).as_posix(),
        "embed_model": EMBED_MODEL,
        "embed_dims": int(embeddings.shape[1]),
        "collection": COLLECTION_NAME,
        "similarity_metric": "cosine",
    }
    (PROCESSED_DIR / "rag_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Built {collection.count():,}-row RAG index at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
