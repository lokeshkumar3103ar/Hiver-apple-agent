"""Helpers for validating cached artifacts against their source rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def dataframe_fingerprint(df: pd.DataFrame, columns: list[str], model_name: str) -> str:
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Cannot fingerprint missing columns: {sorted(missing)}")
    hashed = pd.util.hash_pandas_object(df[columns].fillna(""), index=False).values
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def load_embedding_cache(path: Path, expected_shape: tuple[int, int], fingerprint: str) -> np.ndarray | None:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = np.load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if tuple(cached.shape) != expected_shape or metadata.get("fingerprint") != fingerprint:
        return None
    return cached


def save_embedding_cache(path: Path, embeddings: np.ndarray, fingerprint: str, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    metadata = {
        "fingerprint": fingerprint,
        "model": model_name,
        "shape": list(embeddings.shape),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
