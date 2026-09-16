"""Shared, cwd-independent configuration for the project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "twcs.csv"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"

CONVERSATIONS_PATH = PROCESSED_DIR / "apple_conversations.csv"
LABELED_PATH = PROCESSED_DIR / "labeled_subset.csv"
GOLDEN_SET_PATH = GOLDEN_DIR / "golden_set.csv"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GOOGLE_GENERATION_MODEL", "gemini-3.8-flash")
JUDGE_MODEL = os.getenv("GOOGLE_JUDGE_MODEL", GENERATION_MODEL)
GOOGLE_API_BASE_URL = os.getenv(
    "GOOGLE_API_BASE_URL",
    "https://aiplatform.googleapis.com/v1/publishers/google/models",
).rstrip("/")

# One canonical taxonomy used by discovery, the agent, baselines, and evaluation.
INTENTS = {
    "apple_music_issue": "Cannot access, download, or play Apple Music content",
    "account_and_store_issues": "Apple ID, verification, activation-lock, or store/account problems",
    "macos_update_issues": "Mac problems, especially after a macOS update",
    "customer_service_complaint": "Poor service, long waits, or suspicious Apple messages",
    "all_apps_crashing": "Several apps crash or malfunction",
    "missing_photos": "Photos are missing or appear to have been deleted",
    "ios_update_issues": "General iOS update bugs or complaints",
    "battery_drain_after_update": "Abnormal battery drain after an iOS update",
    "software_update_issues": "Slowness, freezing, or glitches after an update",
    "keyboard_autocorrect_bug": "Keyboard or autocorrect behavior is broken",
    "recurring_device_issues": "Persistent or repeated device defects",
}

