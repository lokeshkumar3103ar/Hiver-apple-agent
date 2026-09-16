"""Small, validated client for Agent Platform express-mode generation."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from src.settings import GENERATION_MODEL, GOOGLE_API_BASE_URL


FLASH_MODEL = GENERATION_MODEL
PRO_MODEL = GENERATION_MODEL  # Backwards-compatible alias for older imports.
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_PLACEHOLDERS = {
    "your_api_key_here",
    "your_google_agent_platform_key_here",
    "replace_me",
}


def _get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key.lower() in _PLACEHOLDERS:
        raise ValueError("GOOGLE_API_KEY is not configured. Copy .env.example to .env and add the key.")
    return key


def _extract_text(payload: dict[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(
            str(part.get("text", ""))
            for part in parts
            if not part.get("thought", False)
        ).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The model returned an unexpected response shape.") from exc
    if not text:
        finish_reason = payload.get("candidates", [{}])[0].get("finishReason", "unknown")
        raise RuntimeError(f"The model returned no text (finish reason: {finish_reason}).")
    return text


def generate(
    prompt: str,
    model: str = FLASH_MODEL,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    *,
    json_mode: bool = False,
    timeout_seconds: int = 45,
) -> str:
    """Generate text with bounded retries and explicit response validation."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    generation_config: dict[str, Any] = {
        # Gemini 3 thinking consumes this same output-token budget. A tiny cap can
        # truncate before the visible answer begins, so enforce a safe floor.
        "maxOutputTokens": max(max_tokens, 1024) if model.startswith("gemini-3") else max_tokens,
    }
    if model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "LOW"}
    else:
        generation_config["temperature"] = temperature

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{GOOGLE_API_BASE_URL}/{model}:generateContent"
    headers = {"x-goog-api-key": _get_api_key(), "Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            if response.status_code == 200:
                return _extract_text(response.json())
            if response.status_code not in _TRANSIENT_STATUS_CODES:
                try:
                    message = response.json().get("error", {}).get("message", "Unknown API error")
                except ValueError:
                    message = response.text[:300] or "Unknown API error"
                raise RuntimeError(f"Agent Platform API error {response.status_code}: {message}")
            last_error = RuntimeError(f"Transient Agent Platform error {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc

        if attempt < 3:
            delay = 2**attempt
            logging.warning("LLM request failed; retrying in %ss (%s/4).", delay, attempt + 1)
            time.sleep(delay)

    raise RuntimeError("LLM request failed after four attempts.") from last_error


def generate_json(
    prompt: str,
    model: str = FLASH_MODEL,
    system: str | None = None,
    *,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Generate a JSON object; malformed or non-object responses are errors."""
    raw = generate(
        prompt=prompt,
        model=model,
        system=system,
        temperature=0.1,
        max_tokens=max_tokens,
        json_mode=True,
    ).strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        raw = raw.removeprefix("json").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected the model to return a JSON object.")
    return parsed


if __name__ == "__main__":
    print(f"Testing Agent Platform connection with model: {FLASH_MODEL}")
    try:
        print(generate("Reply with exactly: connection ok", temperature=0, max_tokens=20))
    except Exception as exc:
        raise SystemExit(f"Connection test failed: {exc}") from exc
