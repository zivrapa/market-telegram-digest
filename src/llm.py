"""Optional OpenAI-compatible summarization."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import httpx

LOG = logging.getLogger(__name__)


def summarize_snapshot(snapshot: Dict[str, Any]) -> str | None:
    """Return a short executive summary grounded on `snapshot`, or None if skipped."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        LOG.info("OPENAI_API_KEY missing; skipping LLM summary")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise weekly market notes for a sophisticated retail reader. "
                    "Use ONLY facts present in the JSON; do not invent levels or tickers. "
                    "If data is missing, say so briefly. Max ~140 words, Markdown bullets."
                ),
            },
            {
                "role": "user",
                "content": "Facts JSON:\n" + json.dumps(snapshot, default=str)[:12000],
            },
        ],
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("LLM summarization failed: %s", exc)
        return None
