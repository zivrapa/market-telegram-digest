"""Send Telegram messages with safe chunking."""

from __future__ import annotations

import logging
from typing import List

import httpx

LOG = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096


def chunk_text(text: str, limit: int = TELEGRAM_LIMIT) -> List[str]:
    """Split long markdown into Telegram-sized chunks without breaking mid-word when possible."""
    text = text.strip()
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit
        piece = remaining[:split_at].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[split_at:].strip()
    return chunks


def send_message(token: str, chat_id: str, text: str, parse_mode: str | None = "Markdown") -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            body = resp.text
            LOG.error("Telegram HTTP error %s: %s", resp.status_code, body)
            if parse_mode == "Markdown":
                LOG.info("Retrying without Markdown parse_mode")
                send_message(token, chat_id, text, parse_mode=None)
                return
            raise


def send_digest(token: str, chat_id: str, markdown: str) -> None:
    parts = chunk_text(markdown)
    for i, part in enumerate(parts):
        prefix = f"(part {i + 1}/{len(parts)})\n\n" if len(parts) > 1 else ""
        send_message(token, chat_id, prefix + part)
