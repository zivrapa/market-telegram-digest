"""Fetch and normalize OHLCV series."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

import pandas as pd
import yfinance as yf

LOG = logging.getLogger(__name__)

_ssl_verify_warning_logged = False


def _session_kwargs() -> Dict[str, Any]:
    """yfinance 0.2.6x uses curl_cffi internally — never pass requests.Session.

    If your company TLS-inspects HTTPS with a private root CA, normal verification fails
    with curl error 60. Set YFINANCE_SSL_VERIFY=0 only as a deliberate workaround
    (traffic can still be inspected by the proxy; you only skip verifying Yahoo's cert chain).
    """
    global _ssl_verify_warning_logged  # noqa: PLW0603
    verify_on = os.getenv("YFINANCE_SSL_VERIFY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if verify_on:
        return {}

    if not _ssl_verify_warning_logged:
        LOG.warning(
            "YFINANCE_SSL_VERIFY is off — certificate verification disabled for Yahoo requests "
            "(common workaround for corporate SSL inspection). Prefer installing your org root CA."
        )
        _ssl_verify_warning_logged = True

    from curl_cffi import requests as curl_requests

    # curl_cffi Session is what yfinance expects when a session is supplied at all.
    return {"session": curl_requests.Session(verify=False)}


def _normalize_history_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=str.lower)

    required = ["close"]
    for col in required:
        if col not in out.columns:
            LOG.warning("Normalized frame missing %s; columns=%s", col, list(out.columns))
            return None

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def fetch_history(symbol: str, period: str = "3y", retries: int = 3) -> pd.DataFrame | None:
    """Download adjusted daily OHLCV for `symbol`. Returns None if unusable."""
    sym = symbol.strip()
    kw = _session_kwargs()
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            t = yf.Ticker(sym, **kw)
            df = t.history(period=period, auto_adjust=True, interval="1d")
            normalized = _normalize_history_df(df)
            if normalized is not None:
                return normalized
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            LOG.warning("Ticker.history failed %s (try %s/%s): %s", sym, attempt + 1, retries, exc)

        try:
            df = yf.download(
                sym,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                **kw,
            )
            normalized = _normalize_history_df(df)
            if normalized is not None:
                return normalized
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            LOG.warning("yfinance.download failed %s (try %s/%s): %s", sym, attempt + 1, retries, exc)

        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))

    LOG.warning("No data returned for %s after %s tries. Last error: %s", sym, retries, last_err)
    return None


def fetch_many(symbols: list[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    """Fetch multiple symbols; skips failures."""
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_history(sym, period=period)
        if df is not None and not df.empty:
            out[sym] = df
        else:
            LOG.warning("Dropped symbol with no data: %s", sym)
    return out
