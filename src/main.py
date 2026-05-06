"""CLI entrypoint: fetch data, compute indicators, render digest, send Telegram."""

from __future__ import annotations

import argparse
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml
from dotenv import load_dotenv

from . import llm
from .data import fetch_many
from .indicators import (
    IndicatorConfig,
    detect_opportunities,
    enrich_ohlcv,
    relative_strength_vs_benchmark,
    snapshot_last_row,
)
from .report import build_markdown, save_snapshot_json, snapshot_for_llm
from .telegram_send import send_digest

LOG = logging.getLogger(__name__)


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _vix_regime(close: Any) -> Tuple[str, float]:
    series = close.dropna().tail(252)
    if series.empty:
        return "unknown", float("nan")
    cur = float(close.iloc[-1])
    med = float(series.median())
    if not math.isfinite(med):
        return "unknown", med
    if cur > med * 1.15:
        return "elevated_vs_median", med
    if cur < med * 0.85:
        return "calm_vs_median", med
    return "near_median", med


def _collect_symbols(cfg: Dict[str, Any]) -> List[str]:
    wanted: Set[str] = set(cfg.get("macro_symbols") or [])
    bench = cfg.get("benchmark_symbol")
    if bench:
        wanted.add(str(bench))
    for syms in (cfg.get("themes") or {}).values():
        wanted.update(str(s) for s in syms)
    wanted.update(str(s) for s in (cfg.get("watchlist") or []))
    return sorted(wanted)


def build_snapshot(cfg_path: Path) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    bench_sym = str(cfg.get("benchmark_symbol", "SPY"))
    rs_lb = int(cfg.get("relative_strength_lookback_days", 20))
    tz_disp = str(cfg.get("timezone_display", "America/New_York"))

    ind_cfg = IndicatorConfig(
        rsi_length=int((cfg.get("indicators") or {}).get("rsi_length", 14)),
        macd_fast=int((cfg.get("indicators") or {}).get("macd_fast", 12)),
        macd_slow=int((cfg.get("indicators") or {}).get("macd_slow", 26)),
        macd_signal=int((cfg.get("indicators") or {}).get("macd_signal", 9)),
        sma_fast=int((cfg.get("indicators") or {}).get("sma_fast", 50)),
        sma_slow=int((cfg.get("indicators") or {}).get("sma_slow", 200)),
    )

    symbols = _collect_symbols(cfg)
    raw_frames = fetch_many(symbols)
    if bench_sym not in raw_frames:
        missing = sorted(set(symbols) - set(raw_frames.keys()))
        preview = missing[:20]
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise RuntimeError(
            f"Benchmark {bench_sym!r} has no price data — cannot compute RS vs {bench_sym}. "
            f"This usually means Yahoo Finance is blocked or unreachable from this network "
            f"(common on locked-down corporate PCs). "
            f"Symbols with no data ({len(missing)}): {preview}{more}. "
            f"Try: personal hotspot/home Wi‑Fi, VPN, or run the digest on GitHub Actions instead. "
            f"If you use an HTTP proxy, set HTTPS_PROXY (and HTTP_PROXY). "
            f"If logs show curl SSL certificate errors behind corporate inspection, set "
            f"YFINANCE_SSL_VERIFY=0 in .env (last resort; ask IT for the proper root CA if possible)."
        )

    bench_close = raw_frames[bench_sym]["close"]

    enriched: Dict[str, Any] = {}
    for sym, frame in raw_frames.items():
        enriched[sym] = enrich_ohlcv(frame, ind_cfg)

    macro_syms = [s for s in (cfg.get("macro_symbols") or []) if s in enriched]
    macro_out: Dict[str, Any] = {}

    for sym in macro_syms:
        df = enriched[sym]
        snap = snapshot_last_row(df)
        if sym.upper().replace("^", "") == "VIX":
            regime, med = _vix_regime(df["close"])
            snap["regime"] = regime
            snap["median_1y"] = med
            macro_out[sym] = snap
            continue

        sym_ret, bench_ret = relative_strength_vs_benchmark(df["close"], bench_close, rs_lb)
        rs_spread = (
            (sym_ret - bench_ret) * 100
            if math.isfinite(sym_ret) and math.isfinite(bench_ret)
            else float("nan")
        )
        snap["rs_vs_benchmark_pct"] = rs_spread
        macro_out[sym] = snap

    themes_cfg = cfg.get("themes") or {}
    themes_out: Dict[str, Any] = {}
    leadership: List[Tuple[str, float]] = []

    for theme_name, syms in themes_cfg.items():
        syms_list = [str(s) for s in syms]
        primary = syms_list[0] if syms_list else None
        symbols_block: Dict[str, Any] = {}
        above_flags: List[float] = []

        for sym in syms_list:
            if sym not in enriched:
                LOG.warning("Theme %s missing data for %s", theme_name, sym)
                continue
            df = enriched[sym]
            snap = snapshot_last_row(df)
            sym_ret, bench_ret = relative_strength_vs_benchmark(df["close"], bench_close, rs_lb)
            rs_spread = (
                (sym_ret - bench_ret) * 100
                if math.isfinite(sym_ret) and math.isfinite(bench_ret)
                else float("nan")
            )
            snap["rs_vs_benchmark_pct"] = rs_spread
            symbols_block[sym] = snap
            above_flags.append(1.0 if snap["above50"] else 0.0)

        breadth = (
            (sum(above_flags) / len(above_flags)) * 100 if above_flags else None
        )

        themes_out[theme_name] = {
            "primary_symbol": primary,
            "alternates": syms_list[1:] if len(syms_list) > 1 else [],
            "symbols": symbols_block,
            "breadth_above50_pct": breadth,
        }

        if primary and primary in symbols_block:
            spread = symbols_block[primary].get("rs_vs_benchmark_pct")
            if isinstance(spread, (int, float)) and math.isfinite(spread):
                leadership.append((theme_name, float(spread)))

    leadership.sort(key=lambda item: item[1], reverse=True)

    opportunities_raw: List[Dict[str, Any]] = []
    seen = set()

    for theme_name, syms in themes_cfg.items():
        for sym in syms:
            sym = str(sym)
            if sym not in enriched:
                continue
            for idea in detect_opportunities(sym, theme_name, enriched[sym], bench_close, rs_lb):
                key = (idea["symbol"], idea["tag"])
                if key in seen:
                    continue
                seen.add(key)
                opportunities_raw.append(idea)

    for sym in cfg.get("watchlist") or []:
        sym = str(sym)
        if sym not in enriched:
            continue
        for idea in detect_opportunities(sym, "watchlist", enriched[sym], bench_close, rs_lb):
            key = (idea["symbol"], idea["tag"])
            if key in seen:
                continue
            seen.add(key)
            opportunities_raw.append(idea)

    def _opp_rs(o: Dict[str, Any]) -> float:
        v = o.get("rs_vs_benchmark_pct")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return float("-inf")

    opportunities_sorted = sorted(opportunities_raw, key=_opp_rs, reverse=True)

    snapshot: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc),
        "timezone_display": tz_disp,
        "benchmark_symbol": bench_sym,
        "relative_strength_lookback_days": rs_lb,
        "macro": macro_out,
        "themes": themes_out,
        "theme_rs_ranking": leadership,
        "opportunities": opportunities_sorted,
    }
    return snapshot


def main(argv: List[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Weekly market digest for Telegram")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("DIGEST_CONFIG", "config.yaml")),
        help="Path to config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print digest only; do not send Telegram")
    parser.add_argument(
        "--snapshot-out",
        type=Path,
        default=Path(os.getenv("SNAPSHOT_OUT", "digest_snapshot.json")),
        help="Where to write JSON snapshot facts",
    )
    args = parser.parse_args(argv)

    env_next_to_config = args.config.resolve().parent / ".env"
    if env_next_to_config.is_file():
        load_dotenv(dotenv_path=env_next_to_config, override=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    snapshot = build_snapshot(args.config)

    llm_summary = None
    if _truthy(os.getenv("USE_LLM_SUMMARY")):
        llm_summary = llm.summarize_snapshot(snapshot_for_llm(snapshot))

    markdown = build_markdown(snapshot)
    if llm_summary:
        markdown = (
            f"*AI executive summary (experimental)*\n\n{llm_summary}\n\n---\n\n" + markdown
        )

    save_snapshot_json(snapshot, str(args.snapshot_out))
    LOG.info("Wrote snapshot JSON to %s", args.snapshot_out)

    print(markdown)

    if args.dry_run:
        LOG.info("Dry-run enabled; skipping Telegram send")
        return 0

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        if os.getenv("GITHUB_ACTIONS") == "true":
            LOG.error("Telegram secrets missing in CI")
            return 1
        LOG.warning("Telegram credentials missing; printed digest only")
        return 0

    send_digest(token, chat_id, markdown)
    LOG.info("Telegram digest sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
