"""Technical indicators and derived snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class IndicatorConfig:
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_fast: int = 50
    sma_slow: int = 200


def enrich_ohlcv(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Append indicator columns to OHLCV dataframe (mutates copy)."""
    out = df.copy()
    out.ta.rsi(length=cfg.rsi_length, append=True)
    out.ta.macd(
        fast=cfg.macd_fast,
        slow=cfg.macd_slow,
        signal=cfg.macd_signal,
        append=True,
    )
    out.ta.sma(length=cfg.sma_fast, append=True)
    out.ta.sma(length=cfg.sma_slow, append=True)
    macd_hist_col = f"MACDh_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
    rsi_col = f"RSI_{cfg.rsi_length}"
    sma_f = f"SMA_{cfg.sma_fast}"
    sma_s = f"SMA_{cfg.sma_slow}"
    rename_map = {
        rsi_col: "rsi",
        macd_hist_col: "macd_hist",
        sma_f: "sma50",
        sma_s: "sma200",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def _pct_change_week(close: pd.Series) -> float:
    if len(close) < 6:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-6] - 1.0)


def relative_strength_vs_benchmark(
    sym_close: pd.Series, bench_close: pd.Series, lookback: int
) -> tuple[float, float]:
    """Return (sym_return, bench_return) over `lookback` trading days."""
    s = sym_close.dropna().iloc[-(lookback + 1) :]
    b = bench_close.reindex(s.index).dropna()
    common = s.index.intersection(b.index)
    if len(common) < lookback + 1:
        return float("nan"), float("nan")
    s = s.loc[common]
    b = b.loc[common]
    sym_ret = float(s.iloc[-1] / s.iloc[-(lookback + 1)] - 1.0)
    bench_ret = float(b.iloc[-1] / b.iloc[-(lookback + 1)] - 1.0)
    return sym_ret, bench_ret


def snapshot_last_row(df: pd.DataFrame) -> Dict[str, Any]:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    close = float(row["close"])
    sma50 = float(row["sma50"]) if pd.notna(row["sma50"]) else float("nan")
    sma200 = float(row["sma200"]) if pd.notna(row["sma200"]) else float("nan")
    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else float("nan")
    macd_hist = float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else float("nan")
    macd_hist_prev = float(prev["macd_hist"]) if pd.notna(prev["macd_hist"]) else float("nan")

    dist50 = (close / sma50 - 1.0) if pd.notna(sma50) and sma50 else float("nan")
    dist200 = (close / sma200 - 1.0) if pd.notna(sma200) and sma200 else float("nan")

    above50 = bool(pd.notna(sma50) and close > sma50)
    above200 = bool(pd.notna(sma200) and close > sma200)

    return {
        "close": close,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "macd_hist_prev": macd_hist_prev,
        "dist50_pct": dist50 * 100 if pd.notna(dist50) else float("nan"),
        "dist200_pct": dist200 * 100 if pd.notna(dist200) else float("nan"),
        "above50": above50,
        "above200": above200,
        "wow_pct": _pct_change_week(df["close"]),
    }


def detect_opportunities(
    symbol: str,
    theme: str,
    df: pd.DataFrame,
    bench_close: pd.Series,
    rs_lookback: int,
) -> List[Dict[str, Any]]:
    snaps = snapshot_last_row(df)
    sym_ret, bench_ret = relative_strength_vs_benchmark(
        df["close"], bench_close, rs_lookback
    )
    rs_spread_pct = (sym_ret - bench_ret) * 100 if np.isfinite(sym_ret + bench_ret) else float("nan")

    trimmed_sym = df["close"].iloc[:-5]
    trimmed_bench = bench_close.reindex(df.index).dropna()
    common_idx = trimmed_sym.index.intersection(trimmed_bench.index)
    sym_ret_prev, bench_ret_prev = relative_strength_vs_benchmark(
        trimmed_sym.loc[common_idx],
        trimmed_bench.loc[common_idx],
        rs_lookback,
    )
    if not np.isfinite(sym_ret_prev):
        rs_improving = False
    else:
        prev_spread = (
            (sym_ret_prev - bench_ret_prev) * 100 if np.isfinite(bench_ret_prev) else float("nan")
        )
        rs_improving = (
            np.isfinite(rs_spread_pct)
            and np.isfinite(prev_spread)
            and rs_spread_pct > prev_spread + 0.1
        )

    ideas: List[Dict[str, Any]] = []

    close = snaps["close"]
    sma50 = snaps["sma50"]
    sma200 = snaps["sma200"]
    rsi = snaps["rsi"]

    if (
        np.isfinite(sma50)
        and np.isfinite(sma200)
        and sma50 > sma200
        and close > sma200 * 0.97
        and abs(close / sma50 - 1.0) <= 0.035
    ):
        ideas.append(
            {
                "theme": theme,
                "symbol": symbol,
                "tag": "pullback_to_50dma_uptrend",
                "thesis": "Pullback toward 50DMA while 50>200 (stage uptrend).",
                "close": close,
                "rsi": rsi,
                "dist50_pct": snaps["dist50_pct"],
                "dist200_pct": snaps["dist200_pct"],
                "rs_vs_benchmark_pct": rs_spread_pct,
                "invalidation": (
                    f"Break/stick below ~{sma50 * 0.97:.2f} (50DMA zone) "
                    f"or lose 200DMA ~{sma200:.2f}."
                ),
            }
        )

    if np.isfinite(rsi) and rsi < 36 and snaps["above200"]:
        ideas.append(
            {
                "theme": theme,
                "symbol": symbol,
                "tag": "rsi_oversold_above200",
                "thesis": "RSI washed out but still above 200DMA (bounce hunting).",
                "close": close,
                "rsi": rsi,
                "dist50_pct": snaps["dist50_pct"],
                "dist200_pct": snaps["dist200_pct"],
                "rs_vs_benchmark_pct": rs_spread_pct,
                "invalidation": f"Close below 200DMA ~{sma200:.2f} or RSI fails to stabilize.",
            }
        )

    if (
        np.isfinite(snaps["macd_hist"])
        and np.isfinite(snaps["macd_hist_prev"])
        and snaps["macd_hist_prev"] < 0
        and snaps["macd_hist"] > snaps["macd_hist_prev"]
        and snaps["above50"]
    ):
        ideas.append(
            {
                "theme": theme,
                "symbol": symbol,
                "tag": "macd_hist_turn",
                "thesis": "MACD histogram curling positive from negative while price >50DMA.",
                "close": close,
                "rsi": rsi,
                "dist50_pct": snaps["dist50_pct"],
                "dist200_pct": snaps["dist200_pct"],
                "rs_vs_benchmark_pct": rs_spread_pct,
                "invalidation": "Histogram resumes lower lows or price loses 50DMA.",
            }
        )

    if rs_improving and np.isfinite(rs_spread_pct) and rs_spread_pct > 0:
        ideas.append(
            {
                "theme": theme,
                "symbol": symbol,
                "tag": "rs_vs_benchmark_improving",
                "thesis": "Relative strength vs SPY rising over recent sessions.",
                "close": close,
                "rsi": rsi,
                "dist50_pct": snaps["dist50_pct"],
                "dist200_pct": snaps["dist200_pct"],
                "rs_vs_benchmark_pct": rs_spread_pct,
                "invalidation": "RS vs benchmark rolls over (spread peaks).",
            }
        )

    return ideas
