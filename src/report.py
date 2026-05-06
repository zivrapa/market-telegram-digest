"""Render Markdown digest + optional JSON artifact."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo


DISCLAIMER = (
    "Disclaimer: Educational automation — not investment advice. "
    "Verify data on your platform; markets gap and indicators lag."
)


def _fmt_pct(x: Any, digits: int = 2) -> str:
    if isinstance(x, (int, float)) and x == x:  # noqa: PLR0124 - NaN check
        return f"{x:.{digits}f}%"
    return "n/a"


def _fmt_num(x: Any, digits: int = 2) -> str:
    if isinstance(x, (int, float)) and x == x:
        return f"{x:.{digits}f}"
    return "n/a"


def build_markdown(snapshot: Dict[str, Any]) -> str:
    tz_name = snapshot.get("timezone_display", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    now = snapshot.get("generated_at")
    if isinstance(now, datetime):
        stamp = now.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
    else:
        stamp = str(now)

    bench = snapshot.get("benchmark_symbol", "SPY")
    rs_lb = snapshot.get("relative_strength_lookback_days", 20)

    lines: List[str] = [
        f"*Sunday market digest* — {stamp}",
        "",
        DISCLAIMER,
        "",
        f"Benchmark for RS spread: **{bench}** ({rs_lb}-session total return difference ×100).",
        "",
        "## Macro snapshot",
    ]

    macro: Dict[str, Any] = snapshot.get("macro") or {}
    vix_block = macro.get("^VIX") or macro.get("VIX")

    order = [s for s in ["SPY", "QQQ", "IWM"] if s in macro]
    for sym in order:
        m = macro[sym]
        lines.append(
            f"- **{sym}** close {_fmt_num(m.get('close'))} | "
            f"WoW {_fmt_pct(m.get('wow_pct'))} | "
            f"vs 50DMA {_fmt_pct(m.get('dist50_pct'))} | "
            f"vs 200DMA {_fmt_pct(m.get('dist200_pct'))} | "
            f"RSI(14) {_fmt_num(m.get('rsi'), 1)} | "
            f"MACDhist {_fmt_num(m.get('macd_hist'), 3)}"
        )

    lines.extend(["", "## Volatility"])
    if vix_block:
        lines.append(
            f"- **VIX** {_fmt_num(vix_block.get('close'), 2)} — {vix_block.get('regime', 'n/a')} "
            f"(vs 1y median {_fmt_num(vix_block.get('median_1y'), 2)})."
        )
    else:
        lines.append("- VIX data unavailable this run.")

    lines.extend(["", "## Theme radar"])

    themes: Dict[str, Any] = snapshot.get("themes") or {}
    leadership = snapshot.get("theme_rs_ranking") or []

    if leadership:
        lines.append("### RS leaderboard (theme primary vs benchmark)")
        for name, spread in leadership:
            lines.append(f"- **{name}** RS spread vs {bench}: {_fmt_pct(spread)}")
        lines.append("")

    for theme_name, payload in themes.items():
        primary = payload.get("primary_symbol")
        prim = (payload.get("symbols") or {}).get(primary, {})
        lines.append(f"### {theme_name.replace('_', ' ').title()} — `{primary}`")
        lines.append(
            f"- Close {_fmt_num(prim.get('close'))} | WoW {_fmt_pct(prim.get('wow_pct'))} | "
            f"vs 50 {_fmt_pct(prim.get('dist50_pct'))} | vs 200 {_fmt_pct(prim.get('dist200_pct'))} | "
            f"RSI {_fmt_num(prim.get('rsi'), 1)} | RS vs {bench} {_fmt_pct(prim.get('rs_vs_benchmark_pct'))}"
        )
        alts = payload.get("alternates") or []
        for sym in alts:
            blk = (payload.get("symbols") or {}).get(sym)
            if not blk:
                continue
            lines.append(
                f"  - `{sym}` close {_fmt_num(blk.get('close'))} | WoW {_fmt_pct(blk.get('wow_pct'))} | "
                f"RS vs {bench} {_fmt_pct(blk.get('rs_vs_benchmark_pct'))}"
            )
        breadth = payload.get("breadth_above50_pct")
        if breadth is not None:
            lines.append(f"- Breadth proxy (% names >50DMA): {_fmt_pct(breadth, 1)}")
        lines.append("")

    opps: List[Dict[str, Any]] = snapshot.get("opportunities") or []
    lines.extend(["## Opportunities (rule-based)", ""])
    if not opps:
        lines.append("_No patterns matched default rules this week._")
    else:
        for i, o in enumerate(opps, start=1):
            lines.append(
                f"{i}. `{o.get('symbol')}` ({o.get('theme')}) — **{o.get('tag')}** — {o.get('thesis')}"
            )
            lines.append(
                f"   - Close {_fmt_num(o.get('close'))} | RSI {_fmt_num(o.get('rsi'), 1)} | "
                f"vs50 {_fmt_pct(o.get('dist50_pct'))} | vs200 {_fmt_pct(o.get('dist200_pct'))} | "
                f"RS vs {bench} {_fmt_pct(o.get('rs_vs_benchmark_pct'))}"
            )
            lines.append(f"   - Invalidate: {o.get('invalidation')}")
            lines.append("")

    return "\n".join(lines).strip()


def snapshot_for_llm(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bulky fields if needed before LLM."""
    slim = dict(snapshot)
    slim.pop("macro_frames", None)
    return slim


def save_snapshot_json(snapshot: Dict[str, Any], path: str) -> None:
    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=_default)
