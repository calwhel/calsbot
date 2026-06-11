"""Execution-level analytics helpers (MFE/MAE, TP tuning)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def strategy_tp_distance_pips(cfg: Optional[dict]) -> Optional[float]:
    """Configured TP distance in pips when the strategy uses pip-based exits."""
    if not cfg:
        return None
    ex = cfg.get("exit") or {}
    for key in ("take_profit_pips", "tp1_pips"):
        val = cfg.get(key) or ex.get(key)
        if val is not None:
            try:
                p = float(val)
                if p > 0:
                    return p
            except (TypeError, ValueError):
                pass
    return None


def compute_tp_tuning_stat(
    execs: List[Any],
    cfg: Optional[dict],
) -> Optional[Dict[str, Any]]:
    """Avg peak profit on non-wins vs configured TP — surfaces TP-too-far hints."""
    tp_pips = strategy_tp_distance_pips(cfg)
    if not tp_pips or tp_pips <= 0:
        return None
    non_wins = [
        e for e in execs
        if getattr(e, "outcome", None) in ("LOSS", "BREAKEVEN")
        and getattr(e, "mfe_pips", None) is not None
    ]
    if len(non_wins) < 3:
        return None
    avg_mfe = sum(float(e.mfe_pips) for e in non_wins) / len(non_wins)
    ratio = avg_mfe / tp_pips if tp_pips > 0 else 0.0
    hint = None
    if ratio < 0.85:
        hint = (
            f"Trades peak at +{avg_mfe:.0f} pips on average but TP is +{tp_pips:.0f} "
            f"— consider a closer target or partial TP."
        )
    return {
        "avg_mfe_non_wins_pips": round(avg_mfe, 1),
        "tp_distance_pips": round(tp_pips, 1),
        "mfe_to_tp_ratio": round(ratio, 2),
        "sample_size": len(non_wins),
        "hint": hint,
    }
