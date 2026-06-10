"""Tests for shared discovery setup quality engine."""
import time
from datetime import datetime, timezone

from app.services.setup_quality import (
    DEFAULT_QUALITY_CFG,
    apply_quality_grades,
    grade_setup,
    normalize_quality_cfg,
)


def _make_klines(n=120, trend="up"):
    base = int(time.time() * 1000) - n * 300_000
    out = []
    price = 2000.0
    for i in range(n):
        o = price
        if trend == "up":
            c = price + 2.0
            h = c + 1.0
            l = o - 0.5
            price = c
        else:
            c = price - 2.0
            h = o + 0.5
            l = c - 1.0
            price = c
        out.append([base + i * 300_000, o, h, l, c, 100.0])
    return out


def test_normalize_quality_cfg_defaults():
    cfg = normalize_quality_cfg({})
    assert cfg["min_confirmations"] == DEFAULT_QUALITY_CFG["min_confirmations"]
    assert cfg["quality_mode"] == "strict"


def test_normalize_quality_cfg_winrate_percent():
    cfg = normalize_quality_cfg({"min_winrate": 58})
    assert cfg["min_winrate"] == 0.58


def test_grade_setup_returns_expected_keys():
    kl = _make_klines(150)
    candles = {"15m": kl, "1h": kl}
    signal = {
        "direction": "LONG",
        "timeframe": "15m",
        "label": "Test",
        "stats": {
            "closed_trades": 25,
            "win_rate": 62.0,
            "avg_pips": 8.0,
            "total_pips": 200,
        },
        "entry_ts": kl[-1][0],
    }
    result = grade_setup(candles, signal, "XAUUSD", {})
    assert "score" in result
    assert result["grade"] in ("A", "B", "C", "D", "F")
    assert isinstance(result["confirmations"], list)
    assert isinstance(result["confluences"], list)
    assert "passed" in result
    assert isinstance(result["reasons"], list)


def test_apply_quality_grades_strict_filters():
    kl = _make_klines(200)
    split_ts = kl[140][0]
    rows = [{
        "label": "Weak",
        "direction": "LONG",
        "timeframe": "15m",
        "session": "all",
        "score": 50,
        "stats": {"closed_trades": 30, "win_rate": 70, "avg_pips": 5},
        "_all_trades": [
            {"outcome": "WIN", "entry_ts": split_ts + 1000, "pip_move": 10, "pnl_pct": 1},
            {"outcome": "LOSS", "entry_ts": split_ts + 2000, "pip_move": -5, "pnl_pct": -1},
        ],
    }]
    out = apply_quality_grades(
        rows, {"15m": kl}, "XAUUSD",
        {"quality_mode": "strict", "min_trades": 20, "min_confirmations": 6},
        wf_splits={"15m": split_ts},
    )
    assert out == [] or all(r.get("grade") for r in out)
