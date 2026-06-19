"""Tests for Claude call cost controls."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.call_gates import (
    passes_quality_gates,
    nearest_level_distance,
    last_closed_body_atr,
    atr_from_klines,
)
from app.gold_ai_trader.scanner import Candidate, setup_cooldown_elapsed, record_claude_invocation


class _Cfg:
    london_start_hour = 7
    london_end_hour = 10
    ny_start_hour = 13
    ny_end_hour = 16


def _bar(ts, o, h, l, c, v=100.0):
    return [int(ts.timestamp() * 1000), o, h, l, c, v]


def test_nearest_level_distance():
    assert nearest_level_distance(4156.0, [4150.0, 4160.0]) == 4.0


def test_body_atr_gate():
    now = datetime(2026, 6, 18, 8, 0, 0)
    k5 = []
    base = now - timedelta(minutes=len(range(20)) * 5)
    price = 4156.0
    for i in range(20):
        ts = base + timedelta(minutes=i * 5)
        k5.append(_bar(ts, price, price + 1, price - 1, price + 0.01, 100))
    k5[-2] = _bar(now - timedelta(minutes=10), 4156, 4162, 4155, 4161, 200)
    atr = atr_from_klines(k5)
    assert last_closed_body_atr(k5, atr) >= 0.5


def test_passes_quality_gates_near_level():
    now = datetime(2026, 6, 18, 8, 30, 0)
    k5 = []
    for i in range(25):
        ts = now - timedelta(minutes=(25 - i) * 5)
        k5.append(_bar(ts, 4156, 4160, 4154, 4156.5, 120 if i == 24 else 100))
    k5[-2] = _bar(now - timedelta(minutes=10), 4154, 4162, 4153, 4160, 250)
    k_daily = [_bar(now - timedelta(days=1), 4140, 4160, 4138, 4155, 1)]
    k1h = k5
    cand = Candidate(
        type="sweep_pdh",
        direction="SHORT",
        detail="Swept PDH 4160.0 then closed below (1.1× ATR)",
        quality_atr=1.1,
        sig_key="sweep_pdh:SHORT",
        raw={},
    )
    ok, reason = passes_quality_gates(
        cand,
        price=4159.5,
        session="london",
        cfg=_Cfg(),
        now=now,
        k5=k5,
        k_daily=k_daily,
        k_1h=k1h,
    )
    assert ok, reason


def test_setup_cooldown_after_record():
    c = Candidate("sweep_pdl", "LONG", "", 1.0, "sweep_pdl:LONG", {})
    assert setup_cooldown_elapsed("sweep_pdl:LONG")
    record_claude_invocation(c)
    assert not setup_cooldown_elapsed("sweep_pdl:LONG")
