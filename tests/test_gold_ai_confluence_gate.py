"""Confluence gate for gold-ai TAKE execution."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.call_gates import meets_min_confluence_for_take, min_confluence_for_take


class _Cand:
    def __init__(self, checklist):
        self.raw = {"readiness_checklist": checklist}


def test_min_confluence_default_is_six():
    os.environ.pop("GOLD_AI_MIN_CONFLUENCE_FOR_TAKE", None)
    assert min_confluence_for_take() == 6


def test_blocks_when_confluence_below_threshold():
    cand = _Cand(
        {
            "htf_aligned": True,
            "at_entry": True,
            "displacement_ok": True,
            "reclaim_held": True,
            "momentum_ok": True,
        }
    )
    ok, reason = meets_min_confluence_for_take(cand)
    assert ok is False
    assert "confluence_5/" in reason


def test_passes_when_confluence_meets_threshold():
    cand = _Cand(
        {
            "htf_aligned": True,
            "at_entry": True,
            "displacement_ok": True,
            "reclaim_held": True,
            "momentum_ok": True,
            "rr_feasible": True,
        }
    )
    ok, reason = meets_min_confluence_for_take(cand)
    assert ok is True
    assert reason == ""
