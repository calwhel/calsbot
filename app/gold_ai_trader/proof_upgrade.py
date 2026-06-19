#!/usr/bin/env python3
"""Generate proof blocks for Gold AI Trader upgrade (run from repo root)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.config import env_defaults
from app.gold_ai_trader.context_levels import build_key_levels_block
from app.gold_ai_trader.context_regime import build_regime_block
from app.gold_ai_trader.learning import (
    compute_r_multiple,
    get_setup_stats,
    _build_learning_prompt,
)
from app.gold_ai_trader.pending_entry import broker_limit_supported
from app.gold_ai_trader.scanner import Candidate


class _Cfg:
    london_start_hour = 7
    london_end_hour = 10
    ny_start_hour = 13
    ny_end_hour = 16


class _DbStats:
    """In-memory fake for setup stats + learning prompt proof."""

    def __init__(self):
        from app.gold_ai_trader.models import GoldAiOutcome, GoldAiDecision

        self.GoldAiOutcome = GoldAiOutcome
        self.GoldAiDecision = GoldAiDecision
        self.outcomes = [
            GoldAiOutcome(
                decision_id=1,
                setup_type="sweep_pdh",
                session="london",
                result="loss",
                pnl=-0.8,
                r_multiple=-0.9,
                closed_ts=datetime.utcnow() - timedelta(days=1),
            ),
            GoldAiOutcome(
                decision_id=2,
                setup_type="sweep_pdh",
                session="london",
                result="loss",
                pnl=-1.1,
                r_multiple=-1.0,
                closed_ts=datetime.utcnow() - timedelta(days=2),
            ),
            GoldAiOutcome(
                decision_id=3,
                setup_type="sweep_pdh",
                session="london",
                result="win",
                pnl=2.4,
                r_multiple=2.1,
                closed_ts=datetime.utcnow() - timedelta(days=3),
            ),
            GoldAiOutcome(
                decision_id=4,
                setup_type="orb_break",
                session="new_york",
                result="win",
                pnl=1.5,
                r_multiple=1.8,
                closed_ts=datetime.utcnow() - timedelta(days=1),
            ),
        ]
        self.decisions = {
            1: GoldAiDecision(
                id=1,
                session="london",
                candidate_type="sweep_pdh",
                confidence=78,
                decision={
                    "direction": "short",
                    "entry": 4170.0,
                    "stop_loss": 4175.0,
                    "rationale": "Entered before reclaim close confirmed.",
                },
            ),
            2: GoldAiDecision(
                id=2,
                session="london",
                candidate_type="sweep_pdh",
                confidence=82,
                decision={
                    "direction": "short",
                    "entry": 4168.0,
                    "stop_loss": 4173.0,
                    "rationale": "Chased after displacement extended.",
                },
            ),
            3: GoldAiDecision(
                id=3,
                session="london",
                candidate_type="sweep_pdh",
                confidence=85,
                decision={
                    "direction": "short",
                    "entry": 4165.0,
                    "stop_loss": 4170.0,
                    "rationale": "Clean sweep + close back inside.",
                },
            ),
            4: GoldAiDecision(
                id=4,
                session="new_york",
                candidate_type="orb_break",
                confidence=80,
                decision={
                    "direction": "long",
                    "entry": 4158.0,
                    "stop_loss": 4153.0,
                    "rationale": "ORB continuation with RVOL.",
                },
            ),
        }

    def query(self, *models):
        self._model = models[0] if len(models) == 1 else models
        return self

    def join(self, other, *a, **k):
        if isinstance(self._model, type):
            self._model = (self._model, other)
        return self

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def first(self):
        return None

    def all(self):
        if isinstance(self._model, tuple):
            pairs = []
            for o in self.outcomes:
                d = self.decisions.get(o.decision_id)
                if d and d.session == "london":
                    pairs.append((o, d))
            return pairs[: getattr(self, "_limit", 12)]
        if hasattr(self._model, "__name__"):
            if self._model.__name__ == "GoldAiOutcome":
                return list(self.outcomes)
            if self._model.__name__ == "GoldAiDecision":
                return list(self.decisions.values())
        return []
    def scalar(self):
        return 4


async def main():
    from app.services.tradfi_prices import get_klines
    from app.gold_ai_trader.config import SYMBOL, ASSET_CLASS

    cfg = env_defaults()
    now = datetime.utcnow()
    k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
    k1h = await get_klines(SYMBOL, ASSET_CLASS, "1h", 50) or []
    k_daily = await get_klines(SYMBOL, ASSET_CLASS, "1d", 5) or []

    spot = float(k5[-1][4]) if k5 else 4156.40
    closes = [float(r[4]) for r in k5 if r and len(r) >= 5]
    atr = 4.2
    if len(closes) >= 15:
        trs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        atr = sum(trs[-14:]) / 14

    session = "london" if 7 <= now.hour < 10 else "new_york"
    levels = build_key_levels_block(
        spot=spot,
        atr=atr,
        session=session,
        cfg=_Cfg(),
        now=now,
        k_daily=k_daily,
        k_1h=k1h,
        k_5m=k5,
    )
    regime = build_regime_block(k1h, k5)

    print("=== PROOF #1 KEY LEVELS ===")
    print("\n".join(levels))
    print()
    print("=== PROOF #6 REGIME ===")
    print("\n".join(regime))
    print()
    print("=== PROOF #4 PENDING ENTRY ===")
    print(f"broker_limit_supported: {broker_limit_supported()}")
    print(
        "Path: use_limit_entry=true → broker LIMIT unavailable → "
        "gold entry-watch pending at Claude entry; fills with market when "
        "spot touches entry; expires min(30m, session end)."
    )
    print()
    print("=== PROOF #3 get_setup_stats() ===")
    db = _DbStats()
    print(json.dumps(get_setup_stats(db), indent=2))
    print()
    print("=== PROOF #2 LEARNING PROMPT (excerpt → digest input) ===")
    prompt = _build_learning_prompt(db, "london")
    print(prompt[:1200])
    print()
    print("=== SAMPLE LESSONS DIGEST (deterministic mock from stats) ===")
    stats = get_setup_stats(db)
    london_sweep = next((s for s in stats if s["setup_type"] == "sweep_pdh"), {})
    digest = (
        f"London sweep_pdh: {london_sweep.get('wins', 0)}/{london_sweep.get('trades', 0)} "
        f"this period (avg R {london_sweep.get('avg_r_multiple')}). "
        "The two losses entered before reclaim/displacement confirmation — "
        "require close back inside the swept level + ≥0.8×ATR displacement body "
        "before entry. The win waited for reclaim at the sweep edge. "
        "Do not chase extended moves away from the entry zone."
    )
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
