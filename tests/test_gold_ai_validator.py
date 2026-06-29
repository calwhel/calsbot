"""Tests for Gold AI validator, entry routing, trade bands, funnel persist."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest

from app.gold_ai_trader.config import env_defaults
from app.gold_ai_trader.decision_validator import validate_take_decision, MIN_RR
from app.gold_ai_trader.entry_routing import use_limit_entry_for_setup
from app.gold_ai_trader.context_bands import build_trade_bands_block
from app.gold_ai_trader.funnel import record as funnel_record, snapshot as funnel_snapshot
from app.gold_ai_trader.learning import format_setup_stats_block
from app.gold_ai_trader.setup_toggles import cisd_modifier_enabled, setup_enabled


class TestDecisionValidator:
    def test_long_valid_rr(self):
        decision = {
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2643.5,
            "take_profit": 2663.0,
        }
        ok, reason, out = validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
            key_levels=[2660.0, 2640.0],
        )
        assert ok is True
        assert reason.startswith("validator:ok")
        assert out.get("validator_note")

    def test_rejects_sl_too_wide_when_cap_enabled(self, monkeypatch):
        monkeypatch.setenv("GOLD_AI_MAX_SL_ATR", "1.0")
        import importlib
        import app.gold_ai_trader.decision_validator as dv

        importlib.reload(dv)
        decision = {
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2642.0,
            "take_profit": 2660.0,
        }
        ok, reason, _ = dv.validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
            key_levels=[],
        )
        assert ok is False
        assert "sl_too_wide" in reason
        monkeypatch.delenv("GOLD_AI_MAX_SL_ATR", raising=False)
        importlib.reload(dv)

    def test_allows_wide_sl_for_swing_by_default(self):
        """Regression: IFVG-style zone invalidation beyond 1×ATR must execute."""
        decision = {
            "direction": "LONG",
            "entry": 4082.40,
            "stop_loss": 4076.00,
            "take_profit": 4098.00,
        }
        ok, reason, out = validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=4082.40,
            atr=1.0,
            setup_detail="IFVG zone 4080.0–4083.0",
            key_levels=[4098.0, 4072.0],
        )
        assert ok is True
        assert reason.startswith("validator:ok")
        assert out.get("validator_sl_atr") == 6.4

    def test_rejects_tight_sl_by_default_floor(self):
        decision = {
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2648.0,  # 20 pips on XAUUSD (below 60-pip floor)
            "take_profit": 2656.0,
        }
        ok, reason, out = validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
            key_levels=[],
        )
        assert ok is False
        assert "sl_too_tight" in reason
        assert out.get("validator_sl_pips") == 20.0

    def test_allows_low_rr_by_default(self):
        """Claude may target session edge with <2R — validator must not block."""
        decision = {
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2643.8,
            "take_profit": 2653.1,
        }
        ok, reason, out = validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
            key_levels=[],
        )
        assert ok is True
        assert reason.startswith("validator:ok")
        assert out.get("validator_note") == "rr=0.50"

    def test_tp_adjusted_when_min_rr_enabled(self, monkeypatch):
        monkeypatch.setenv("GOLD_AI_MIN_RR", "2.0")
        import importlib
        import app.gold_ai_trader.decision_validator as dv

        importlib.reload(dv)
        decision = {
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2643.8,
            "take_profit": 2653.1,
        }
        ok, reason, out = dv.validate_take_decision(
            decision,
            candidate_direction="LONG",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
            key_levels=[2663.0],
        )
        assert ok is True
        assert reason == "validator:ok_tp_adjusted"
        assert float(out["take_profit"]) == 2663.0
        monkeypatch.delenv("GOLD_AI_MIN_RR", raising=False)
        importlib.reload(dv)

    def test_rejects_bad_price_order(self):
        decision = {
            "direction": "SHORT",
            "entry": 2650.0,
            "stop_loss": 2648.0,
            "take_profit": 2655.0,
        }
        ok, reason, _ = validate_take_decision(
            decision,
            candidate_direction="SHORT",
            spot=2650.0,
            atr=2.0,
            setup_detail="",
        )
        assert ok is False
        assert "short_price_order" in reason


class TestEntryRouting:
    def test_zone_setup_uses_limit(self):
        cfg = env_defaults()
        assert use_limit_entry_for_setup("fvg_retrace_bear", cfg) is True

    def test_ob_setup_uses_market_by_default(self):
        cfg = env_defaults()
        assert use_limit_entry_for_setup("ob_bull", cfg) is False
        assert use_limit_entry_for_setup("ob_bear", cfg) is False

    def test_ob_limit_when_disabled(self, monkeypatch):
        monkeypatch.setenv("GOLD_AI_OB_MARKET_ENTRY", "false")
        cfg = env_defaults()
        assert use_limit_entry_for_setup("ob_bull", cfg) is True

    def test_momentum_setup_uses_market(self):
        cfg = env_defaults()
        assert use_limit_entry_for_setup("sweep_pdh", cfg) is False
        assert use_limit_entry_for_setup("judas_bull", cfg) is False
        assert use_limit_entry_for_setup("asian_sweep_bull", cfg) is False

    def test_force_market_env(self, monkeypatch):
        monkeypatch.setenv("GOLD_AI_FORCE_MARKET_ENTRY", "true")
        cfg = env_defaults()
        assert use_limit_entry_for_setup("ob_bull", cfg) is False


class TestTradeBands:
    def test_builds_bands_with_zone(self):
        lines = build_trade_bands_block(
            spot=2650.0,
            atr=2.0,
            direction="LONG",
            setup_detail="zone 2649.0–2651.0 reclaim",
            key_levels=[2660.0, 2640.0],
        )
        text = "\n".join(lines)
        assert "TRADE BANDS" in text
        assert "2651.0" in text or "2649.0" in text


class TestFunnelCounters:
    def test_validator_and_news_counters(self):
        funnel_record("validator_rejected", setup="ob_bull", reason="test")
        funnel_record("news_blocked", reason="NFP")
        snap = funnel_snapshot()
        assert snap["validator_rejected"] >= 1
        assert snap["news_blocked"] >= 1


class TestSetupTogglesTier2:
    def test_judas_default_off(self):
        os.environ.pop("GOLD_AI_SETUP_JUDAS_BULL", None)
        assert setup_enabled("judas_bull") is False

    def test_asian_sweep_default_on(self):
        os.environ.pop("GOLD_AI_SETUP_ASIAN_SWEEP_BULL", None)
        assert setup_enabled("asian_sweep_bull") is True

    def test_cisd_modifier_default_off(self):
        os.environ.pop("GOLD_AI_CISD_MODIFIER", None)
        assert cisd_modifier_enabled() is False


class TestSetupStatsBlock:
    def test_empty_stats_block(self, monkeypatch):
        monkeypatch.setattr(
            "app.gold_ai_trader.learning.get_setup_stats",
            lambda db, days=14: [],
        )
        lines = format_setup_stats_block(None, session="london")
        assert any("SETUP STATS" in ln for ln in lines)
        assert any("No closed trades" in ln for ln in lines)
