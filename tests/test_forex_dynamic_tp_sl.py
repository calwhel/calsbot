"""Tests for live forex Dynamic TP/SL (Claude-set levels)."""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services import forex_claude_confirm as fcc


class TestClampValidateLevels(unittest.TestCase):
    def test_valid_sl35_tp90(self):
        out = fcc.clamp_validate_dynamic_levels(35, 90)
        self.assertIsNotNone(out)
        sl, tp, rr = out
        self.assertEqual(sl, 35)
        self.assertEqual(tp, 90)
        self.assertAlmostEqual(rr, 90 / 35, places=2)

    def test_sl80_clamped_to_60(self):
        out = fcc.clamp_validate_dynamic_levels(80, 150)
        self.assertIsNotNone(out)
        sl, tp, rr = out
        self.assertEqual(sl, 60)
        self.assertEqual(tp, 150)
        self.assertGreaterEqual(rr, 2.0)

    def test_tp50_sl30_rr_rejected(self):
        self.assertIsNone(fcc.clamp_validate_dynamic_levels(30, 50))

    def test_sl_below_min_clamped(self):
        out = fcc.clamp_validate_dynamic_levels(10, 40)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], 20)


class TestDynamicTpSlFire(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fcc.reset_confirm_cache_for_tests()
        self._env = dict(os.environ)
        os.environ.pop("ENABLE_FOREX_CLAUDE_CONFIRM", None)
        os.environ.pop("DISABLE_FOREX_CLAUDE_CONFIRM", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        fcc.reset_confirm_cache_for_tests()

    def _base_kwargs(self):
        return dict(
            strategy_id=1,
            symbol="EURUSD",
            direction="LONG",
            entry=1.0800,
            static_sl=1.0780,
            static_tp=1.0840,
            static_sl_pct=0.18,
            static_tp_pct=0.37,
            config={"dynamic_tp_sl": True},
            conditions_met=["✅ test"],
            price_data={"price": 1.08, "price_source": "spot_live", "rsi": 55},
            timeframe="15m",
            now_utc=datetime(2026, 6, 18, 7, 30),
        )

    async def test_valid_claude_levels_applied(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(
                {"sl_pips": 35, "tp_pips": 90, "reason": "clean structure"},
                0.001,
                "{}",
            ),
        ):
            res = await fcc.resolve_forex_claude_fire(**self._base_kwargs())
        self.assertTrue(res.allowed_to_fire)
        self.assertTrue(res.used_dynamic)
        self.assertEqual(res.sl_pips, 35)
        self.assertEqual(res.tp_pips, 90)
        self.assertAlmostEqual(res.tp_pips / res.sl_pips, 90 / 35, places=2)
        self.assertTrue(fcc.assert_valid_sl_price(res.sl_price))

    async def test_sl80_over_max_clamped_or_static_fallback(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(
                {"sl_pips": 80, "tp_pips": 100, "reason": "wide"},
                0.001,
                "{}",
            ),
        ):
            res = await fcc.resolve_forex_claude_fire(**self._base_kwargs())
        if res.used_dynamic:
            self.assertLessEqual(res.sl_pips, 60)
        else:
            self.assertTrue(res.static_fallback)
            self.assertEqual(res.sl_price, 1.0780)

    async def test_bad_rr_static_fallback(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(
                {"sl_pips": 30, "tp_pips": 50, "reason": "tight"},
                0.001,
                "{}",
            ),
        ):
            res = await fcc.resolve_forex_claude_fire(**self._base_kwargs())
        self.assertTrue(res.allowed_to_fire)
        self.assertFalse(res.used_dynamic)
        self.assertTrue(res.static_fallback)
        self.assertEqual(res.sl_price, 1.0780)

    async def test_api_error_static_fallback_still_fires(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(None, 0.0, "timeout"),
        ):
            res = await fcc.resolve_forex_claude_fire(**self._base_kwargs())
        self.assertTrue(res.allowed_to_fire)
        self.assertTrue(res.static_fallback)
        self.assertTrue(fcc.assert_valid_sl_price(res.sl_price))

    async def test_unticked_no_claude_call(self):
        kw = self._base_kwargs()
        kw["config"] = {"dynamic_tp_sl": False}
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            res = await fcc.resolve_forex_claude_fire(**kw)
        mock_claude.assert_not_called()
        self.assertFalse(res.claude_called)
        self.assertEqual(res.sl_price, 1.0780)

    async def test_stale_price_blocks_dynamic_fire(self):
        kw = self._base_kwargs()
        kw["price_data"] = {"price": 1.08, "price_source": "unknown"}
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            res = await fcc.resolve_forex_claude_fire(**kw)
        mock_claude.assert_not_called()
        self.assertFalse(res.allowed_to_fire)
        self.assertIn("price_gate", res.reason)

    async def test_non_ctrader_metal_price_blocks_dynamic(self):
        kw = self._base_kwargs()
        kw["symbol"] = "XAUUSD"
        kw["price_data"] = {
            "price": 2650.0,
            "price_source": "spot_live",
            "live_source": "coinbase",
        }
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            res = await fcc.resolve_forex_claude_fire(**kw)
        mock_claude.assert_not_called()
        self.assertFalse(res.allowed_to_fire)
        self.assertIn("non_ctrader_price", res.reason)

    async def test_gate_skipped_logged_on_static_fallback(self):
        kw = self._base_kwargs()
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(None, 0.0, "timeout"),
        ), patch.object(fcc.logger, "warning") as mock_log:
            res = await fcc.resolve_forex_claude_fire(**kw)
        self.assertTrue(res.allowed_to_fire)
        self.assertTrue(res.static_fallback)
        logged = " ".join(str(c) for c in mock_log.call_args_list)
        self.assertIn("[forex-claude] gate FALLBACK", logged)

    async def test_combined_confirm_false_skips(self):
        os.environ["ENABLE_FOREX_CLAUDE_CONFIRM"] = "1"
        kw = self._base_kwargs()
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(
                {"confirm": False, "sl_pips": 35, "tp_pips": 90, "reason": "no edge"},
                0.001,
                "{}",
            ),
        ):
            res = await fcc.resolve_forex_claude_fire(**kw)
        self.assertFalse(res.allowed_to_fire)
        self.assertFalse(res.used_dynamic)

    async def test_combined_confirm_true_applies_levels(self):
        os.environ["ENABLE_FOREX_CLAUDE_CONFIRM"] = "1"
        kw = self._base_kwargs()
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(
                {"confirm": True, "sl_pips": 35, "tp_pips": 90, "reason": "go"},
                0.001,
                "{}",
            ),
        ) as mock_claude:
            res = await fcc.resolve_forex_claude_fire(**kw)
            await fcc.resolve_forex_claude_fire(**kw)
        self.assertTrue(res.allowed_to_fire)
        self.assertTrue(res.used_dynamic)
        self.assertEqual(mock_claude.await_count, 1)


class TestAssertValidSl(unittest.TestCase):
    def test_rejects_zero_and_none(self):
        self.assertFalse(fcc.assert_valid_sl_price(0))
        self.assertFalse(fcc.assert_valid_sl_price(None))
        self.assertTrue(fcc.assert_valid_sl_price(1.078))


if __name__ == "__main__":
    unittest.main()
