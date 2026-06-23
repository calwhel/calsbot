"""Tests for live forex Claude final confirmation gate."""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services import anthropic_budget_guard as budget
from app.services import forex_claude_confirm as fcc


class TestForexClaudeConfirmSessions(unittest.TestCase):
    def test_asia_window_01_to_04_utc_only(self):
        inside = [
            datetime(2026, 6, 18, 1, 0),
            datetime(2026, 6, 18, 3, 59),
        ]
        outside = [
            datetime(2026, 6, 18, 0, 59),
            datetime(2026, 6, 18, 4, 0),
            datetime(2026, 6, 18, 10, 0),
        ]
        for dt in inside:
            self.assertEqual(fcc.active_confirm_session(dt), "asia", msg=str(dt))
            self.assertTrue(fcc.in_forex_claude_confirm_session(dt), msg=str(dt))
        for dt in outside:
            self.assertNotEqual(fcc.active_confirm_session(dt), "asia", msg=str(dt))

    def test_london_and_ny_windows(self):
        self.assertEqual(
            fcc.active_confirm_session(datetime(2026, 6, 18, 7, 0)),
            "london",
        )
        self.assertEqual(
            fcc.active_confirm_session(datetime(2026, 6, 18, 13, 0)),
            "new_york",
        )

    def test_confirm_matches_forex_sessions_module(self):
        from app.services import forex_sessions as fs

        for hour in (7, 13, 2):
            dt = datetime(2026, 6, 18, hour, 0)
            self.assertEqual(
                fcc.in_forex_claude_confirm_session(dt),
                fs.in_live_forex_session(dt),
            )


class TestForexClaudeConfirmGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fcc.reset_confirm_cache_for_tests()
        budget.reset_daily_spend_for_tests()
        self._env = dict(os.environ)
        os.environ["ENABLE_FOREX_CLAUDE_CONFIRM"] = "1"
        os.environ["ANTHROPIC_DAILY_BUDGET_USD"] = "5.0"
        os.environ.pop("DISABLE_FOREX_CLAUDE_CONFIRM", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        fcc.reset_confirm_cache_for_tests()
        budget.reset_daily_spend_for_tests()

    async def test_local_session_gate_blocks_without_claude_call(self):
        """Outside confirm session → skip, no API call."""
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=1,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ RSI"],
                price_data={"price": 1.08},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 10, 0),
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_confirm_session")
        mock_claude.assert_not_called()

    async def test_fail_closed_on_api_error(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=(None, 0.0, "timeout"),
        ):
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=2,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ EMA"],
                price_data={"price": 1.08},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 7, 30),
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "timeout")

    async def test_budget_cap_blocks_without_claude_call(self):
        budget.record_call("other", 5.0, now=datetime(2026, 6, 18, 7, 0))
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=3,
                symbol="GBPUSD",
                direction="SHORT",
                entry=1.27,
                sl=1.28,
                tp=1.25,
                conditions_met=["✅ trend"],
                price_data={"price": 1.27},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 7, 15),
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "anthropic_daily_budget_exceeded")
        mock_claude.assert_not_called()

    async def test_confirm_true_allows_fire(self):
        with patch.object(
            fcc,
            "_call_claude",
            new_callable=AsyncMock,
            return_value=({"confirm": True, "reason": "clean setup"}, 0.001, "{}"),
        ):
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=4,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ all"],
                price_data={"price": 1.08, "rsi": 55},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 12, 30),
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "clean setup")

    async def test_cache_prevents_duplicate_calls_same_bar(self):
        mock = AsyncMock(
            return_value=({"confirm": True, "reason": "ok"}, 0.001, "{}"),
        )
        with patch.object(fcc, "_call_claude", mock):
            args = dict(
                strategy_id=5,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ x"],
                price_data={"price": 1.08},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 2, 15),
            )
            ok1, _ = await fcc.maybe_forex_claude_confirm(**args)
            ok2, reason2 = await fcc.maybe_forex_claude_confirm(**args)
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertTrue(reason2.startswith("cache:"))
        self.assertEqual(mock.await_count, 1)

    async def test_disabled_passes_through_without_call(self):
        os.environ.pop("ENABLE_FOREX_CLAUDE_CONFIRM", None)
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=6,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ x"],
                price_data={"price": 1.08},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 2, 0),
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "disabled")
        mock_claude.assert_not_called()


class TestSessionFilterAsia(unittest.TestCase):
    def test_asia_session_in_session_filter(self):
        from app.services.session_filter import is_in_allowed_session

        cfg = {"sessions_enabled": True, "allowed_sessions": ["asia"]}
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 18, 2, 30))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 18, 5, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_asia_matches_live_forex_canonical_window(self):
        from app.services import forex_sessions as fs
        from app.services.session_filter import is_in_allowed_session

        cfg = {"sessions_enabled": True, "allowed_sessions": ["asia"]}
        dt = datetime(2026, 6, 18, 2, 30)
        ok_filter, _ = is_in_allowed_session(cfg, dt)
        self.assertEqual(ok_filter, fs.in_live_forex_session(dt))


if __name__ == "__main__":
    unittest.main()
