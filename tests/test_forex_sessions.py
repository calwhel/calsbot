"""Tests for unified live forex session windows (single source of truth)."""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services import forex_claude_confirm as fcc
from app.services import forex_sessions as fs
from app.services.session_filter import is_in_allowed_session
from app.services.strategy_ta import eval_fx_killzone


class TestLiveForexSessionWindows(unittest.TestCase):
    OUTSIDE = [
        datetime(2026, 6, 18, 16, 0),
        datetime(2026, 6, 18, 10, 0),
        datetime(2026, 6, 18, 11, 0),
        datetime(2026, 6, 18, 9, 0),   # London ends 09:00 exclusive
        datetime(2026, 6, 18, 5, 0),
    ]
    INSIDE = [
        (datetime(2026, 6, 18, 7, 0), "london"),
        (datetime(2026, 6, 18, 8, 0), "london"),  # 08:00 UTC is inside London 06–09
        (datetime(2026, 6, 18, 13, 0), "new_york"),
        (datetime(2026, 6, 18, 2, 0), "asia"),
    ]

    def test_outside_windows_not_in_live_session(self):
        for dt in self.OUTSIDE:
            self.assertFalse(
                fs.in_live_forex_session(dt),
                msg=f"expected outside at {dt}",
            )

    def test_inside_windows_active_session(self):
        for dt, expected in self.INSIDE:
            self.assertTrue(fs.in_live_forex_session(dt), msg=str(dt))
            self.assertEqual(fs.active_live_forex_session(dt), expected)

    def test_session_filter_matches_confirm_gate(self):
        for dt in self.OUTSIDE:
            self.assertFalse(fcc.in_forex_claude_confirm_session(dt))
            self.assertFalse(fs.in_live_forex_session(dt))
        for dt, _ in self.INSIDE:
            self.assertEqual(
                fcc.in_forex_claude_confirm_session(dt),
                fs.in_live_forex_session(dt),
            )

    def test_wizard_london_session_uses_canonical_window(self):
        cfg = {"sessions_enabled": True, "allowed_sessions": ["london"]}
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 18, 7, 30))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 18, 10, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_live_forex_session_allowed_requires_canonical_window(self):
        ok, reason = fs.live_forex_session_allowed({}, datetime(2026, 6, 18, 11, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_live_forex_session")
        ok, _ = fs.live_forex_session_allowed({}, datetime(2026, 6, 18, 7, 0))
        self.assertTrue(ok)


class TestAsianKzUnchanged(unittest.TestCase):
    def test_asian_kz_ict_window_20_to_23(self):
        with patch("app.services.strategy_ta.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 6, 18, 21, 0)
            inside, msg = eval_fx_killzone({"killzone": "asian_kz"})
        self.assertTrue(inside)
        self.assertIn("20-23", msg)

        with patch("app.services.strategy_ta.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 6, 18, 2, 0)
            inside, _ = eval_fx_killzone({"killzone": "asian_kz"})
        self.assertFalse(inside)


class TestLiveForexScanGate(unittest.TestCase):
    def test_outside_hours_block_scan_gate(self):
        for dt in (
            datetime(2026, 6, 18, 16, 0),
            datetime(2026, 6, 18, 10, 0),
            datetime(2026, 6, 18, 11, 0),
            datetime(2026, 6, 18, 9, 0),
        ):
            ok, reason = fs.live_forex_session_allowed({}, dt)
            self.assertFalse(ok, msg=str(dt))
            self.assertEqual(reason, "outside_live_forex_session")

    def test_inside_hours_allow_scan_gate(self):
        for dt in (
            datetime(2026, 6, 18, 7, 0),
            datetime(2026, 6, 18, 13, 0),
            datetime(2026, 6, 18, 2, 0),
        ):
            ok, _ = fs.live_forex_session_allowed({}, dt)
            self.assertTrue(ok, msg=str(dt))

    def test_inside_window_must_match_for_claude_confirm(self):
        """Option A: confirm gate and scan gate share forex_sessions."""
        dt = datetime(2026, 6, 18, 7, 30)
        self.assertTrue(fs.in_live_forex_session(dt))
        self.assertTrue(fcc.in_forex_claude_confirm_session(dt))
        dt_out = datetime(2026, 6, 18, 10, 0)
        self.assertFalse(fs.in_live_forex_session(dt_out))
        self.assertFalse(fcc.in_forex_claude_confirm_session(dt_out))


class TestForexClaudeConfirmUsesSharedSessions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fcc.reset_confirm_cache_for_tests()
        self._env = dict(os.environ)
        os.environ["ENABLE_FOREX_CLAUDE_CONFIRM"] = "1"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        fcc.reset_confirm_cache_for_tests()

    async def test_confirm_gate_blocks_outside_shared_window_without_api(self):
        with patch.object(fcc, "_call_claude", new_callable=AsyncMock) as mock_claude:
            ok, reason = await fcc.maybe_forex_claude_confirm(
                strategy_id=1,
                symbol="EURUSD",
                direction="LONG",
                entry=1.08,
                sl=1.07,
                tp=1.09,
                conditions_met=["✅ x"],
                price_data={"price": 1.08},
                timeframe="15m",
                now_utc=datetime(2026, 6, 18, 10, 0),
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_confirm_session")
        mock_claude.assert_not_called()


if __name__ == "__main__":
    unittest.main()
