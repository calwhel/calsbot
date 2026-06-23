"""Tests for unified live forex session windows (single source of truth)."""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services import forex_claude_confirm as fcc
from app.services import forex_sessions as fs
from app.services.forex_engine import in_session
from app.services.session_filter import is_in_allowed_session
from app.services.strategy_ta import eval_forex_session, eval_fx_killzone


class TestLiveForexSessionWindows(unittest.TestCase):
    OUTSIDE = [
        datetime(2026, 6, 18, 5, 0),
        datetime(2026, 6, 18, 21, 0),
        datetime(2026, 6, 18, 22, 0),
        datetime(2026, 6, 18, 0, 30),   # before Asia 01:00
    ]
    INSIDE_LONDON = [
        datetime(2026, 6, 18, 10, 0),
        datetime(2026, 6, 18, 13, 0),
        datetime(2026, 6, 18, 15, 0),
    ]

    def test_canonical_windows(self):
        self.assertEqual(fs.LIVE_FOREX_SESSIONS["london"], (7, 0, 16, 0))
        self.assertEqual(fs.LIVE_FOREX_SESSIONS["new_york"], (12, 0, 21, 0))
        self.assertEqual(fs.LIVE_FOREX_SESSIONS["asia"], (1, 0, 4, 0))
        self.assertEqual(fs.overlap_window(), (12, 0, 16, 0))

    def test_outside_windows_not_in_live_session(self):
        for dt in self.OUTSIDE:
            self.assertFalse(
                fs.in_live_forex_session(dt),
                msg=f"expected outside at {dt}",
            )

    def test_inside_london_windows(self):
        for dt in self.INSIDE_LONDON:
            self.assertTrue(fs.in_live_forex_session(dt), msg=str(dt))
            self.assertEqual(fs.active_live_forex_session(dt), "london")

    def test_asia_window(self):
        self.assertTrue(fs.in_live_forex_session(datetime(2026, 6, 18, 2, 0)))
        self.assertEqual(
            fs.active_live_forex_session(datetime(2026, 6, 18, 2, 0)), "asia"
        )

    def test_session_filter_matches_confirm_gate(self):
        for dt in self.OUTSIDE:
            self.assertFalse(fcc.in_forex_claude_confirm_session(dt))
            self.assertFalse(fs.in_live_forex_session(dt))
        for dt in self.INSIDE_LONDON:
            self.assertEqual(
                fcc.in_forex_claude_confirm_session(dt),
                fs.in_live_forex_session(dt),
            )

    def test_wizard_london_session_uses_canonical_window(self):
        cfg = {"sessions_enabled": True, "allowed_sessions": ["london"]}
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 18, 7, 30))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 18, 10, 0))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 18, 17, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_live_forex_session_allowed_requires_canonical_window(self):
        ok, reason = fs.live_forex_session_allowed({}, datetime(2026, 6, 18, 5, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_live_forex_session")
        for dt in self.INSIDE_LONDON:
            ok, _ = fs.live_forex_session_allowed({}, dt)
            self.assertTrue(ok, msg=str(dt))


class TestCrossSurfaceAlignment(unittest.TestCase):
    """UI / wizard / forex_engine / fire-gate must agree for the same timestamp."""

    LONDON_CFG = {"sessions_enabled": True, "allowed_sessions": ["london"]}

    def _surfaces_london(self, dt: datetime) -> dict:
        ok_filter, _ = is_in_allowed_session(self.LONDON_CFG, dt)
        ok_gate, _ = fs.live_forex_session_allowed(self.LONDON_CFG, dt)
        with patch("app.services.forex_engine.datetime") as mock_dt:
            mock_dt.utcnow.return_value = dt
            ok_eval, _ = eval_forex_session(
                {"sessions": ["london"], "condition": "in_session"}
            )
        return {
            "unified": fs.session_active_unified("london", dt),
            "forex_engine": in_session("london", dt),
            "session_filter": ok_filter,
            "fire_gate": ok_gate,
            "eval_forex_session": ok_eval,
        }

    def test_10_13_15_utc_in_london_everywhere(self):
        for dt in (
            datetime(2026, 6, 18, 10, 0),
            datetime(2026, 6, 18, 13, 0),
            datetime(2026, 6, 18, 15, 0),
        ):
            surfaces = self._surfaces_london(dt)
            self.assertTrue(all(surfaces.values()), msg=f"{dt} -> {surfaces}")

    def test_05_17_22_utc_blocked_everywhere(self):
        for dt in (
            datetime(2026, 6, 18, 5, 0),
            datetime(2026, 6, 18, 17, 0),
            datetime(2026, 6, 18, 22, 0),
        ):
            surfaces = self._surfaces_london(dt)
            self.assertFalse(any(surfaces.values()), msg=f"{dt} -> {surfaces}")


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
                now_utc=datetime(2026, 6, 18, 5, 0),
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_confirm_session")
        mock_claude.assert_not_called()


if __name__ == "__main__":
    unittest.main()
