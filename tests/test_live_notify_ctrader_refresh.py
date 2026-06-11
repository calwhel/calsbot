"""Live open notify gating + cTrader token refresh hardening."""
import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestLiveNotifyGating(unittest.TestCase):
    def test_queued_path_does_not_claim_open_notify(self):
        from app.services import strategy_executor as se

        src = inspect.getsource(se.evaluate_and_fire)
        self.assertIn("_fmt_queued_open_notice", src)
        self.assertIn(
            "full LIVE card is sent after broker fill confirmation",
            src,
        )
        idx = src.find("order_result.get(\"queued\")")
        block = src[idx:idx + 2500]
        self.assertNotIn("_claim_tg_open_notify(db, execution.id)", block)

    def test_queue_apply_requires_fill_for_live_notify(self):
        from app.services import ctrader_order_queue as coq

        src = inspect.getsource(coq._apply_order_result)
        self.assertIn("actual_fill and actual_fill > 0", src)
        self.assertIn("_release_tg_open_notify", src)


class TestCtraderRefreshHardening(unittest.TestCase):
    def test_refresh_re_reads_prefs_before_oauth(self):
        from app.services import ctrader_client as cc

        src = inspect.getsource(cc.refresh_user_ctrader_token)
        self.assertIn("_read_fresh_ctrader_prefs", src)
        self.assertIn("Immediately before OAuth", src)

    def test_demo_host_pinned_when_is_live_false(self):
        from app.services.ctrader_client import (
            CTRADER_HOST_DEMO,
            _host_for_account,
        )

        prefs = MagicMock()
        prefs.ctrader_accounts = (
            '[{"ctidTraderAccountId": 12345, "isLive": false}]'
        )
        self.assertEqual(_host_for_account(prefs, 12345), CTRADER_HOST_DEMO)

    def test_relink_alert_helper_exists(self):
        from app.services import ctrader_client as cc

        self.assertTrue(hasattr(cc, "_notify_ctrader_relink_needed"))


if __name__ == "__main__":
    unittest.main()
