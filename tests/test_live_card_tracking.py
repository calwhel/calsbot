"""Live strategy card tracking — metals/commodity close-reconcile inclusion."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import unittest
from unittest.mock import MagicMock, patch

from app.services.trade_management import CTRADER_LIVE_ASSET_CLASSES


class TestCtraderLiveAssetClasses(unittest.TestCase):
    def test_includes_metals_and_commodity(self):
        self.assertIn("metals", CTRADER_LIVE_ASSET_CLASSES)
        self.assertIn("commodity", CTRADER_LIVE_ASSET_CLASSES)
        self.assertIn("forex", CTRADER_LIVE_ASSET_CLASSES)
        self.assertIn("index", CTRADER_LIVE_ASSET_CLASSES)


class TestReconcileWorklistMetals(unittest.TestCase):
    def test_metals_open_execution_in_worklist(self):
        from app.services.strategy_executor import _build_forex_reconcile_worklist

        fake_ex = MagicMock()
        fake_ex.id = 101
        fake_ex.user_id = 7
        fake_ex.strategy_id = 55
        fake_ex.symbol = "XAUUSD"
        fake_ex.direction = "LONG"
        fake_ex.entry_price = 2650.0
        fake_ex.tp_price = 2660.0
        fake_ex.tp2_price = None
        fake_ex.sl_price = 2640.0
        fake_ex.notes = "live | pos=999001"
        fake_ex.ctrader_position_id = "999001"
        fake_ex.asset_class = "metals"

        fake_user = MagicMock()
        fake_user.id = 7

        fake_strat = MagicMock()
        fake_strat.ctrader_account_id = "12345"

        fake_pref = MagicMock()
        fake_pref.ctrader_account_id = "12345"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [fake_ex]
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            fake_user,
            fake_pref,
            fake_strat,
        ]

        with patch("app.database.BgSessionLocal", return_value=mock_db):
            with patch(
                "app.services.ctrader_client.resolve_ctrader_ctid",
                return_value="12345",
            ):
                work = _build_forex_reconcile_worklist()

        self.assertEqual(len(work), 1)
        self.assertEqual(work[0]["symbol"], "XAUUSD")
        self.assertEqual(work[0]["position_id"], 999001)


class TestExecutionEventMetalsMatch(unittest.TestCase):
    def test_find_open_exec_matches_metals_asset_class(self):
        from app.services.ctrader_execution_events import _find_open_exec

        fake_ex = MagicMock()
        fake_ex.id = 202
        fake_ex.user_id = 3
        fake_ex.symbol = "XAUUSD"
        fake_ex.direction = "LONG"
        fake_ex.entry_price = 2650.0
        fake_ex.notes = "pos=888777"
        fake_ex.ctrader_position_id = None
        fake_ex.asset_class = "metals"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.filter.return_value.all.return_value = [
            fake_ex,
        ]

        with patch("app.database.BgSessionLocal", return_value=mock_db):
            match = _find_open_exec(888777, user_id=3)

        self.assertIsNotNone(match)
        self.assertEqual(match["exec_id"], 202)
        self.assertEqual(match["symbol"], "XAUUSD")


if __name__ == "__main__":
    unittest.main()
