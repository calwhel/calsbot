"""Live broker fire requires status=active AND enabled account assignment."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.strategy_account_assignments import (
    TRADFI_BROKER_ASSET_CLASSES,
    resolve_live_fire_intent,
)


class TestResolveLiveFireIntent(unittest.TestCase):
    def test_paper_with_assignment_stays_paper(self):
        """Paper + live account ticked must NOT fire broker orders."""
        strategy = SimpleNamespace(id=42, status="paper", user_id=1)
        prefs = SimpleNamespace(ctrader_account_id="99999")
        targets = [{"ctrader_account_id": "47465772", "lot_size": 0.01}]
        db = MagicMock()
        with patch(
            "app.services.strategy_account_assignments.get_enabled_fire_targets",
            return_value=targets,
        ):
            wants, out = resolve_live_fire_intent(db, strategy, "forex", prefs)
        self.assertFalse(wants)
        self.assertEqual(out, [])

    def test_active_with_assignment_fires_live(self):
        strategy = SimpleNamespace(id=42, status="active", user_id=1)
        targets = [{"ctrader_account_id": "47465772", "lot_size": 0.01}]
        db = MagicMock()
        with patch(
            "app.services.strategy_account_assignments.get_enabled_fire_targets",
            return_value=targets,
        ):
            wants, out = resolve_live_fire_intent(db, strategy, "forex", None)
        self.assertTrue(wants)
        self.assertEqual(out, targets)

    def test_active_without_assignment_stays_off(self):
        strategy = SimpleNamespace(id=1, status="active", user_id=1)
        db = MagicMock()
        with patch(
            "app.services.strategy_account_assignments.get_enabled_fire_targets",
            return_value=[],
        ):
            wants, out = resolve_live_fire_intent(db, strategy, "forex", None)
        self.assertFalse(wants)
        self.assertEqual(out, [])

    def test_paper_without_assignments_stays_paper(self):
        strategy = SimpleNamespace(id=1, status="paper", user_id=1)
        db = MagicMock()
        with patch(
            "app.services.strategy_account_assignments.get_enabled_fire_targets",
            return_value=[],
        ):
            wants, _ = resolve_live_fire_intent(db, strategy, "forex", None)
        self.assertFalse(wants)

    def test_crypto_ignores_assignments(self):
        strategy = SimpleNamespace(id=1, status="paper", user_id=1)
        db = MagicMock()
        wants, out = resolve_live_fire_intent(db, strategy, "crypto", None)
        self.assertFalse(wants)
        self.assertEqual(out, [])

    def test_evaluate_and_fire_blocks_paper_status(self):
        from app.services import strategy_executor as se

        src = inspect.getsource(se.evaluate_and_fire)
        self.assertIn("resolve_live_fire_intent", src)
        self.assertIn("reason=status=paper", src)
        self.assertNotIn("account toggle is source of truth", src)

    def test_tradfi_asset_classes_include_metals(self):
        self.assertIn("metals", TRADFI_BROKER_ASSET_CLASSES)
        self.assertIn("forex", TRADFI_BROKER_ASSET_CLASSES)


if __name__ == "__main__":
    unittest.main()
