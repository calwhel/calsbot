"""Tests for automatic strategy repair before executor scanning."""
import unittest
from unittest.mock import MagicMock

from app.services.strategy_heal import heal_strategy_row, resolve_strategy_asset_class


class TestStrategyHeal(unittest.TestCase):
    def test_draft_forex_promoted_to_paper(self):
        strat = MagicMock()
        strat.status = "draft"
        strat.name = "EURUSD BOS 15m"
        strat.asset_class = "crypto"
        strat.config = {
            "asset_class": "forex",
            "universe": {"type": "specific", "symbols": ["EURUSD"]},
            "entry_conditions": {"conditions": [{"type": "market_structure"}]},
        }
        stats = {}
        self.assertTrue(heal_strategy_row(strat, stats))
        self.assertEqual(strat.status, "paper")
        self.assertEqual(stats["promoted_to_paper"], 1)
        self.assertEqual(strat.asset_class, "forex")

    def test_paused_index_resumed_as_paper(self):
        strat = MagicMock()
        strat.status = "paused"
        strat.name = "NAS100 scalp"
        strat.asset_class = "index"
        strat.config = {
            "asset_class": "index",
            "universe": {"type": "specific", "symbols": ["NAS100"]},
            "entry_conditions": {"conditions": []},
        }
        stats = {}
        heal_strategy_row(strat, stats)
        self.assertEqual(strat.status, "paper")

    def test_infer_universe_from_name(self):
        strat = MagicMock()
        strat.status = "paper"
        strat.name = "GBPUSD FVG retest"
        strat.asset_class = "forex"
        strat.config = {
            "asset_class": "forex",
            "universe": {"type": "specific", "symbols": []},
            "entry_conditions": {"conditions": []},
        }
        stats = {}
        self.assertTrue(heal_strategy_row(strat, stats))
        self.assertEqual(strat.config["universe"]["symbols"], ["GBPUSD"])

    def test_webhook_not_promoted(self):
        strat = MagicMock()
        strat.status = "draft"
        strat.name = "TV webhook"
        strat.asset_class = "crypto"
        strat.config = {
            "entry_conditions": {"entry_type": "tradingview_webhook", "conditions": []},
        }
        stats = {}
        self.assertFalse(heal_strategy_row(strat, stats))
        self.assertEqual(strat.status, "draft")

    def test_resolve_asset_class_prefers_config(self):
        strat = MagicMock()
        strat.asset_class = "crypto"
        strat.config = {"asset_class": "forex"}
        self.assertEqual(resolve_strategy_asset_class(strat), "forex")


if __name__ == "__main__":
    unittest.main()
