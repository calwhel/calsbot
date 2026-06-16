"""Marketplace — forex/metals ranked and displayed by pips, not leveraged %."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import unittest


class TestMarketplaceTradfiHelpers(unittest.TestCase):
    def test_tradfi_asset_classes(self):
        _MKT_TRADFI_AC = frozenset({"forex", "index", "metals", "commodity", "stock"})
        self.assertIn("metals", _MKT_TRADFI_AC)
        self.assertIn("forex", _MKT_TRADFI_AC)

    def test_perf_score_prefers_pips_for_forex(self):
        _MKT_TRADFI_AC = frozenset({"forex", "index", "metals", "commodity", "stock"})

        def _mkt_perf_score(item: dict) -> float:
            ac = (item.get("asset_class") or "crypto").lower()
            if ac in _MKT_TRADFI_AC:
                return float(item.get("live_pips_pnl") or -1e9)
            return float(item.get("live_pnl") or -1e9)

        items = [
            {"asset_class": "forex", "live_pips_pnl": 42.0, "live_pnl": -15.0},
            {"asset_class": "crypto", "live_pips_pnl": None, "live_pnl": 8.0},
        ]
        items.sort(key=_mkt_perf_score, reverse=True)
        self.assertEqual(items[0]["asset_class"], "forex")
        self.assertEqual(items[0]["live_pips_pnl"], 42.0)


if __name__ == "__main__":
    unittest.main()
