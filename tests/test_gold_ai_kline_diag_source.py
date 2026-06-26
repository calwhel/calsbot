from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app" / "gold_ai_trader" / "loop.py").read_text()
QUALITY = (ROOT / "app" / "gold_ai_trader" / "data_quality.py").read_text()
REFRESH = (ROOT / "app" / "gold_ai_trader" / "data_refresh.py").read_text()
PRICES = (ROOT / "app" / "services" / "tradfi_prices.py").read_text()


class GoldAiKlineDiagSourceTests(unittest.TestCase):
    def test_stale_skip_logs_diagnostic_payload(self):
        self.assertIn("[gold-ai] stale skip source=%s bar_age_s=%s fetch_age_s=%s", LOOP)
        self.assertIn("trendbar_blocked=%s trendbar_reason=%s detail=%s", LOOP)
        self.assertIn("ctrader_trendbar_blocked", LOOP)
        self.assertIn("ctrader_trendbar_block_reason", LOOP)

    def test_market_data_exposes_stale_debug_fields(self):
        self.assertIn("def _ctrader_trendbar_block_state() -> Tuple[bool, str]:", QUALITY)
        self.assertIn("\"kline_bar_age_s\": kline_bar_age_s", QUALITY)
        self.assertIn("\"kline_fetch_age_s\": kline_fetch_age_s", QUALITY)
        self.assertIn("cache_fetched_at=kline_fetched_at", QUALITY)

    def test_refresh_hardening_runs_ctrader_sweep_and_restart_path(self):
        self.assertIn("sweep_stale_klines(symbols=[SYMBOL], timeframes=[\"5m\", \"15m\", \"1h\"])", REFRESH)
        self.assertIn("async def _maybe_restart_ctrader_builder(reason: str) -> bool:", REFRESH)
        self.assertIn("GOLD_AI_KLINE_RESTART_MIN_INTERVAL_S", REFRESH)
        self.assertIn("trendbar_fetch_blocked_reason", REFRESH)

    def test_prices_expose_fetch_age_and_stale_prefetch_bypass(self):
        self.assertIn("def get_metal_kline_fetch_age_s(", PRICES)
        self.assertIn("def get_metal_kline_fetched_at(", PRICES)
        self.assertIn("prefetch-fast stale cache bypass", PRICES)


if __name__ == "__main__":
    unittest.main()
