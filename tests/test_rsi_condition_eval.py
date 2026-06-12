"""RSI threshold evaluation — operator and threshold field normalization."""
import asyncio
import unittest

from app.services.strategy_ta import eval_indicator


class TestRsiConditionEval(unittest.IsolatedAsyncioTestCase):
    async def test_rsi_gt_55_fails_at_38(self):
        cond = {"name": "rsi", "condition": "gt", "threshold": 55, "timeframe": "1h"}
        passed, detail = await eval_indicator(
            cond, {}, {"rsi_1h": 38.5}, "XAUUSD", None, {},
        )
        self.assertFalse(passed)
        self.assertIn("38.5", detail)
        self.assertIn("> 55", detail)

    async def test_rsi_gt_55_passes_at_60(self):
        cond = {"name": "rsi", "operator": "gt", "value": 55, "timeframe": "1h"}
        passed, detail = await eval_indicator(
            cond, {}, {"rsi_1h": 60.0}, "XAUUSD", None, {},
        )
        self.assertTrue(passed)
        self.assertIn("> 55", detail)

    async def test_rsi_type_alias(self):
        from app.services.strategy_ta import evaluate_strategy_conditions

        cfg = {
            "entry_conditions": {
                "operator": "AND",
                "conditions": [
                    {"type": "rsi", "operator": "gt", "value": 55, "timeframe": "1h"},
                ],
            },
        }
        passed, details = await evaluate_strategy_conditions(
            cfg, "XAUUSD", {"price": 4200}, {"rsi_1h": 38.5}, None,
        )
        self.assertFalse(passed)
        self.assertTrue(any("38.5" in d for d in details))


if __name__ == "__main__":
    unittest.main()
