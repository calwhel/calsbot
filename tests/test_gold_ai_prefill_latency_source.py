import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAUDE = (ROOT / "app/gold_ai_trader/claude.py").read_text(encoding="utf-8")
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")
EXECUTOR = (ROOT / "app/gold_ai_trader/executor.py").read_text(encoding="utf-8")
PENDING = (ROOT / "app/gold_ai_trader/pending_entry.py").read_text(encoding="utf-8")


class GoldAiPrefillLatencySourceTests(unittest.TestCase):
    def test_assistant_prefill_removed_from_claude_paths(self):
        self.assertNotIn("GOLD_AI_CLAUDE_JSON_PREFILL", CLAUDE)
        self.assertNotIn('"role": "assistant"', CLAUDE)
        self.assertIn("malformed Claude JSON (attempting salvage)", CLAUDE)
        self.assertIn("malformed JSON salvaged via repair pass", CLAUDE)

    def test_latency_and_stale_guard_hooks_present(self):
        self.assertIn("async def _stale_entry_recheck(", LOOP)
        self.assertIn("[gold-ai-latency]", LOOP)
        self.assertIn("timing_ctx[\"broker_ack_ts\"]", EXECUTOR)
        self.assertIn("new_order_latency(", EXECUTOR)
        self.assertIn("stale_guard_blocked", PENDING)
        self.assertIn("[gold-ai-latency]", PENDING)


if __name__ == "__main__":
    unittest.main()
