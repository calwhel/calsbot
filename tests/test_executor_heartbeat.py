"""Executor process heartbeat persistence."""
import unittest
from unittest import mock


class TestExecutorHeartbeat(unittest.TestCase):
    def test_mark_heartbeat_updates_memory(self):
        from app.services.strategy_executor import mark_heartbeat, get_heartbeats
        mark_heartbeat("test_loop")
        self.assertIn("test_loop", get_heartbeats())

    def test_persist_executor_process_heartbeat_sets_memory_hb(self):
        from app.services.strategy_executor import (
            get_heartbeats,
            persist_executor_process_heartbeat,
        )

        async def _run():
            try:
                await persist_executor_process_heartbeat()
            except Exception:
                pass

        import asyncio
        asyncio.run(_run())
        self.assertIn("executor_process", get_heartbeats())


if __name__ == "__main__":
    unittest.main()
