"""Standalone executor must force-reclaim external lock holders."""

import unittest
from unittest import mock

from app.executor_lock import EXECUTOR_LOCK_ID, reclaim_executor_lock


class TestStandaloneReclaim(unittest.TestCase):
    def test_force_reclaim_uses_zero_idle_threshold(self):
        with mock.patch(
            "app.services.telegram_poller_lock.terminate_advisory_lock_holders",
            return_value=1,
        ) as term:
            n = reclaim_executor_lock(force=True)
        self.assertEqual(n, 1)
        term.assert_called_once_with(EXECUTOR_LOCK_ID, min_idle_seconds=0.0)

    def test_gentle_reclaim_uses_idle_guard(self):
        with mock.patch(
            "app.services.telegram_poller_lock.terminate_advisory_lock_holders",
            return_value=0,
        ) as term:
            reclaim_executor_lock(force=False)
        term.assert_called_once_with(EXECUTOR_LOCK_ID, min_idle_seconds=120.0)


if __name__ == "__main__":
    unittest.main()
