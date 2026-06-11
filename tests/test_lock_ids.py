"""Canonical lock ID registry."""
import unittest

from app.advisory_lock_ids import ALL_ADVISORY_LOCK_IDS
from app.lock_ids import EXECUTOR_LOCK_ID, TG_POLLER_LOCK_ID


class TestLockIds(unittest.TestCase):
    def test_executor_and_poller_distinct(self):
        self.assertNotEqual(EXECUTOR_LOCK_ID, TG_POLLER_LOCK_ID)

    def test_all_lock_ids_unique(self):
        ids = sorted(ALL_ADVISORY_LOCK_IDS)
        self.assertEqual(len(ids), len(set(ids)), f"duplicate lock IDs: {ids}")


if __name__ == "__main__":
    unittest.main()
