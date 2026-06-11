"""Advisory lock ID registry must stay collision-free."""
import unittest

from app.advisory_lock_ids import ALL_ADVISORY_LOCK_IDS


class TestAdvisoryLockIds(unittest.TestCase):
    def test_all_lock_ids_unique(self):
        ids = sorted(ALL_ADVISORY_LOCK_IDS)
        self.assertEqual(len(ids), len(set(ids)), f"duplicate lock IDs: {ids}")


if __name__ == "__main__":
    unittest.main()
