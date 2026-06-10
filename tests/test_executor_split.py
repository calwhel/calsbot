"""Tests for split crypto/forex executor env flags and advisory lock ids."""
import os
import unittest
from unittest import mock

from app.executor_lock import (
    EXECUTOR_LOCK_ID,
    FOREX_EXECUTOR_LOCK_ID,
    get_executor_lock_id,
)


class TestExecutorSplitLocks(unittest.TestCase):
    def test_portal_lock_id_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_executor_lock_id(), EXECUTOR_LOCK_ID)

    def test_forex_only_lock_id(self):
        with mock.patch.dict(os.environ, {"EXECUTOR_ONLY": "1"}, clear=True):
            self.assertEqual(get_executor_lock_id(), FOREX_EXECUTOR_LOCK_ID)

    def test_lock_ids_are_distinct(self):
        self.assertNotEqual(EXECUTOR_LOCK_ID, FOREX_EXECUTOR_LOCK_ID)


class TestExecutorSplitFlags(unittest.TestCase):
    def test_crypto_disabled_flag(self):
        from app.services import strategy_executor as se

        with mock.patch.dict(os.environ, {"DISABLE_CRYPTO_EXECUTOR": "1"}, clear=True):
            self.assertTrue(se.crypto_executor_disabled())
            self.assertFalse(se.forex_executor_disabled())

    def test_forex_disabled_flag(self):
        from app.services import strategy_executor as se

        with mock.patch.dict(os.environ, {"DISABLE_FOREX_EXECUTOR": "1"}, clear=True):
            self.assertTrue(se.forex_executor_disabled())
            self.assertFalse(se.crypto_executor_disabled())

    def test_runtime_profile_booleans(self):
        from app.services.strategy_executor import executor_runtime_profile

        with mock.patch.dict(
            os.environ,
            {"EXECUTOR_ONLY": "1", "DISABLE_CRYPTO_EXECUTOR": "1"},
            clear=False,
        ):
            prof = executor_runtime_profile()
            self.assertTrue(prof["executor_only"])
            self.assertTrue(prof["crypto_disabled"])
            self.assertFalse(prof["forex_disabled"])
            self.assertIn("crypto_scan_interval_s", prof)
            self.assertIn("forex_scan_interval_s", prof)


if __name__ == "__main__":
    unittest.main()
