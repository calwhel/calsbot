"""Single-runner guarantee for the X / Twitter auto-poster advisory lock.

The poster now runs on BOTH the always-up web worker and the bot companion,
each gated by ``try_acquire_persistent_lock(TWITTER_POSTER_LOCK_ID)``. Exactly
one process must win the lock (no double tweets), the loser must stand by, and
the helper must NEVER terminate the live holder.
"""
import sys
import types
import unittest
from unittest import mock


class _FakeCursor:
    def __init__(self, acquire_result):
        self._acquire_result = acquire_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._sql = sql

    def fetchone(self):
        if "pg_try_advisory_lock" in getattr(self, "_sql", ""):
            return (self._acquire_result,)
        return (None,)


class _FakeConn:
    def __init__(self, acquire_result):
        self._acquire_result = acquire_result
        self.autocommit = False

    def cursor(self):
        return _FakeCursor(self._acquire_result)

    def close(self):
        pass


class TestTwitterPosterLock(unittest.TestCase):
    def setUp(self):
        from app.services import telegram_poller_lock as tpl
        self.tpl = tpl
        tpl._lock_conns.clear()

    def tearDown(self):
        self.tpl._lock_conns.clear()

    def _patch_psycopg2(self, acquire_result):
        fake = types.ModuleType("psycopg2")
        fake.connect = lambda *a, **k: _FakeConn(acquire_result)
        return mock.patch.dict(sys.modules, {"psycopg2": fake})

    def test_winner_acquires_loser_stands_by(self):
        tpl = self.tpl
        lock_id = tpl.TWITTER_POSTER_LOCK_ID
        with mock.patch.object(tpl, "_get_db_url", return_value="postgres://x"), \
                mock.patch.object(tpl, "_start_keepalive"):
            # First process: Postgres grants the lock.
            with self._patch_psycopg2(True):
                self.assertTrue(tpl.try_acquire_persistent_lock(lock_id))
            self.assertTrue(tpl.holds_lock(lock_id))
            # Already-held → fast True, no re-acquire.
            self.assertTrue(tpl.try_acquire_persistent_lock(lock_id))

        # A *separate* process (clear local state) where the lock is held
        # elsewhere → Postgres refuses → stands by (False), never terminates.
        tpl._lock_conns.clear()
        with mock.patch.object(tpl, "_get_db_url", return_value="postgres://x"), \
                mock.patch.object(tpl, "_start_keepalive"):
            with self._patch_psycopg2(False):
                self.assertFalse(tpl.try_acquire_persistent_lock(lock_id))
            self.assertFalse(tpl.holds_lock(lock_id))

    def test_distinct_lock_ids(self):
        from app.services import telegram_poller_lock as tpl
        from app.advisory_lock_ids import TWITTER_POSTER_LOCK_ID
        from app.lock_ids import EXECUTOR_LOCK_ID, TG_POLLER_LOCK_ID

        self.assertEqual(tpl.MAIN_POLLER_LOCK_ID, TG_POLLER_LOCK_ID)
        self.assertEqual(tpl.FOREX_POLLER_LOCK_ID, TG_POLLER_LOCK_ID)
        self.assertNotEqual(TG_POLLER_LOCK_ID, EXECUTOR_LOCK_ID)
        self.assertNotEqual(TG_POLLER_LOCK_ID, TWITTER_POSTER_LOCK_ID)
        self.assertEqual(tpl.TWITTER_POSTER_LOCK_ID, TWITTER_POSTER_LOCK_ID)


if __name__ == "__main__":
    unittest.main()
