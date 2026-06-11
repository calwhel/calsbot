"""Regression test for the two-worker executor advisory-lock thrash.

With GUNICORN_WORKERS=2 the HTTP worker's claim loop used to terminate the
executor worker's lock connection on every attempt (it only checked that the
holder was "idle"), thrashing the executor so no trades fired. The fix adds an
idle-age guard: a holder is only reclaimed when it has been idle long enough to
be a genuine zombie, never a live sibling that keeps its lock connection warm.
"""
import sys
import types
import unittest
from unittest import mock

from app.advisory_lock_ids import APP_NAME_EXECUTOR, APP_NAME_TG_POLLER, EXECUTOR_LOCK_ID
from app.lock_ids import TG_POLLER_LOCK_ID


class _FakeCursor:
    def __init__(self, rows, terminate_log, lock_id):
        self._rows = rows
        self._terminate_log = terminate_log
        self._lock_id = lock_id
        self._last_terminated_pid = None
        self._sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._sql = sql
        if "pg_terminate_backend" in sql:
            self._last_terminated_pid = params[0]
            self._terminate_log.append(params[0])

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if "pg_terminate_backend" in self._sql:
            return (True,)
        if "pg_locks" in self._sql and "LIMIT 1" in self._sql:
            return (1,)
        return (None,)


class _FakeConn:
    def __init__(self, rows, terminate_log, lock_id):
        self._rows = rows
        self._terminate_log = terminate_log
        self._lock_id = lock_id
        self.autocommit = False

    def cursor(self):
        return _FakeCursor(self._rows, self._terminate_log, self._lock_id)

    def close(self):
        pass


def _run_executor_with_rows(rows, min_idle_seconds, owner_app=APP_NAME_EXECUTOR):
    terminate_log = []
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = lambda *a, **k: _FakeConn(
        rows, terminate_log, EXECUTOR_LOCK_ID
    )

    from app import executor_lock as el

    with mock.patch.dict(sys.modules, {"psycopg2": fake_psycopg2}), \
            mock.patch("app.config.settings.get_database_url", return_value="postgres://x"):
        n = el.terminate_lock_holders(
            EXECUTOR_LOCK_ID,
            min_idle_seconds=min_idle_seconds,
            owner_app=owner_app,
            log_prefix="[executor_lock]",
        )
    return n, terminate_log


def _run_poller_with_rows(rows, min_idle_seconds=0.0):
    terminate_log = []
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = lambda *a, **k: _FakeConn(
        rows, terminate_log, TG_POLLER_LOCK_ID
    )

    from app.services import telegram_poller_lock as tpl

    with mock.patch.dict(sys.modules, {"psycopg2": fake_psycopg2}), \
            mock.patch.object(tpl, "_get_db_url", return_value="postgres://x"):
        n = tpl._terminate_poller_lock_holders(
            min_idle_seconds=min_idle_seconds,
            log_prefix="[tg-lock]",
        )
    return n, terminate_log


class TestExecutorLockReclaim(unittest.TestCase):
    def test_live_sibling_is_not_terminated(self):
        rows = [(27734, "idle", APP_NAME_EXECUTOR, 12.0)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 0, "live sibling holder must NOT be terminated")
        self.assertEqual(log, [])

    def test_stale_zombie_is_terminated(self):
        rows = [(99999, "idle", APP_NAME_EXECUTOR, 600.0)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 1, "genuinely stale holder should be reclaimed")
        self.assertEqual(log, [99999])

    def test_gone_session_is_terminated(self):
        rows = [(88888, "gone", APP_NAME_EXECUTOR, None)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 1)
        self.assertEqual(log, [88888])

    def test_other_subsystem_is_never_terminated(self):
        rows = [(27734, "idle", APP_NAME_TG_POLLER, 600.0)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=0.0)
        self.assertEqual(n, 0, "must not kill another subsystem's backend")
        self.assertEqual(log, [])

    def test_executor_never_terminates_tg_poller_on_executor_lock(self):
        rows = [(27734, "idle", APP_NAME_TG_POLLER, 600.0)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=0.0)
        self.assertEqual(n, 0)
        self.assertEqual(log, [])

    def test_empty_app_is_not_terminated(self):
        rows = [(27734, "idle", "", 600.0)]
        n, log = _run_executor_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 0, "legacy empty app must not match — no cross-kill")
        self.assertEqual(log, [])

    def test_poller_never_terminates_executor(self):
        rows = [(27734, "idle", APP_NAME_EXECUTOR, 600.0)]
        n, log = _run_poller_with_rows(rows)
        self.assertEqual(n, 0)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()
