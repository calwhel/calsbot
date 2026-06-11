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

from app.advisory_lock_ids import APP_NAME_EXECUTOR, EXECUTOR_LOCK_ID


class _FakeCursor:
    def __init__(self, rows, terminate_log):
        self._rows = rows
        self._terminate_log = terminate_log
        self._last_terminated_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if "pg_terminate_backend" in sql:
            self._last_terminated_pid = params[0]
            self._terminate_log.append(params[0])
        self._sql = sql

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if "pg_terminate_backend" in getattr(self, "_sql", ""):
            return (True,)
        return (None,)


class _FakeConn:
    def __init__(self, rows, terminate_log):
        self._rows = rows
        self._terminate_log = terminate_log
        self.autocommit = False

    def cursor(self):
        return _FakeCursor(self._rows, self._terminate_log)

    def close(self):
        pass


def _run_with_rows(rows, min_idle_seconds, owner_app_prefix=APP_NAME_EXECUTOR):
    """Invoke _terminate_other_lock_holders against a faked psycopg2 + DB url."""
    terminate_log = []
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = lambda *a, **k: _FakeConn(rows, terminate_log)

    from app.services import telegram_poller_lock as tpl

    with mock.patch.dict(sys.modules, {"psycopg2": fake_psycopg2}), \
            mock.patch.object(tpl, "_get_db_url", return_value="postgres://x"):
        n = tpl._terminate_other_lock_holders(
            EXECUTOR_LOCK_ID,
            min_idle_seconds=min_idle_seconds,
            owner_app_prefix=owner_app_prefix,
            log_prefix="[executor_lock]",
        )
    return n, terminate_log


class TestExecutorLockReclaim(unittest.TestCase):
    def test_live_sibling_is_not_terminated(self):
        # Live holder: idle only 12s — a sibling worker pinging its lock.
        rows = [(27734, "idle", APP_NAME_EXECUTOR, 12.0)]
        n, log = _run_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 0, "live sibling holder must NOT be terminated")
        self.assertEqual(log, [])

    def test_stale_zombie_is_terminated(self):
        # Zombie holder from a dead deploy: idle 600s, no keepalive.
        rows = [(99999, "idle", APP_NAME_EXECUTOR, 600.0)]
        n, log = _run_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 1, "genuinely stale holder should be reclaimed")
        self.assertEqual(log, [99999])

    def test_gone_session_is_terminated(self):
        # Lock with no live backend row (state 'gone') — always reclaimable.
        rows = [(88888, "gone", APP_NAME_EXECUTOR, None)]
        n, log = _run_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 1)
        self.assertEqual(log, [88888])

    def test_other_subsystem_is_never_terminated(self):
        rows = [(27734, "idle", "th-tgpoller", 600.0)]
        n, log = _run_with_rows(rows, min_idle_seconds=0.0)
        self.assertEqual(n, 0, "must not kill another subsystem's backend")
        self.assertEqual(log, [])

    def test_legacy_behaviour_without_guard(self):
        # Default min_idle_seconds=0 preserves the original "terminate any holder"
        # behaviour relied on by the single-poller telegram path (same subsystem).
        rows = [(27734, "idle", APP_NAME_EXECUTOR, 12.0)]
        n, log = _run_with_rows(rows, min_idle_seconds=0.0)
        self.assertEqual(n, 1)
        self.assertEqual(log, [27734])

    def test_legacy_empty_app_on_same_lock_still_reclaimable(self):
        rows = [(27734, "idle", "", 600.0)]
        n, log = _run_with_rows(rows, min_idle_seconds=90.0)
        self.assertEqual(n, 1)
        self.assertEqual(log, [27734])


if __name__ == "__main__":
    unittest.main()
