"""Executor lock session reconnect — survives Neon SSL blips without full restart."""
import unittest
from unittest import mock


class TestExecutorLockReconnect(unittest.TestCase):
    def test_neon_lock_connect_kwargs(self):
        from app.executor_lock import NEON_LOCK_CONNECT_KWARGS

        self.assertEqual(NEON_LOCK_CONNECT_KWARGS["keepalives_idle"], 30)
        self.assertEqual(NEON_LOCK_CONNECT_KWARGS["keepalives_interval"], 10)
        self.assertEqual(NEON_LOCK_CONNECT_KWARGS["keepalives_count"], 5)
        self.assertEqual(NEON_LOCK_CONNECT_KWARGS["sslmode"], "require")
        self.assertIn("tcp_keepalives_idle=30", NEON_LOCK_CONNECT_KWARGS["options"])

    def test_reconnect_silent_skips_success_log(self):
        from app import executor_lock as el

        with mock.patch.object(el, "create_lock_connection", return_value=object()), \
                mock.patch.object(el, "try_acquire_lock", return_value=True), \
                mock.patch.object(el, "close_lock_connection"), \
                mock.patch.object(el.logger, "info") as m_info:
            conn = el.reconnect_lock_connection(None, max_attempts=1, silent=True)
            self.assertIsNotNone(conn)
            m_info.assert_not_called()

    def test_reconnect_succeeds_on_second_attempt(self):
        from app import executor_lock as el

        closed = []
        attempts = {"n": 0}

        def _fake_create():
            attempts["n"] += 1
            return object()

        def _fake_acquire(conn, lock_id=el.EXECUTOR_LOCK_ID):
            return attempts["n"] >= 2

        old_conn = mock.Mock()
        old_conn.closed = 0

        with mock.patch.object(el, "create_lock_connection", side_effect=_fake_create), \
                mock.patch.object(el, "try_acquire_lock", side_effect=_fake_acquire), \
                mock.patch.object(el, "close_lock_connection", side_effect=lambda c: closed.append(c)), \
                mock.patch.object(el, "LOCK_RECONNECT_DELAY_SECS", 0.0):
            new_conn = el.reconnect_lock_connection(
                old_conn, max_attempts=3, retry_delay=0.0
            )

        self.assertIsNotNone(new_conn)
        self.assertEqual(attempts["n"], 2)
        self.assertIn(old_conn, closed)

    def test_reconnect_returns_none_when_exhausted(self):
        from app import executor_lock as el

        with mock.patch.object(el, "create_lock_connection", return_value=object()), \
                mock.patch.object(el, "try_acquire_lock", return_value=False), \
                mock.patch.object(el, "close_lock_connection"), \
                mock.patch.object(el, "LOCK_RECONNECT_DELAY_SECS", 0.0):
            new_conn = el.reconnect_lock_connection(
                None, max_attempts=2, retry_delay=0.0
            )

        self.assertIsNone(new_conn)


if __name__ == "__main__":
    unittest.main()
