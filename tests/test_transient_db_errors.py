"""Transient Neon/Postgres connection error detection."""
import unittest

from app.db_resilience import is_transient_db_error, run_with_db_retry


class TestTransientDbErrors(unittest.TestCase):
    def test_ssl_closed_is_transient(self):
        exc = Exception("SSL connection has been closed unexpectedly")
        self.assertTrue(is_transient_db_error(exc))

    def test_eof_is_transient(self):
        exc = Exception("SSL SYSCALL error: EOF detected")
        self.assertTrue(is_transient_db_error(exc))

    def test_normal_error_is_not_transient(self):
        self.assertFalse(is_transient_db_error(ValueError("bad uid")))

    def test_run_with_db_retry_recovers(self):
        calls = {"n": 0}

        def _fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise Exception("SSL connection has been closed unexpectedly")
            return "ok"

        self.assertEqual(
            run_with_db_retry(_fn, max_attempts=3, retry_delay=0.0, label="test"),
            "ok",
        )
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
