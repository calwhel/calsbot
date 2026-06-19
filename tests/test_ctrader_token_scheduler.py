"""cTrader OAuth token scheduler — single refresh owner."""
import inspect
import unittest


class TestCtraderTokenScheduler(unittest.TestCase):
    def test_scheduler_uses_owner_advisory_lock(self):
        from app.services import ctrader_token_scheduler as sched

        src = inspect.getsource(sched._try_acquire_owner_lock)
        self.assertIn("pg_try_advisory_lock", src)
        self.assertIn("_POLL_LOCK_ID", src)

    def test_refresh_cycle_uses_single_flight_refresh(self):
        from app.services import ctrader_token_scheduler as sched

        src = inspect.getsource(sched.run_token_refresh_cycle)
        self.assertIn("refresh_user_ctrader_token", src)
        self.assertNotIn("refresh_access_token", src)


if __name__ == "__main__":
    unittest.main()
