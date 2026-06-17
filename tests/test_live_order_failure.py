"""Live order failure classification, retry, and durable logging."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from app.services.live_order_failure import (
    ClassifiedFailure,
    FailureCategory,
    classify_live_order_failure,
    humanize_skip_reason,
    live_order_retry_backoff_s,
    max_live_order_attempts,
    record_live_fire_failure,
)
from app.services import ctrader_order_queue as coq


class TestLiveOrderFailureClassification(unittest.TestCase):
    def test_insufficient_margin_permanent(self):
        c = classify_live_order_failure("ORDER_CANCELLED: insufficient margin")
        self.assertFalse(c.transient)
        self.assertEqual(c.reason, "insufficient margin")
        self.assertEqual(c.category, FailureCategory.BROKER_REJECTED)

    def test_timeout_transient(self):
        c = classify_live_order_failure("broker timeout")
        self.assertTrue(c.transient)

    def test_token_auth_transient(self):
        c = classify_live_order_failure("account auth failed")
        self.assertTrue(c.transient)
        self.assertIn("token", c.reason.lower())

    def test_forex_not_approved_permanent(self):
        c = classify_live_order_failure("forex not approved")
        self.assertFalse(c.transient)

    def test_exception_captured(self):
        c = classify_live_order_failure(None, exception=RuntimeError("socket reset"))
        self.assertEqual(c.category, FailureCategory.EXCEPTION)
        self.assertIn("socket reset", c.reason)

    def test_skip_reason(self):
        c = classify_live_order_failure(None, skip_reason="blk_max_open")
        self.assertEqual(c.category, FailureCategory.SKIPPED)

    def test_humanize_blockers(self):
        self.assertIn(
            "forex not approved",
            humanize_skip_reason(["forex live not approved"]),
        )


class TestLiveOrderRetryWiring(unittest.TestCase):
    def test_run_order_job_has_retry_loop(self):
        src = inspect.getsource(coq._run_order_job)
        self.assertIn("max_live_order_attempts", src)
        self.assertIn("_execution_already_filled", src)
        self.assertIn("live_order_retry_backoff_s", src)

    def test_apply_order_result_records_failure(self):
        src = inspect.getsource(coq._apply_order_result)
        self.assertIn("record_live_fire_failure", src)
        self.assertIn("notify_live_order_failure", src)

    def test_retry_backoff_schedule(self):
        self.assertEqual(live_order_retry_backoff_s(0), 0.5)
        self.assertEqual(live_order_retry_backoff_s(1), 1.5)
        self.assertEqual(max_live_order_attempts(), 3)


class TestLiveFireFailurePersist(unittest.TestCase):
    def test_model_avoids_reserved_postgres_ctid_column(self):
        from app.strategy_models import LiveFireFailure

        self.assertIsNone(LiveFireFailure.__table__.columns.get("ctid"))
        self.assertIsNotNone(LiveFireFailure.__table__.columns.get("ctrader_account_id"))

    def test_record_live_fire_failure(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import app.models  # noqa — registers users table
        from app.models import User
        from app.strategy_models import LiveFireFailure

        engine = create_engine("sqlite:///:memory:")
        User.__table__.metadata.create_all(
            bind=engine, tables=[User.__table__, LiveFireFailure.__table__],
        )
        Session = sessionmaker(bind=engine)
        classified = ClassifiedFailure(
            reason="insufficient margin",
            category=FailureCategory.BROKER_REJECTED,
            transient=False,
        )
        with mock.patch("app.database.SessionLocal", Session):
            record_live_fire_failure(
                user_id=1,
                strategy_id=10,
                execution_id=99,
                signal_group_id="grp1",
                ctid="47465772",
                symbol="EURUSD",
                direction="LONG",
                lots="0.10",
                classified=classified,
                attempts=3,
                sibling_summary="#demo filled",
            )
        db = Session()
        try:
            row = db.query(LiveFireFailure).filter_by(execution_id=99).first()
            self.assertIsNotNone(row)
            self.assertEqual(row.reason, "insufficient margin")
            self.assertEqual(row.attempts, 3)
            self.assertEqual(row.ctrader_account_id, "47465772")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
