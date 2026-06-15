"""Assignment save endpoint — table migration + JSON hardening."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import json
import unittest
from decimal import Decimal

from app.services.strategy_account_assignments import (
    serialize_assignment_row,
    upsert_strategy_assignments,
)


class TestSerializeAssignment(unittest.TestCase):
    def test_decimal_lot_coerced_to_float(self):
        row = serialize_assignment_row("111", True, Decimal("0.25"))
        self.assertEqual(row["lot_size"], 0.25)
        json.dumps(row)

    def test_null_lot(self):
        row = serialize_assignment_row("111", False, None)
        self.assertIsNone(row["lot_size"])


class TestAssignmentSaveEndpoint(unittest.TestCase):
    def test_ensure_table_then_upsert_works(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.strategy_models import StrategyAccountAssignment, UserStrategy
        from app.models import User, UserPreference
        from app.services.strategy_account_assignments import (
            ensure_strategy_account_assignments_table,
            upsert_strategy_assignments,
        )

        engine = create_engine("sqlite:///:memory:")
        User.__table__.create(engine)
        UserPreference.__table__.create(engine)
        UserStrategy.__table__.create(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        u = User(telegram_id=1, username="u", first_name="U")
        db.add(u)
        db.commit()
        db.refresh(u)
        db.add(UserPreference(user_id=u.id, ctrader_added_accounts='["111"]'))
        s = UserStrategy(user_id=u.id, name="T", config={}, status="active", asset_class="forex")
        db.add(s)
        db.commit()
        ensure_strategy_account_assignments_table(engine)
        saved = upsert_strategy_assignments(
            db, s.id, [{"ctid": "111", "enabled": True, "lot_size": 0.25}],
            allowed_ctids={"111"},
        )
        db.commit()
        self.assertEqual(saved[0]["lot_size"], 0.25)

    def test_decimal_response_serializes(self):
        import strategy_portal_server as sps
        payload = _lf_json_safe({
            "ok": True,
            "assignments": [serialize_assignment_row("1", True, Decimal("0.01"))],
        })
        json.dumps(payload)
        self.assertEqual(payload["assignments"][0]["lot_size"], 0.01)


def _lf_json_safe(obj):
    from strategy_portal_server import _lf_json_safe as fn
    return fn(obj)


if __name__ == "__main__":
    unittest.main()
