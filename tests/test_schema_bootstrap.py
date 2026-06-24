"""Schema bootstrap — fresh DB, idempotency, Gold AI write paths."""
import os
import unittest
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


class TestSchemaBootstrap(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        from app.database import Base

        Base.metadata.drop_all(bind=self.engine)

    def test_create_orm_tables_creates_users_and_gold_ai(self):
        from app.schema_bootstrap import create_orm_tables, orm_table_names, verify_orm_tables

        create_orm_tables(self.engine)

        insp = inspect(self.engine)
        self.assertTrue(insp.has_table("users"))
        self.assertTrue(insp.has_table("gold_ai_config"))
        self.assertTrue(insp.has_table("gold_ai_decisions"))
        self.assertTrue(insp.has_table("gold_ai_outcomes"))

        missing = verify_orm_tables(self.engine)
        self.assertEqual(missing, [], msg=f"missing ORM tables: {missing}")
        self.assertEqual(len(orm_table_names()) - len(missing), len(orm_table_names()))

    def test_create_orm_tables_idempotent(self):
        from app.schema_bootstrap import create_orm_tables, verify_orm_tables

        create_orm_tables(self.engine)
        create_orm_tables(self.engine)
        orm_missing, _lazy = verify_orm_tables(self.engine), []
        from app.schema_bootstrap import verify_orm_tables as verify_orm

        self.assertEqual(verify_orm(self.engine), [])

    def test_gold_ai_decision_and_outcome_write_paths(self):
        from app.schema_bootstrap import create_orm_tables
        from app.gold_ai_trader.models import GoldAiDecision, GoldAiOutcome
        from app.gold_ai_trader.learning import record_outcome_from_execution

        create_orm_tables(self.engine)

        Session = sessionmaker(bind=self.engine)
        db = Session()
        try:
            dec = GoldAiDecision(
                session="london",
                candidate_type="ict_ob",
                context_snapshot="{}",
                reasoning="test",
                decision={"entry": 2650.0, "stop_loss": 2640.0, "direction": "long"},
                action="take",
                confidence=72,
                executed=True,
            )
            db.add(dec)
            db.commit()
            db.refresh(dec)

            execution = SimpleNamespace(
                outcome="WIN",
                entry_price=2650.0,
                sl_price=2640.0,
                direction="LONG",
                pnl_pct=1.5,
                mfe_pips=12.0,
                mae_pips=3.0,
                closed_at=datetime.utcnow(),
            )
            self.assertTrue(record_outcome_from_execution(db, dec.id, execution))

            outcome = (
                db.query(GoldAiOutcome)
                .filter(GoldAiOutcome.decision_id == dec.id)
                .one()
            )
            self.assertEqual(outcome.result, "win")
            self.assertEqual(outcome.setup_type, "ict_ob")
            self.assertEqual(outcome.session, "london")
            self.assertIsNotNone(outcome.r_multiple)
        finally:
            db.close()

    def test_sorted_tables_respects_fk_order_users_before_dependents(self):
        from app.schema_bootstrap import orm_table_names

        names = orm_table_names()
        self.assertIn("users", names)
        self.assertIn("user_strategies", names)
        self.assertIn("strategy_executions", names)
        self.assertLess(names.index("users"), names.index("user_strategies"))
        self.assertLess(names.index("users"), names.index("strategy_executions"))


if __name__ == "__main__":
    unittest.main()
