"""Gemini Gold schema unit tests."""
import os
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.models import (
    GeminiGoldConfig,
    GeminiGoldDecision,
    GeminiGoldOutcome,
)
from app.gemini_gold_trader.schema import ensure_gemini_gold_trader_schema, seed_config_if_missing


def test_schema_idempotent():
    with patch("app.gemini_gold_trader.schema.run_with_db_retry", side_effect=lambda fn, **kw: fn()):
        with patch("app.gemini_gold_trader.schema.Base.metadata.create_all") as create_all:
            with patch("app.gemini_gold_trader.schema.inspect") as insp_mock:
                with patch("app.gemini_gold_trader.schema._apply_column_alters"):
                    insp_mock.return_value.has_table.return_value = True
                    ensure_gemini_gold_trader_schema(force=True)
                    ensure_gemini_gold_trader_schema(force=True)
                    assert create_all.call_count == 2


def test_schema_review_columns_required():
    from app.gemini_gold_trader.schema import _REQUIRED_COLUMNS

    assert "gemini_gold_reviews" in _REQUIRED_COLUMNS
    cols = _REQUIRED_COLUMNS["gemini_gold_reviews"]
    assert "timing_insights" in cols
    assert "aggressiveness_insights" in cols
    assert "ctrader_account_notes" in cols

    db = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    existing = GeminiGoldConfig(id=1, max_calls_day=340, dry_run=True)
    db.query.return_value.filter.return_value.first.side_effect = [None, existing]
    row1 = seed_config_if_missing(db)
    row2 = seed_config_if_missing(db)
    assert row1.id == 1
    assert row2.id == 1
    assert row1.max_calls_day == 340
    assert row1.dry_run is True
