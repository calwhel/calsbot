"""Gemini Gold AI review model selection."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import (
    GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
    gemini_gold_review_model,
)
from app.gemini_gold_trader.review import _is_review_model_not_found, review_model_name


def test_default_review_model_is_gemini_31_pro_preview():
    os.environ.pop("GEMINI_GOLD_REVIEW_MODEL", None)
    assert gemini_gold_review_model() == "gemini-3.1-pro-preview"
    assert review_model_name() == GEMINI_GOLD_REVIEW_MODEL_DEFAULT


def test_deprecated_25_pro_env_is_remapped():
    os.environ["GEMINI_GOLD_REVIEW_MODEL"] = "gemini-2.5-pro"
    try:
        assert gemini_gold_review_model() == "gemini-3.1-pro-preview"
    finally:
        os.environ.pop("GEMINI_GOLD_REVIEW_MODEL", None)


def test_is_review_model_not_found_detects_404():
    exc = Exception(
        "404 NOT_FOUND. {'error': {'message': 'models/gemini-2.5-pro is no longer available.'}}"
    )
    assert _is_review_model_not_found(exc) is True
    assert _is_review_model_not_found(Exception("timeout")) is False
