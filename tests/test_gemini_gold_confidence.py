"""Confidence threshold defaults for gemini-gold."""
from app.gemini_gold_trader.config import env_defaults


def test_env_default_confidence_is_90():
    cfg = env_defaults()
    assert cfg.confidence_threshold == 90
