"""Tests for scheduled discovery scan runner helpers."""
from app.services.scan_schedule_runner import _dedupe_ok, _grade_gte


def test_grade_gte():
    assert _grade_gte("A", "B") is True
    assert _grade_gte("B", "B") is True
    assert _grade_gte("C", "B") is False
    assert _grade_gte("F", "A") is False


def test_dedupe_within_interval():
    key = (1, "gold", "XAUUSD", "EMA Cross", "ema", "15m", "london")
    assert _dedupe_ok(key, 60) is True
    assert _dedupe_ok(key, 60) is False
