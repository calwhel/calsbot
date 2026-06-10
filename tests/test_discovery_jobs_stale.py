"""Tests for discovery job stale/orphan recovery."""
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.services import discovery_jobs as dj


def test_job_is_stale_when_running_too_long():
    row = SimpleNamespace(
        status="running",
        updated_at=datetime.utcnow() - timedelta(seconds=dj._DISCOVERY_STALE_SECS + 30),
        created_at=datetime.utcnow() - timedelta(hours=1),
    )
    assert dj._job_is_stale(row) is True


def test_job_not_stale_when_recent():
    row = SimpleNamespace(
        status="running",
        updated_at=datetime.utcnow() - timedelta(seconds=30),
        created_at=datetime.utcnow() - timedelta(minutes=2),
    )
    assert dj._job_is_stale(row) is False
