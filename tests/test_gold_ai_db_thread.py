"""Tests for gold-ai db_thread wrapper arg injection."""
from __future__ import annotations

import asyncio
import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_load_merged_config_expects_db_plus_env():
    from app.gold_ai_trader.loop import _load_merged_config

    params = list(inspect.signature(_load_merged_config).parameters)
    assert params == ["db", "env"]


@pytest.mark.asyncio
async def test_run_with_db_injects_single_db_arg():
    from app.gold_ai_trader.db_thread import run_with_db

    seen: list[tuple] = []

    def _fn(db, value):
        seen.append((db, value))
        return value

    mock_db = MagicMock()
    with patch("app.gold_ai_trader.db_thread.SessionLocal", return_value=mock_db):
        result = await run_with_db(_fn, "env-defaults")

    assert result == "env-defaults"
    assert len(seen) == 1
    assert seen[0] == (mock_db, "env-defaults")
    mock_db.close.assert_called_once()


@pytest.mark.asyncio
async def test_load_merged_config_via_run_with_db():
    from app.gold_ai_trader.config import env_defaults
    from app.gold_ai_trader.db_thread import run_with_db
    from app.gold_ai_trader.loop import _load_merged_config

    env = env_defaults()
    mock_cfg_row = object()
    mock_db = MagicMock()

    with patch("app.gold_ai_trader.db_thread.SessionLocal", return_value=mock_db), patch(
        "app.gold_ai_trader.loop.seed_config_if_missing",
        return_value=mock_cfg_row,
    ) as seed, patch(
        "app.gold_ai_trader.loop.merge_config",
        return_value=env,
    ) as merge:
        cfg = await run_with_db(_load_merged_config, env)

    assert cfg is env
    seed.assert_called_once_with(mock_db)
    merge.assert_called_once_with(mock_cfg_row, env)
    mock_db.close.assert_called_once()


@pytest.mark.asyncio
async def test_double_with_db_session_raises_type_error():
    from app.gold_ai_trader.db_thread import run_with_db, with_db_session

    @with_db_session
    def _already_wrapped(db, value):
        return value

    with patch("app.gold_ai_trader.db_thread.SessionLocal", return_value=MagicMock()):
        with pytest.raises(TypeError, match="positional arguments"):
            await run_with_db(_already_wrapped, "x")
