"""Tests for gemini-gold db_thread wrapper + SSL retry."""
from __future__ import annotations

import asyncio
import inspect
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_load_merged_config_expects_db_plus_env():
    from app.gemini_gold_trader.loop import _load_merged_config

    params = list(inspect.signature(_load_merged_config).parameters)
    assert params == ["db", "env"]


def test_run_with_db_injects_single_db_arg():
    from app.gemini_gold_trader.db_thread import run_with_db

    seen: list[tuple] = []

    def _fn(db, value):
        seen.append((db, value))
        return value

    async def _run():
        mock_db = MagicMock()
        with patch("app.gemini_gold_trader.db_thread.BgSessionLocal", return_value=mock_db):
            result = await run_with_db(_fn, "env-defaults")
        assert result == "env-defaults"
        assert len(seen) == 1
        assert seen[0] == (mock_db, "env-defaults")
        mock_db.close.assert_called_once()

    asyncio.run(_run())


def test_with_db_session_retries_transient_ssl_error():
    from app.gemini_gold_trader.db_thread import with_db_session

    calls = {"n": 0}
    mock_db = MagicMock()

    @with_db_session
    def _fn(db):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("SSL connection has been closed unexpectedly")
        return "ok"

    with patch("app.gemini_gold_trader.db_thread.BgSessionLocal", return_value=mock_db):
        assert _fn() == "ok"
    assert calls["n"] == 2
    assert mock_db.close.call_count == 2


def test_load_merged_config_via_run_with_db():
    from app.gemini_gold_trader.config import env_defaults
    from app.gemini_gold_trader.db_thread import run_with_db
    from app.gemini_gold_trader.loop import _load_merged_config

    env = env_defaults()

    async def _run():
        mock_cfg_row = object()
        mock_db = MagicMock()
        with patch("app.gemini_gold_trader.db_thread.BgSessionLocal", return_value=mock_db), patch(
            "app.gemini_gold_trader.loop.seed_config_if_missing",
            return_value=mock_cfg_row,
        ) as seed, patch(
            "app.gemini_gold_trader.loop.merge_config",
            return_value=env,
        ) as merge:
            cfg = await run_with_db(_load_merged_config, env)
        assert cfg is env
        seed.assert_called_once_with(mock_db)
        merge.assert_called_once_with(mock_cfg_row, env)
        mock_db.close.assert_called_once()

    asyncio.run(_run())
