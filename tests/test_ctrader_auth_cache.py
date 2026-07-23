"""Connection-scoped account-auth caching (order-path latency reduction)."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services import ctrader_client as cc


def _reset():
    cc._ACCOUNT_AUTH_FRESH.clear()


def test_account_auth_cached_skips_rpc_within_ttl():
    _reset()
    reader = MagicMock()
    writer = MagicMock()
    host = "demo.example.com"
    ctid = 123

    auth = AsyncMock(return_value=True)

    async def _run():
        with patch.object(cc, "_account_auth", new=auth), patch.object(
            cc, "_account_auth_cache_ttl_s", return_value=30.0
        ):
            ok1 = await cc._account_auth_cached(reader, writer, "tok", ctid, host)
            ok2 = await cc._account_auth_cached(reader, writer, "tok", ctid, host)
        assert ok1 and ok2
        # Second call served from cache — only one real auth RPC.
        assert auth.await_count == 1

    asyncio.run(_run())


def test_account_auth_cached_reauths_on_token_change():
    _reset()
    reader, writer = MagicMock(), MagicMock()
    host, ctid = "demo.example.com", 123
    auth = AsyncMock(return_value=True)

    async def _run():
        with patch.object(cc, "_account_auth", new=auth), patch.object(
            cc, "_account_auth_cache_ttl_s", return_value=30.0
        ):
            await cc._account_auth_cached(reader, writer, "tok1", ctid, host)
            await cc._account_auth_cached(reader, writer, "tok2", ctid, host)
        assert auth.await_count == 2

    asyncio.run(_run())


def test_account_auth_cached_reauths_after_invalidation():
    _reset()
    reader, writer = MagicMock(), MagicMock()
    host, ctid = "demo.example.com", 123
    auth = AsyncMock(return_value=True)

    async def _run():
        with patch.object(cc, "_account_auth", new=auth), patch.object(
            cc, "_account_auth_cache_ttl_s", return_value=30.0
        ):
            await cc._account_auth_cached(reader, writer, "tok", ctid, host)
            cc._clear_account_auth_fresh(host, ctid)
            await cc._account_auth_cached(reader, writer, "tok", ctid, host)
        assert auth.await_count == 2

    asyncio.run(_run())


def test_account_auth_cached_not_recorded_on_failure():
    _reset()
    reader, writer = MagicMock(), MagicMock()
    host, ctid = "demo.example.com", 123
    auth = AsyncMock(return_value=False)

    async def _run():
        with patch.object(cc, "_account_auth", new=auth), patch.object(
            cc, "_account_auth_cache_ttl_s", return_value=30.0
        ):
            ok = await cc._account_auth_cached(reader, writer, "tok", ctid, host)
        assert ok is False
        assert cc._acct_conn_key(host, ctid) not in cc._ACCOUNT_AUTH_FRESH

    asyncio.run(_run())
