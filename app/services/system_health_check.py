"""
Hourly system health monitor.

Runs once an hour inside the single advisory-locked executor worker and verifies
that every moving part of the platform is in working order:

    • Neon database (the single source of truth for all envs)
    • Public website / portal edge (TLS + proxy + app reachable)
    • Telegram bot API (the channel this very report is sent over)
    • FMP price feed (forex / metals / index market data)
    • cTrader spot feed + broker link (live forex execution)
    • MEXC market data (crypto strategy evaluation)
    • Bitunix API (crypto broker execution)
    • Executor scan loops (crypto / forex / live-manager heartbeats)

It then DMs the owner on Telegram a single summary card — every hour, whether
everything is green or something needs attention — so the owner always has an
up-to-date confirmation that the platform is alive.

Each individual check is fully guarded: one failing probe never aborts the rest,
and a probe that depends on an externally-closed market (forex on weekends) is
downgraded to an informational note instead of a false alarm.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# How often to run the full sweep (seconds). Override with EXECUTOR_HEALTH_INTERVAL.
HEALTH_INTERVAL_SECONDS = int(os.getenv("EXECUTOR_HEALTH_INTERVAL", "3600"))

# Public site to probe for the edge/TLS check. Override with PUBLIC_SITE_URL.
PUBLIC_SITE_URL = os.getenv("PUBLIC_SITE_URL", "https://tradehubmarkets.com").rstrip("/")

# A loop is considered "alive" if it ran a cycle within this many seconds.
_HEARTBEAT_FRESH_SECS = 600  # 10 minutes (loops cycle every 1–15s)

# Severity → icon
_ICON = {"ok": "✅", "warn": "⚠️", "crit": "❌", "info": "ℹ️"}

# In-process guard so a second monitor loop can't start (see run_system_health_monitor).
_MONITOR_ACTIVE = False


def _result(key: str, label: str, status: str, detail: str = "") -> Dict[str, Any]:
    return {"key": key, "label": label, "status": status, "detail": detail}


# ── Individual checks ─────────────────────────────────────────────────────────

async def _check_database() -> Dict[str, Any]:
    """Neon reachability + a couple of light operational counts."""
    def _run() -> Dict[str, Any]:
        from sqlalchemy import text
        from app.database import SessionLocal
        from app.strategy_models import UserStrategy, StrategyExecution
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            active = (
                db.query(UserStrategy)
                .filter(UserStrategy.status.in_(["active", "paper"]))
                .count()
            )
            open_live = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.outcome == "OPEN",
                    StrategyExecution.is_paper.is_(False),
                )
                .count()
            )
            return {"active": active, "open_live": open_live}
        finally:
            db.close()

    try:
        info = await asyncio.wait_for(asyncio.to_thread(_run), timeout=15)
        return _result(
            "database", "Database (Neon)", "ok",
            f"{info['active']} active/paper strategies · {info['open_live']} open live trades",
        )
    except Exception as e:
        return _result("database", "Database (Neon)", "crit", f"{type(e).__name__}: {e}")


async def _check_website(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Hit the public health endpoint through the production edge."""
    url = f"{PUBLIC_SITE_URL}/health"
    try:
        r = await client.get(url, timeout=12)
        if r.status_code == 200:
            return _result("website", "Website", "ok", f"{PUBLIC_SITE_URL} → HTTP 200")
        return _result("website", "Website", "crit", f"{url} → HTTP {r.status_code}")
    except Exception as e:
        return _result("website", "Website", "crit", f"{type(e).__name__}: {e}")


async def _check_telegram(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Verify the bot token(s) with getMe."""
    tokens = []
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        tokens.append(("main", os.getenv("TELEGRAM_BOT_TOKEN")))
    if os.getenv("FOREX_BOT_TOKEN") and os.getenv("FOREX_BOT_TOKEN") != os.getenv("TELEGRAM_BOT_TOKEN"):
        tokens.append(("forex", os.getenv("FOREX_BOT_TOKEN")))
    if not tokens:
        return _result("telegram", "Telegram bot", "crit", "no bot token configured")
    names: List[str] = []
    for label, tok in tokens:
        try:
            r = await client.get(f"https://api.telegram.org/bot{tok}/getMe", timeout=10)
            j = r.json()
            if r.status_code == 200 and j.get("ok"):
                names.append(f"{label}=@{j['result'].get('username', '?')}")
            else:
                return _result("telegram", "Telegram bot", "crit", f"{label} getMe failed (HTTP {r.status_code})")
        except Exception as e:
            return _result("telegram", "Telegram bot", "crit", f"{label}: {type(e).__name__}: {e}")
    return _result("telegram", "Telegram bot", "ok", " · ".join(names))


async def _check_fmp(forex_open: bool) -> Dict[str, Any]:
    """FMP feed powers forex / metals / index prices."""
    def _run() -> Optional[float]:
        try:
            from app.services.spot_price_store import get_mid
            px = get_mid("EURUSD")
            if px:
                return px
        except Exception:
            pass
        from app.services import fmp_price_feed
        return fmp_price_feed.get_price("EURUSD")

    if not os.getenv("FMP_API_KEY"):
        return _result("fmp", "FMP price feed", "crit", "FMP_API_KEY not set")
    try:
        price = await asyncio.wait_for(asyncio.to_thread(_run), timeout=15)
        if price and price > 0:
            return _result("fmp", "FMP price feed", "ok", f"EURUSD={price}")
        # No price: only an issue while the forex market is open.
        if forex_open:
            return _result("fmp", "FMP price feed", "warn", "no EURUSD price returned")
        return _result("fmp", "FMP price feed", "info", "no price (forex market closed)")
    except Exception as e:
        return _result("fmp", "FMP price feed", "crit", f"{type(e).__name__}: {e}")


async def _check_ctrader(forex_open: bool) -> Dict[str, Any]:
    """cTrader spot feed + broker link (only meaningful if a user has linked cTrader)."""
    def _connected_count() -> int:
        from app.database import SessionLocal
        from app.models import UserPreference
        db = SessionLocal()
        try:
            return (
                db.query(UserPreference)
                .filter(UserPreference.ctrader_account_id.isnot(None))
                .count()
            )
        finally:
            db.close()

    try:
        from app.services import ctrader_price_feed
        try:
            linked = await asyncio.wait_for(asyncio.to_thread(_connected_count), timeout=10)
        except Exception:
            linked = -1  # unknown

        st = ctrader_price_feed.feed_status()
        sym_n = int(st.get("symbol_count") or 0)
        syms = st.get("cached_symbols") or ctrader_price_feed.cached_symbols()
        live = bool(st.get("live")) or sym_n > 0

        if live and sym_n > 0:
            sample = syms[0] if syms else "?"
            return _result(
                "ctrader", "cTrader feed", "ok",
                f"{sym_n} live ticks (e.g. {sample}) · {linked} linked account(s)",
            )
        # Not live. Dormant-when-nobody-linked is expected, not a fault.
        if linked == 0:
            return _result("ctrader", "cTrader feed", "info", "dormant (no linked cTrader accounts)")
        if forex_open:
            return _result("ctrader", "cTrader feed", "warn", f"feed not live ({linked} linked account(s))")
        return _result("ctrader", "cTrader feed", "info", "feed idle (forex market closed)")
    except Exception as e:
        return _result("ctrader", "cTrader feed", "crit", f"{type(e).__name__}: {e}")


async def _check_mexc(client: httpx.AsyncClient) -> Dict[str, Any]:
    """MEXC supplies crypto klines for strategy evaluation (24/7 market)."""
    try:
        r = await client.get(
            "https://api.mexc.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1},
            timeout=12,
        )
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return _result("mexc", "Crypto data (MEXC)", "ok", "BTCUSDT klines flowing")
        return _result("mexc", "Crypto data (MEXC)", "warn", f"HTTP {r.status_code} / unexpected body")
    except Exception as e:
        return _result("mexc", "Crypto data (MEXC)", "crit", f"{type(e).__name__}: {e}")


async def _check_bitunix(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Bitunix is the crypto broker; verify its public market endpoint is reachable."""
    try:
        r = await client.get(
            "https://fapi.bitunix.com/api/v1/futures/market/tickers", timeout=12
        )
        if r.status_code == 200:
            data = (r.json() or {}).get("data") or []
            n = len([t for t in data if str(t.get("symbol", "")).endswith("USDT")])
            if n > 0:
                return _result("bitunix", "Bitunix broker", "ok", f"{n} USDT perps listed")
            return _result("bitunix", "Bitunix broker", "warn", "reachable but empty ticker list")
        return _result("bitunix", "Bitunix broker", "crit", f"HTTP {r.status_code}")
    except Exception as e:
        return _result("bitunix", "Bitunix broker", "crit", f"{type(e).__name__}: {e}")


def _check_executor(forex_open: bool) -> Dict[str, Any]:
    """Verify the background scan loops are cycling via their heartbeats."""
    try:
        from app.services.strategy_executor import get_heartbeats
        hb = get_heartbeats()
        now = time.time()

        def fresh(name: str) -> bool:
            ts = hb.get(name)
            return ts is not None and (now - ts) <= _HEARTBEAT_FRESH_SECS

        # Crypto loop should always be cycling (24/7 market).
        loops = {
            "crypto_executor": True,
            "paper_monitor": True,
            "live_monitor": True,
        }
        # Forex loops only cycle meaningfully while the forex market is open.
        if forex_open:
            loops["forex_executor"] = True
            loops["forex_live_manager"] = True

        stale = [name for name in loops if not fresh(name)]
        if not stale:
            checked = ", ".join(sorted(loops))
            return _result("executor", "Executor loops", "ok", f"cycling: {checked}")
        return _result("executor", "Executor loops", "crit", f"stale/never-ran: {', '.join(sorted(stale))}")
    except Exception as e:
        return _result("executor", "Executor loops", "crit", f"{type(e).__name__}: {e}")


# ── Orchestration ─────────────────────────────────────────────────────────────

async def run_system_health_check() -> List[Dict[str, Any]]:
    """Run every probe, guarded, and return the list of result dicts."""
    try:
        from app.services.asset_classes import is_market_open
        forex_open = bool(is_market_open("forex", datetime.now(timezone.utc)))
    except Exception:
        forex_open = True  # assume open → never silently skip forex checks

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        probes = await asyncio.gather(
            _check_database(),
            _check_website(client),
            _check_telegram(client),
            _check_fmp(forex_open),
            _check_ctrader(forex_open),
            _check_mexc(client),
            _check_bitunix(client),
            return_exceptions=True,
        )
    for p in probes:
        if isinstance(p, Exception):
            results.append(_result("unknown", "Check", "crit", f"{type(p).__name__}: {p}"))
        else:
            results.append(p)
    # Executor heartbeat check is purely in-process (no awaiting).
    results.append(_check_executor(forex_open))
    return results


def format_report(results: List[Dict[str, Any]]) -> str:
    """Build the HTML Telegram summary card. All dynamic text is escaped."""
    crit = [r for r in results if r["status"] == "crit"]
    warn = [r for r in results if r["status"] == "warn"]

    if crit:
        header = "❌ <b>TradeHub Health — ACTION NEEDED</b>"
    elif warn:
        header = "⚠️ <b>TradeHub Health — minor issues</b>"
    else:
        header = "✅ <b>TradeHub Health — all systems healthy</b>"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [header, f"<i>{html.escape(ts)}</i>", ""]
    for r in results:
        icon = _ICON.get(r["status"], "•")
        label = html.escape(str(r["label"]))
        detail = html.escape(str(r["detail"]))
        if detail:
            lines.append(f"{icon} <b>{label}</b> — {detail}")
        else:
            lines.append(f"{icon} <b>{label}</b>")

    if crit:
        lines.append("")
        lines.append("👉 <b>One or more systems need attention.</b>")
    return "\n".join(lines)


async def _send_report(text: str) -> bool:
    """DM the owner the report. Returns True only on confirmed delivery."""
    from app.services.telegram_dm import owner_chat_id, send_dm

    owner = owner_chat_id()
    if not owner:
        logger.warning("[health] OWNER_TELEGRAM_ID missing — cannot send report")
        return False
    sent = await send_dm(owner, text)
    if not sent:
        logger.warning("[health] failed to deliver report to owner %s", owner)
    return sent


async def run_system_health_monitor() -> None:
    """Hourly loop: run the sweep and DM the owner a summary every cycle."""
    # In-process singleton guard: the executor reclaim path doesn't cancel
    # previously-started background tasks on advisory-lock churn, so a reclaim
    # in the SAME process could otherwise spawn a second monitor loop → duplicate
    # owner DMs. Refuse to start a second concurrent loop in this process.
    global _MONITOR_ACTIVE
    if _MONITOR_ACTIVE:
        logger.info("[health] monitor already running in this process — skipping duplicate loop")
        return
    _MONITOR_ACTIVE = True
    try:
        # Brief startup delay so executor loops can record their first heartbeat
        # before the first sweep (avoids a spurious "stale loops" first report).
        await asyncio.sleep(60)
        while True:
            try:
                results = await run_system_health_check()
                report = format_report(results)
                sent = await _send_report(report)
                n_crit = sum(1 for r in results if r["status"] == "crit")
                if sent:
                    logger.info(f"[health] hourly sweep delivered — {len(results)} checks, {n_crit} critical")
                else:
                    logger.warning(f"[health] hourly sweep NOT delivered — {len(results)} checks, {n_crit} critical")
            except Exception as e:
                logger.error(f"[health] sweep failed: {e}")
            await asyncio.sleep(HEALTH_INTERVAL_SECONDS)
    finally:
        _MONITOR_ACTIVE = False
