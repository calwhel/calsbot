"""FMP economic calendar cache — entry blocking and open-position news warnings."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL_S = 4 * 3600
_EVENTS: List[Dict] = []
_NEWS_WARN_SENT: Set[Tuple[int, str]] = set()
_task: Optional[asyncio.Task] = None


def _parse_event_time(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "").replace("T", " ")[:19]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def _normalize_impact(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    if "high" in s:
        return "high"
    if "med" in s:
        return "medium"
    return "low"


def _symbol_currencies(symbol: str) -> List[str]:
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym in ("XAUUSD", "XAGUSD", "GOLD", "SILVER"):
        return ["USD"]
    if len(sym) >= 6:
        return [sym[:3], sym[3:6]]
    return []


def _impact_matches(cfg_impact: str, event_impact: str) -> bool:
    cfg = (cfg_impact or "high").lower()
    if cfg == "high_medium":
        return event_impact in ("high", "medium")
    return event_impact == "high"


async def _refresh_once() -> None:
    global _EVENTS
    try:
        from app.services.tradfi_prices import _env_fmp_api_key
        from app.services.fmp_price_feed import fetch_economic_calendar
    except Exception as exc:
        logger.warning("[calendar] import failed: %s", exc)
        return

    if not _env_fmp_api_key():
        return

    today = datetime.utcnow().date()
    from_d = today.strftime("%Y-%m-%d")
    to_d = (today + timedelta(days=3)).strftime("%Y-%m-%d")
    raw = await fetch_economic_calendar(from_d, to_d)
    if not raw:
        logger.warning("[calendar] FMP fetch failed — keeping stale (%d events)", len(_EVENTS))
        return

    parsed: List[Dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        t = _parse_event_time(row.get("date") or row.get("time") or "")
        if not t:
            continue
        currency = (row.get("currency") or row.get("country") or "").upper()[:3]
        if not currency:
            continue
        parsed.append({
            "time_utc": t,
            "currency": currency,
            "impact": _normalize_impact(row.get("impact") or ""),
            "event_name": str(row.get("event") or row.get("name") or "Economic event"),
        })

    parsed.sort(key=lambda e: e["time_utc"])
    if parsed:
        _EVENTS = parsed
        logger.info("[calendar] refreshed %d events (%s → %s)", len(_EVENTS), from_d, to_d)


async def _refresh_loop() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            await _refresh_once()
        except Exception as exc:
            logger.warning("[calendar] refresh error: %s", exc)
        await asyncio.sleep(_REFRESH_INTERVAL_S)


def start() -> None:
    """Schedule the 4h economic-calendar refresh (executor worker only)."""
    global _task
    if _task and not _task.done():
        return
    _task = asyncio.create_task(_refresh_loop())
    logger.info("[calendar] economic calendar refresh scheduled (every 4h)")


def blocking_event(symbol: str, cfg: dict, now_utc: datetime) -> Optional[str]:
    """Return event name when entries should be blocked, else None. Fail-open if empty."""
    if not cfg.get("news_filter_enabled"):
        return None
    if not _EVENTS:
        return None
    before = int(cfg.get("news_buffer_before_min", 30))
    after = int(cfg.get("news_buffer_after_min", 30))
    impact_cfg = cfg.get("news_impact", "high")
    currencies = _symbol_currencies(symbol)
    for ev in _EVENTS:
        if ev["currency"] not in currencies:
            continue
        if not _impact_matches(impact_cfg, ev["impact"]):
            continue
        t0 = ev["time_utc"] - timedelta(minutes=before)
        t1 = ev["time_utc"] + timedelta(minutes=after)
        if t0 <= now_utc <= t1:
            return ev["event_name"]
    return None


def upcoming_high_event(symbol: str, now_utc: datetime, within_min: int = 15) -> Optional[str]:
    """HIGH-impact event starting within within_min minutes (warning only)."""
    if not _EVENTS:
        return None
    currencies = _symbol_currencies(symbol)
    horizon = now_utc + timedelta(minutes=within_min)
    for ev in _EVENTS:
        if ev["currency"] not in currencies or ev["impact"] != "high":
            continue
        t = ev["time_utc"]
        if now_utc <= t <= horizon:
            return ev["event_name"]
    return None


async def maybe_warn_open_position(
    execution_id: int,
    symbol: str,
    user_id: int,
    asset_class: str = "forex",
) -> None:
    """Telegram warning once per (execution, event) when HIGH news is within 15 min."""
    if asset_class not in ("forex", "metals", "commodity", "index"):
        return
    evt = upcoming_high_event(symbol, datetime.utcnow(), within_min=15)
    if not evt:
        return
    key = (execution_id, evt)
    if key in _NEWS_WARN_SENT:
        return
    _NEWS_WARN_SENT.add(key)
    try:
        from app.services.strategy_executor import _tg_send, _telegram_int_id
        from app.models import User
        from app.database import BgSessionLocal

        db = BgSessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            tg_id = _telegram_int_id(user) if user else None
        finally:
            db.close()
        if tg_id:
            await _tg_send(
                tg_id,
                f"⚠️ {evt} in 15 min — open {symbol} position",
                asset_class=asset_class,
            )
    except Exception as exc:
        logger.debug("[calendar] news warn failed exec#%s: %s", execution_id, exc)
