"""Unified open-position management — partial TP, breakeven, trailing."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_PARTIAL_SKIP_LOGGED: set = set()


def effective_trade_mgmt_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new root-level trade-mgmt keys with legacy exit.* keys."""
    ex = cfg.get("exit") or {}
    be_enabled = cfg.get("breakeven_enabled")
    if be_enabled is None:
        be_enabled = bool(ex.get("breakeven_at_pips") or ex.get("breakeven_pct"))
    trail_enabled = cfg.get("trailing_enabled")
    if trail_enabled is None:
        trail_enabled = bool(ex.get("trailing_stop"))
    partial_enabled = cfg.get("partial_tp_enabled")
    if partial_enabled is None:
        partial_enabled = bool(ex.get("partial_close_pct") and ex.get("take_profit2_pips"))
    return {
        "breakeven_enabled": bool(be_enabled),
        "breakeven_trigger_pips": float(
            cfg.get("breakeven_trigger_pips")
            or ex.get("breakeven_at_pips")
            or 20
        ),
        "breakeven_offset_pips": float(cfg.get("breakeven_offset_pips", 1)),
        "trailing_enabled": bool(trail_enabled),
        "trailing_distance_pips": float(
            cfg.get("trailing_distance_pips")
            or ex.get("trailing_stop_pips")
            or 15
        ),
        "trailing_step_pips": float(
            cfg.get("trailing_step_pips") or 5
        ),
        "partial_tp_enabled": bool(partial_enabled),
        "tp1_pips": float(
            cfg.get("tp1_pips")
            or ex.get("take_profit_pips")
            or 20
        ),
        "tp1_close_percent": float(
            cfg.get("tp1_close_percent")
            or ex.get("partial_close_pct")
            or 50
        ),
        "tp1_move_sl_breakeven": bool(
            cfg.get("tp1_move_sl_breakeven", True)
        ),
    }


def active_sl(execution) -> Optional[float]:
    """Effective stop for SL-hit checks — prefers current_sl over sl_price."""
    cur = getattr(execution, "current_sl", None)
    if cur is not None:
        return float(cur)
    sl = getattr(execution, "sl_price", None)
    return float(sl) if sl is not None else None


def original_sl(execution) -> Optional[float]:
    """Original stop at fire time (before any breakeven/trailing move)."""
    notes = getattr(execution, "notes", None) or ""
    m = re.search(r"orig_sl=([0-9.+-eE]+)", notes)
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            pass
    sl = getattr(execution, "sl_price", None)
    return float(sl) if sl is not None else None


def breakeven_sl_price(
    symbol: str,
    entry: float,
    direction: str,
    offset_pips: float = 0.0,
) -> float:
    """Broker-style breakeven stop: LONG entry+offset, SHORT entry−offset."""
    from app.services.forex_engine import pip_size as _pip_size

    pip = _pip_size(symbol)
    off = float(offset_pips or 0.0) * pip
    if (direction or "").upper() == "SHORT":
        return round(float(entry) - off, 6)
    return round(float(entry) + off, 6)


def _sl_tolerance(symbol: str, entry: float) -> float:
    try:
        from app.services.forex_engine import pip_size as _pip_size

        pip = _pip_size(symbol)
        return max(float(pip) * 0.5, abs(float(entry)) * 1e-7)
    except Exception:
        return abs(float(entry)) * 1e-5


def breakeven_was_claimed(execution) -> bool:
    return bool(getattr(execution, "breakeven_applied", False)) or (
        "be_moved" in ((getattr(execution, "notes", None) or ""))
    )


def _classify_sl_outcome(sl: float, entry: float, direction: str) -> str:
    try:
        if not entry:
            return "LOSS"
        tol = abs(entry) * 1e-7
        if abs(sl - entry) <= tol:
            return "BREAKEVEN"
        if direction == "LONG" and sl > entry + tol:
            return "WIN"
        if direction == "SHORT" and sl < entry - tol:
            return "WIN"
    except Exception:
        pass
    return "LOSS"


def classify_sl_close_outcome(
    execution,
    exit_price: float,
    *,
    log_prefix: str = "[trade-mgmt]",
) -> str:
    """Classify a stop-out — never label original-SL hits as BREAKEVEN."""
    entry = float(getattr(execution, "entry_price", 0) or 0)
    direction = getattr(execution, "direction", "LONG") or "LONG"
    symbol = getattr(execution, "symbol", "") or ""
    tol = _sl_tolerance(symbol, entry)
    exit_px = float(exit_price)
    eff_sl = active_sl(execution)
    orig = original_sl(execution)

    if breakeven_was_claimed(execution) and entry > 0:
        if abs(exit_px - entry) <= tol:
            return "BREAKEVEN"
        if orig is not None and abs(exit_px - orig) <= tol and abs(orig - entry) > tol:
            exec_id = getattr(execution, "id", 0)
            logger.critical(
                "%s breakeven was applied but close occurred at original SL — "
                "SL move failed exec=%s exit=%s orig_sl=%s entry=%s current_sl=%s",
                log_prefix,
                exec_id,
                exit_px,
                orig,
                entry,
                eff_sl,
            )
            return "LOSS"

    if eff_sl is not None and entry > 0:
        return _classify_sl_outcome(float(eff_sl), entry, direction)
    return "LOSS"


def persist_paper_stop_level(
    execution,
    new_sl: float,
    session,
    *,
    mark_breakeven: bool = False,
) -> bool:
    """Write current_sl + sl_price and commit — required before BE notifications."""
    new_sl = round(float(new_sl), 6)
    execution.current_sl = new_sl
    execution.sl_price = new_sl
    if mark_breakeven:
        execution.breakeven_applied = True
    try:
        session.commit()
        session.refresh(execution)
        return True
    except Exception as exc:
        logger.warning(
            "[trade-mgmt] paper SL persist failed exec=%s: %s",
            getattr(execution, "id", 0),
            exc,
        )
        try:
            session.rollback()
        except Exception:
            pass
        return False


def audit_mislabeled_breakeven_paper_closes(session, limit: int = 20) -> int:
    """Log paper closes that were labeled BREAKEVEN but exited at original SL."""
    from app.strategy_models import StrategyExecution

    rows = (
        session.query(StrategyExecution)
        .filter(
            StrategyExecution.is_paper == True,  # noqa: E712
            StrategyExecution.outcome == "BREAKEVEN",
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(limit)
        .all()
    )
    affected = 0
    for ex in rows:
        if not breakeven_was_claimed(ex):
            continue
        entry = float(ex.entry_price or 0)
        exit_px = float(ex.exit_price or 0)
        orig = original_sl(ex)
        tol = _sl_tolerance(ex.symbol or "", entry)
        if (
            orig is not None
            and entry > 0
            and abs(exit_px - orig) <= tol
            and abs(orig - entry) > tol
        ):
            affected += 1
            logger.warning(
                "[trade-mgmt] mislabeled BREAKEVEN paper exec=%s %s %s "
                "entry=%s exit=%s orig_sl=%s pips_pnl=%s",
                ex.id,
                ex.symbol,
                ex.direction,
                entry,
                exit_px,
                orig,
                ex.pips_pnl,
            )
    if affected:
        logger.warning(
            "[trade-mgmt] breakeven mislabel audit: %s/%s recent BREAKEVEN "
            "paper closes exited at original SL",
            affected,
            len(rows),
        )
    return affected


def correct_mislabeled_breakeven_closes(session, limit: int = 20) -> int:
    """Re-label paper BREAKEVEN closes that actually exited at original SL."""
    from app.strategy_models import StrategyExecution

    rows = (
        session.query(StrategyExecution)
        .filter(
            StrategyExecution.is_paper == True,  # noqa: E712
            StrategyExecution.outcome == "BREAKEVEN",
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(limit)
        .all()
    )
    fixed = 0
    for ex in rows:
        if not breakeven_was_claimed(ex):
            continue
        entry = float(ex.entry_price or 0)
        exit_px = float(ex.exit_price or 0)
        orig = original_sl(ex)
        tol = _sl_tolerance(ex.symbol or "", entry)
        if (
            orig is not None
            and entry > 0
            and abs(exit_px - orig) <= tol
            and abs(orig - entry) > tol
        ):
            ex.outcome = "LOSS"
            fixed += 1
            logger.warning(
                "[trade-mgmt] corrected mislabeled BREAKEVEN → LOSS exec=%s "
                "%s exit=%s orig_sl=%s",
                ex.id,
                ex.symbol,
                exit_px,
                orig,
            )
    if fixed:
        try:
            session.commit()
        except Exception:
            session.rollback()
            return 0
    return fixed


def _profit_pips(symbol: str, entry: float, live_price: float, direction: str) -> float:
    from app.services.forex_engine import pip_size as _pip_size

    pip = _pip_size(symbol)
    if pip <= 0:
        return 0.0
    if direction == "SHORT":
        return (entry - live_price) / pip
    return (live_price - entry) / pip


def _locked_pips(symbol: str, entry: float, sl: float, direction: str) -> float:
    """Pips locked in by the current stop level (0 if stop is still behind entry)."""
    from app.services.forex_engine import pip_size as _pip_size

    pip = _pip_size(symbol)
    if pip <= 0 or sl is None:
        return 0.0
    if direction == "SHORT":
        return max(0.0, (entry - float(sl)) / pip)
    return max(0.0, (float(sl) - entry) / pip)


def update_excursion_pips(execution, profit_pips: float, session) -> bool:
    """Track peak profit (MFE) and peak drawdown (MAE) — commit only when improved."""
    changed = False
    try:
        cur_mfe = getattr(execution, "mfe_pips", None)
        if profit_pips > float(cur_mfe or 0.0):
            execution.mfe_pips = round(profit_pips, 1)
            changed = True
        adverse = max(0.0, -float(profit_pips))
        cur_mae = getattr(execution, "mae_pips", None)
        if adverse > float(cur_mae or 0.0):
            execution.mae_pips = round(adverse, 1)
            changed = True
        if changed:
            session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass
        changed = False
    return changed


def _validate_sl(direction: str, new_sl: float, live_price: float) -> bool:
    if direction == "LONG":
        return new_sl < live_price
    return new_sl > live_price


def _round_close_volume(remaining: float, pct: float, step: int, min_vol: int) -> int:
    if remaining <= 0 or pct <= 0:
        return 0
    raw = int(remaining * pct / 100.0)
    if step > 0:
        raw = (raw // step) * step
    if min_vol > 0 and raw < min_vol:
        return 0
    if raw <= 0 or raw >= int(remaining):
        return 0
    if min_vol > 0 and (int(remaining) - raw) < min_vol:
        return 0
    return raw


async def _telegram_trade(
    user_id: int,
    text: str,
    asset_class: str = "forex",
    *,
    msg_type: str = "trade",
    symbol: str = "",
    exec_id: int = 0,
) -> None:
    try:
        from app.database import BgSessionLocal
        from app.models import User
        from app.services.strategy_executor import _telegram_int_id
        from app.services.telegram_dm import deliver_trade_telegram

        db = BgSessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            tg = _telegram_int_id(user) if user else None
        finally:
            db.close()
        if tg:
            await deliver_trade_telegram(
                tg,
                text,
                msg_type=msg_type,
                symbol=symbol,
                exec_id=exec_id,
                asset_class=asset_class,
            )
    except Exception as exc:
        logger.warning(
            "[telegram] FAILED %s %s exec=%s: %s",
            msg_type,
            (symbol or "?").upper(),
            exec_id or 0,
            exc,
        )


async def manage_open_position(
    execution,
    strategy_cfg: Dict[str, Any],
    live_price: float,
    session,
) -> None:
    """Partial TP → breakeven → trailing for one OPEN execution."""
    if not execution or execution.outcome != "OPEN":
        return
    if not live_price or live_price <= 0:
        return
    if not execution.entry_price:
        return

    symbol = execution.symbol
    direction = execution.direction
    entry = float(execution.entry_price)
    profit_pips = _profit_pips(symbol, entry, live_price, direction)
    update_excursion_pips(execution, profit_pips, session)

    cfg = effective_trade_mgmt_cfg(strategy_cfg or {})
    if not any((
        cfg["partial_tp_enabled"],
        cfg["breakeven_enabled"],
        cfg["trailing_enabled"],
    )):
        return
    is_paper = bool(getattr(execution, "is_paper", False))
    asset_class = getattr(execution, "asset_class", "forex") or "forex"

    if execution.current_sl is None and execution.sl_price is not None:
        execution.current_sl = float(execution.sl_price)
    if execution.remaining_volume is None and execution.broker_volume_units:
        execution.remaining_volume = float(execution.broker_volume_units)
    elif execution.remaining_volume is None:
        execution.remaining_volume = 1.0

    cur_sl = active_sl(execution)
    force_be = False

    # ── 3. Partial TP ─────────────────────────────────────────────────────
    if (
        cfg["partial_tp_enabled"]
        and not bool(getattr(execution, "tp1_done", False))
        and profit_pips >= cfg["tp1_pips"]
    ):
        remaining = float(execution.remaining_volume or 0)
        close_vol = _round_close_volume(
            remaining, cfg["tp1_close_percent"], step=1, min_vol=1,
        )
        if is_paper:
            if close_vol > 0:
                execution.tp1_done = True
                execution.tp1_closed_volume = float(close_vol)
                execution.tp1_realized_pips = round(profit_pips, 1)
                execution.remaining_volume = remaining - close_vol
                if cfg["tp1_move_sl_breakeven"]:
                    force_be = True
                session.commit()
                await _telegram_trade(
                    execution.user_id,
                    f"💰 TP1 {symbol}: closed {cfg['tp1_close_percent']:.0f}% "
                    f"@ {live_price:.5g} (+{profit_pips:.1f} pips) — runner active",
                    asset_class=asset_class,
                )
        else:
            from app.models import User
            from app.services.ctrader_client import close_partial_position_for_user

            user = session.query(User).filter(User.id == execution.user_id).first()
            pos_id = getattr(execution, "ctrader_position_id", None)
            total_vol = int(execution.broker_volume_units or remaining)
            frac = cfg["tp1_close_percent"] / 100.0
            if user and pos_id and total_vol > 0 and frac > 0:
                closed = await close_partial_position_for_user(
                    user, symbol, int(pos_id), total_vol, frac,
                )
                if closed > 0:
                    execution.tp1_done = True
                    execution.tp1_closed_volume = float(closed)
                    execution.tp1_realized_pips = round(profit_pips, 1)
                    execution.remaining_volume = remaining - float(closed)
                    if cfg["tp1_move_sl_breakeven"]:
                        force_be = True
                    session.commit()
                    await _telegram_trade(
                        execution.user_id,
                        f"💰 TP1 {symbol}: closed {cfg['tp1_close_percent']:.0f}% "
                        f"@ {live_price:.5g} (+{profit_pips:.1f} pips) — runner active",
                        asset_class=asset_class,
                    )
            elif close_vol <= 0 and execution.id not in _PARTIAL_SKIP_LOGGED:
                _PARTIAL_SKIP_LOGGED.add(execution.id)
                logger.warning(
                    "[trade-mgmt] partial TP skipped %s exec#%s — volume below broker min",
                    symbol, execution.id,
                )

    # ── 4. Breakeven ──────────────────────────────────────────────────────
    be_ready = (
        cfg["breakeven_enabled"]
        and not bool(getattr(execution, "breakeven_applied", False))
        and (force_be or profit_pips >= cfg["breakeven_trigger_pips"])
    )
    if be_ready:
        new_sl = breakeven_sl_price(
            symbol, entry, direction, cfg["breakeven_offset_pips"],
        )
        if _validate_sl(direction, new_sl, live_price):
            amended = await _apply_sl_amend(
                execution,
                new_sl,
                session,
                is_paper,
                keep_tp=True,
                mark_breakeven=True,
            )
            if amended:
                if not is_paper:
                    execution.breakeven_applied = True
                    session.commit()
                await _telegram_trade(
                    execution.user_id,
                    f"🔒 Breakeven set @ {new_sl:.5g} — reached +{profit_pips:.1f} "
                    f"pips (trigger {cfg['breakeven_trigger_pips']:.0f})",
                    asset_class=asset_class,
                    msg_type="breakeven",
                    symbol=symbol,
                    exec_id=int(getattr(execution, "id", 0) or 0),
                )
        else:
            logger.warning(
                "[trade-mgmt] breakeven SL invalid %s %s new_sl=%s live=%s",
                symbol, direction, new_sl, live_price,
            )

    # ── 5. Trailing ───────────────────────────────────────────────────────
    if cfg["trailing_enabled"] and (
        bool(getattr(execution, "breakeven_applied", False))
        or not cfg["breakeven_enabled"]
    ):
        from app.services.forex_engine import pip_size as _pip_size

        pip = _pip_size(symbol)
        dist = cfg["trailing_distance_pips"] * pip
        step = cfg["trailing_step_pips"] * pip
        cur_sl = active_sl(execution)
        if direction == "LONG":
            candidate = live_price - dist
            if cur_sl is None or candidate > cur_sl + step:
                if _validate_sl(direction, candidate, live_price):
                    if await _apply_sl_amend(
                        execution, candidate, session, is_paper, keep_tp=True,
                    ):
                        execution.current_sl = candidate
                        execution.sl_price = candidate
                        session.commit()
                        _locked = _locked_pips(symbol, entry, candidate, direction)
                        _peak = float(getattr(execution, "mfe_pips", None) or profit_pips)
                        await _telegram_trade(
                            execution.user_id,
                            f"📈 Trailing SL → {candidate:.5g} "
                            f"(+{_locked:.1f} pips locked, peak +{_peak:.1f} pips)",
                            asset_class=asset_class,
                            msg_type="breakeven",
                            symbol=symbol,
                            exec_id=int(getattr(execution, "id", 0) or 0),
                        )
        else:
            candidate = live_price + dist
            if cur_sl is None or candidate < cur_sl - step:
                if _validate_sl(direction, candidate, live_price):
                    if await _apply_sl_amend(
                        execution, candidate, session, is_paper, keep_tp=True,
                    ):
                        execution.current_sl = candidate
                        execution.sl_price = candidate
                        session.commit()
                        _locked = _locked_pips(symbol, entry, candidate, direction)
                        _peak = float(getattr(execution, "mfe_pips", None) or profit_pips)
                        await _telegram_trade(
                            execution.user_id,
                            f"📈 Trailing SL → {candidate:.5g} "
                            f"(+{_locked:.1f} pips locked, peak +{_peak:.1f} pips)",
                            asset_class=asset_class,
                            msg_type="breakeven",
                            symbol=symbol,
                            exec_id=int(getattr(execution, "id", 0) or 0),
                        )


async def _apply_sl_amend(
    execution,
    new_sl: float,
    session,
    is_paper: bool,
    *,
    keep_tp: bool = True,
    mark_breakeven: bool = False,
) -> bool:
    new_sl = round(float(new_sl), 6)
    if is_paper:
        return persist_paper_stop_level(
            execution, new_sl, session, mark_breakeven=mark_breakeven,
        )
    from app.models import User
    from app.services.ctrader_client import amend_position_sl

    user = session.query(User).filter(User.id == execution.user_id).first()
    pos_id = getattr(execution, "ctrader_position_id", None)
    if not user or not pos_id:
        logger.warning(
            "[trade-mgmt] live SL amend skipped exec=%s — no broker position",
            getattr(execution, "id", 0),
        )
        return False
    tp = float(execution.tp2_price or execution.tp_price) if keep_tp else None
    ok = await amend_position_sl(user.id, int(pos_id), new_sl, keep_tp=tp)
    if not ok:
        logger.warning(
            "[trade-mgmt] live SL amend FAILED exec=%s %s new_sl=%s — "
            "breakeven notification suppressed",
            getattr(execution, "id", 0),
            execution.symbol,
            new_sl,
        )
        await _telegram_trade(
            execution.user_id,
            f"⚠️ SL amend failed {execution.symbol}: broker rejected amend",
            asset_class=getattr(execution, "asset_class", "forex") or "forex",
        )
        return False
    execution.current_sl = new_sl
    execution.sl_price = new_sl
    try:
        session.commit()
    except Exception:
        session.rollback()
        return False
    return True
