"""Unified open-position management — partial TP, breakeven, trailing."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PARTIAL_SKIP_LOGGED: set = set()


def effective_trade_mgmt_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new root-level trade-mgmt keys with legacy exit.* keys."""
    ex = cfg.get("exit") or {}
    be_enabled = cfg.get("breakeven_enabled")
    if be_enabled is None:
        # Pip-based trigger only — legacy breakeven_pct/breakeven_at_pct are ignored
        # (they fed the removed PaperMonitor ROI AUTO-BREAKEVEN path).
        be_enabled = bool(
            ex.get("breakeven_at_pips")
            or cfg.get("breakeven_trigger_pips")
        )
    trail_enabled = cfg.get("trailing_enabled")
    if trail_enabled is None:
        trail_enabled = bool(ex.get("trailing_stop"))
    partial_enabled = cfg.get("partial_tp_enabled")
    if partial_enabled is None:
        partial_enabled = bool(ex.get("partial_close_pct") and ex.get("take_profit2_pips"))
    return {
        "breakeven_enabled": bool(be_enabled),
        "breakeven_trigger_pips": max(
            1.0,
            float(
                cfg.get("breakeven_trigger_pips")
                or ex.get("breakeven_at_pips")
                or 20
            ),
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


def position_id_from_execution(execution) -> Optional[int]:
    """Broker position id from column or pos= token in notes."""
    if getattr(execution, "ctrader_position_id", None):
        try:
            return int(execution.ctrader_position_id)
        except (TypeError, ValueError):
            pass
    m = re.search(r"pos=(\d+)", (getattr(execution, "notes", None) or ""))
    return int(m.group(1)) if m else None


def _normalize_operator(raw: Any) -> str:
    op = str(raw or "gt").strip().lower()
    aliases = {
        "above": "gt",
        "below": "lt",
        "greater": "gt",
        "greater_than": "gt",
        "less": "lt",
        "less_than": "lt",
        "overbought": "gt",
        "oversold": "lt",
        "rising": "gt",
        "falling": "lt",
    }
    return aliases.get(op, op)


def indicator_threshold(cond: Dict[str, Any]) -> float:
    for key in ("value", "threshold", "level", "rsi_level", "rsi_threshold"):
        if cond.get(key) is not None:
            try:
                return float(cond[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def normalize_indicator_condition(cond: Dict[str, Any]) -> Dict[str, Any]:
    """Unify operator/threshold from wizard + AI config variants."""
    op = _normalize_operator(cond.get("operator") or cond.get("condition"))
    val = indicator_threshold(cond)
    return {**cond, "operator": op, "value": val}


def effective_tp(execution) -> Optional[float]:
    """Active take-profit — TP2 when partial runner still open."""
    notes = getattr(execution, "notes", None) or ""
    if ("partial_close_pct=" in notes or "partial_close_done" in notes) and getattr(
        execution, "tp2_price", None
    ):
        return float(execution.tp2_price)
    tp = getattr(execution, "tp_price", None)
    return float(tp) if tp is not None else None


def _close_level_tolerance(symbol: str, entry: float, level: float) -> float:
    try:
        from app.services.forex_engine import pip_size as _pip_size

        pip = _pip_size(symbol)
        return max(float(pip) * 3.0, abs(float(level)) * 1e-6)
    except Exception:
        return max(abs(float(level)) * 1e-5, abs(float(entry)) * 1e-5)


def check_directional_exit_hit(
    execution,
    high: float,
    low: float,
) -> Optional[Tuple[str, float, str]]:
    """
    Return (outcome, exit_price, hit_kind) when TP or SL is touched.
    hit_kind is 'tp' or 'sl'. Uses active_sl / effective_tp only.
    """
    direction = (getattr(execution, "direction", "LONG") or "LONG").upper()
    tp = effective_tp(execution)
    sl = active_sl(execution)
    if sl is None:
        sl = getattr(execution, "sl_price", None)
    if tp is None and sl is None:
        return None
    if direction == "LONG":
        tp_hit = tp is not None and high >= float(tp)
        sl_hit = sl is not None and low <= float(sl)
        if tp_hit and not sl_hit:
            return "WIN", float(tp), "tp"
        if sl_hit and not tp_hit:
            return classify_sl_close_outcome(execution, float(sl)), float(sl), "sl"
    else:
        tp_hit = tp is not None and low <= float(tp)
        sl_hit = sl is not None and high >= float(sl)
        if tp_hit and not sl_hit:
            return "WIN", float(tp), "tp"
        if sl_hit and not tp_hit:
            return classify_sl_close_outcome(execution, float(sl)), float(sl), "sl"
    return None


def validate_close_sanity(
    execution,
    outcome: str,
    exit_price: float,
    hit_kind: str,
) -> Tuple[str, str]:
    """
    Reject phantom TP/SL labels. Returns (outcome, note_label).
    Logs CRITICAL on mismatch.
    """
    symbol = getattr(execution, "symbol", "") or ""
    entry = float(getattr(execution, "entry_price", 0) or 0)
    exit_px = float(exit_price)
    exec_id = getattr(execution, "id", 0)
    tp = effective_tp(execution)
    sl = active_sl(execution) or getattr(execution, "sl_price", None)

    direction = (getattr(execution, "direction", "LONG") or "LONG").upper()
    tol = _sl_tolerance(symbol, entry)

    if breakeven_was_claimed(execution) and entry > 0:
        if sl is not None and abs(exit_px - float(sl)) <= tol:
            return "BREAKEVEN", "breakeven stop"
        if abs(exit_px - entry) <= tol:
            return "BREAKEVEN", "breakeven stop"

    if entry > 0 and outcome == "WIN":
        if direction == "LONG" and exit_px <= entry + tol:
            outcome = "BREAKEVEN" if breakeven_was_claimed(execution) else "LOSS"
        elif direction == "SHORT" and exit_px >= entry - tol:
            outcome = "BREAKEVEN" if breakeven_was_claimed(execution) else "LOSS"

    if hit_kind == "tp" and tp is not None:
        tol = _close_level_tolerance(symbol, entry, tp)
        if abs(exit_px - float(tp)) > tol:
            logger.critical(
                "[close-sanity] exec=%s labeled TP but exit %s vs tp %s (tol=%s)",
                exec_id, exit_px, tp, tol,
            )
            return outcome if outcome != "WIN" else "LOSS", "market close"
    if hit_kind == "sl" and sl is not None:
        tol = _close_level_tolerance(symbol, entry, float(sl))
        if abs(exit_px - float(sl)) > tol:
            logger.critical(
                "[close-sanity] exec=%s labeled SL but exit %s vs sl %s (tol=%s)",
                exec_id, exit_px, sl, tol,
            )
            return classify_sl_close_outcome(execution, exit_px), "market close"

    return outcome, close_note_label(execution, outcome, exit_px, hit_kind)


def classify_and_close_paper(
    execution,
    outcome: str,
    exit_price: float,
    session,
    *,
    hit_kind: str = "",
) -> None:
    """Shared classifier + paper close — use from every paper close site."""
    outcome, label = validate_close_sanity(
        execution, outcome, float(exit_price), hit_kind,
    )
    from app.services.strategy_executor import _close_paper_execution

    _close_paper_execution(
        execution, outcome, float(exit_price), session, close_label=label,
    )


def close_note_label(
    execution,
    outcome: str,
    exit_price: float,
    hit_kind: str = "",
) -> str:
    """Human-readable close label — never 'SL hit' on breakeven scratch."""
    symbol = getattr(execution, "symbol", "") or ""
    entry = float(getattr(execution, "entry_price", 0) or 0)
    exit_px = float(exit_price)
    if breakeven_was_claimed(execution) and entry > 0:
        if abs(exit_px - entry) <= _sl_tolerance(symbol, entry):
            return "breakeven stop"
    o = (outcome or "").upper()
    if o == "BREAKEVEN":
        return "breakeven stop"
    if o == "WIN" and hit_kind == "tp":
        return "TP hit"
    if o in ("LOSS", "WIN") and hit_kind == "sl":
        if o == "BREAKEVEN":
            return "breakeven stop"
        return "SL hit" if o == "LOSS" else "trailing stop"
    if o == "WIN":
        return "TP hit"
    if o == "LOSS":
        return "SL hit"
    return "Closed"


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
        if (
            orig is not None
            and abs(exit_px - orig) <= tol
            and abs(orig - entry) > tol
        ):
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
        if eff_sl is not None and abs(exit_px - float(eff_sl)) <= tol:
            if abs(float(eff_sl) - entry) <= tol * 3:
                return "BREAKEVEN"
        if abs(exit_px - entry) <= tol:
            return "BREAKEVEN"

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
    from datetime import datetime

    new_sl = round(float(new_sl), 6)
    execution.current_sl = new_sl
    execution.sl_price = new_sl
    if mark_breakeven:
        execution.breakeven_applied = True
    sleff_ms = int(datetime.utcnow().timestamp() * 1000) + 1
    notes = (execution.notes or "").strip()
    notes = re.sub(r"\s*\|?\s*sleff=\d+", "", notes).strip(" |")
    if mark_breakeven and "be_moved" not in notes:
        notes = (f"{notes} | be_moved".strip(" |")) if notes else "be_moved"
    notes = (f"{notes} | sleff={sleff_ms}".strip(" |")) if notes else f"sleff={sleff_ms}"
    execution.notes = notes
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


def reclassify_legacy_autobe_paper_closes(
    session,
    *,
    hours: int = 168,
    limit: int = 500,
) -> int:
    """
    Reclassify paper WIN/LOSS closes polluted by legacy PaperMonitor AUTO-BREAKEVEN
    (flat scratch at entry, often mislabeled TP hit). Returns rows corrected.
    """
    from datetime import datetime, timedelta

    from app.strategy_models import StrategyExecution

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        session.query(StrategyExecution)
        .filter(
            StrategyExecution.is_paper == True,  # noqa: E712
            StrategyExecution.outcome.in_(("WIN", "LOSS")),
            StrategyExecution.closed_at >= cutoff,
            StrategyExecution.entry_price.isnot(None),
            StrategyExecution.exit_price.isnot(None),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(limit)
        .all()
    )
    fixed = 0
    fixed_strat_ids: set = set()
    for ex in rows:
        entry = float(ex.entry_price or 0)
        exit_px = float(ex.exit_price or 0)
        if entry <= 0:
            continue
        notes = (ex.notes or "")
        tol = _sl_tolerance(ex.symbol or "", entry)
        at_entry = abs(exit_px - entry) <= tol
        if not at_entry:
            continue
        legacy_be = (
            "be_moved" in notes
            or "AUTO-BREAKEVEN" in notes
            or (
                "TP hit" in notes
                and breakeven_was_claimed(ex)
            )
        )
        if not legacy_be:
            continue
        pnl = float(ex.pnl_pct or 0)
        if abs(pnl) > 0.15 and not breakeven_was_claimed(ex):
            continue
        ex.outcome = "BREAKEVEN"
        label = close_note_label(ex, "BREAKEVEN", exit_px, "sl")
        pnl_sign = "+" if pnl >= 0 else ""
        ex.notes = (
            f"{label} · {pnl_sign}{pnl}% · exit {exit_px:.6g} "
            f"| legacy-autobe-reclassified"
        )
        fixed += 1
        fixed_strat_ids.add(ex.strategy_id)
        logger.info(
            "[trade-mgmt] legacy AUTO-BE reclassified exec=%s %s → BREAKEVEN "
            "(entry=%s exit=%s pnl=%s)",
            ex.id,
            ex.symbol,
            entry,
            exit_px,
            pnl,
        )
    if fixed:
        try:
            session.commit()
            from app.services.strategy_executor import _update_performance

            for sid in fixed_strat_ids:
                _update_performance(sid, session)
            session.commit()
        except Exception as exc:
            logger.warning("[trade-mgmt] legacy autobe reclassify commit failed: %s", exc)
            try:
                session.rollback()
            except Exception:
                pass
            return 0
    if fixed:
        logger.warning(
            "[trade-mgmt] reclassified %s legacy AUTO-BREAKEVEN paper closes → BREAKEVEN",
            fixed,
        )
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
        and cfg["breakeven_trigger_pips"] > 0
        and not bool(getattr(execution, "breakeven_applied", False))
        and (force_be or profit_pips >= cfg["breakeven_trigger_pips"])
    )
    if be_ready:
        logger.info(
            "[trade-mgmt] BE trigger exec=%s profit=%.1f >= trigger=%.0f",
            getattr(execution, "id", 0),
            profit_pips,
            cfg["breakeven_trigger_pips"],
        )
        new_sl = breakeven_sl_price(
            symbol, entry, direction, cfg["breakeven_offset_pips"],
        )
        if _validate_sl(direction, new_sl, live_price):
            amended, _be_ms = await _apply_sl_amend(
                execution,
                new_sl,
                session,
                is_paper,
                keep_tp=True,
                mark_breakeven=True,
                trigger_pips=profit_pips,
                trigger_cfg=cfg["breakeven_trigger_pips"],
            )
            if amended and not is_paper:
                await _notify_live_breakeven(
                    execution,
                    new_sl,
                    profit_pips,
                    cfg["breakeven_trigger_pips"],
                    session,
                    asset_class=asset_class,
                    confirm_ms=_be_ms,
                )
            elif amended and is_paper:
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
                    amended_trail, _ = await _apply_sl_amend(
                        execution, candidate, session, is_paper, keep_tp=True,
                    )
                    if amended_trail:
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
                    amended_trail, _ = await _apply_sl_amend(
                        execution, candidate, session, is_paper, keep_tp=True,
                    )
                    if amended_trail:
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


async def _notify_live_breakeven(
    execution,
    new_sl: float,
    profit_pips: float,
    trigger_pips: float,
    session,
    *,
    asset_class: str = "forex",
    confirm_ms: float = 0.0,
) -> None:
    """Push + Telegram breakeven alert — only after broker-confirmed amend."""
    try:
        from app.services.strategy_executor import (
            _claim_tg_be_notify,
            _notify_breakeven_alert,
            _telegram_int_id,
        )
        from app.models import User
        from app.strategy_models import UserStrategy

        exec_id = int(getattr(execution, "id", 0) or 0)
        if not _claim_tg_be_notify(session, exec_id):
            return
        user = session.query(User).filter(User.id == execution.user_id).first()
        strat = session.query(UserStrategy).filter(
            UserStrategy.id == execution.strategy_id
        ).first()
        entry = float(execution.entry_price or 0)
        move_pct = abs(profit_pips) * 0.01 if entry else 0.0
        await _notify_breakeven_alert(
            user_id=execution.user_id,
            telegram_id=_telegram_int_id(user) if user else None,
            strategy_name=(strat.name if strat else None) or "Strategy",
            symbol=execution.symbol,
            direction=execution.direction,
            leverage=int(getattr(execution, "leverage", 1) or 1),
            move_pct=move_pct,
            strategy_id=execution.strategy_id,
            execution_id=exec_id,
            kind="live",
        )
        logger.info(
            "[trade-mgmt] breakeven via tick (+%.0fms broker confirm) exec=%s @ %s",
            confirm_ms,
            exec_id,
            new_sl,
        )
    except Exception as exc:
        logger.warning(
            "[trade-mgmt] live breakeven notify failed exec=%s: %s",
            getattr(execution, "id", 0),
            exc,
        )


async def _apply_sl_amend(
    execution,
    new_sl: float,
    session,
    is_paper: bool,
    *,
    keep_tp: bool = True,
    mark_breakeven: bool = False,
    trigger_pips: float = 0.0,
    trigger_cfg: float = 0.0,
) -> Tuple[bool, float]:
    """Returns (success, broker_confirm_ms). DB updated only on broker confirm."""
    import time as _time

    new_sl = round(float(new_sl), 6)
    exec_id = int(getattr(execution, "id", 0) or 0)
    if is_paper:
        ok = persist_paper_stop_level(
            execution, new_sl, session, mark_breakeven=mark_breakeven,
        )
        return ok, 0.0

    pos_id = position_id_from_execution(execution)
    if not pos_id:
        logger.warning(
            "[trade-mgmt] live SL amend skipped exec=%s — no broker position",
            exec_id,
        )
        return False, 0.0

    tp = None
    if keep_tp:
        tp = execution.tp2_price or execution.tp_price
        tp = float(tp) if tp is not None else None
        if tp is None:
            logger.warning(
                "[sl-amend] exec=%s missing tp_price — amend may clear broker TP",
                exec_id,
            )

    orig_sl = active_sl(execution) or getattr(execution, "sl_price", None)
    t0 = _time.monotonic()
    try:
        from app.services.ctrader_order_queue import enqueue_ctrader_sl_amend

        res = await enqueue_ctrader_sl_amend(
            user_id=int(execution.user_id),
            exec_id=exec_id,
            position_id=int(pos_id),
            new_sl=new_sl,
            keep_tp=tp,
        )
    except Exception as exc:
        res = {"ok": False, "result": "failed", "error": str(exc)}
    confirm_ms = (_time.monotonic() - t0) * 1000.0

    result = res.get("result") or ("confirmed" if res.get("ok") else "failed")
    logger.info(
        "[sl-amend] exec=%s requested=%s result=%s broker_reply=%s",
        exec_id,
        new_sl,
        result,
        res.get("broker_reply"),
    )

    if not res.get("ok"):
        stop_rem = orig_sl if orig_sl is not None else "unknown"
        await _telegram_trade(
            execution.user_id,
            f"⚠️ SL amend FAILED — stop remains at {stop_rem}",
            asset_class=getattr(execution, "asset_class", "forex") or "forex",
        )
        return False, confirm_ms

    execution.current_sl = new_sl
    execution.sl_price = new_sl
    if mark_breakeven:
        execution.breakeven_applied = True
        notes = (execution.notes or "").strip()
        if "be_moved" not in notes:
            execution.notes = (f"{notes} | be_moved".strip(" |")) if notes else "be_moved"
    try:
        session.commit()
    except Exception:
        session.rollback()
        return False, confirm_ms
    return True, confirm_ms


_FOREX_LIKE_ASSET_CLASSES = frozenset({
    "forex", "metals", "commodity", "index", "stock",
})


def is_forex_like_asset(asset_class: Optional[str]) -> bool:
    return (asset_class or "").lower() in _FOREX_LIKE_ASSET_CLASSES


def compute_pips_from_prices(
    entry: float,
    exit_price: float,
    direction: str,
    symbol: str,
) -> Optional[float]:
    """Signed pips from original entry vs exit — same formula for paper and live."""
    try:
        from app.services.forex_engine import pip_size as _pip_size

        ps = _pip_size(symbol or "")
        ent = float(entry)
        ex = float(exit_price)
        if not ps or ps <= 0 or ent <= 0 or ex <= 0:
            return None
        if (direction or "LONG").upper() == "LONG":
            return round((ex - ent) / ps, 1)
        return round((ent - ex) / ps, 1)
    except Exception:
        return None


def compute_execution_pips_pnl(execution, exit_price: Optional[float] = None) -> Optional[float]:
    """
    Pips P&L for an execution — always from stored entry fill vs exit.
    Paper and live share this path; broker deal is only for exit price on LIVE.
    """
    ac = (getattr(execution, "asset_class", None) or "crypto").lower()
    if not is_forex_like_asset(ac):
        return None
    entry = float(getattr(execution, "entry_price", 0) or 0)
    if entry <= 0:
        return None
    xp = float(exit_price if exit_price is not None else getattr(execution, "exit_price", 0) or 0)
    if xp <= 0:
        return None
    try:
        from app.services.strategy_executor import _tp1_blend_exit

        pnl_exit = float(_tp1_blend_exit(execution, xp))
    except Exception:
        pnl_exit = xp
    direction = getattr(execution, "direction", "LONG") or "LONG"
    symbol = getattr(execution, "symbol", "") or ""
    pips = compute_pips_from_prices(entry, pnl_exit, direction, symbol)
    return guard_pips_sanity(
        getattr(execution, "id", 0),
        entry,
        pnl_exit,
        symbol,
        direction,
        pips,
    )


def guard_pips_sanity(
    exec_id: int,
    entry: float,
    exit_price: float,
    symbol: str,
    direction: str,
    pips: Optional[float],
) -> Optional[float]:
    """If rounded pips is 0 but price move exceeds half a pip, recompute and log."""
    if pips is None:
        return None
    try:
        from app.services.forex_engine import pip_size as _pip_size

        ps = _pip_size(symbol or "")
        if not ps or ps <= 0:
            return pips
        move_pips = abs(float(exit_price) - float(entry)) / ps
        if abs(pips) < 0.05 and move_pips > 0.5:
            logger.critical(
                "[pnl-sanity] exec=%s pips=0 but entry=%s exit=%s symbol=%s move=%.1fp",
                exec_id,
                entry,
                exit_price,
                symbol,
                move_pips,
            )
            raw = compute_pips_from_prices(entry, exit_price, direction, symbol)
            return raw if raw is not None else pips
    except Exception:
        pass
    return pips


def format_pips_display(
    *,
    pips_pnl: Optional[float],
    entry: float,
    exit_price: float,
    symbol: str,
    direction: str,
    notes: Optional[str] = None,
    tp1_blend_line: Optional[str] = None,
    pnl_pct: Optional[float] = None,
) -> str:
    """Trade-card P/L line — NULL is 'pending', never fake '0 pips'."""
    if tp1_blend_line:
        return str(tp1_blend_line)
    if pips_pnl is None:
        if notes and "pending_reconcile" in (notes or ""):
            return "pending"
        computed = compute_pips_from_prices(entry, exit_price, direction, symbol)
        if computed is not None:
            pips_pnl = computed
        else:
            return "pending"
    sign = "+" if pips_pnl >= 0 else "−"
    if abs(pips_pnl) >= 1:
        return f"{sign}{abs(pips_pnl):.0f} pips"
    return f"{sign}{abs(pips_pnl):.1f} pips"


def backfill_missing_pips_pnl(hours: int = 168, limit: int = 500) -> int:
    """
    Recompute pips_pnl for closes with NULL/0 pips but entry != exit.
    Returns count of rows corrected.
    """
    from datetime import datetime, timedelta

    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    fixed = 0
    fixed_strat_ids: set = set()
    db = SessionLocal()
    try:
        rows = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN")),
                StrategyExecution.closed_at >= cutoff,
                StrategyExecution.entry_price.isnot(None),
                StrategyExecution.exit_price.isnot(None),
            )
            .order_by(StrategyExecution.closed_at.desc())
            .limit(limit)
            .all()
        )
        for ex in rows:
            if not is_forex_like_asset(getattr(ex, "asset_class", None)):
                continue
            cur = ex.pips_pnl
            if cur is not None and abs(float(cur)) >= 0.05:
                continue
            entry = float(ex.entry_price or 0)
            exit_px = float(ex.exit_price or 0)
            if entry <= 0 or exit_px <= 0:
                continue
            try:
                from app.services.forex_engine import pip_size as _pip_size

                ps = _pip_size(ex.symbol or "")
                if ps and abs(exit_px - entry) <= ps * 0.25:
                    continue
            except Exception:
                if abs(exit_px - entry) <= max(abs(entry) * 1e-8, 1e-6):
                    continue
            new_pips = compute_execution_pips_pnl(ex, exit_px)
            if new_pips is None or abs(new_pips) < 0.05:
                continue
            ex.pips_pnl = new_pips
            fixed += 1
            fixed_strat_ids.add(ex.strategy_id)
            logger.info(
                "[pnl-backfill] exec=%s %s %s pips %s→%s (entry=%s exit=%s)",
                ex.id,
                ex.symbol,
                ex.outcome,
                cur,
                new_pips,
                entry,
                exit_px,
            )
        if fixed:
            db.commit()
            try:
                from app.services.strategy_executor import _update_performance

                for sid in fixed_strat_ids:
                    _update_performance(sid, db)
                db.commit()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[pnl-backfill] failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
    if fixed:
        logger.warning("[pnl-backfill] corrected %s closes with missing/zero pips_pnl", fixed)
    return fixed


def mark_pending_reconcile(execution, session, reason: str = "") -> None:
    """Flag OPEN execution awaiting broker deal data — never fabricate close."""
    notes = (execution.notes or "").strip()
    if "pending_reconcile" not in notes:
        tag = "pending_reconcile"
        if reason:
            tag = f"{tag}:{reason[:40]}"
        execution.notes = (f"{notes} | {tag}".strip(" |")) if notes else tag
        try:
            session.commit()
        except Exception:
            session.rollback()


def _broker_pips_from_deal(
    entry: float,
    exit_price: float,
    direction: str,
    symbol: str,
    *,
    exec_id: int = 0,
) -> Optional[float]:
    pips = compute_pips_from_prices(entry, exit_price, direction, symbol)
    return guard_pips_sanity(exec_id, entry, exit_price, symbol, direction, pips)


async def reconcile_broker_pnl_for_recent_closes(hours: int = 48) -> dict:
    """
    Compare recorded live closes vs broker deal P/L; correct DB when they diverge.
    Returns audit counts {corrected, phantom_paper, broker_mismatch_checked}.
    """
    from datetime import datetime, timedelta

    from app.database import BgSessionLocal as SessionLocal
    from app.models import User
    from app.strategy_models import StrategyExecution
    from app.services.ctrader_client import get_position_close_detail_for_user

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    stats = {"corrected": 0, "phantom_paper": 0, "checked": 0, "broker_mismatch": 0}

    db = SessionLocal()
    try:
        live_rows = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.is_paper == False,  # noqa: E712
                StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN")),
                StrategyExecution.asset_class.in_(("forex", "index")),
                StrategyExecution.closed_at >= cutoff,
            )
            .all()
        )
        paper_rows = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.is_paper == True,  # noqa: E712
                StrategyExecution.outcome == "WIN",
                StrategyExecution.asset_class.in_(("forex", "index")),
                StrategyExecution.closed_at >= cutoff,
            )
            .all()
        )
        for ex in paper_rows:
            if not ex.fired_at or not ex.closed_at:
                continue
            age_s = (ex.closed_at - ex.fired_at).total_seconds()
            if age_s >= 60:
                continue
            tp = effective_tp(ex)
            exit_px = float(ex.exit_price or 0)
            if tp and exit_px and abs(exit_px - float(tp)) > _close_level_tolerance(
                ex.symbol or "", float(ex.entry_price or 0), float(tp),
            ):
                stats["phantom_paper"] += 1
                logger.critical(
                    "[close-sanity] phantom paper TP exec=%s age=%.0fs exit=%s tp=%s",
                    ex.id,
                    age_s,
                    exit_px,
                    tp,
                )

        user_cache: Dict[int, object] = {}
        for ex in live_rows:
            pos_id = position_id_from_execution(ex)
            if not pos_id:
                continue
            stats["checked"] += 1
            user = user_cache.get(ex.user_id)
            if user is None:
                user = db.query(User).filter(User.id == ex.user_id).first()
                user_cache[ex.user_id] = user
            if not user:
                continue
            broker = await get_position_close_detail_for_user(
                user,
                int(pos_id),
                entry_hint=float(ex.entry_price or 0),
                direction=ex.direction,
            )
            if not broker or not broker.get("exit_price"):
                continue
            broker_exit = float(broker["exit_price"])
            broker_outcome = broker.get("outcome") or "LOSS"
            our_pips = float(ex.pips_pnl or 0)
            entry_f = float(ex.entry_price or 0)
            our_exit = float(ex.exit_price or 0)
            our_recomputed = compute_execution_pips_pnl(ex, our_exit)
            broker_pips = _broker_pips_from_deal(
                entry_f,
                broker_exit,
                ex.direction or "LONG",
                ex.symbol or "",
                exec_id=int(ex.id),
            )
            if broker_pips is None:
                continue
            # LIVE only — never trust broker price normalization that collapses
            # exit to entry (would zero pips on real wins).
            try:
                from app.services.forex_engine import pip_size as _pip_size

                ps = _pip_size(ex.symbol or "")
                if (
                    ps
                    and abs(broker_exit - entry_f) <= ps * 0.25
                    and our_recomputed is not None
                    and abs(our_recomputed) > 1.0
                    and abs(broker_pips) < 0.05
                ):
                    logger.warning(
                        "[reconcile] exec=%s broker exit≈entry — keeping our exit=%s pips=%s",
                        ex.id,
                        our_exit,
                        our_recomputed,
                    )
                    broker_exit = our_exit
                    broker_pips = our_recomputed
            except Exception:
                pass
            if broker_pips is None:
                continue
            tol = 2.0
            cmp_our = our_recomputed if our_recomputed is not None else our_pips
            if abs(cmp_our - broker_pips) > tol or ex.outcome != broker_outcome:
                stats["broker_mismatch"] += 1
                logger.warning(
                    "[reconcile] corrected exec=%s our=%s pips=%s broker=%s pips=%s "
                    "outcome %s→%s exit %s→%s",
                    ex.id,
                    ex.outcome,
                    our_pips,
                    broker_outcome,
                    broker_pips,
                    ex.outcome,
                    broker_outcome,
                    ex.exit_price,
                    broker_exit,
                )
                ex.exit_price = broker_exit
                ex.outcome = broker_outcome
                ex.pips_pnl = broker_pips
                lev = ex.leverage or 1
                entry = float(ex.entry_price or 0)
                if entry > 0:
                    if ex.direction == "LONG":
                        ex.pnl_pct = round(
                            (broker_exit - entry) / entry * 100 * lev, 2,
                        )
                    else:
                        ex.pnl_pct = round(
                            (entry - broker_exit) / entry * 100 * lev, 2,
                        )
                label = close_note_label(ex, broker_outcome, broker_exit)
                pnl_sign = "+" if (ex.pnl_pct or 0) >= 0 else ""
                ex.notes = (
                    f"{label} · {pnl_sign}{ex.pnl_pct}% · exit {broker_exit:.6g} "
                    f"| reconcile-corrected"
                )
                stats["corrected"] += 1
        bf = backfill_missing_pips_pnl(hours=hours, limit=200)
        stats["backfilled"] = bf
        if stats["corrected"] or bf:
            db.commit()
    except Exception as exc:
        logger.warning("[reconcile] broker P/L audit failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
    if stats["phantom_paper"] or stats["broker_mismatch"]:
        logger.warning(
            "[reconcile] 48h audit: phantom_paper=%s broker_mismatch=%s corrected=%s",
            stats["phantom_paper"],
            stats["broker_mismatch"],
            stats["corrected"],
        )
    return stats
