"""
AI Strategy Generator — autonomous content engine for the marketplace.

Every cycle (default: hourly), this service:
  1. Asks an LLM to invent a fresh wizard-format trading strategy with
     diversity hints (random symbol / style / direction / TF / risk band).
  2. Validates the spec against the wizard schema.
  3. Backtests it via the existing backtest engine over N days.
  4. If it clears the gates (Sharpe / max-DD / trade-count / win-rate),
     creates a UserStrategy + StrategyPerformance + StrategyMarketplace
     row owned by the system "AI Curator" account, tagged is_ai_generated.

All thresholds + cadence are env-tunable. Runs only in production by
default (REPL_DEPLOYMENT=1) or when ENABLE_AI_GENERATOR=1, mirroring
the strategy executor pattern so dev never duplicates against the
shared Neon DB.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── System AI Curator user ──────────────────────────────────────────────────
AI_CURATOR_TELEGRAM_ID = "ai_curator_system"
AI_CURATOR_UID         = "ai-curator"
AI_CURATOR_NAME        = "AI Curator"

# ── Tunable knobs (env-overridable) ─────────────────────────────────────────
def _envf(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except Exception: return default

def _envi(name: str, default: int) -> int:
    try: return int(os.environ.get(name, default))
    except Exception: return default

GEN_INTERVAL_SEC   = _envi("AIGEN_INTERVAL_SEC", 3600)        # 1h
GEN_PER_CYCLE      = _envi("AIGEN_PER_CYCLE", 3)              # 3/hr (was 6) — cost-tuned
BT_DAYS            = _envi("AIGEN_BT_DAYS", 60)
BT_TIMEOUT_SEC     = _envi("AIGEN_BT_TIMEOUT_SEC", 90)
THRESH_SHARPE      = _envf("AIGEN_MIN_SHARPE", 1.5)
THRESH_MAX_DD      = _envf("AIGEN_MAX_DD", 15.0)              # absolute % max drawdown
THRESH_TRADES      = _envi("AIGEN_MIN_TRADES", 30)
THRESH_WIN_RATE    = _envf("AIGEN_MIN_WIN_RATE", 45.0)
LOG_FAILURES       = os.environ.get("AIGEN_LOG_FAILURES", "1") not in ("0", "false", "")
# Cost guard: stop generating once the marketplace already has plenty of AI
# listings — the promotion loop (Phase 2) thins them out, generation refills.
MAX_AI_LISTINGS    = _envi("AIGEN_MAX_LISTINGS", 200)
LLM_MAX_TOKENS     = _envi("AIGEN_LLM_MAX_TOKENS", 500)       # was 700

# ── Phase 2: promotion + unpublish thresholds (post-paper performance) ─────
PROMOTE_AFTER_DAYS  = _envi("AIGEN_PROMOTE_AFTER_DAYS", 14)
PROMOTE_MIN_TRADES  = _envi("AIGEN_PROMOTE_MIN_TRADES", 20)
PROMOTE_WIN_RATE    = _envf("AIGEN_PROMOTE_WIN_RATE", 50.0)
PROMOTE_MIN_PNL     = _envf("AIGEN_PROMOTE_MIN_PNL_PCT", 5.0)
UNPUBLISH_MIN_TRADES = _envi("AIGEN_UNPUBLISH_MIN_TRADES", 10)
UNPUBLISH_WIN_RATE   = _envf("AIGEN_UNPUBLISH_WIN_RATE", 30.0)
UNPUBLISH_PNL        = _envf("AIGEN_UNPUBLISH_PNL_PCT", -8.0)
INACTIVE_AFTER_DAYS  = _envi("AIGEN_INACTIVE_DAYS", 30)
INACTIVE_MAX_TRADES  = _envi("AIGEN_INACTIVE_MAX_TRADES", 5)
PROMOTE_INTERVAL_SEC = _envi("AIGEN_PROMOTE_INTERVAL_SEC", 86400)  # 24h

# Diverse universe — top liquid USDT perpetuals.
COIN_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "ADAUSDT",
    "TRXUSDT", "DOTUSDT", "MATICUSDT", "NEARUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "INJUSDT", "TIAUSDT", "APTUSDT",
]

STYLE_HINTS = ["scalp", "swing", "reversal", "momentum", "breakout", "smc"]
DIR_HINTS   = ["LONG", "SHORT", "BOTH"]
PRIMARY_HINTS = [
    "rsi", "macd", "ema_cross", "bollinger", "supertrend",
    "price_momentum", "volume_spike",
]

# Module-level cycle stats for /api/admin/ai-generator/stats
_STATS = {
    "started_at": None,
    "cycles_run": 0,
    "cycles_skipped_cap": 0,
    "specs_generated": 0,
    "specs_published": 0,
    "specs_failed_validate": 0,
    "specs_failed_threshold": 0,
    "specs_failed_backtest": 0,
    "last_cycle_at": None,
    "last_cycle_published": 0,
    "last_cycle_generated": 0,
    "promotion_runs": 0,
    "promoted_total": 0,
    "unpublished_total": 0,
    "last_promotion_at": None,
    "recent": [],   # last 20 outcomes
}


# ── Bootstrap: ensure the AI Curator system user exists ────────────────────
def ensure_ai_curator_user(db) -> Optional[int]:
    """Create the AI Curator user if it doesn't exist; return its user_id."""
    from app.models import User
    u = db.query(User).filter(User.uid == AI_CURATOR_UID).first()
    if u:
        return u.id
    u = db.query(User).filter(User.telegram_id == AI_CURATOR_TELEGRAM_ID).first()
    if u:
        if not u.uid:
            u.uid = AI_CURATOR_UID
            db.commit()
        return u.id
    try:
        u = User(
            telegram_id   = AI_CURATOR_TELEGRAM_ID,
            uid           = AI_CURATOR_UID,
            username      = "ai_curator",
            first_name    = AI_CURATOR_NAME,
            grandfathered = True,        # never killed by sub checks
            approved      = True,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        logger.info(f"[AIGen] Created AI Curator system user id={u.id}")
        return u.id
    except Exception as e:
        db.rollback()
        logger.error(f"[AIGen] Failed to create AI Curator: {e}")
        return None


# ── LLM-based strategy spec generator ──────────────────────────────────────
SYSTEM_PROMPT = """You are an expert algorithmic crypto-perpetuals strategist.
Your job is to invent ONE concrete trading strategy spec in strict JSON.

Output only valid JSON. No markdown, no commentary. Schema:

{
  "name": "<2-4 word punchy name, no quotes inside>",
  "description": "<one-sentence plain-english explanation>",
  "category": "scalp" | "swing" | "reversal" | "momentum" | "breakout" | "smc",
  "direction": "LONG" | "SHORT" | "BOTH",
  "entry": {
    "primary": <one of the condition objects below>,
    "confirm": <optional second condition object, or null>
  },
  "exit": {
    "take_profit_pct": <float 0.5..15>,
    "take_profit2_pct": <float or null, > take_profit_pct if set>,
    "stop_loss_pct":  <float 0.3..8>,
    "trailing_stop":  <bool>
  },
  "risk": {
    "leverage": <int 3..25>,
    "position_size_pct": <float 1..15>,
    "max_trades_per_day": <int 1..20>,
    "max_open_positions": <int 1..5>,
    "cooldown_minutes":   <int 5..240>,
    "daily_loss_limit_pct": <float 3..20>
  }
}

Allowed condition object shapes (USE EXACTLY ONE PER CONDITION):
- {"type":"rsi","period":14,"operator":"lt"|"gt"|"lte"|"gte","value":<float 5..95>}
- {"type":"macd","condition":"bullish"|"bearish"|"crosses_above"|"crosses_below","operator":"gt"|"lt","value":0}
- {"type":"ema","period":<int 5..50>,"period2":<int 20..200>,"condition":"above"|"below"|"crosses_above"|"crosses_below"}
- {"type":"bb","period":20,"std_mult":2.0,"condition":"price_above_upper"|"price_below_lower"|"squeeze"|"expand"}
- {"type":"supertrend","period":10,"multiplier":3.0,"condition":"bullish"|"bearish"}
- {"type":"price_momentum","window_minutes":<int 5..60>,"operator":"gt","value":<float 0.5..15>,"direction":"up"|"down"|"any"}
- {"type":"volume_spike","multiplier":<float 1.2..4.0>}

Hard rules:
- LONG strategies must use bullish/oversold conditions; SHORT must use bearish/overbought; BOTH may use either.
- Leverage * position_size_pct / 100 should not exceed 1.5 (risk sanity).
- take_profit_pct / stop_loss_pct ratio should be at least 1.0.
- Output JSON ONLY.
"""

def _user_prompt(symbol: str, style: str, direction: str, primary_hint: str, seed: int) -> str:
    return f"""Generate ONE strategy spec.

Hints (use them, but you may adjust if it makes the strategy stronger):
- target coin: {symbol}
- style: {style}
- direction: {direction}
- preferred primary signal family: {primary_hint}
- creativity seed: {seed}

The spec will be backtested over {BT_DAYS} days on the 5-minute chart against {symbol}.
Aim for Sharpe > {THRESH_SHARPE}, max drawdown < {THRESH_MAX_DD}%, win rate > {THRESH_WIN_RATE}%.
Return ONLY the JSON object."""


async def _call_anthropic_json(system: str, prompt: str, max_tokens: int = None) -> Optional[Dict]:
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Defensive: strip code fences if model included them.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[AIGen] Anthropic call failed: {e}")
        return None


async def _call_gemini_json(system: str, prompt: str) -> Optional[Dict]:
    try:
        from app.services.ai_chat_assistant import call_gemini_chat
        text = await call_gemini_chat(prompt, system_prompt=system + "\n\nReturn ONLY valid JSON.")
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[AIGen] Gemini call failed: {e}")
        return None


async def generate_one_spec() -> Optional[Dict]:
    """Call the LLM to invent one strategy spec. Returns parsed dict or None."""
    symbol      = random.choice(COIN_UNIVERSE)
    style       = random.choice(STYLE_HINTS)
    direction   = random.choice(DIR_HINTS)
    primary     = random.choice(PRIMARY_HINTS)
    seed        = random.randint(1, 9999)
    prompt      = _user_prompt(symbol, style, direction, primary, seed)

    spec = await _call_anthropic_json(SYSTEM_PROMPT, prompt)
    if spec is None:
        spec = await _call_gemini_json(SYSTEM_PROMPT, prompt)
    if not spec:
        return None
    spec["__symbol"] = symbol  # stash for the wizard config builder
    return spec


# ── Spec → wizard config converter + validator ─────────────────────────────
ALLOWED_COND_TYPES = {
    "rsi", "macd", "ema", "bb", "supertrend", "price_momentum", "volume_spike",
}

def _validate_cond(cond: Dict) -> bool:
    if not isinstance(cond, dict): return False
    t = cond.get("type")
    if t not in ALLOWED_COND_TYPES: return False
    return True

def spec_to_wizard_config(spec: Dict) -> Optional[Dict]:
    """Convert an LLM spec into the wizard JSON shape the executor + backtest expect."""
    try:
        symbol = spec.get("__symbol")
        if not symbol:
            return None
        primary = (spec.get("entry") or {}).get("primary")
        if not _validate_cond(primary):
            return None
        confirm = (spec.get("entry") or {}).get("confirm")
        if confirm is not None and not _validate_cond(confirm):
            confirm = None  # silently drop bad confirmation rather than reject
        conds = [primary]
        if confirm:
            conds.append(confirm)

        ex = spec.get("exit") or {}
        rk = spec.get("risk") or {}

        # Hard sanity bounds — clamp anything wild.
        leverage  = max(2, min(int(rk.get("leverage", 10)), 30))
        pos_pct   = max(1.0, min(float(rk.get("position_size_pct", 5.0)), 15.0))
        tp        = max(0.5, min(float(ex.get("take_profit_pct", 3.0)), 20.0))
        tp2_raw   = ex.get("take_profit2_pct")
        tp2       = float(tp2_raw) if tp2_raw is not None else None
        if tp2 is not None and tp2 <= tp: tp2 = None
        sl        = max(0.3, min(float(ex.get("stop_loss_pct", 1.5)), 10.0))
        max_td    = max(1, min(int(rk.get("max_trades_per_day", 5)), 25))
        max_open  = max(1, min(int(rk.get("max_open_positions", 2)), 6))
        cooldown  = max(2, min(int(rk.get("cooldown_minutes", 30)), 360))
        dll       = max(2.0, min(float(rk.get("daily_loss_limit_pct", 8.0)), 30.0))

        direction = spec.get("direction", "LONG").upper()
        if direction not in ("LONG", "SHORT", "BOTH"):
            direction = "LONG"

        category = spec.get("category", "general").lower()
        if category not in ("scalp", "swing", "reversal", "momentum", "breakout", "smc"):
            category = "general"

        name = (spec.get("name") or f"AI {category.title()} {symbol[:-4]}").strip()[:80]
        desc = (spec.get("description") or "")[:280]

        cfg = {
            "version": "1.0",
            "name": name,
            "description": desc,
            "universe": {"type": "specific", "symbols": [symbol]},
            "direction": direction,
            "entry_conditions": {"operator": "AND", "conditions": conds},
            "exit": {
                "take_profit_pct":  tp,
                "take_profit2_pct": tp2,
                "stop_loss_pct":    sl,
                "trailing_stop":    bool(ex.get("trailing_stop", False)),
            },
            "risk": {
                "leverage":             leverage,
                "position_size_pct":    pos_pct,
                "max_trades_per_day":   max_td,
                "max_open_positions":   max_open,
                "cooldown_minutes":     cooldown,
                "daily_loss_limit_pct": dll,
                "no_duplicate_symbol":  True,
            },
            "filters": {"time_filter": None, "btc_regime": None},
            "_build_mode": "paper",
            "_category": category,
            "_ai_generated": True,
        }
        return cfg
    except Exception as e:
        logger.warning(f"[AIGen] spec_to_wizard_config failed: {e}")
        return None


def _config_fingerprint(cfg: Dict) -> str:
    """Stable hash so we can dedupe near-identical strategies."""
    primary = ((cfg.get("entry_conditions") or {}).get("conditions") or [{}])[0]
    key = {
        "symbol": (cfg.get("universe") or {}).get("symbols", []),
        "direction": cfg.get("direction"),
        "primary": primary,
        "tp":  round(float(cfg.get("exit", {}).get("take_profit_pct", 0)), 2),
        "sl":  round(float(cfg.get("exit", {}).get("stop_loss_pct",   0)), 2),
        "lev": cfg.get("risk", {}).get("leverage"),
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


# ── Backtest gate ──────────────────────────────────────────────────────────
async def backtest_and_score(cfg: Dict) -> Tuple[bool, Dict, str]:
    """Run the backtest. Returns (passed, result_dict, reason)."""
    from app.services.backtest_engine import run_backtest
    try:
        result = await asyncio.wait_for(run_backtest(cfg, BT_DAYS), timeout=BT_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False, {}, "timeout"
    except Exception as e:
        return False, {}, f"error:{e}"

    if not isinstance(result, dict) or "error" in result:
        return False, result or {}, f"engine_error:{(result or {}).get('error','?')}"

    stats = result.get("stats") or {}
    sharpe   = float(stats.get("sharpe", 0) or 0)
    max_dd   = abs(float(stats.get("max_drawdown", 0) or 0))
    trades   = int(stats.get("closed_trades", 0) or 0)
    win_rate = float(stats.get("win_rate", 0) or 0)
    pnl      = float(stats.get("total_pnl", 0) or 0)

    # Hard gates.
    if trades < THRESH_TRADES:
        return False, result, f"low_trades({trades}<{THRESH_TRADES})"
    if sharpe < THRESH_SHARPE:
        return False, result, f"low_sharpe({sharpe:.2f}<{THRESH_SHARPE})"
    if max_dd > THRESH_MAX_DD:
        return False, result, f"high_dd({max_dd:.2f}>{THRESH_MAX_DD})"
    if win_rate < THRESH_WIN_RATE:
        return False, result, f"low_wr({win_rate:.1f}<{THRESH_WIN_RATE})"
    if pnl <= 0:
        return False, result, f"negative_pnl({pnl:.2f})"

    return True, result, "ok"


# ── Publish to marketplace ─────────────────────────────────────────────────
def publish_strategy(db, curator_user_id: int, cfg: Dict, bt: Dict) -> Optional[int]:
    """Insert UserStrategy + StrategyPerformance + StrategyMarketplace rows.
    Returns new listing id, or None on dedupe / failure."""
    from app.strategy_models import (
        UserStrategy, StrategyPerformance, StrategyMarketplace,
    )
    try:
        fp = _config_fingerprint(cfg)
        # Dedupe — skip if curator already has a strategy with this fingerprint
        existing = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == curator_user_id)
            .filter(UserStrategy.config["__fingerprint"].astext == fp)
            .first()
        ) if False else None  # JSON-key query is dialect-specific; use Python check
        if existing is None:
            existing = (
                db.query(UserStrategy)
                .filter(UserStrategy.user_id == curator_user_id, UserStrategy.name == cfg.get("name"))
                .first()
            )
        if existing:
            return None

        cfg["__fingerprint"] = fp

        strategy = UserStrategy(
            user_id     = curator_user_id,
            name        = cfg.get("name", "AI Strategy")[:120],
            description = cfg.get("description", ""),
            config      = cfg,
            status      = "paper",        # AI strategies start in paper mode
            is_public   = True,
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)

        stats = bt.get("stats") or {}
        perf = StrategyPerformance(
            strategy_id   = strategy.id,
            total_trades  = int(stats.get("closed_trades", 0) or 0),
            wins          = int(stats.get("wins", 0) or 0),
            losses        = int(stats.get("losses", 0) or 0),
            win_rate      = float(stats.get("win_rate", 0) or 0),
            total_pnl_pct = float(stats.get("total_pnl", 0) or 0),
            avg_win_pct   = float(stats.get("avg_win", 0) or 0),
            avg_loss_pct  = float(stats.get("avg_loss", 0) or 0),
        )
        db.add(perf)

        # Tags include category + ai
        cat  = cfg.get("_category", "general")
        tags = ["ai", cat]
        if cfg.get("direction") in ("LONG", "SHORT"):
            tags.append(cfg["direction"].lower())

        listing = StrategyMarketplace(
            strategy_id        = strategy.id,
            author_id          = curator_user_id,
            title              = strategy.name,
            summary            = cfg.get("description", "") or f"AI-generated {cat} strategy backtested over {BT_DAYS} days.",
            tags               = tags,
            category           = cat if cat != "general" else "momentum",
            pricing_model      = "free",
            price_usdt         = 0.0,
            is_ai_generated    = True,
            backtest_sharpe    = float(stats.get("sharpe", 0) or 0),
            backtest_pnl_pct   = float(stats.get("total_pnl", 0) or 0),
            backtest_trades    = int(stats.get("closed_trades", 0) or 0),
            backtest_win_rate  = float(stats.get("win_rate", 0) or 0),
            backtest_max_dd    = abs(float(stats.get("max_drawdown", 0) or 0)),
            backtest_days      = BT_DAYS,
        )
        db.add(listing)
        db.commit()
        db.refresh(listing)
        return listing.id
    except Exception as e:
        db.rollback()
        logger.error(f"[AIGen] publish failed: {e}", exc_info=True)
        return None


# ── Cycle + loop ───────────────────────────────────────────────────────────
async def run_one_cycle(per_cycle: int = None) -> Dict:
    """Run a single generation cycle. Returns summary dict."""
    n = per_cycle if per_cycle is not None else GEN_PER_CYCLE
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyMarketplace
    summary = {"generated": 0, "published": 0, "failed": 0, "started_at": datetime.utcnow().isoformat()}

    db = SessionLocal()
    curator_id = ensure_ai_curator_user(db)
    if not curator_id:
        db.close()
        logger.error("[AIGen] no curator user — aborting cycle")
        summary["error"] = "no_curator"
        return summary

    # Cost guard: don't keep burning LLM credits if the marketplace is full.
    # The promotion loop will thin out under-performers, then generation resumes.
    try:
        active_count = (
            db.query(StrategyMarketplace)
            .filter(StrategyMarketplace.is_ai_generated == True)
            .filter(StrategyMarketplace.author_id == curator_id)
            .count()
        )
    except Exception:
        active_count = 0
    db.close()

    if active_count >= MAX_AI_LISTINGS:
        _STATS["cycles_skipped_cap"] += 1
        logger.info(
            f"[AIGen] skipping cycle — {active_count} AI listings already published "
            f"(cap={MAX_AI_LISTINGS}). Promotion loop will thin them out."
        )
        summary["skipped"] = "cap_reached"
        summary["active_listings"] = active_count
        return summary

    for i in range(n):
        spec = await generate_one_spec()
        if not spec:
            _STATS["specs_failed_validate"] += 1
            summary["failed"] += 1
            continue
        cfg = spec_to_wizard_config(spec)
        if not cfg:
            _STATS["specs_failed_validate"] += 1
            summary["failed"] += 1
            _record_recent("invalid_spec", spec.get("name") or "?", None, None)
            continue
        _STATS["specs_generated"] += 1
        summary["generated"] += 1

        passed, bt, reason = await backtest_and_score(cfg)
        if not passed:
            if reason.startswith("timeout") or reason.startswith("error") or reason.startswith("engine_error"):
                _STATS["specs_failed_backtest"] += 1
            else:
                _STATS["specs_failed_threshold"] += 1
            if LOG_FAILURES:
                logger.info(f"[AIGen] reject {cfg['name']}: {reason}")
            _record_recent(reason, cfg["name"], cfg, bt.get("stats") if isinstance(bt, dict) else None)
            continue

        db = SessionLocal()
        try:
            listing_id = publish_strategy(db, curator_id, cfg, bt)
        finally:
            db.close()

        if listing_id:
            _STATS["specs_published"] += 1
            summary["published"] += 1
            stats = bt.get("stats") or {}
            logger.info(
                f"[AIGen] PUBLISHED {cfg['name']} ({cfg['universe']['symbols'][0]}) "
                f"sharpe={stats.get('sharpe',0):.2f} pnl={stats.get('total_pnl',0):.1f}% "
                f"wr={stats.get('win_rate',0):.0f}% trades={stats.get('closed_trades',0)} listing={listing_id}"
            )
            _record_recent("published", cfg["name"], cfg, stats)
        else:
            summary["failed"] += 1
            _record_recent("dedupe_or_db_fail", cfg["name"], cfg, None)

        # Pace LLM + backtest pressure across the cycle.
        await asyncio.sleep(2)

    _STATS["cycles_run"] += 1
    _STATS["last_cycle_at"] = datetime.utcnow().isoformat()
    _STATS["last_cycle_published"] = summary["published"]
    _STATS["last_cycle_generated"] = summary["generated"]
    summary["finished_at"] = datetime.utcnow().isoformat()
    return summary


def _record_recent(outcome: str, name: str, cfg: Optional[Dict], stats: Optional[Dict]):
    entry = {
        "ts":      datetime.utcnow().isoformat(),
        "outcome": outcome,
        "name":    name,
    }
    if cfg:
        entry["symbol"] = (cfg.get("universe") or {}).get("symbols", [None])[0]
        entry["direction"] = cfg.get("direction")
    if stats:
        entry["sharpe"]   = round(float(stats.get("sharpe", 0) or 0), 2)
        entry["pnl"]      = round(float(stats.get("total_pnl", 0) or 0), 2)
        entry["win_rate"] = round(float(stats.get("win_rate", 0) or 0), 1)
        entry["trades"]   = int(stats.get("closed_trades", 0) or 0)
    _STATS["recent"].insert(0, entry)
    if len(_STATS["recent"]) > 30:
        _STATS["recent"] = _STATS["recent"][:30]


def get_stats() -> Dict:
    """Snapshot of generator stats for the admin endpoint."""
    s = dict(_STATS)
    s["thresholds"] = {
        "sharpe":   THRESH_SHARPE,
        "max_dd":   THRESH_MAX_DD,
        "trades":   THRESH_TRADES,
        "win_rate": THRESH_WIN_RATE,
    }
    s["config"] = {
        "interval_sec":  GEN_INTERVAL_SEC,
        "per_cycle":     GEN_PER_CYCLE,
        "backtest_days": BT_DAYS,
    }
    return s


# ── Phase 2: promotion / unpublish cycle ───────────────────────────────────
def run_promotion_cycle() -> Dict:
    """
    Walk all AI Curator strategies that have been live in paper for ≥
    PROMOTE_AFTER_DAYS days and act on their measured performance:

      - Winners (≥ PROMOTE_MIN_TRADES, win_rate ≥ PROMOTE_WIN_RATE,
        total_pnl_pct ≥ PROMOTE_MIN_PNL) → mark listing.is_verified
        + is_featured + record verified_pnl/win_rate (gives them a
        green-check + featured row treatment in the marketplace).
      - Losers (≥ UNPUBLISH_MIN_TRADES, AND
        (win_rate < UNPUBLISH_WIN_RATE OR total_pnl_pct < UNPUBLISH_PNL))
        → delete the listing + flip strategy.status to 'archived'.
      - Stagnant (older than INACTIVE_AFTER_DAYS, fewer than
        INACTIVE_MAX_TRADES trades) → unpublish, archive.

    Idempotent — re-running won't double-promote or double-archive.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import (
        UserStrategy, StrategyPerformance, StrategyMarketplace,
    )
    from datetime import timedelta

    summary = {"checked": 0, "promoted": 0, "unpublished": 0, "started_at": datetime.utcnow().isoformat()}
    db = SessionLocal()
    try:
        curator_id = ensure_ai_curator_user(db)
        if not curator_id:
            return summary
        cutoff_eligible = datetime.utcnow() - timedelta(days=PROMOTE_AFTER_DAYS)
        cutoff_inactive = datetime.utcnow() - timedelta(days=INACTIVE_AFTER_DAYS)

        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == curator_id)
            .filter(UserStrategy.created_at <= cutoff_eligible)
            .filter(UserStrategy.status != "archived")
            .all()
        )
        for s in strategies:
            summary["checked"] += 1
            perf = (
                db.query(StrategyPerformance)
                .filter(StrategyPerformance.strategy_id == s.id)
                .first()
            )
            listing = (
                db.query(StrategyMarketplace)
                .filter(StrategyMarketplace.strategy_id == s.id)
                .first()
            )
            if not listing:
                continue

            trades = int(getattr(perf, "total_trades", 0) or 0) if perf else 0
            wr     = float(getattr(perf, "win_rate", 0) or 0) if perf else 0.0
            pnl    = float(getattr(perf, "total_pnl_pct", 0) or 0) if perf else 0.0

            promote = (
                trades >= PROMOTE_MIN_TRADES
                and wr   >= PROMOTE_WIN_RATE
                and pnl  >= PROMOTE_MIN_PNL
                and not listing.is_verified
            )
            unpublish_loser = (
                trades >= UNPUBLISH_MIN_TRADES
                and (wr < UNPUBLISH_WIN_RATE or pnl < UNPUBLISH_PNL)
            )
            unpublish_inactive = (
                s.created_at <= cutoff_inactive
                and trades < INACTIVE_MAX_TRADES
            )

            try:
                if promote:
                    listing.is_verified      = True
                    listing.is_featured      = True
                    listing.verified_win_rate = wr
                    listing.verified_pnl      = pnl
                    db.commit()
                    summary["promoted"] += 1
                    _STATS["promoted_total"] += 1
                    logger.info(
                        f"[AIGen] PROMOTED {s.name} → verified "
                        f"(trades={trades} wr={wr:.0f}% pnl={pnl:+.1f}%)"
                    )
                elif unpublish_loser or unpublish_inactive:
                    reason = "loser" if unpublish_loser else "inactive"
                    # Delete child rows first to avoid ForeignKeyViolation if
                    # any user has cloned or rated this listing. We use raw
                    # SQL DELETE so we don't need to import the ext models
                    # (some deploys run before that table exists).
                    try:
                        from sqlalchemy import text as _text
                        db.execute(_text("DELETE FROM strategy_purchases WHERE listing_id=:lid"), {"lid": listing.id})
                    except Exception:
                        pass
                    try:
                        from sqlalchemy import text as _text
                        db.execute(_text("DELETE FROM strategy_ratings WHERE listing_id=:lid"), {"lid": listing.id})
                    except Exception:
                        pass
                    try:
                        from sqlalchemy import text as _text
                        db.execute(_text("DELETE FROM strategy_offers WHERE listing_id=:lid"), {"lid": listing.id})
                    except Exception:
                        pass
                    db.delete(listing)
                    s.status = "archived"
                    db.commit()
                    summary["unpublished"] += 1
                    _STATS["unpublished_total"] += 1
                    logger.info(
                        f"[AIGen] UNPUBLISHED {s.name} ({reason}) "
                        f"trades={trades} wr={wr:.0f}% pnl={pnl:+.1f}%"
                    )
            except Exception as e:
                db.rollback()
                logger.warning(f"[AIGen] promotion action failed for {s.id}: {e}")

        _STATS["promotion_runs"] += 1
        _STATS["last_promotion_at"] = datetime.utcnow().isoformat()
    finally:
        db.close()
    summary["finished_at"] = datetime.utcnow().isoformat()
    return summary


_LAST_PROMOTION_TS = 0.0  # monotonic wall-clock seconds since last promote pass


async def run_loop():
    """Forever-loop. Logs and sleeps between cycles. Catches all errors."""
    global _LAST_PROMOTION_TS
    import time as _time

    _STATS["started_at"] = datetime.utcnow().isoformat()
    logger.info(
        f"[AIGen] loop starting — every {GEN_INTERVAL_SEC}s, {GEN_PER_CYCLE} specs/cycle, "
        f"{BT_DAYS}d backtest, gates(sharpe>{THRESH_SHARPE} dd<{THRESH_MAX_DD}% "
        f"trades>={THRESH_TRADES} wr>={THRESH_WIN_RATE}%) cap={MAX_AI_LISTINGS} "
        f"promote≥{PROMOTE_AFTER_DAYS}d/{PROMOTE_MIN_TRADES}tr/{PROMOTE_WIN_RATE}%wr/{PROMOTE_MIN_PNL}%pnl"
    )
    # Brief delay so startup isn't slowed by the first generation cycle.
    await asyncio.sleep(30)
    while True:
        try:
            await run_one_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[AIGen] cycle crashed: {e}", exc_info=True)

        # Promotion / unpublish pass — runs roughly once per PROMOTE_INTERVAL_SEC
        # (default 24h). Cheap pure-DB operation, no LLM cost.
        try:
            now = _time.time()
            if now - _LAST_PROMOTION_TS >= PROMOTE_INTERVAL_SEC:
                logger.info("[AIGen] running promotion / unpublish cycle…")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_promotion_cycle)
                _LAST_PROMOTION_TS = now
        except Exception as e:
            logger.error(f"[AIGen] promotion cycle crashed: {e}", exc_info=True)

        await asyncio.sleep(GEN_INTERVAL_SEC)
