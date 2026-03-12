"""
AI Strategy Builder — Build Your Own Strategy Portal

Compiles a natural-language strategy description into a precise JSON config.
The compiled config is evaluated in real-time by strategy_ta.py against live
Binance Futures OHLCV data.
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Full condition type schema — passed verbatim to the compiler AI
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_SCHEMA = """
=== COMPLETE CONDITION TYPE REFERENCE ===

All conditions share a "type" field. Every field marked [opt] has a default.

── INDICATOR ──────────────────────────────────────────────────────────────────
{"type":"indicator", "name":"<NAME>", "timeframe":"15m", ...}

Timeframes: 1m | 3m | 5m | 15m | 30m | 1h | 4h | 1d

  RSI
    {"type":"indicator","name":"rsi","timeframe":"15m","operator":"lt","value":30}
    operators: gt | gte | lt | lte | eq
    Typical: oversold <30, overbought >70

  MACD
    {"type":"indicator","name":"macd","condition":"bullish_cross"}
    conditions: bullish_cross | bearish_cross | bullish | bearish
    {"type":"indicator","name":"macd_hist","operator":"gt","value":0}  ← histogram

  EMA (simple cross/direction)
    {"type":"indicator","name":"ema","condition":"golden_cross"}
    conditions: golden_cross | death_cross | bullish | bearish
    {"type":"indicator","name":"ema","condition":"bullish","fast":9,"slow":21,"timeframe":"1h"}

  EMA Ribbon (multiple EMA alignment)
    {"type":"indicator","name":"ema_ribbon","condition":"aligned_bullish","timeframe":"1h","periods":[9,21,55,100,200]}
    conditions: aligned_bullish | aligned_bearish

  Bollinger Bands
    {"type":"indicator","name":"bb","condition":"squeeze"}
    conditions: squeeze | above_upper | below_lower | upper_touch | lower_touch |
                overbought | oversold | mean_reversion
    {"type":"indicator","name":"bb","operator":"gt","value":90}  ← %B value

  VWAP
    {"type":"indicator","name":"vwap","operator":"lt","value":-2.0}  ← % deviation from VWAP
    {"type":"indicator","name":"vwap","condition":"below"}  ← price below VWAP
    {"type":"indicator","name":"vwap","condition":"above"}

  Volume
    {"type":"indicator","name":"volume_ratio","operator":"gt","value":1.5}

  Stochastic RSI
    {"type":"indicator","name":"stoch_rsi","condition":"oversold","timeframe":"15m"}
    conditions: oversold | overbought | bullish_cross | bearish_cross

  SuperTrend
    {"type":"indicator","name":"supertrend","condition":"bullish_flip","timeframe":"15m","period":10,"multiplier":3.0}
    conditions: bullish | bearish | bullish_flip | bearish_flip

  ADX (trend strength)
    {"type":"indicator","name":"adx","operator":"gt","value":25,"timeframe":"1h"}
    {"type":"indicator","name":"adx","condition":"trending"}      ← ADX > 25
    {"type":"indicator","name":"adx","condition":"strong_trend"}  ← ADX > 40
    {"type":"indicator","name":"adx","condition":"weak"}          ← ADX < 20

  ATR Expansion
    {"type":"indicator","name":"atr_expansion","condition":"expanding","timeframe":"15m","multiplier":1.2}
    conditions: expanding | contracting

  Williams %R
    {"type":"indicator","name":"williams_r","condition":"oversold","timeframe":"15m","period":14}
    conditions: oversold (< -80) | overbought (> -20)
    {"type":"indicator","name":"williams_r","operator":"lt","value":-80}

  CCI
    {"type":"indicator","name":"cci","condition":"oversold","timeframe":"15m","period":20}
    conditions: oversold (< -100) | overbought (> 100)
    {"type":"indicator","name":"cci","operator":"lt","value":-100}

  OBV
    {"type":"indicator","name":"obv","condition":"bullish","timeframe":"15m"}
    conditions: bullish | bearish | divergence_bullish | divergence_bearish

  Heikin Ashi
    {"type":"indicator","name":"heikin_ashi","condition":"bullish_flip","timeframe":"5m"}
    conditions: bullish | bearish | bullish_flip | bearish_flip | strong_bull | strong_bear

  Ichimoku
    {"type":"indicator","name":"ichimoku","condition":"above_cloud","timeframe":"1h"}
    conditions: above_cloud | below_cloud | in_cloud | tk_cross_bullish | tk_cross_bearish |
                bullish_cloud | bearish_cloud

  Squeeze Momentum (LazyBear)
    {"type":"indicator","name":"squeeze","condition":"firing","timeframe":"15m"}
    conditions: firing | on | off | bull_mom | bear_mom

── PRICE MOMENTUM ─────────────────────────────────────────────────────────────
{"type":"price_momentum","window_minutes":10,"operator":"gt","value":8,"direction":"up"}
direction: up | down | any
→ "pumped 8% in 10 minutes" | "dropped 5% in 15 minutes" | "moved 3% in 5 minutes"

── VOLUME SPIKE ───────────────────────────────────────────────────────────────
{"type":"volume_spike","multiplier":2.5}
→ "volume spike" | "unusual volume" | "3x normal volume"

── SUPPORT / RESISTANCE ───────────────────────────────────────────────────────
{"type":"support_resistance","condition":"at_support","tolerance_pct":1.0}
{"type":"support_resistance","condition":"breakout_above"}
{"type":"support_resistance","condition":"breakout_below"}
{"type":"support_resistance","condition":"at_resistance","tolerance_pct":0.5}
{"type":"support_resistance","condition":"between"}  ← ranging between S and R

── FAIR VALUE GAP (FVG / Imbalance) ───────────────────────────────────────────
{"type":"fvg","direction":"bullish","condition":"price_in_gap","timeframe":"15m"}
{"type":"fvg","direction":"bearish","condition":"approaching","timeframe":"5m"}
{"type":"fvg","direction":"any","condition":"gap_exists","timeframe":"1h"}
conditions: price_in_gap | approaching | gap_exists | gap_filled
→ "FVG fill" | "fair value gap" | "imbalance" | "price returning to gap"

── CANDLESTICK PATTERNS ───────────────────────────────────────────────────────
{"type":"candlestick","pattern":"bullish_engulfing","timeframe":"15m"}

patterns:
  bullish_engulfing  | bearish_engulfing
  hammer             | inverted_hammer
  shooting_star      | pin_bar
  doji               | dragonfly_doji | gravestone_doji
  morning_star       | evening_star
  three_white_soldiers | three_black_crows
  tweezer_bottom     | tweezer_top
  inside_bar         | outside_bar
  marubozu

→ "pin bar" | "hammer" | "doji" | "engulfing candle" | "morning star" | etc.

── CONSECUTIVE CANDLES ────────────────────────────────────────────────────────
{"type":"consecutive_candles","direction":"red","count":3,"timeframe":"15m"}
direction: green | red
→ "3 red candles in a row" | "5 consecutive green candles"

── MARKET STRUCTURE (SMC) ─────────────────────────────────────────────────────
{"type":"market_structure","condition":"bos_bullish","timeframe":"15m"}
conditions:
  bos_bullish   ← break of structure to the upside
  bos_bearish   ← break of structure to the downside
  choch_bullish ← change of character bullish
  choch_bearish ← change of character bearish

→ "BOS" | "break of structure" | "CHoCH" | "change of character" | "structure break"

── ORDER BLOCKS (SMC) ────────────────────────────────────────────────────────
{"type":"order_block","ob_type":"bullish","timeframe":"15m","tolerance_pct":1.0}
ob_type: bullish | bearish
→ "order block" | "OB" | "institutional level" | "order block mitigation"

── FIBONACCI ──────────────────────────────────────────────────────────────────
{"type":"fibonacci","level":0.618,"condition":"at_retracement","timeframe":"4h","tolerance_pct":1.0}
{"type":"fibonacci","level":1.618,"condition":"at_extension","timeframe":"1h"}
common levels: 0.236 | 0.382 | 0.5 | 0.618 | 0.786 | 1.0 | 1.272 | 1.618
→ "61.8% fib" | "golden ratio" | "50% retrace" | "fib extension 1.618"

── DIVERGENCE ─────────────────────────────────────────────────────────────────
{"type":"divergence","indicator":"rsi","direction":"bullish","timeframe":"15m"}
{"type":"divergence","indicator":"macd","direction":"bearish","timeframe":"1h"}
direction: bullish | bearish
→ "RSI divergence" | "MACD divergence" | "hidden divergence" | "bearish divergence"

── FUNDING RATE ───────────────────────────────────────────────────────────────
{"type":"funding_rate","operator":"lt","value":-0.05}  ← funding < -0.05% (very negative)
{"type":"funding_rate","operator":"gt","value":0.1}    ← funding > 0.1% (very positive)
→ "funding rate negative" | "extreme funding" | "funding < -0.1%" | "funding arbitrage"

── OPEN INTEREST ──────────────────────────────────────────────────────────────
{"type":"open_interest","condition":"rising","window_minutes":60}
{"type":"open_interest","operator":"gt","change_pct":5,"window_minutes":30}
conditions: rising | falling
→ "OI rising" | "open interest spike" | "OI increasing with price"

── SESSION FILTER ─────────────────────────────────────────────────────────────
{"type":"session","sessions":["london","new_york"]}
sessions: asian | tokyo | london | europe | new_york | ny | overlap
→ "London session" | "NY session" | "Asian session" | "London/NY overlap"

── PRICE RELATIVE ─────────────────────────────────────────────────────────────
{"type":"price_relative","reference":"daily_open","condition":"above"}
{"type":"price_relative","reference":"session_high","operator":"gt","value":0}
{"type":"price_relative","reference":"weekly_open","condition":"below"}
references: daily_open | session_high | session_low | weekly_open
→ "price above daily open" | "trading above yesterday's open" | "session high"

── SENTIMENT ──────────────────────────────────────────────────────────────────
{"type":"sentiment","operator":"gt","value":60}
→ "high social sentiment" | "bullish sentiment" | "trending on social"

── LIQUIDATION ────────────────────────────────────────────────────────────────
{"type":"liquidation","direction":"below","tolerance_pct":2.0}
direction: below (long liquidations below price) | above (short liquidations above)
→ "near liquidation cluster" | "liquidity pool" | "liquidation magnet"
"""

STRATEGY_SCHEMA = """
{
  "version": "1.0",
  "name": "Strategy Name",
  "description": "Plain English description",
  "universe": {
    "type": "all",              // "all" | "specific"
    "symbols": [],              // ["SOLUSDT","ETHUSDT"] if type=specific
    "exclude_slow_highcap": true,
    "min_volume_usd": 500000,
    "min_24h_change": null,     // null = no filter, e.g. 5.0 = only coins up 5%+
    "max_24h_change": null
  },
  "direction": "LONG",          // "LONG" | "SHORT" | "BOTH"
  "entry_conditions": {
    "operator": "AND",          // "AND" = all must pass | "OR" = any can pass
    "conditions": [ /* see condition reference above */ ]
  },
  "exit": {
    "take_profit_pct": 3.0,
    "take_profit2_pct": null,   // optional second TP, null to disable
    "stop_loss_pct": 1.5,
    "trailing_stop": false,
    "trailing_stop_pct": null,
    "breakeven_at_pct": null    // move SL to entry when price is up this %
  },
  "risk": {
    "leverage": 10,
    "position_size_pct": 5,
    "max_trades_per_day": 3,
    "max_open_positions": 1,
    "cooldown_minutes": 30,
    "daily_loss_limit_pct": 5
  },
  "filters": {
    "time_filter": null,        // null | {"start_hour":8,"end_hour":20}  (UTC)
    "btc_regime": null          // null | "bullish" | "bearish" | "neutral"
  }
}
"""

COMPILER_SYSTEM_PROMPT = f"""You are an expert crypto perpetual futures trading strategy compiler.
Your ONLY job: translate a trader's natural-language description into a precise JSON config.

OUTPUT: Return ONLY valid JSON. No markdown fences, no explanation, no comments in JSON.

{STRATEGY_SCHEMA}

{CONDITION_SCHEMA}

=== COMPILATION RULES ===

DIRECTION
  • User describes entering on pumps / overbought / shorts → direction = SHORT
  • User describes entering on dips / oversold / longs → direction = LONG
  • User says "both" or RSI-adaptive → direction = BOTH

STYLE PRESETS (override with explicit values if given)
  scalp     → max_trades_per_day 4–8, cooldown 15–30min, tp ≤3%, sl ≤1.5%, leverage 10–20
  swing     → max_trades_per_day 1–2, cooldown 4h+, tp 5–15%, sl 2–5%, leverage 3–8
  momentum  → max_trades_per_day 3–6, cooldown 20–40min, tp 3–6%, sl 1.5–2%, leverage 10–15
  reversal  → max_trades_per_day 2–4, cooldown 45–90min, tp 3–6%, sl 1.5–2.5%, leverage 8–12
  smc       → max_trades_per_day 2–4, cooldown 60–120min, tp 5–10%, sl 2–3%, leverage 5–10
  sniper    → max_trades_per_day 1–2, cooldown 2h, position_size 2–5%
  custom    → use explicit values from user; apply reasonable defaults for anything unspecified

CONDITION SELECTION
  "RSI oversold" → indicator rsi lt 30
  "RSI overbought" → indicator rsi gt 70
  "MACD cross" → indicator macd bullish_cross or bearish_cross
  "EMA cross" / "golden cross" → indicator ema golden_cross
  "death cross" → indicator ema death_cross
  "above EMA 200" → indicator ema bullish + slow=200
  "EMA ribbon" / "all EMAs aligned" → indicator ema_ribbon aligned_bullish/bearish
  "BB squeeze" / "Bollinger squeeze" → indicator bb squeeze
  "above upper BB" → indicator bb above_upper
  "below lower BB" → indicator bb below_lower
  "price above VWAP" → indicator vwap condition=above
  "price below VWAP" → indicator vwap condition=below
  "StochRSI" → indicator stoch_rsi
  "SuperTrend bullish" → indicator supertrend bullish
  "SuperTrend flip" → indicator supertrend bullish_flip / bearish_flip
  "trending market" / "ADX" → indicator adx trending
  "ATR expanding" / "volatility breakout" → indicator atr_expansion expanding
  "Williams R" → indicator williams_r
  "CCI" → indicator cci
  "OBV" → indicator obv
  "Heikin Ashi" → indicator heikin_ashi
  "Ichimoku" / "cloud" → indicator ichimoku
  "squeeze momentum" / "TTM squeeze" → indicator squeeze
  "pump X% in Y minutes" → price_momentum direction=up
  "dump / dropped X%" → price_momentum direction=down
  "volume spike" / "3x volume" → volume_spike
  "at support" → support_resistance at_support
  "resistance breakout" → support_resistance breakout_above
  "support breakdown" → support_resistance breakout_below
  "FVG" / "fair value gap" / "imbalance" → fvg
  "hammer" / "pin bar" / "engulfing" / "doji" / "morning star" → candlestick
  "3 red candles" / "consecutive candles" → consecutive_candles
  "BOS" / "break of structure" → market_structure bos_bullish or bos_bearish
  "CHoCH" / "change of character" → market_structure choch_bullish or choch_bearish
  "order block" / "OB" → order_block
  "fib 61.8%" / "golden ratio" / "fibonacci" → fibonacci
  "RSI divergence" / "MACD divergence" → divergence
  "funding rate" → funding_rate
  "open interest" / "OI" → open_interest
  "London session" / "NY session" / "Asian session" → session
  "price above daily open" / "above session high" → price_relative
  "sentiment" / "social score" → sentiment
  "liquidation cluster" / "liquidity pool" → liquidation

RISK/REWARD
  • Never set stop_loss_pct > take_profit_pct (minimum 1:1 R:R)
  • Never set leverage > 25 unless user explicitly requests higher
  • For scalps with ≤3% TP, keep SL ≤ 2%

OPERATOR GROUPS
  Use AND for confirming setups (most strategies)
  Use OR for breakout/momentum screens (checking multiple coins for any signal)

UNIVERSE
  If user mentions specific coins → type="specific", symbols=[...]
  If user mentions "top gainers" / "movers" → min_24h_change=3 (or as specified)
  If user mentions "mid-caps" / "altcoins" → exclude_slow_highcap=true (default)
  If user mentions "BTC/ETH only" → type="specific", symbols=["BTCUSDT","ETHUSDT"]
  Default volume: 500000 USD/24h

FILTERS
  If user mentions time restriction → time_filter with UTC hours
  If user says "only in bull market" → btc_regime="bullish"
  If user says "only in bear market" → btc_regime="bearish"

ALWAYS INCLUDE
  • Reasonable defaults for any missing fields
  • A clear description field summarising the strategy in plain English
  • At least one entry condition
"""


async def compile_strategy_from_conversation(
    conversation: List[Dict[str, str]],
    user_description: str,
) -> Optional[Dict]:
    """
    Takes user description, returns compiled strategy config dict or None on failure.
    Uses Claude as the compiler — it knows the full condition schema.
    """
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=3000,
            system=COMPILER_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Compile this trading strategy into the JSON config format:\n\n"
                    f"{user_description}"
                )
            }],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"): part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"Strategy compiler JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Strategy compiler error: {e}")
        return None


async def validate_strategy(config: Dict) -> Dict:
    """
    Run the compiled strategy through Claude for logic/risk review.
    Returns {valid, warnings, suggestions, summary, risk_rating}
    Uses AsyncAnthropic so it never blocks the event loop.
    """
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        prompt = f"""Review this crypto perpetual futures strategy config for:
1. Logic errors or conflicting conditions
2. Risk management issues (R:R ratio, leverage vs TP/SL)
3. Practical firing frequency (will it fire too often or never?)
4. Missing confirmations that experienced traders would add

Strategy:
{json.dumps(config, indent=2)}

Reply ONLY with this JSON (no other text):
{{
  "valid": true,
  "warnings": ["string", ...],
  "suggestions": ["string", ...],
  "summary": "2-3 sentences: what this strategy does, when it fires, why it makes sense",
  "risk_rating": "LOW | MEDIUM | HIGH"
}}"""

        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"): part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        return json.loads(raw)

    except Exception as e:
        logger.error(f"Strategy validation error: {e}")
        return {
            "valid": True, "warnings": [], "suggestions": [],
            "summary": config.get("description", "Custom strategy"),
            "risk_rating": "MEDIUM",
        }


async def generate_strategy_summary(config: Dict) -> str:
    """Generate a short human-readable summary for the marketplace listing."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise this trading strategy in 2 clear sentences for traders "
                    f"browsing a marketplace. Be specific about the signal used:\n"
                    f"{json.dumps(config, indent=2)}"
                )
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return config.get("description", "Custom trading strategy")


def format_config_for_display(config: Dict) -> str:
    """Format a compiled config as readable Telegram-ready text."""
    lines = []
    lines.append(f"<b>{config.get('name', 'Untitled')}</b>")
    lines.append(f"<i>{config.get('description', '')}</i>\n")
    direction = config.get("direction", "LONG")
    lines.append(f"{'🟢' if direction=='LONG' else '🔴' if direction=='SHORT' else '⚡'} <b>Direction:</b> {direction}")
    uni = config.get("universe", {})
    if uni.get("type") == "specific":
        lines.append(f"🎯 <b>Coins:</b> {', '.join(uni.get('symbols', []))}")
    else:
        lines.append(f"🎯 <b>Coins:</b> All eligible")
    entry = config.get("entry_conditions", {})
    conds = entry.get("conditions", [])
    lines.append(f"\n<b>Entry ({entry.get('operator','AND')}):</b>")
    for c in conds:
        ct = c.get("type", "")
        if ct == "indicator":
            n = c.get("name","").upper()
            lines.append(f"  • {n} {c.get('timeframe','')} {c.get('condition','')} {c.get('operator','')} {c.get('value','')}")
        elif ct == "price_momentum":
            lines.append(f"  • Price {c.get('direction','moved')} {c.get('value','')}%+ in {c.get('window_minutes','')}min")
        elif ct == "volume_spike":
            lines.append(f"  • Vol spike {c.get('multiplier','')}×")
        elif ct == "support_resistance":
            lines.append(f"  • {c.get('condition','').replace('_',' ')} (±{c.get('tolerance_pct',1)}%)")
        elif ct == "fvg":
            lines.append(f"  • {c.get('direction','').title()} FVG {c.get('condition','').replace('_',' ')}")
        elif ct == "candlestick":
            lines.append(f"  • Pattern: {c.get('pattern','').replace('_',' ').title()} on {c.get('timeframe','')}")
        elif ct == "market_structure":
            lines.append(f"  • {c.get('condition','').replace('_',' ').upper()}")
        elif ct == "order_block":
            lines.append(f"  • {c.get('ob_type','').title()} Order Block")
        elif ct == "fibonacci":
            lines.append(f"  • Fib {float(c.get('level',0.618))*100:.1f}% {c.get('condition','retracement')}")
        elif ct == "divergence":
            lines.append(f"  • {c.get('direction','').title()} {c.get('indicator','RSI').upper()} Divergence")
        elif ct == "funding_rate":
            lines.append(f"  • Funding rate {c.get('operator','')} {c.get('value','')}%")
        elif ct == "session":
            lines.append(f"  • Session: {', '.join(c.get('sessions',[]))}")
        elif ct == "consecutive_candles":
            lines.append(f"  • {c.get('count',3)} consecutive {c.get('direction','')} candles")
        elif ct == "open_interest":
            lines.append(f"  • OI {c.get('condition','change')} {c.get('change_pct','')}%")
        else:
            lines.append(f"  • {ct}: {c.get('condition','')} {c.get('operator','')} {c.get('value','')}")

    ex = config.get("exit", {})
    tp2 = f" / TP2 {ex.get('take_profit2_pct','')}%" if ex.get("take_profit2_pct") else ""
    lines.append(f"\n<b>Exit:</b> TP {ex.get('take_profit_pct',3)}%{tp2}  ·  SL {ex.get('stop_loss_pct',1.5)}%")
    if ex.get("trailing_stop"):
        lines.append(f"  Trailing: {ex.get('trailing_stop_pct',1)}%")
    if ex.get("breakeven_at_pct"):
        lines.append(f"  Breakeven at: +{ex.get('breakeven_at_pct')}%")

    risk = config.get("risk", {})
    lines.append(f"<b>Risk:</b> {risk.get('leverage',10)}× lev  ·  {risk.get('position_size_pct',5)}% size  ·  max {risk.get('max_trades_per_day',3)}/day")

    return "\n".join(lines)
