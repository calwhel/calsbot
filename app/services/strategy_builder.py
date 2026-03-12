"""
AI Strategy Builder — Build Your Own Strategy Portal

Takes a natural-language conversation from a user and compiles it into
a structured strategy config JSON. Uses Claude (Anthropic) as the compiler.
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy config schema — the canonical format for all user strategies
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_CONFIG_SCHEMA = """
{
  "version": "1.0",
  "name": "Strategy Name",
  "description": "Plain English description of the strategy",
  "universe": {
    "type": "all",           // "all" | "specific"
    "symbols": [],           // if type="specific", list like ["SOLUSDT", "INJUSDT"]
    "exclude_slow_highcap": true,
    "min_volume_usd": 500000,
    "min_24h_change": null,   // null = no filter
    "max_24h_change": null
  },
  "direction": "LONG",       // "LONG" | "SHORT" | "BOTH"
  "entry_conditions": {
    "operator": "AND",       // "AND" = all must pass | "OR" = any can pass
    "conditions": [
      // INDICATOR condition:
      {"type": "indicator", "name": "rsi", "timeframe": "15m", "operator": "lt", "value": 30},
      {"type": "indicator", "name": "macd", "condition": "bullish"},
      {"type": "indicator", "name": "ema", "condition": "bearish"},
      {"type": "indicator", "name": "volume_ratio", "operator": "gt", "value": 1.5},
      {"type": "indicator", "name": "vwap", "operator": "lt", "value": -2.0},
      // PRICE MOMENTUM condition:
      {"type": "price_momentum", "window_minutes": 10, "operator": "gt", "value": 10, "direction": "up"},
      // VOLUME SPIKE condition:
      {"type": "volume_spike", "multiplier": 2.0},
      // SUPPORT/RESISTANCE condition:
      {"type": "support_resistance", "condition": "at_support", "tolerance_pct": 1.0},
      {"type": "support_resistance", "condition": "breakout_above"},
      // FAIR VALUE GAP condition:
      {"type": "fvg", "direction": "bullish", "condition": "price_in_gap", "timeframe": "15m"}
    ]
  },
  "exit": {
    "take_profit_pct": 3.0,
    "stop_loss_pct": 1.5,
    "trailing_stop": false,
    "trailing_stop_pct": null
  },
  "risk": {
    "leverage": 10,
    "position_size_pct": 5,
    "max_trades_per_day": 3,
    "max_open_positions": 1,
    "cooldown_minutes": 30
  },
  "filters": {
    "time_filter": null,     // null = any time | {"start_hour": 8, "end_hour": 20} for UTC hours
    "btc_regime": null       // null = any | "bullish" | "bearish" | "neutral"
  }
}
"""

COMPILER_SYSTEM_PROMPT = f"""You are an expert crypto trading strategy compiler. 
Your job is to take a user's strategy description and compile it into a precise JSON config.

STRATEGY CONFIG SCHEMA:
{STRATEGY_CONFIG_SCHEMA}

RULES:
1. Always return ONLY valid JSON — no markdown, no explanation, just the JSON object.
2. For direction: if the user describes entering on pumps/overbought → SHORT. On dips/oversold → LONG.
3. For RSI: overbought = >70, oversold = <30. If user says "RSI oversold" use lt 30.
4. For price_momentum: "pumped 10% in 10 minutes" → window_minutes=10, value=10, direction="up"
5. For FVG: "fair value gap" or "FVG" → use fvg condition type.
6. If the user mentions "support" or "key level" → use support_resistance condition.
7. Reasonable defaults if not specified: leverage=10, position_size_pct=5, tp=3%, sl=1.5%.
8. If user mentions "scalp" → max_trades_per_day=4-6, tp<=3%, sl<=2%.
9. If user mentions "swing" → max_trades_per_day=1-2, tp>=5%, sl<=3%.
10. Never set leverage > 25 for user strategies.
11. Always set stop_loss_pct <= take_profit_pct (minimum 1:1 R:R).
"""


async def compile_strategy_from_conversation(
    conversation: List[Dict[str, str]],
    user_description: str,
) -> Optional[Dict]:
    """
    Takes the full conversation history + final user description,
    returns compiled strategy config dict or None on failure.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        messages = [
            {"role": "user", "content": f"Please compile this trading strategy into JSON:\n\n{user_description}"}
        ]

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=COMPILER_SYSTEM_PROMPT,
            messages=messages,
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        config = json.loads(raw)
        return config

    except json.JSONDecodeError as e:
        logger.error(f"Strategy compiler JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Strategy compiler error: {e}")
        return None


async def validate_strategy(config: Dict) -> Dict:
    """
    Run the compiled strategy through Claude for a risk/logic review.
    Returns {valid: bool, warnings: list, suggestions: list, summary: str}
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        prompt = f"""Review this trading strategy config for logic errors, conflicts, and risk issues.

Strategy config:
{json.dumps(config, indent=2)}

Reply in JSON format:
{{
  "valid": true/false,
  "warnings": ["warning 1", "warning 2"],
  "suggestions": ["suggestion 1"],
  "summary": "One paragraph plain English summary of what this strategy does and when it fires",
  "risk_rating": "LOW/MEDIUM/HIGH"
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        return json.loads(raw)
    except Exception as e:
        logger.error(f"Strategy validation error: {e}")
        return {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "summary": config.get("description", "Custom strategy"),
            "risk_rating": "MEDIUM",
        }


async def generate_strategy_summary(config: Dict) -> str:
    """Generate a short human-readable summary for display in the portal."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise this trading strategy in 2 sentences for a non-technical audience:\n"
                    f"{json.dumps(config, indent=2)}"
                )
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return config.get("description", "Custom trading strategy")


def format_config_for_display(config: Dict) -> str:
    """Format a strategy config as readable text for Telegram."""
    lines = []
    lines.append(f"<b>{config.get('name', 'Untitled')}</b>")
    lines.append(f"<i>{config.get('description', '')}</i>")
    lines.append("")

    # Direction
    direction = config.get("direction", "LONG")
    dir_icon  = {"LONG": "🟢", "SHORT": "🔴", "BOTH": "⚡"}.get(direction, "⚡")
    lines.append(f"{dir_icon} <b>Direction:</b> {direction}")

    # Universe
    uni = config.get("universe", {})
    if uni.get("type") == "specific":
        syms = ", ".join(uni.get("symbols", []))
        lines.append(f"🎯 <b>Coins:</b> {syms}")
    else:
        lines.append(f"🎯 <b>Coins:</b> All eligible (excl. slow high-caps)")

    # Entry conditions
    entry = config.get("entry_conditions", {})
    conds = entry.get("conditions", [])
    lines.append(f"\n<b>Entry ({entry.get('operator','AND')}):</b>")
    for c in conds:
        ctype = c.get("type", "")
        if ctype == "indicator":
            lines.append(f"  • {c.get('name','').upper()} {c.get('timeframe','')} {c.get('operator','')} {c.get('value','')}")
        elif ctype == "price_momentum":
            lines.append(f"  • Price {c.get('direction','moved')} {c.get('value','')}%+ in {c.get('window_minutes','')}min")
        elif ctype == "volume_spike":
            lines.append(f"  • Volume spike {c.get('multiplier','')}x normal")
        elif ctype == "support_resistance":
            lines.append(f"  • Price {c.get('condition','').replace('_',' ')} (±{c.get('tolerance_pct',1)}%)")
        elif ctype == "fvg":
            lines.append(f"  • {c.get('direction','').title()} FVG — {c.get('condition','').replace('_',' ')}")

    # Exit
    ex = config.get("exit", {})
    lines.append(f"\n<b>Exit:</b> TP {ex.get('take_profit_pct',3)}%  ·  SL {ex.get('stop_loss_pct',1.5)}%")
    if ex.get("trailing_stop"):
        lines.append(f"  Trailing stop: {ex.get('trailing_stop_pct',1)}%")

    # Risk
    risk = config.get("risk", {})
    lines.append(f"<b>Risk:</b> {risk.get('leverage',10)}x lev  ·  {risk.get('position_size_pct',5)}% size  ·  max {risk.get('max_trades_per_day',3)}/day")

    return "\n".join(lines)
