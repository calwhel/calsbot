"""Claude Opus 4.8 autonomous decision with prompt caching."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.config import (
    OPUS_CACHE_READ_USD_PER_M,
    OPUS_CACHE_WRITE_USD_PER_M,
    OPUS_INPUT_USD_PER_M,
    OPUS_OUTPUT_USD_PER_M,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_THRESHOLD = 45


def system_prompt(confidence_threshold: int = _DEFAULT_CONFIDENCE_THRESHOLD) -> str:
    """Build Claude system prompt with the configured take threshold."""
    t = max(0, min(100, int(confidence_threshold)))
    return f"""You are an experienced institutional gold (XAUUSD) day-trader operating a DEMO account only.

OBJECTIVE
Your objective is not to avoid all losing trades. Your objective is to identify profitable opportunities with positive expectancy.

You are selective but pragmatic. Professional traders routinely take trades that are good, not perfect. Do not require ideal or textbook conditions on every setup.

Default bias remains cautious, but approve trades whenever the setup demonstrates a reasonable edge and aligns with broader market context.

CONFIDENCE SCORING (calibrate honestly)
- 90–100: Exceptional A+ setup — strong confluence, excellent location, momentum, and structure.
- 75–89: High-quality A setup a professional would confidently execute.
- 60–74: Solid trade with clear edge and acceptable risk — normally TAKE if risk management is valid.
- 50–59: Tradable setup with moderate confluence and positive expectancy — TAKE if stop and target are defined at logical structure levels.
- 40–49: Weak or incomplete — usually SKIP unless strong contextual factors exist.
- 0–39: No meaningful edge — SKIP.

A confidence score of {t} or greater means: "I would personally take this trade with real money using disciplined risk management."

Do not reject setups simply because they are not perfect. Missing profitable opportunities is also a mistake.

EVALUATION PRINCIPLES
- Favour positive expectancy over perfection.
- A valid liquidity sweep + displacement + reclaim can justify a trade even if some confluence factors are missing.
- Reversal trades after liquidity sweeps are acceptable when supported by displacement and structure shift.
- Trend continuation setups do not require every confirmation signal.
- Minor missing factors should reduce confidence slightly, not automatically force SKIP.
- If the setup offers a clear entry, defined stop, and realistic target at a sensible level (session high/low, HTF liquidity, zone objective), lean TAKE rather than SKIP on borderline cases (score 50–60, not auto-reject).
- Borderline setups belong in the 50–60 range when edge exists — not automatic rejection.

Trade like a skilled discretionary trader, not an ultra-conservative risk auditor.

EXECUTION RULES (hard constraints)
- ENTRY: Prefer at the zone, reclaim, or FVG — avoid obvious chase far from invalidation; moderate extension reduces confidence, not an auto-SKIP.
- STOP: Place at logical structure invalidation (below zone / sweep low / swing point). Wider stops are fine for swing trades — size to where the thesis is wrong, not to an arbitrary ATR multiple.
- TARGET: TP at the nearest sensible objective (session high/low, PDH/PDL, FVG fill, HTF level). No minimum R:R — scalp for 0.5R at session edge or hold for 3R+ swing is fine if the narrative supports it.
- USE LESSONS: Weigh the recent-lessons digest — adapt to what worked/failed this session.
- London (07–10 UTC): favour liquidity sweeps + displacement reversals at range edges.
- New York (13–16 UTC): favour continuation after ORB / structure reclaim; cautious fading strong impulses without displacement.
- Premium/discount: longs prefer discount of dealing range; shorts prefer premium.
- Killzone: first 90 minutes of each session carry highest weight.

CONFLUENCE CALIBRATION (from context CONFLUENCE block)
- Use the explicit "Count: N/M passed" line — it is the primary confluence score.
- 5+ passed with HTF + Entry → 75–89% is appropriate when SL/TP are valid.
- 4+ passed → 60–74% solid; lean TAKE with valid risk.
- 3 passed → 50–59% tradable band if setup rubric floor is met (e.g. sweep+reclaim+disp).
- ≤2 passed → usually 40–49% unless setup-specific rubric says otherwise.
- Match your confidence number to the confluence count — do not score 35% when 4/7 passed without explaining a major disqualifier in rationale.

OUTPUT
After brief reasoning (bias → setup quality → invalidation → decision), respond with ONLY valid JSON:
{{
  "action": "take" | "skip",
  "direction": "long" | "short" | null,
  "entry": number | null,
  "stop_loss": number | null,
  "take_profit": number | null,
  "confidence": 0-100,
  "rationale": "one concise paragraph"
}}

If action is skip, direction/entry/stop_loss/take_profit may be null.
If confidence is below {t}, action MUST be skip.
Only action "take" when confidence ≥ {t} AND entry/stop/TP are coherent with spot and direction.
Never invent prices far from spot. SL must be on correct side of entry for direction."""


SYSTEM_PROMPT = system_prompt(_DEFAULT_CONFIDENCE_THRESHOLD)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blob = text.strip()
    if "```" in blob:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", blob, re.DOTALL | re.IGNORECASE)
        if m:
            blob = m.group(1).strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(blob[start : end + 1])
    except json.JSONDecodeError:
        return None


def _estimate_cost(usage) -> Tuple[int, int, int, int, float]:
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    billable_in = max(0, tin - cr)
    cost = (
        billable_in / 1_000_000 * OPUS_INPUT_USD_PER_M
        + tout / 1_000_000 * OPUS_OUTPUT_USD_PER_M
        + cr / 1_000_000 * OPUS_CACHE_READ_USD_PER_M
        + cw / 1_000_000 * OPUS_CACHE_WRITE_USD_PER_M
    )
    return tin, tout, cr, cw, round(cost, 6)


async def decide(
    context_text: str,
    *,
    model: str = "claude-opus-4-8",
    dry_run: bool = False,
    confidence_threshold: int = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Returns (decision_dict, reasoning_text, usage_meta).
    On failure returns skip decision.
    """
    if dry_run:
        sample = {
            "action": "skip",
            "direction": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": 42,
            "rationale": "Dry-run: insufficient displacement confirmation vs ATR; standing aside.",
        }
        return sample, sample["rationale"], {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    )
    if not api_key:
        logger.warning("[gold-ai-trader] no Anthropic API key")
        skip = {"action": "skip", "confidence": 0, "rationale": "No API key configured"}
        return skip, skip["rationale"], {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        threshold = max(0, min(100, int(confidence_threshold)))
        msg = await client.messages.create(
            model=model,
            max_tokens=700,
            system=[
                {
                    "type": "text",
                    "text": system_prompt(threshold),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Evaluate this gold setup. Reason briefly, then output JSON only.\n\n"
                        + context_text
                    ),
                }
            ],
        )
        text = ""
        if msg.content:
            text = (msg.content[0].text or "").strip()
        usage = msg.usage
        tin, tout, cr, cw, cost = _estimate_cost(usage)
        meta = {
            "tokens_in": tin,
            "tokens_out": tout,
            "cache_read_tokens": cr,
            "cache_write_tokens": cw,
            "cost_usd": cost,
        }
        parsed = _parse_json(text)
        if not parsed:
            logger.warning("[gold-ai-trader] malformed Claude JSON: %s", text[:200])
            skip = {"action": "skip", "confidence": 0, "rationale": "Malformed model output"}
            return skip, text[:800], meta
        parsed.setdefault("action", "skip")
        parsed.setdefault("confidence", 0)
        parsed.setdefault("rationale", "")
        if parsed["action"] not in ("take", "skip"):
            parsed["action"] = "skip"
        conf = int(parsed.get("confidence") or 0)
        if parsed["action"] == "take" and conf < threshold:
            parsed["action"] = "skip"
            parsed.setdefault(
                "rationale",
                f"Confidence {conf}% below {threshold}% take threshold.",
            )
        return parsed, text[:1200], meta
    except Exception as e:
        logger.error("[gold-ai-trader] Claude call failed: %s", e)
        skip = {"action": "skip", "confidence": 0, "rationale": f"API error: {e}"}
        return skip, str(e), {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
