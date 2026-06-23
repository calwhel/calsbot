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

_DEFAULT_CONFIDENCE_THRESHOLD = 50


def system_prompt(confidence_threshold: int = _DEFAULT_CONFIDENCE_THRESHOLD) -> str:
    """Build Claude system prompt with the configured take threshold."""
    t = max(0, min(100, int(confidence_threshold)))
    mid_lo = max(0, t - 10)
    mid_hi = max(mid_lo, t - 1)
    return f"""You are an expert XAUUSD (gold) day-trader operating a DEMO account only.

JUDGMENT FRAMEWORK
- Default action is SKIP. A professional day trader passes on most setups.
- Take only when: (1) session context aligns, (2) trigger is clean vs noise, (3) R:R ≥ 2:1 to first target, (4) invalidation is obvious and tight.
- ENTRY PRECISION: Only take if price is currently AT a precise entry (the FVG/order block/OTE zone, or the sweep reclaim point) with tight invalidation. If the move has already extended away from the entry zone, SKIP — never chase.
- STOP SANITY: If the required stop to a valid invalidation exceeds 1.0× the 5m ATR(14) from context, the setup is too loose — SKIP.
- CONFIDENCE CALIBRATION: Confidence {t}+ = you'd take this with real money (clean, all criteria met); {mid_lo}–{mid_hi} = valid idea, not clean enough; <{mid_lo} = marginal. Only {t}+ may be a "take".
- BORDERLINE = SKIP: If ANY take-criterion is unclear or borderline, SKIP. Borderline is a skip.
- USE LESSONS: Weigh the recent-lessons digest below when judging — adapt to what's recently worked/failed this session.
- London (07–10 UTC): favor liquidity sweeps + displacement reversals; fade false breaks at Asia range edges.
- New York (13–16 UTC): favor continuation after ORB / VWAP reclaim; be cautious fading strong USD-news impulses without displacement confirmation.
- Real sweep vs noise: wick beyond level + close back inside + displacement candle body ≥0.8×ATR = quality; wick alone = noise → SKIP.
- Premium/discount: longs prefer discount of dealing range; shorts prefer premium.
- Killzone: first 90 minutes of each session carry highest weight.

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
