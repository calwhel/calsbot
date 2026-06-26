"""Gold AI Claude decisioning with model-aware cost estimation."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.config import (
    HAIKU_CACHE_READ_USD_PER_M,
    HAIKU_CACHE_WRITE_USD_PER_M,
    HAIKU_INPUT_USD_PER_M,
    HAIKU_OUTPUT_USD_PER_M,
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
- USE LESSONS: If lessons/history are provided, treat them as secondary context only.
  Do not treat tiny samples as predictive; prioritize current structure, displacement, confirmation, and R:R.
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

OUTPUT (STRICT FORMAT)
Respond with ONLY ONE valid JSON object matching this schema.
No prose. No markdown. No code fences. No commentary before or after JSON.
{{
  "action": "take" | "skip",
  "direction": "long" | "short" | null,
  "entry": number | null,
  "stop_loss": number | null,
  "take_profit": number | null,
  "confidence": 0-100,
  "rationale": "one concise paragraph"
}}
Do not include keys outside this schema.

If action is skip, direction/entry/stop_loss/take_profit may be null.
If confidence is below {t}, action MUST be skip.
Only action "take" when confidence ≥ {t} AND entry/stop/TP are coherent with spot and direction.
Never invent prices far from spot. SL must be on correct side of entry for direction."""


SYSTEM_PROMPT = system_prompt(_DEFAULT_CONFIDENCE_THRESHOLD)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return first balanced {...} object, string-aware."""
    if not text:
        return None
    start = -1
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tolerates markdown fences and leading/trailing prose around one JSON object.
    """
    if not text:
        return None
    blob = text.strip()
    # Try full payload first.
    try:
        loaded = json.loads(blob)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass

    candidates = []
    if "```" in blob:
        for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", blob, re.DOTALL | re.IGNORECASE):
            fenced = (m.group(1) or "").strip()
            if fenced:
                candidates.append(fenced)
    candidates.append(blob)

    for cand in candidates:
        obj = _extract_first_json_object(cand)
        if not obj:
            continue
        try:
            loaded = json.loads(obj)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            continue
    return None


def _json_only_user_prompt(context_text: str) -> str:
    return (
        "Return ONLY one valid JSON object that matches the schema in system instructions.\n"
        "No prose, no markdown, no code fences, and no text before or after JSON.\n\n"
        f"{context_text}"
    )


def orb_system_prompt(confidence_threshold: int = 55) -> str:
    """Dedicated ORB prompt (separate from ICT/SMC prompt)."""
    t = max(0, min(100, int(confidence_threshold)))
    return f"""You are evaluating an Opening Range Breakout (ORB) trade on XAUUSD for a DEMO strategy.

OBJECTIVE
- Judge this setup ONLY in ORB terms: opening range quality, breakout quality, fakeout risk, retest quality, and chase risk vs break level.
- Do NOT use ICT/SMC killzone/OB language unless explicitly provided in context.

ORB RUBRIC
- Opening range should be well-defined (clear high/low, reasonable height vs ATR).
- Breakout should show directional commitment (close/bodies/displacement), not just a thin wick.
- Penalize fakeouts/sweeps that immediately reject back inside the range.
- If entry is too extended from the break level, confidence must drop or SKIP.
- Reward clean breakout + controlled pullback/retest + continuation potential.

RISK RULES
- SL should invalidate the ORB thesis (typically opposite side of range or ATR-invalidated level).
- TP should be range-relative and realistic for session conditions.
- Borderline but tradable setups can be TAKE when risk is coherent and confidence justifies it.

OUTPUT (STRICT JSON)
Respond with ONLY valid JSON:
{{
  "action": "take" | "skip",
  "direction": "long" | "short" | null,
  "entry": number | null,
  "stop_loss": number | null,
  "take_profit": number | null,
  "confidence": 0-100,
  "rationale": "one concise paragraph"
}}

If confidence is below {t}, action MUST be "skip"."""


def _pricing_for_model(model: str) -> Tuple[float, float, float, float]:
    m = (model or "").strip().lower()
    if "haiku" in m:
        return (
            HAIKU_INPUT_USD_PER_M,
            HAIKU_OUTPUT_USD_PER_M,
            HAIKU_CACHE_READ_USD_PER_M,
            HAIKU_CACHE_WRITE_USD_PER_M,
        )
    return (
        OPUS_INPUT_USD_PER_M,
        OPUS_OUTPUT_USD_PER_M,
        OPUS_CACHE_READ_USD_PER_M,
        OPUS_CACHE_WRITE_USD_PER_M,
    )


def _estimate_cost(
    usage,
    *,
    model: str = "claude-opus-4-8",
) -> Tuple[int, int, int, int, float]:
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    in_rate, out_rate, read_rate, write_rate = _pricing_for_model(model)
    billable_in = max(0, tin - cr)
    cost = (
        billable_in / 1_000_000 * in_rate
        + tout / 1_000_000 * out_rate
        + cr / 1_000_000 * read_rate
        + cw / 1_000_000 * write_rate
    )
    return tin, tout, cr, cw, round(cost, 6)


def _merge_usage_meta(
    base: Dict[str, Any],
    *,
    usage,
    model: str,
    suffix: str = "",
) -> Dict[str, Any]:
    tin, tout, cr, cw, cost = _estimate_cost(usage, model=model)
    base["tokens_in"] = int(base.get("tokens_in", 0)) + tin
    base["tokens_out"] = int(base.get("tokens_out", 0)) + tout
    base["cache_read_tokens"] = int(base.get("cache_read_tokens", 0)) + cr
    base["cache_write_tokens"] = int(base.get("cache_write_tokens", 0)) + cw
    base["cost_usd"] = round(float(base.get("cost_usd", 0.0)) + cost, 6)
    if suffix:
        base[f"tokens_in_{suffix}"] = tin
        base[f"tokens_out_{suffix}"] = tout
        base[f"cache_read_tokens_{suffix}"] = cr
        base[f"cache_write_tokens_{suffix}"] = cw
        base[f"cost_usd_{suffix}"] = cost
    return base


async def _repair_json_once(
    *,
    client,
    raw_text: str,
    model: str,
    threshold: int,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[Any]]:
    """
    One repair pass: ask Claude to convert its own prose to schema JSON.
    """
    repair_prompt = (
        "Convert the following analysis into EXACTLY one JSON object matching this schema:\n"
        "{"
        '"action":"take|skip",'
        '"direction":"long|short|null",'
        '"entry":"number|null",'
        '"stop_loss":"number|null",'
        '"take_profit":"number|null",'
        '"confidence":"0-100 integer",'
        '"rationale":"string"'
        "}\n"
        "Rules: keep original trade intent; do not add prose/markdown/code fences.\n"
        "If a value is missing, use null.\n\n"
        "Analysis to convert:\n"
        f"{raw_text}"
    )
    msg = await client.messages.create(
        model=model,
        max_tokens=250,
        system=[
            {
                "type": "text",
                "text": (
                    "You are a strict JSON formatter. "
                    "Return only one valid JSON object and nothing else."
                ),
            }
        ],
        messages=[
            {"role": "user", "content": repair_prompt},
        ],
    )
    repaired_text = ""
    if msg.content:
        repaired_text = (msg.content[0].text or "").strip()
    parsed = _parse_json(repaired_text)
    if parsed and int(parsed.get("confidence") or 0) < threshold and parsed.get("action") == "take":
        parsed["action"] = "skip"
        parsed.setdefault(
            "rationale",
            f"Confidence below {threshold}% take threshold.",
        )
    return parsed, repaired_text, getattr(msg, "usage", None)


async def _decide_with_prompt(
    context_text: str,
    *,
    model: str = "claude-opus-4-8",
    dry_run: bool = False,
    confidence_threshold: int = _DEFAULT_CONFIDENCE_THRESHOLD,
    system_text: str,
    user_intro: str,
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
        prompt_body = (
            f"{(user_intro or '').strip()}\n\n{context_text}"
            if (user_intro or "").strip()
            else context_text
        )
        user_prompt = _json_only_user_prompt(prompt_body)
        msg = await client.messages.create(
            model=model,
            max_tokens=700,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = ""
        if msg.content:
            text = (msg.content[0].text or "").strip()
        meta = {
            "tokens_in": 0,
            "tokens_out": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
        }
        _merge_usage_meta(meta, usage=msg.usage, model=model)
        parsed = _parse_json(text)
        if not parsed:
            logger.warning(
                "[gold-ai-trader] malformed Claude JSON (attempting salvage): %s",
                text[:200],
            )
            parsed, repaired_text, repaired_usage = await _repair_json_once(
                client=client,
                raw_text=text[:3000],
                model=model,
                threshold=threshold,
            )
            meta["salvage_attempted"] = True
            if repaired_usage is not None:
                _merge_usage_meta(meta, usage=repaired_usage, model=model, suffix="salvage")
            if parsed:
                meta["salvage_success"] = True
                logger.info("[gold-ai-trader] malformed JSON salvaged via repair pass")
            else:
                logger.warning(
                    "[gold-ai-trader] malformed Claude JSON (salvage failed): %s",
                    repaired_text[:200] if repaired_text else text[:200],
                )
                skip = {
                    "action": "skip",
                    "confidence": 0,
                    "rationale": "Malformed model output",
                }
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


async def decide(
    context_text: str,
    *,
    model: str = "claude-opus-4-8",
    dry_run: bool = False,
    confidence_threshold: int = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    threshold = max(0, min(100, int(confidence_threshold)))
    return await _decide_with_prompt(
        context_text,
        model=model,
        dry_run=dry_run,
        confidence_threshold=threshold,
        system_text=system_prompt(threshold),
        user_intro="Evaluate this gold setup. Reason briefly, then output JSON only.",
    )


async def decide_orb(
    context_text: str,
    *,
    model: str = "claude-opus-4-8",
    dry_run: bool = False,
    confidence_threshold: int = 55,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    threshold = max(0, min(100, int(confidence_threshold)))
    return await _decide_with_prompt(
        context_text,
        model=model,
        dry_run=dry_run,
        confidence_threshold=threshold,
        system_text=orb_system_prompt(threshold),
        user_intro=(
            "Evaluate this Opening Range Breakout candidate in ORB terms only. "
            "Return JSON only."
        ),
    )
