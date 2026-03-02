"""
AI Signal Filter - Uses Claude to validate trading signals before broadcast.

Reviews each signal candidate with market data and approves/rejects based on:
- Technical analysis quality
- Risk factors (BTC correlation, overextension)
- Market conditions
- Entry timing
"""

import os
import logging
import json
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── Grok Macro Cache ─────────────────────────────────────────────────────────
_grok_macro_cache: Dict = {}
_grok_macro_last_refresh: Optional[datetime] = None
GROK_MACRO_CACHE_MINUTES = 20


def _get_grok_client():
    """Return an AsyncOpenAI client pointed at xAI, or None."""
    xai_key = os.getenv('XAI_API_KEY')
    if not xai_key:
        return None
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")


async def _grok_agent_search(prompt: str, max_tokens: int = 350, timeout: float = 80.0) -> str:
    """
    Call xAI Agent Tools API (POST /v1/responses) with live web_search + x_search.
    Returns the text response or raises on failure with full error detail.
    """
    import aiohttp
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        raise ValueError("No XAI_API_KEY set in environment")

    payload = {
        "model": "grok-4",
        "input": [{"role": "user", "content": prompt}],
        "tools": [{"type": "web_search"}, {"type": "x_search"}],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/responses",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            status = resp.status
            raw = await resp.text()

    # Log the raw response so we can debug exactly what xAI is returning
    if status != 200:
        logger.error(f"❌ xAI Agent API HTTP {status}: {raw[:500]}")
        raise ValueError(f"HTTP {status}: {raw[:300]}")

    try:
        data = json.loads(raw)
    except Exception as je:
        logger.error(f"❌ xAI Agent API non-JSON response (HTTP {status}): {raw[:300]}")
        raise ValueError(f"Non-JSON response: {raw[:200]}")

    if data.get("error"):
        logger.error(f"❌ xAI Agent API error field: {data['error']}")
        raise ValueError(f"Agent API error: {data['error']}")

    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"].strip()

    logger.error(f"❌ xAI Agent API — no output_text found. Full response: {raw[:600]}")
    raise ValueError(f"No output_text in response. Keys: {list(data.keys())}")


async def refresh_grok_macro_context(force_fresh: bool = False) -> Dict:
    """
    Ask Grok-4 with LIVE web+X search for a global macro/geopolitical briefing.
    Uses xAI Agent Tools API — Grok actively searches the web and X/Twitter right now.
    Falls back to grok-3-beta static knowledge if Agent API fails.
    Cached for GROK_MACRO_CACHE_MINUTES minutes.
    """
    global _grok_macro_cache, _grok_macro_last_refresh

    prompt = (
        "You are a real-time macro and geopolitical intelligence analyst for crypto traders. "
        "Search the web and X/Twitter RIGHT NOW and tell me what is driving crypto markets. "
        "Cover ALL of the following if relevant:\n"
        "- GEOPOLITICAL: Active wars, ceasefire talks, trade wars, tariffs, sanctions, "
        "diplomatic tensions, elections, government crypto seizures, nation-state moves.\n"
        "- MACRO/ECONOMIC: Fed decisions or recent commentary, CPI/PPI/jobs data releases today, "
        "DXY strength, interest rate futures, recession signals, oil/gold moves.\n"
        "- CRYPTO-SPECIFIC: BTC key levels and current trend, ETF flows today, major liquidations, "
        "exchange issues, regulatory actions, protocol events, whale on-chain moves.\n"
        "- SENTIMENT: Current fear vs greed reading, what's trending on X/Twitter right now, "
        "institutional positioning shifts.\n"
        "Give a sharp 3-4 sentence briefing with CURRENT SPECIFIC FACTS (prices, names, numbers). "
        "Tell traders whether to lean long or short right now and why. "
        "End with exactly one of these tags on its own line:\n"
        "MACRO_BIAS: BULLISH  or  MACRO_BIAS: BEARISH  or  MACRO_BIAS: NEUTRAL"
    )

    text = ""
    live_search_used = False
    last_error = ""

    # Try Agent Tools API with live search first (grok-4 with web search takes 45-75s)
    try:
        text = await asyncio.wait_for(_grok_agent_search(prompt, max_tokens=350), timeout=90.0)
        live_search_used = True
        logger.info("🌐 Grok-4 live search (web+X) used for macro briefing")
    except Exception as e:
        err_type = type(e).__name__
        last_error = f"grok-4 Agent Tools API: {err_type}: {e}"
        logger.error(f"❌ Grok-4 Agent API FAILED ({err_type}): {e} — falling back to grok-3-beta static knowledge")

    # Fallback: grok-3-beta without live search
    if not text:
        try:
            grok = _get_grok_client()
            if not grok:
                last_error += " | grok-4: No XAI_API_KEY set in environment"
                logger.error("❌ XAI_API_KEY missing — cannot call any Grok model")
            else:
                response = await asyncio.wait_for(
                    grok.chat.completions.create(
                        model="grok-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.3,
                    ),
                    timeout=40.0,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    logger.warning("⚠️ Using grok-4 STATIC knowledge (no live search) — data may be outdated")
        except Exception as e2:
            last_error += f" | grok-4 fallback: {e2}"
            logger.error(f"❌ Grok-4 fallback also failed: {e2}")

    if not text:
        # force_fresh=True (from /briefing command) — never serve stale cache, return error
        if force_fresh:
            return {"error": last_error or "Both API calls returned empty text"}
        # Background refresh — fall back to last known cache to keep signals flowing
        if _grok_macro_cache:
            return _grok_macro_cache
        return {"error": last_error or "Both API calls returned empty text"}

    bias = "NEUTRAL"
    for line in text.splitlines():
        if line.strip().upper().startswith("MACRO_BIAS:"):
            tag = line.split(":", 1)[1].strip().upper()
            if tag in ("BULLISH", "BEARISH", "NEUTRAL"):
                bias = tag
            break

    summary = text
    for tag in ("MACRO_BIAS: BULLISH", "MACRO_BIAS: BEARISH", "MACRO_BIAS: NEUTRAL"):
        summary = summary.replace(tag, "").strip()

    _grok_macro_cache = {
        "summary": summary,
        "bias": bias,
        "live_search": live_search_used,
        "agent_error": last_error if not live_search_used and last_error else None,
    }
    _grok_macro_last_refresh = datetime.utcnow()
    source = "🌐 live web+X" if live_search_used else "📚 static"
    logger.info(f"🌍 Grok macro [{source}] → bias={bias} | {summary[:120]}...")
    return _grok_macro_cache


async def get_cached_grok_macro() -> Dict:
    """Return cached macro context, auto-refreshing if older than GROK_MACRO_CACHE_MINUTES."""
    if (
        not _grok_macro_last_refresh
        or datetime.utcnow() - _grok_macro_last_refresh > timedelta(minutes=GROK_MACRO_CACHE_MINUTES)
    ):
        return await refresh_grok_macro_context()
    return _grok_macro_cache


async def get_grok_coin_intelligence(symbol: str, direction: str) -> Dict:
    """
    Scalp-focused coin intelligence from Grok.
    Returns:
      summary          : str  — 2-3 sentence context for Claude
      hard_no          : bool — True if Grok flags a serious red flag
      hard_no_reason   : str  — reason if hard_no is True
      momentum_fading  : bool — True if X/social momentum is dying (bad for scalps)
      momentum_verdict : str  — RISING | FADING | NEUTRAL
    Times out in 15 seconds so it never blocks signal generation.
    """
    result = {
        "summary": "",
        "hard_no": False,
        "hard_no_reason": "",
        "momentum_fading": False,
        "momentum_verdict": "NEUTRAL",
    }
    grok = _get_grok_client()
    if not grok:
        return result
    try:
        coin = symbol.replace("USDT", "").replace("PERP", "").replace("-", "")
        prompt = (
            f"You are analyzing ${coin} for a SHORT-TERM SCALP ({direction}, 5-20 min hold). "
            f"Answer these questions concisely:\n"
            f"(1) MOMENTUM: Is X/Twitter attention on ${coin} RISING, FADING, or NEUTRAL "
            f"in the LAST 30 MINUTES? Are posts increasing or were they from 1-2h ago? "
            f"This is the most important question for a scalp.\n"
            f"(2) RED FLAGS: Any hacks, exploits, delistings, rug signals, or emergency "
            f"announcements in the last few hours?\n"
            f"(3) CONTEXT: Any whale moves, influencer calls, or news that could spike or "
            f"dump ${coin} in the next 20 minutes?\n\n"
            f"FORMAT your response exactly like this:\n"
            f"MOMENTUM: RISING or FADING or NEUTRAL\n"
            f"SUMMARY: [2 sentence factual summary]\n"
            f"If there is a critical red flag, instead reply: HARD_NO: [reason]"
        )
        response = await asyncio.wait_for(
            grok.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2,
            ),
            timeout=30.0,
        )
        text = (response.choices[0].message.content or "").strip()

        if text.upper().startswith("HARD_NO:"):
            reason = text.split(":", 1)[1].strip() if ":" in text else text
            result["hard_no"] = True
            result["hard_no_reason"] = reason[:200]
            logger.warning(f"🚨 Grok HARD VETO on {symbol} {direction}: {reason[:100]}")
            return result

        # Parse structured response
        momentum_verdict = "NEUTRAL"
        summary_lines = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("MOMENTUM:"):
                tag = line.split(":", 1)[1].strip().upper()
                if tag in ("RISING", "FADING", "NEUTRAL"):
                    momentum_verdict = tag
            elif line.upper().startswith("SUMMARY:"):
                summary_lines.append(line.split(":", 1)[1].strip())
            elif line and not line.upper().startswith("MOMENTUM:"):
                summary_lines.append(line)

        result["momentum_verdict"] = momentum_verdict
        result["momentum_fading"] = momentum_verdict == "FADING"
        result["summary"] = " ".join(summary_lines).strip()[:300]

        momentum_icon = {"RISING": "📈", "FADING": "📉", "NEUTRAL": "➡️"}.get(momentum_verdict, "")
        logger.info(
            f"{momentum_icon} Grok momentum for {symbol}: {momentum_verdict} | "
            f"{result['summary'][:100]}..."
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Grok coin intelligence timed out for {symbol} — skipping")
        return result
    except Exception as e:
        logger.warning(f"Grok coin intelligence error for {symbol}: {e}")
        return result


async def get_grok_chart_vision(symbol: str, direction: str) -> Dict:
    """
    Generate a 5m candlestick chart with EMA8/21 + volume and send it to
    grok-2-vision-1212 for visual TA validation.

    Returns:
      chart_verdict    : str  — CONFIRMS_LONG | CONFIRMS_SHORT | NEUTRAL | AGAINST
      pattern          : str  — detected chart pattern
      visual_analysis  : str  — Grok's 1-2 sentence visual description
      chart_ok         : bool — False if chart could not be generated or timed out
    """
    result = {
        "chart_verdict": "NEUTRAL",
        "pattern": "",
        "visual_analysis": "",
        "chart_ok": False,
    }
    grok = _get_grok_client()
    if not grok:
        return result

    try:
        import aiohttp
        import base64
        from io import BytesIO

        # ── 1. Fetch 5m klines from Binance Futures ──────────────────────────
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&limit=60"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                klines = await resp.json()

        if not klines or len(klines) < 10:
            logger.warning(f"Grok chart vision: too few klines for {symbol}")
            return result

        opens  = [float(k[1]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        vols   = [float(k[5]) for k in klines]

        # ── 2. Generate chart in executor (matplotlib is sync) ───────────────
        def _render_chart() -> str:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            def _ema_series(vals, period):
                if len(vals) < period:
                    return [None] * len(vals)
                k = 2.0 / (period + 1)
                e = sum(vals[:period]) / period
                out = [None] * (period - 1) + [e]
                for v in vals[period:]:
                    e = v * k + e * (1 - k)
                    out.append(e)
                return out

            ema8  = _ema_series(closes, 8)
            ema21 = _ema_series(closes, 21)
            n = len(klines)
            xs = list(range(n))

            bg = "#0d0d1a"
            fig = plt.figure(figsize=(12, 7), facecolor=bg)
            gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            for ax in (ax1, ax2):
                ax.set_facecolor(bg)
                for spine in ax.spines.values():
                    spine.set_color("#2a2a3e")
                ax.tick_params(colors="#aaaacc", labelsize=7)

            # Candles
            for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
                col = "#00e676" if c >= o else "#ff1744"
                ax1.plot([i, i], [l, h], color=col, linewidth=0.8, zorder=2)
                ax1.bar(i, abs(c - o) or (h - l) * 0.01,
                        bottom=min(o, c), color=col, width=0.6, alpha=0.88, zorder=2)

            # EMAs
            e8_x  = [i for i, v in enumerate(ema8)  if v is not None]
            e8_y  = [v for v in ema8  if v is not None]
            e21_x = [i for i, v in enumerate(ema21) if v is not None]
            e21_y = [v for v in ema21 if v is not None]
            ax1.plot(e8_x,  e8_y,  color="#ff8c00", linewidth=1.3, label="EMA8",  zorder=3)
            ax1.plot(e21_x, e21_y, color="#2979ff", linewidth=1.3, label="EMA21", zorder=3)

            ax1.set_title(f"{symbol} · 5m · {direction} scalp analysis",
                          color="#ddddff", fontsize=11, pad=8)
            ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8,
                       loc="upper left", framealpha=0.6)
            ax1.set_ylabel("Price", color="#aaaacc", fontsize=8)
            plt.setp(ax1.get_xticklabels(), visible=False)

            # Volume
            for i, (o, c, v) in enumerate(zip(opens, closes, vols)):
                col = "#00e676" if c >= o else "#ff1744"
                ax2.bar(i, v, color=col, alpha=0.6)
            ax2.set_ylabel("Vol", color="#aaaacc", fontsize=7)

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=bg)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        loop = asyncio.get_event_loop()
        chart_b64 = await loop.run_in_executor(None, _render_chart)

        # ── 3. Send to Grok Vision ───────────────────────────────────────────
        coin = symbol.replace("USDT", "").replace("PERP", "").replace("-", "")
        prompt = (
            f"You are analyzing a 5-minute candlestick chart for ${coin} "
            f"to validate a {direction} scalp entry (5-20 min hold).\n"
            f"Green candles = bullish, Red candles = bearish. "
            f"Orange line = EMA8, Blue line = EMA21. Bottom panel = volume.\n\n"
            f"Answer:\n"
            f"1. Does the chart CONFIRM or work AGAINST a {direction} entry right now?\n"
            f"2. What pattern is forming? (bull flag, bear pennant, EMA crossover, range, breakdown, etc.)\n"
            f"3. Is volume supporting the move?\n"
            f"4. Any rejection wicks, reversal candles, or exhaustion signals?\n\n"
            f"Reply in EXACTLY this format:\n"
            f"CHART_VERDICT: CONFIRMS_{direction} or AGAINST or NEUTRAL\n"
            f"PATTERN: [name]\n"
            f"VISUAL_ANALYSIS: [2 sentence max]"
        )

        response = await asyncio.wait_for(
            grok.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{chart_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=160,
                temperature=0.2,
            ),
            timeout=20.0,
        )

        text = (response.choices[0].message.content or "").strip()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("CHART_VERDICT:"):
                result["chart_verdict"] = line.split(":", 1)[1].strip()
            elif line.startswith("PATTERN:"):
                result["pattern"] = line.split(":", 1)[1].strip()
            elif line.startswith("VISUAL_ANALYSIS:"):
                result["visual_analysis"] = line.split(":", 1)[1].strip()

        result["chart_ok"] = True
        logger.info(
            f"👁️ Grok vision {symbol}: {result['chart_verdict']} | "
            f"{result['pattern']} | {result['visual_analysis'][:80]}"
        )
        return result

    except asyncio.TimeoutError:
        logger.warning(f"Grok chart vision timed out for {symbol}")
        return result
    except Exception as e:
        logger.warning(f"Grok chart vision error for {symbol}: {e}")
        return result


def get_anthropic_client():
    """Get Anthropic Claude client - checks Replit AI Integrations first, then standalone key."""
    try:
        import anthropic
        
        # Check for Replit AI Integrations first
        base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        
        # Fall back to standalone Anthropic API key (for Railway)
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            base_url = None
        
        if not api_key:
            logger.warning("No Anthropic API key found")
            return None
        
        if base_url:
            return anthropic.Anthropic(base_url=base_url, api_key=api_key)
        else:
            return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Anthropic client: {e}")
        return None


def build_signal_prompt(
    signal_data: Dict,
    market_context: Optional[Dict] = None,
    grok_context: Optional[str] = None,
    grok_macro: Optional[Dict] = None,
) -> str:
    """Build the analysis prompt for Claude."""
    symbol = signal_data.get('symbol', 'UNKNOWN')
    direction = signal_data.get('direction', 'LONG')
    entry_price = signal_data.get('entry_price', 0)
    stop_loss = signal_data.get('stop_loss', 0)
    take_profit = signal_data.get('take_profit_1', signal_data.get('take_profit', 0))
    confidence = signal_data.get('confidence', 0)
    reasoning = signal_data.get('reasoning', '')
    change_24h = signal_data.get('24h_change', 0)
    volume_24h = signal_data.get('24h_volume', 0)
    is_parabolic = signal_data.get('is_parabolic_reversal', False)
    leverage = signal_data.get('leverage', 10)
    trade_type_label = signal_data.get('trade_type', 'STANDARD')

    # Calculate risk metrics
    if direction == 'LONG':
        sl_pct = ((entry_price - stop_loss) / entry_price) * 100 if entry_price > 0 else 0
        tp_pct = ((take_profit - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:
        sl_pct = ((stop_loss - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        tp_pct = ((entry_price - take_profit) / entry_price) * 100 if entry_price > 0 else 0

    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0

    # Build BTC short-term context
    btc_context = ""
    if market_context:
        btc_summary = market_context.get('btc_summary', '')
        btc_verdict = market_context.get('btc_verdict', 'NEUTRAL')
        if btc_summary:
            btc_context = btc_summary
            if direction == 'LONG' and btc_verdict in ('BEARISH', 'OVERBOUGHT'):
                btc_context += f"\n⚠️ WARNING: BTC 5m is {btc_verdict} — be very selective with LONG entries."
            elif direction == 'SHORT' and btc_verdict in ('BULLISH', 'OVERSOLD'):
                btc_context += f"\n⚠️ WARNING: BTC 5m is {btc_verdict} — be very selective with SHORT entries."

    # Build Grok macro context string
    macro_bias = ""
    macro_summary = ""
    if grok_macro:
        macro_bias = grok_macro.get('bias', 'NEUTRAL')
        macro_summary = grok_macro.get('summary', '')
        if macro_bias == 'BEARISH' and direction == 'LONG':
            btc_context += f"\n⚠️ GROK MACRO ALERT: Current macro is BEARISH — extra caution on LONG entries."
        elif macro_bias == 'BULLISH' and direction == 'SHORT':
            btc_context += f"\n⚠️ GROK MACRO ALERT: Current macro is BULLISH — extra caution on SHORT entries."

    trade_type = "PARABOLIC REVERSAL SHORT" if is_parabolic else f"{direction}"

    # Append past trade lessons for this direction/type to inform decision
    lessons_context = ""
    try:
        from app.services.ai_trade_learner import format_lessons_for_ai_prompt
        lessons_context = format_lessons_for_ai_prompt(
            trade_type=trade_type_label,
            direction=direction,
            symbol=symbol
        )
    except Exception:
        pass

    # Inject live system performance context
    live_context = ""
    try:
        from app.services.ai_trade_learner import get_live_trading_context
        live_context = get_live_trading_context()
    except Exception:
        pass

    # Per-coin trade history — most impactful context for grok-4's decision
    coin_history = ""
    try:
        from app.services.ai_trade_learner import get_coin_trade_history
        coin_history = get_coin_trade_history(symbol=symbol, direction=direction, limit=10)
    except Exception:
        pass

    return f"""You are a professional crypto trading analyst. Analyze this trade signal and decide if it should be executed.

SIGNAL DETAILS:
- Symbol: {symbol}
- Direction: {trade_type}
- Entry: ${entry_price:.6f}
- Stop Loss: ${stop_loss:.6f} ({sl_pct:.2f}% risk)
- Take Profit: ${take_profit:.6f} ({tp_pct:.2f}% target)
- Risk/Reward: {rr_ratio:.2f}:1
- Leverage: {leverage}x
- 24h Change: {change_24h:+.1f}%
- 24h Volume: ${volume_24h:,.0f}
- Signal Confidence: {confidence}%
- Technical Reasoning: {reasoning}

MARKET CONTEXT:
{btc_context if btc_context else "No BTC context available."}
{live_context}
{coin_history}
{lessons_context}

GROK INTELLIGENCE (real-time world & crypto awareness):
Macro environment ({macro_bias if macro_bias else "UNKNOWN"}): {macro_summary if macro_summary else "No macro data available."}
Coin-specific intel: {grok_context if grok_context else "No coin-specific intelligence available."}

STRATEGY RULES:
- LONGS: Enter early momentum (0-12% pumps), TP at 67%, SL at 65% @ 20x
- SHORTS: Mean reversion on 35%+ gainers, target pullback
- PARABOLIC: Aggressive shorts on 50%+ exhausted pumps, 200% TP @ 20x

Analyze this signal considering:
1. Is the entry timing good? (not chasing, not too early)
2. Is the risk/reward acceptable?
3. Does the technical setup support this trade?
4. Is BTC's short-term momentum ALIGNED with this trade direction? A LONG during a BTC bearish 15m phase or a SHORT during a BTC bullish 15m phase is a significant red flag — lower confidence or reject unless the coin's setup is exceptionally strong.
5. Does the X/Twitter sentiment support or contradict this trade?
6. Would you take this trade with real money?
7. What is the single most important reason to take or skip this trade?

Respond in JSON format only:
{{
    "approved": true or false,
    "confidence": 1-10 (how confident you are in this trade),
    "recommendation": "STRONG BUY" or "BUY" or "HOLD" or "AVOID",
    "reasoning": "2-3 sentence plain English explanation for traders",
    "why_this_trade": "1-2 sentence plain English explanation of WHY this specific trade is being taken right now - focus on the key edge/catalyst. Make it actionable and easy for non-technical traders to understand.",
    "risks": ["list", "of", "key", "risks"],
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR"
}}"""


async def analyze_signal_with_ai(
    signal_data: Dict,
    market_context: Optional[Dict] = None
) -> Dict:
    """
    Use Claude to analyze a trading signal and decide if it should be broadcast.
    
    Args:
        signal_data: The signal candidate with all technical data
        market_context: Optional BTC/market data for correlation analysis
    
    Returns:
        {
            'approved': True/False,
            'confidence': 1-10,
            'reasoning': 'Plain English explanation',
            'risks': ['risk1', 'risk2'],
            'recommendation': 'STRONG BUY / BUY / HOLD / AVOID'
        }
    """
    try:
        client = get_anthropic_client()
        if not client:
            raise ValueError("Claude client not available")

        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'LONG')

        # Run all three Grok checks in parallel — macro, coin intel, chart vision
        grok_macro_result, coin_intel, chart_vision = await asyncio.gather(
            get_cached_grok_macro(),
            get_grok_coin_intelligence(symbol, direction),
            get_grok_chart_vision(symbol, direction),
            return_exceptions=True,
        )
        if isinstance(grok_macro_result, Exception):
            grok_macro_result = {}
        if isinstance(coin_intel, Exception):
            coin_intel = {"summary": "", "hard_no": False, "hard_no_reason": ""}
        if isinstance(chart_vision, Exception):
            chart_vision = {"chart_verdict": "NEUTRAL", "pattern": "", "visual_analysis": "", "chart_ok": False}

        # Grok hard veto — block immediately without spending Claude tokens
        if isinstance(coin_intel, dict) and coin_intel.get('hard_no'):
            reason = coin_intel.get('hard_no_reason', 'Serious red flag detected by AI analysis')
            logger.warning(f"🚨 GROK HARD VETO: {symbol} {direction} blocked — {reason}")
            return {
                'approved': False,
                'confidence': 1,
                'recommendation': 'AVOID',
                'reasoning': f"AI analysis flagged a serious risk: {reason}",
                'why_this_trade': '',
                'risks': [reason],
                'entry_quality': 'POOR',
            }

        coin_summary = coin_intel.get('summary', '') if isinstance(coin_intel, dict) else ''
        momentum_verdict = coin_intel.get('momentum_verdict', 'NEUTRAL') if isinstance(coin_intel, dict) else 'NEUTRAL'
        momentum_fading = coin_intel.get('momentum_fading', False) if isinstance(coin_intel, dict) else False

        # Momentum freshness gate — scalps need RISING or NEUTRAL momentum
        # A FADING momentum signal means the move is already over on X
        if momentum_fading:
            logger.warning(
                f"📉 Grok: {symbol} momentum FADING on X — "
                f"scalp entry too late, social move already peaked"
            )
            return {
                'approved': False,
                'confidence': 2,
                'recommendation': 'AVOID',
                'reasoning': (
                    f"AI analysis: {symbol} social momentum is FADING — "
                    f"the move has already peaked, scalp entry is too late."
                ),
                'why_this_trade': '',
                'risks': ['Social momentum fading — scalp entry window closed'],
                'entry_quality': 'POOR',
            }

        # Enrich coin summary with momentum verdict for Claude context
        momentum_icon = {"RISING": "📈", "NEUTRAL": "➡️"}.get(momentum_verdict, "")
        if coin_summary and momentum_verdict != "NEUTRAL":
            coin_summary = f"[X Momentum: {momentum_icon} {momentum_verdict}] {coin_summary}"

        # Chart vision gate — block if chart actively works against the trade direction
        chart_verdict = chart_vision.get('chart_verdict', 'NEUTRAL') if isinstance(chart_vision, dict) else 'NEUTRAL'
        chart_pattern = chart_vision.get('pattern', '') if isinstance(chart_vision, dict) else ''
        chart_analysis = chart_vision.get('visual_analysis', '') if isinstance(chart_vision, dict) else ''
        chart_ok = chart_vision.get('chart_ok', False) if isinstance(chart_vision, dict) else False

        if chart_ok and chart_verdict == 'AGAINST':
            logger.warning(
                f"📉 Grok chart vision: {symbol} chart AGAINST {direction} — "
                f"pattern={chart_pattern}"
            )
            return {
                'approved': False,
                'confidence': 2,
                'recommendation': 'AVOID',
                'reasoning': (
                    f"AI chart analysis: {symbol} 5m chart is working "
                    f"AGAINST a {direction} entry. Pattern: {chart_pattern}. {chart_analysis}"
                ),
                'why_this_trade': '',
                'risks': [f'Chart pattern works against {direction}: {chart_pattern}'],
                'entry_quality': 'POOR',
            }

        # Enrich coin summary with chart context for Claude
        if chart_ok and chart_verdict != 'NEUTRAL':
            chart_icon = "✅" if f"CONFIRMS_{direction}" in chart_verdict else "⚠️"
            chart_note = (
                f" | Chart: {chart_icon} {chart_verdict} ({chart_pattern})"
                f"{' — ' + chart_analysis[:80] if chart_analysis else ''}"
            )
            coin_summary = (coin_summary + chart_note).strip()

        # Build the prompt with Grok macro + coin intel + chart vision injected
        prompt = build_signal_prompt(
            signal_data,
            market_context,
            grok_context=coin_summary,
            grok_macro=grok_macro_result if isinstance(grok_macro_result, dict) else {},
        )

        macro_bias = grok_macro_result.get('bias', 'NEUTRAL') if isinstance(grok_macro_result, dict) else 'NEUTRAL'
        logger.info(
            f"🤖 Grok analyzing {symbol} {direction} | "
            f"macro={macro_bias} | momentum={momentum_verdict} | "
            f"chart={chart_verdict} | coin_intel={'✅' if coin_summary else '⬜'}"
        )

        # ── Grok-4 final approval ─────────────────────────────────────────────
        grok_client = _get_grok_client()
        if not grok_client:
            raise ValueError("No XAI_API_KEY — Grok final approval unavailable")

        full_prompt = (
            "You are a professional crypto scalp trading analyst. "
            "Be concise and decisive. Respond in valid JSON only.\n\n"
            + prompt
        )
        grok_response = await asyncio.wait_for(
            grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=350,
                temperature=0.2,
            ),
            timeout=45.0,
        )
        result_text = (grok_response.choices[0].message.content or "{}").strip()

        # Parse JSON from response
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            result = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning(f"Grok returned non-JSON: {result_text[:200]}")
            raise

        # Ensure all required fields exist
        result.setdefault('approved', False)
        result.setdefault('confidence', 5)
        result.setdefault('recommendation', 'HOLD')
        result.setdefault('reasoning', 'Unable to analyze signal.')
        result.setdefault('why_this_trade', '')
        result.setdefault('risks', [])
        result.setdefault('entry_quality', 'FAIR')

        logger.info(f"🤖 Grok Final: {symbol} {direction}: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'} ({result['recommendation']})")
        logger.info(f"   Reasoning: {result['reasoning']}")

        # ── Macro-adaptive TP/SL adjustment ──────────────────────────────────
        # Apply AFTER approval so only approved signals get adjusted
        if result.get('approved') and macro_bias in ('BULLISH', 'BEARISH'):
            entry = signal_data.get('entry_price', 0)
            tp1 = signal_data.get('take_profit_1', signal_data.get('take_profit', 0))
            sl  = signal_data.get('stop_loss', 0)

            # Alignment = macro matches direction (BULLISH+LONG or BEARISH+SHORT)
            aligned = (macro_bias == 'BULLISH' and direction == 'LONG') or \
                      (macro_bias == 'BEARISH' and direction == 'SHORT')

            if aligned and entry > 0 and tp1 > 0:
                # Macro confirms direction — widen TP 20% to let winner run
                if direction == 'LONG':
                    tp_dist = tp1 - entry
                    new_tp = entry + tp_dist * 1.20
                else:
                    tp_dist = entry - tp1
                    new_tp = entry - tp_dist * 1.20
                signal_data['take_profit_1'] = round(new_tp, 8)
                if 'take_profit' in signal_data:
                    signal_data['take_profit'] = round(new_tp, 8)
                result['macro_tp_sl_note'] = f"Macro {macro_bias} confirms {direction} — TP widened +20%"
                logger.info(f"📊 Macro TP widened: {tp1:.6f} → {new_tp:.6f} ({macro_bias} + {direction})")

            elif not aligned and entry > 0 and tp1 > 0 and sl > 0:
                # Macro opposes direction — tighten TP 20% (take profits sooner), SL 15%
                if direction == 'LONG':
                    tp_dist = tp1 - entry
                    new_tp = entry + tp_dist * 0.80
                    sl_dist = entry - sl
                    new_sl = entry - sl_dist * 0.85
                else:
                    tp_dist = entry - tp1
                    new_tp = entry - tp_dist * 0.80
                    sl_dist = sl - entry
                    new_sl = entry + sl_dist * 0.85
                signal_data['take_profit_1'] = round(new_tp, 8)
                if 'take_profit' in signal_data:
                    signal_data['take_profit'] = round(new_tp, 8)
                signal_data['stop_loss'] = round(new_sl, 8)
                result['macro_tp_sl_note'] = f"Macro {macro_bias} opposes {direction} — TP tightened -20%, SL tightened -15%"
                logger.info(f"📊 Macro TP/SL tightened: TP {tp1:.6f}→{new_tp:.6f}, SL {sl:.6f}→{new_sl:.6f} ({macro_bias} vs {direction})")

        return result

    except Exception as e:
        logger.error(f"Grok Signal Filter error: {e}")
        return {
            'approved': True,
            'confidence': 5,
            'recommendation': 'BUY',
            'reasoning': f'AI analysis unavailable, proceeding with technical signals. (Error: {str(e)[:50]})',
            'why_this_trade': '',
            'risks': ['AI analysis failed'],
            'entry_quality': 'UNKNOWN'
        }


def _ema(values: list, period: int) -> float:
    """Calculate EMA from a list of floats."""
    if len(values) < period:
        return values[-1] if values else 0.0
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(closes: list, period: int = 14) -> float:
    """Calculate RSI from a list of close prices."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── BTC State Cache ───────────────────────────────────────────────────────────
_btc_state_cache: Optional[Dict] = None
_btc_state_last_fetch: Optional[datetime] = None
_BTC_STATE_TTL_SECONDS = 180  # refresh every 3 minutes


async def _fetch_btc_klines(interval: str, limit: int) -> list:
    import aiohttp
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=6)) as r:
            return await r.json()


async def get_btc_state() -> Dict:
    """
    Comprehensive BTC analysis across 5m and 1h timeframes.
    Cached for 3 minutes. Returns block_longs / block_shorts flags
    that scanners use to gate altcoin signals.
    """
    global _btc_state_cache, _btc_state_last_fetch

    now = datetime.now()
    if (
        _btc_state_cache is not None
        and _btc_state_last_fetch is not None
        and (now - _btc_state_last_fetch).total_seconds() < _BTC_STATE_TTL_SECONDS
    ):
        return _btc_state_cache

    try:
        async def _fetch_btc_orderbook() -> dict:
            import aiohttp
            url = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=100"
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=6)) as r:
                    return await r.json()

        klines_5m, klines_1h, ob_raw = await asyncio.gather(
            _fetch_btc_klines("5m", 60),
            _fetch_btc_klines("1h", 50),
            _fetch_btc_orderbook(),
            return_exceptions=True
        )

        def _ema_list(closes, period):
            if len(closes) < period:
                return closes[-1] if closes else 0
            k = 2 / (period + 1)
            ema = sum(closes[:period]) / period
            for p in closes[period:]:
                ema = p * k + ema * (1 - k)
            return ema

        # ── 5m analysis ──
        c5 = [float(k[4]) for k in klines_5m] if isinstance(klines_5m, list) else []
        rsi_5m = _rsi(c5) if len(c5) >= 15 else 50.0
        ema8_5m = _ema_list(c5, 8) if len(c5) >= 8 else (c5[-1] if c5 else 0)
        ema21_5m = _ema_list(c5, 21) if len(c5) >= 21 else (c5[-1] if c5 else 0)
        price_5m = c5[-1] if c5 else 0
        mom_5m = sum(1 if c5[i] > c5[i-1] else -1 for i in range(-4, 0)) if len(c5) >= 5 else 0
        change_5m = ((price_5m - c5[-12]) / c5[-12] * 100) if len(c5) >= 12 and c5[-12] > 0 else 0

        bullish_5m = price_5m > ema8_5m > ema21_5m and rsi_5m > 50 and mom_5m >= 2
        bearish_5m = price_5m < ema8_5m < ema21_5m and rsi_5m < 50 and mom_5m <= -2

        # ── 1h analysis ──
        c1h = [float(k[4]) for k in klines_1h] if isinstance(klines_1h, list) else []
        h1h = [float(k[2]) for k in klines_1h] if isinstance(klines_1h, list) else []
        l1h = [float(k[3]) for k in klines_1h] if isinstance(klines_1h, list) else []
        rsi_1h = _rsi(c1h) if len(c1h) >= 15 else 50.0
        ema20_1h = _ema_list(c1h, 20) if len(c1h) >= 20 else (c1h[-1] if c1h else 0)
        ema50_1h = _ema_list(c1h, 50) if len(c1h) >= 50 else (c1h[-1] if c1h else 0)
        price_1h = c1h[-1] if c1h else 0
        mom_1h = sum(1 if c1h[i] > c1h[i-1] else -1 for i in range(-3, 0)) if len(c1h) >= 4 else 0
        change_1h = ((price_1h - c1h[-4]) / c1h[-4] * 100) if len(c1h) >= 4 and c1h[-4] > 0 else 0

        bullish_1h = price_1h > ema20_1h and rsi_1h > 48 and mom_1h >= 1
        bearish_1h = price_1h < ema20_1h and rsi_1h < 52 and mom_1h <= -1

        # ── S/R levels from 1h pivot highs/lows ──
        sr_levels = {'resistances': [], 'supports': [], 'nearest_resistance_pct': None, 'nearest_support_pct': None}
        at_resistance = False
        at_support = False
        if len(h1h) >= 10 and price_5m > 0:
            price_range = max(h1h) - min(l1h)
            cluster_threshold = price_range * 0.008
            pivot_highs, pivot_lows = [], []
            for i in range(2, len(h1h) - 2):
                if h1h[i] >= h1h[i-1] and h1h[i] >= h1h[i-2] and h1h[i] >= h1h[i+1] and h1h[i] >= h1h[i+2]:
                    pivot_highs.append(h1h[i])
                if l1h[i] <= l1h[i-1] and l1h[i] <= l1h[i-2] and l1h[i] <= l1h[i+1] and l1h[i] <= l1h[i+2]:
                    pivot_lows.append(l1h[i])
            all_levels = sorted(pivot_highs + pivot_lows)
            clusters = []
            if all_levels:
                cur = [all_levels[0]]
                for lv in all_levels[1:]:
                    if lv - cur[-1] <= cluster_threshold:
                        cur.append(lv)
                    else:
                        clusters.append(sum(cur) / len(cur))
                        cur = [lv]
                clusters.append(sum(cur) / len(cur))
            resistances = sorted([l for l in clusters if l > price_5m])
            supports = sorted([l for l in clusters if l < price_5m], reverse=True)
            sr_levels['resistances'] = resistances[:3]
            sr_levels['supports'] = supports[:3]
            if resistances:
                pct = (resistances[0] - price_5m) / price_5m * 100
                sr_levels['nearest_resistance_pct'] = round(pct, 2)
                at_resistance = pct < 0.8  # within 0.8% of resistance = danger zone for longs
            if supports:
                pct = (price_5m - supports[0]) / price_5m * 100
                sr_levels['nearest_support_pct'] = round(pct, 2)
                at_support = pct < 0.8  # within 0.8% of support = good zone for longs

        # ── Order book wall detection ──
        ob_signal = 'NEUTRAL'
        ask_wall_near = False
        bid_wall_near = False
        ob_detail = 'no data'
        if isinstance(ob_raw, dict) and price_5m > 0:
            bids = [[float(b[0]), float(b[1])] for b in ob_raw.get('bids', [])]
            asks = [[float(a[0]), float(a[1])] for a in ob_raw.get('asks', [])]
            # Total volume in top 20 levels
            bid_vol_top = sum(b[1] for b in bids[:20])
            ask_vol_top = sum(a[1] for a in asks[:20])
            total_vol = bid_vol_top + ask_vol_top
            bid_ratio = bid_vol_top / total_vol * 100 if total_vol > 0 else 50
            # Detect walls: levels with > 3x average size within 1.5% of price
            if asks:
                avg_ask_size = sum(a[1] for a in asks[:50]) / len(asks[:50])
                near_asks = [a for a in asks if a[0] <= price_5m * 1.015]
                ask_wall_near = any(a[1] >= avg_ask_size * 3 for a in near_asks)
            if bids:
                avg_bid_size = sum(b[1] for b in bids[:50]) / len(bids[:50])
                near_bids = [b for b in bids if b[0] >= price_5m * 0.985]
                bid_wall_near = any(b[1] >= avg_bid_size * 3 for b in near_bids)
            if bid_ratio >= 60 and not ask_wall_near:
                ob_signal = 'BUY_HEAVY'
            elif bid_ratio <= 40 and not bid_wall_near:
                ob_signal = 'SELL_HEAVY'
            else:
                ob_signal = 'BALANCED'
            ob_detail = (
                f"bid_ratio={bid_ratio:.0f}% | "
                f"ask_wall_near={ask_wall_near} | bid_wall_near={bid_wall_near} | "
                f"signal={ob_signal}"
            )
            logger.info(f"📗 BTC ORDER BOOK → {ob_detail}")

        if sr_levels['nearest_resistance_pct'] is not None:
            logger.info(
                f"📐 BTC S/R → nearest resistance {sr_levels['nearest_resistance_pct']:+.2f}% | "
                f"nearest support {sr_levels.get('nearest_support_pct', 'N/A')} % | "
                f"at_resistance={at_resistance} at_support={at_support}"
            )

        # ── Combined verdict ──
        if bullish_5m and bullish_1h:
            verdict = "STRONGLY_BULLISH"
        elif bearish_5m and bearish_1h:
            verdict = "STRONGLY_BEARISH"
        elif rsi_5m >= 72 and rsi_1h >= 65:
            verdict = "OVERBOUGHT"
        elif rsi_5m <= 28 and rsi_1h <= 35:
            verdict = "OVERSOLD"
        elif bullish_5m or bullish_1h:
            verdict = "BULLISH"
        elif bearish_5m or bearish_1h:
            verdict = "BEARISH"
        else:
            verdict = "NEUTRAL"

        # ── Gate logic: base + S/R + order book overlay ──
        block_longs = verdict in ("STRONGLY_BEARISH", "OVERBOUGHT")
        block_shorts = verdict in ("STRONGLY_BULLISH", "OVERSOLD")

        # S/R overlap check: if price is simultaneously at resistance AND support, BTC is
        # in a tight consolidation/squeeze range — neither direction is clearly blocked.
        # Applying both blocks would incorrectly kill all signals during ranging markets.
        both_sr = at_resistance and at_support

        # Resistance wall: block longs only when NOT simultaneously at support (not ranging)
        if not block_longs and at_resistance and not both_sr and (ask_wall_near or ob_signal == 'SELL_HEAVY'):
            block_longs = True
            verdict = verdict + "+AT_RESISTANCE"
            logger.info(f"🧱 BTC LONG BLOCKED: price at resistance with ask wall / sell-heavy book")

        # Support wall: block shorts only when NOT simultaneously at resistance (not ranging)
        if not block_shorts and at_support and not both_sr and (bid_wall_near or ob_signal == 'BUY_HEAVY'):
            block_shorts = True
            verdict = verdict + "+AT_SUPPORT"
            logger.info(f"🧱 BTC SHORT BLOCKED: price at support with bid wall / buy-heavy book")

        if both_sr:
            verdict = verdict + "+RANGING"
            logger.info(f"📊 BTC at both S/R simultaneously — tight range/squeeze, no S/R blocks applied")

        state = {
            'verdict': verdict,
            'block_longs': block_longs,
            'block_shorts': block_shorts,
            'rsi_5m': rsi_5m,
            'rsi_1h': rsi_1h,
            'change_5m': change_5m,
            'change_1h': change_1h,
            'bullish_5m': bullish_5m,
            'bullish_1h': bullish_1h,
            'bearish_5m': bearish_5m,
            'bearish_1h': bearish_1h,
            'price': price_5m,
            'at_resistance': at_resistance,
            'at_support': at_support,
            'ask_wall_near': ask_wall_near,
            'bid_wall_near': bid_wall_near,
            'ob_signal': ob_signal,
            'resistances': sr_levels['resistances'],
            'supports': sr_levels['supports'],
            'nearest_resistance_pct': sr_levels['nearest_resistance_pct'],
            'nearest_support_pct': sr_levels['nearest_support_pct'],
            'summary': (
                f"BTC verdict: {verdict} | "
                f"5m RSI={rsi_5m:.0f} | 1h RSI={rsi_1h:.0f} | "
                f"S/R: res={sr_levels['nearest_resistance_pct']}% sup={sr_levels['nearest_support_pct']}% | "
                f"OB: {ob_signal} ask_wall={ask_wall_near} bid_wall={bid_wall_near} | "
                f"blocks: longs={block_longs} shorts={block_shorts}"
            ),
        }

        _btc_state_cache = state
        _btc_state_last_fetch = now
        logger.info(f"📊 BTC STATE → {state['summary']}")
        return state

    except Exception as e:
        logger.warning(f"BTC state fetch failed: {e}")
        fallback = {
            'verdict': 'NEUTRAL', 'block_longs': False, 'block_shorts': False,
            'rsi_5m': 50, 'rsi_1h': 50, 'change_5m': 0, 'change_1h': 0,
            'ob_signal': 'NEUTRAL', 'at_resistance': False, 'at_support': False,
            'ask_wall_near': False, 'bid_wall_near': False,
            'summary': 'BTC state unavailable — allowing all signals',
        }
        if _btc_state_cache:
            return _btc_state_cache
        return fallback


async def get_btc_context() -> Optional[Dict]:
    """
    Get BTC short-term context: 15m RSI, 8/21 EMA trend, recent candle direction.
    Returns a structured dict with a human-readable 'summary' and a 'verdict'.
    """
    try:
        import aiohttp
        url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=50"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                klines = await resp.json()

        if not klines or len(klines) < 5:
            return None

        closes = [float(k[4]) for k in klines]
        current_price = closes[-1]

        rsi_val = _rsi(closes)
        ema8 = _ema(closes, 8)
        ema21 = _ema(closes, 21)

        last3 = closes[-4:]
        candle_changes = [(last3[i] - last3[i - 1]) / last3[i - 1] * 100 for i in range(1, 4)]
        recent_direction = sum(1 if c > 0 else -1 for c in candle_changes)

        above_ema8 = current_price > ema8
        above_ema21 = current_price > ema21
        ema_bullish = above_ema8 and above_ema21
        ema_bearish = not above_ema8 and not above_ema21

        if ema_bullish and rsi_val > 50 and recent_direction >= 1:
            verdict = "BULLISH"
        elif ema_bearish and rsi_val < 50 and recent_direction <= -1:
            verdict = "BEARISH"
        elif rsi_val > 70:
            verdict = "OVERBOUGHT"
        elif rsi_val < 30:
            verdict = "OVERSOLD"
        else:
            verdict = "NEUTRAL"

        change_str = ", ".join(f"{c:+.2f}%" for c in candle_changes)
        summary = (
            f"BTC 5m: RSI {rsi_val:.0f} | "
            f"{'above' if above_ema8 else 'below'} 8EMA, "
            f"{'above' if above_ema21 else 'below'} 21EMA | "
            f"Last 3 candles: {change_str} | "
            f"Verdict: {verdict}"
        )

        logger.info(f"📊 BTC short-term → {summary}")
        return {
            'btc_price': current_price,
            'btc_rsi_15m': rsi_val,
            'btc_ema8': ema8,
            'btc_ema21': ema21,
            'btc_verdict': verdict,
            'btc_summary': summary,
        }

    except Exception as e:
        logger.warning(f"Could not fetch BTC short-term context: {e}")
        return None


def format_ai_analysis_for_signal(ai_result: Dict) -> str:
    """Format AI analysis for inclusion in signal message (hidden - returns empty)."""
    # AI analysis is internal only - don't show in signal output
    return ""


# Minimum confidence to approve a signal
MIN_AI_CONFIDENCE = 8


async def should_broadcast_signal(signal_data: Dict) -> tuple[bool, str]:
    """
    Main entry point - check if signal should be broadcast.

    Returns:
        (should_broadcast: bool, ai_analysis_text: str)
        ai_analysis_text contains the WHY THIS TRADE explanation when approved.
    """
    # Short-circuit: very high confidence signals skip Claude to reduce API costs
    pre_score = signal_data.get('confidence', 0)
    if isinstance(pre_score, (int, float)) and pre_score >= 90:
        logger.info(f"⚡ Signal pre-score {pre_score}% — auto-approved, skipping Claude")
        return True, ""

    # Get market context
    btc_context = await get_btc_context()

    # Analyze with Claude
    ai_result = await analyze_signal_with_ai(signal_data, btc_context)

    # Decision logic
    approved = ai_result.get('approved', False)
    confidence = ai_result.get('confidence', 0)

    # Require both approval AND minimum confidence
    should_broadcast = approved and confidence >= MIN_AI_CONFIDENCE

    if should_broadcast:
        why = ai_result.get('why_this_trade', '').strip()
        if why:
            analysis_text = f"\n<b>💡 Why this trade:</b> <i>{why}</i>\n"
        else:
            analysis_text = ""
    else:
        rejection_reason = ai_result.get('reasoning', 'Did not meet quality standards')
        risks = ai_result.get('risks', [])
        logger.info(f"🚫 Signal REJECTED by Claude: {rejection_reason}")
        if risks:
            logger.info(f"   Risks: {', '.join(risks)}")
        analysis_text = ""

    return should_broadcast, analysis_text
