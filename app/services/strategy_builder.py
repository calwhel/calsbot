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

  SMA (Simple Moving Average — price vs level, or cross)
    {"type":"indicator","name":"sma","timeframe":"1h","period":200,"source":"close","condition":"price_above"}
    {"type":"indicator","name":"sma","timeframe":"1h","period":200,"source":"close","condition":"price_below"}
    {"type":"indicator","name":"sma","timeframe":"1h","period":50,"source":"close","condition":"bullish_cross","period2":200}
    {"type":"indicator","name":"sma","timeframe":"1h","period":50,"source":"close","condition":"bearish_cross","period2":200}
    conditions: price_above | price_below | bullish_cross | bearish_cross | above_high | below_low | inside_band
    source options: close | high | low  (default: close)
    → Use "price_above" to filter longs above a key SMA (e.g. 200 SMA trend filter).
    → Use "bullish_cross" / "bearish_cross" for golden cross / death cross setups (requires period2).
    → Use name "sma_ribbon" for multi-SMA alignment: {"type":"indicator","name":"sma_ribbon","periods":[20,50,100,200],"condition":"aligned_bullish"}

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
    {"type":"indicator","name":"adx","condition":"ranging"}       ← ADX < 25 (market is ranging, not trending)
    → Use "ranging" for Range Trader and mean-reversion strategies to avoid trending markets.

  ATR Expansion / Volatility
    {"type":"indicator","name":"atr_expansion","condition":"expanding","timeframe":"15m","multiplier":1.2}
    {"type":"indicator","name":"atr_expansion","condition":"contracting","timeframe":"15m","multiplier":1.2}
    conditions: expanding | contracting
    → Use "contracting" before a squeeze-breakout entry. Use "expanding" to confirm a move is underway.

  Keltner Channel
    {"type":"indicator","name":"keltner","condition":"squeeze","timeframe":"15m"}
    {"type":"indicator","name":"keltner","condition":"above_upper","timeframe":"15m"}
    {"type":"indicator","name":"keltner","condition":"below_lower","timeframe":"15m"}
    {"type":"indicator","name":"keltner","condition":"inside_bands","timeframe":"15m"}
    conditions: squeeze | above_upper | below_lower | inside_bands
    → "squeeze" = Bollinger Bands are inside Keltner Channel (high-probability breakout setup).
    → Use for Range Trader (squeeze) or Momentum styles (above_upper = strong uptrend).

  Williams %R
    {"type":"indicator","name":"williams_r","condition":"oversold","timeframe":"15m","period":14}
    conditions: oversold (< -80) | overbought (> -20)
    {"type":"indicator","name":"williams_r","operator":"lt","value":-80}

  CCI
    {"type":"indicator","name":"cci","condition":"oversold","timeframe":"15m","period":20}
    conditions: oversold (< -100) | overbought (> 100) | bullish (> 0) | bearish (< 0)
    {"type":"indicator","name":"cci","operator":"lt","value":-100}
    Optional MA smoothing — add "ma_type" and "ma_period" to smooth the CCI series before comparison:
    {"type":"indicator","name":"cci","condition":"bullish","period":20,"ma_type":"EMA","ma_period":3}
    ma_type: "SMA" | "EMA" | "SMMA" | "WMA" | "VWMA"    ma_period: integer (default 3)
    → Use "bullish"/"bearish" for Trend Magic / zero-cross strategies (CCI > 0 = trend up).
    → Use ma_type when PineScript applies an MA to CCI before comparing to threshold (smoothMagicTrend pattern).

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

── DAY-TRADER FILTERS (intraday volatility / volume / VWAP) ────────────────────
These are dedicated top-level types (NOT under "indicator"). Prefer them when the
user talks about intraday volatility gates, relative volume, or VWAP bands/bias.

  ATR Filter (volatility gate — only trade when there's enough range)
    {"type":"atr_filter","condition":"volatile","min_atr_pct":0.3,"period":14,"timeframe":"5m"}
    {"type":"atr_filter","condition":"expanding","period":14,"lookback":5,"timeframe":"5m"}
    conditions: volatile (ATR ≥ min_atr_pct % of price) | expanding (ATR rising vs lookback bars ago)
    min_atr_pct: minimum ATR as a % of price (default 0.3). Direction-neutral — use as a confirmation.
    → Kills dead-tape entries. Great confirmation for breakout / momentum day-trade setups.

  Relative Volume (RVOL — current bar volume vs its recent average)
    {"type":"rvol","condition":"high","threshold":1.5,"period":20,"timeframe":"5m"}
    conditions: high (RVOL ≥ threshold) | low (RVOL < threshold)
    threshold: RVOL multiple (default 1.5). period: bars to average (default 20). Direction-neutral.
    → "high" confirms real participation behind a move. Pairs with breakouts and ORB.

  VWAP Bands (session VWAP ± standard-deviation bands)
    {"type":"vwap_bands","condition":"below_lower","num_std":2.0,"timeframe":"5m"}
    conditions: below_lower (price ≤ VWAP − N·SD → oversold/LONG bias) |
                above_upper (price ≥ VWAP + N·SD → overbought/SHORT bias) |
                inside (price between the bands)
    num_std: band width in standard deviations (default 2.0).
    → Intraday mean-reversion: fade +2SD short, buy −2SD long. The classic VWAP-band scalp.

  VWAP Bias (directional filter — price above/below session VWAP)
    {"type":"vwap_bias","condition":"above","timeframe":"5m"}
    conditions: above (price > VWAP → LONG bias) | below (price < VWAP → SHORT bias)
    → Pure directional filter. "Longs only above VWAP" is the #1 intraday discipline rule.

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
Sub-conditions (`condition` field):
  gap_exists      → any qualifying FVG present (default)
  just_formed     → FVG completed in the last 1-2 bars (CREATION signal —
                    use this for "detect a strong impulsive move creating an FVG")
  price_in_gap    → current price is inside the gap zone (retest / mitigation)
  tap_and_reject  → last bar wicked into the gap and closed outside in the
                    FVG's directional bias (rejection / liquidity grab)
  approaching     → price within 0.5% of the gap edge
  gap_filled      → price has fully traded through the gap

Optional quality filters (omit or set to 0 = disabled — match legacy behaviour):
  min_gap_pct      → minimum width as % of mid price (e.g. 0.3)
  min_gap_atr_mult → minimum width × ATR(14)  ← volatility-aware sizing.
                     e.g. 2.5 = "FVG ≥ 2.5×ATR" (high imbalance)
  disp_atr_mult    → ICT displacement — formation candle body must be ≥
                     this × ATR (e.g. 1.5 = strong impulsive bar, no dojis)
  min_gap_usd      → absolute USD floor on gap width (e.g. 100)
  only_unfilled    → true → drop FVGs already touched
  max_age_bars     → only consider FVGs formed within the last N bars

Examples:
{"type":"fvg","direction":"bullish","condition":"price_in_gap","timeframe":"15m"}
{"type":"fvg","direction":"bearish","condition":"approaching","timeframe":"5m"}
{"type":"fvg","direction":"any","condition":"gap_exists","timeframe":"1h"}
{"type":"fvg","direction":"bearish","condition":"just_formed","timeframe":"5m",
 "min_gap_atr_mult":2.5,"disp_atr_mult":1.5,"only_unfilled":true}
{"type":"fvg","direction":"bearish","condition":"tap_and_reject","timeframe":"15m",
 "min_gap_atr_mult":2.0,"only_unfilled":true,"max_age_bars":20}

→ Triggers: "FVG fill" | "fair value gap" | "imbalance" | "price returning to gap"
            "FVG ≥ 2.5x ATR" | "high-imbalance FVG" | "ICT displacement creating FVG"
            "FVG just formed" | "tap into FVG and reject" | "rejection from FVG" | "FVG mitigation"

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
{"type":"price_relative","reference":"session_low","condition":"near","threshold_pct":2}
{"type":"price_relative","reference":"session_high","condition":"near","threshold_pct":1.5}
references: daily_open | session_high | session_low | weekly_open
conditions: above | below | near (within threshold_pct of the reference level)
→ "price above daily open" | "trading above yesterday's open" | "session high"
→ "price near session low" | "within 2% of session low" → use condition "near" + threshold_pct
→ "price near session high" → condition "near", reference "session_high"

── SENTIMENT ──────────────────────────────────────────────────────────────────
{"type":"sentiment","operator":"gt","value":60}
→ "high social sentiment" | "bullish sentiment" | "trending on social"

── LIQUIDATION ────────────────────────────────────────────────────────────────
{"type":"liquidation","direction":"below","tolerance_pct":2.0}
direction: below (long liquidations below price) | above (short liquidations above)
→ "near liquidation cluster" | "liquidity pool" | "liquidation magnet"

── INVERTED FAIR VALUE GAP (IFVG) ────────────────────────────────────────────
{"type":"ifvg","direction":"bullish","timeframe":"15m"}
direction: bullish | bearish | any
conditions (same as fvg): gap_exists | just_formed | price_in_gap | tap_and_reject | approaching | gap_filled
→ "IFVG" | "inverted FVG" | "price re-enters old gap" | "mitigated gap retest"

── FOREX — SESSION BREAK ─────────────────────────────────────────────────────
{"type":"forex_session_break","condition":"high_break","session":"asian","range_minutes":60,"timeframe":"15m"}
condition: high_break | low_break | either_break
session: asian | sydney | london | new_york
→ "London breakout" | "Asian range break" | "session high/low break"

── FOREX — PREVIOUS LEVEL ────────────────────────────────────────────────────
{"type":"forex_prev_level","condition":"sweep_pdh","timeframe":"15m"}
conditions: sweep_pdh | sweep_pdl | break_pdh | break_pdl | at_pdh | at_pdl
(pdh=previous-day-high, pdl=previous-day-low)
→ "sweep previous day high" | "break above PDH" | "previous day low liquidity"

── FOREX — CURRENCY STRENGTH ─────────────────────────────────────────────────
{"type":"forex_currency_strength","window":"4h","min_diff":0.6,"direction":"either"}
direction: either | base_strong | quote_strong
→ "currency strength" | "strong USD" | "weak GBP" | "AUD outperforming"

── FOREX — LIQUIDITY / PRICE ACTION ─────────────────────────────────────────
{"type":"forex_liquidity_pa","pattern":"sweep_eqh","timeframe":"15m","lookback":20,"tolerance_pips":3}
patterns: sweep_eqh | sweep_eql | stop_hunt_high | stop_hunt_low | equal_highs | equal_lows
→ "equal highs sweep" | "stop hunt" | "liquidity grab" | "SSL/BSL sweep"

── FOREX — NEWS AVOIDANCE ────────────────────────────────────────────────────
{"type":"forex_news_avoidance","minutes_before":30,"minutes_after":30,"min_impact":"high"}
min_impact: low | medium | high
→ "avoid news" | "no trade during NFP" | "high-impact news filter"

── FOREX — COT SENTIMENT ─────────────────────────────────────────────────────
{"type":"forex_cot","condition":"specs_extreme_long","extreme_pct":75,"lookback_weeks":52}
conditions: specs_extreme_long | specs_extreme_short | commercials_extreme_long | commercials_extreme_short
→ "COT report" | "commitment of traders" | "speculator positioning" | "institutional sentiment"

── ICT KILLZONE ───────────────────────────────────────────────────────────────
{"type":"fx_killzone","killzone":"london_kz"}
killzone options: london_kz (07:00–09:00 UTC) | ny_kz (12:00–14:00 UTC) | asian_kz (20:00–23:00 UTC) | any_kz
→ "killzone" | "London KZ" | "NY killzone" | "Asian open" | "ICT time window" | "high-probability window"

── ICT OTE — OPTIMAL TRADE ENTRY ─────────────────────────────────────────────
{"type":"fx_ote","direction":"bullish","swing_lookback":20,"fib_low":61.8,"fib_high":78.6,"timeframe":"15m"}
direction: bullish | bearish
fib_low/fib_high: Fibonacci retracement zone % (default 61.8–78.6 = golden pocket)
→ "OTE" | "optimal trade entry" | "golden zone" | "61.8 fib" | "78.6 fib" | "golden pocket" | "retracement entry"

── ICT DISPLACEMENT ──────────────────────────────────────────────────────────
{"type":"fx_displacement","direction":"bullish","min_body_ratio":3,"timeframe":"15m"}
direction: bullish | bearish | any
min_body_ratio: body must be ≥ N × average body size (default 3 = institutional candle)
→ "displacement" | "impulse candle" | "institutional candle" | "large body candle" | "displacement move"

── ICT EQUAL HIGHS / EQUAL LOWS ──────────────────────────────────────────────
{"type":"fx_equal_hl","type":"eqh","lookback":30,"tolerance_pips":3,"timeframe":"15m"}
type: eqh (equal highs) | eql (equal lows)
→ "equal highs" | "equal lows" | "EQH" | "EQL" | "double top liquidity" | "double bottom liquidity" | "BSL/SSL"

── ICT CISD — CHANGE IN STATE OF DELIVERY ────────────────────────────────────
{"type":"fx_cisd","direction":"bullish","max_run":10,"timeframe":"5m"}
direction: bullish (close back above the open of the last bearish run = sellers done, buyers in)
           bearish (close back below the open of the last bullish run = buyers done, sellers in)
max_run: max length of the opposing delivery run to scan (default 10)
→ "CISD" | "change in state of delivery" | "delivery flip" | "change of delivery" | "state of delivery shift"

── ICT BREAKER BLOCK ──────────────────────────────────────────────────────────
{"type":"fx_breaker","direction":"bullish","lookback":50,"tolerance_pct":0.5,"timeframe":"15m"}
direction: bullish (former supply → support) | bearish (former demand → resistance)
→ "breaker block" | "breaker" | "failed order block" | "broken OB returning" | "mitigation block"

── ICT PREMIUM / DISCOUNT ARRAY ──────────────────────────────────────────────
{"type":"fx_pd_array","bias":"discount","lookback":50,"timeframe":"1h"}
bias: discount (price below 50% of swing = buy zone) | premium (price above 50% = sell zone)
→ "premium zone" | "discount zone" | "PD array" | "premium/discount" | "below equilibrium" | "50% level" | "equilibrium"

── ICT JUDAS SWING ───────────────────────────────────────────────────────────
{"type":"fx_judas_swing","session":"london","swing_pips":10,"reversal_pips":5,"timeframe":"15m"}
session: london (08:00 UTC) | ny (13:30 UTC)
→ "Judas swing" | "fake move" | "false break" | "manipulation leg" | "stop hunt reversal" | "fake breakout then reversal"

── ICT SILVER BULLET ─────────────────────────────────────────────────────────
{"type":"fx_silver_bullet","window":"any"}
window: early_am (03:00–04:00 NY) | am (10:00–11:00 NY) | pm (15:00–16:00 NY) | any
→ "silver bullet" | "ICT silver bullet" | "3 AM setup" | "10 AM setup" | "3 PM setup" | "ICT precision entry"

── OPENING RANGE BREAKOUT (all asset classes) ────────────────────────────────
{"type":"opening_range_break","session_start":"london","orb_minutes":30,"direction":"both","timeframe":"5m"}
session_start: london (08:00 UTC) | ny (13:30 UTC) | asia (00:00 UTC) | midnight (00:00 UTC)
direction: up | down | both
orb_minutes: 5 | 15 | 30 | 60
→ "ORB" | "opening range breakout" | "first 30 minutes high/low" | "range break at open" | "opening range"

── VWAP CROSS (all asset classes) ────────────────────────────────────────────
{"type":"vwap_cross","direction":"cross_above","timeframe":"5m"}
direction: cross_above (bullish) | cross_below (bearish)
→ "VWAP cross" | "cross above VWAP" | "cross below VWAP" | "price crosses VWAP" | "VWAP momentum"

── STOCHASTIC OSCILLATOR (all asset classes) ────────────────────────────────
{"type":"stochastic","condition":"bullish_cross","k_period":14,"d_period":3,"timeframe":"15m"}
condition: oversold (<20) | overbought (>80) | bullish_cross (%K crosses above %D) | bearish_cross (%K crosses below %D)
→ "stochastic" | "stoch cross" | "%K %D" | "stoch oversold" | "stochastic oscillator" | "slow stochastic" | "stoch 14 3"

── ICT POWER OF 3 — PO3 (forex/index, session-based) ────────────────────────
{"type":"fx_po3","direction":"bullish","sweep_pips":5,"timeframe":"15m"}
direction: bullish (Asian low swept → distribution up) | bearish (Asian high swept → distribution down)
sweep_pips: minimum pip extension beyond Asian range to confirm manipulation (default 5)
→ "Power of 3" | "PO3" | "ICT PO3" | "accumulation manipulation distribution" | "AMD cycle"
→ "Asian range swept then reverses" | "Judas swing into distribution" | "fake break then real move"

── WYCKOFF PHASES (all asset classes) ───────────────────────────────────────
{"type":"wyckoff","phase":"spring","lookback":30,"timeframe":"1h"}
phase: spring (bullish — price wicks below support, closes back inside)
       upthrust (bearish — price wicks above resistance, closes back inside)
       test (low-volume re-test of spring/upthrust level — confirmation)
       markup (strong bullish close above midpoint, expanding volume)
       markdown (strong bearish close below midpoint, expanding volume)
→ "Wyckoff" | "spring" | "shakeout" | "upthrust" | "test of support" | "Wyckoff accumulation"
→ "markup phase" | "markdown phase" | "Wyckoff distribution" | "Wyckoff re-accumulation"
"""

STRATEGY_SCHEMA = """
{
  "version": "1.0",
  "name": "Strategy Name",
  "description": "Plain English description",
  "asset_class": "crypto",      // "crypto" | "forex" | "stock" | "index"
  "universe": {
    "type": "all",              // "all" | "specific"
    "symbols": [],              // ["SOLUSDT","ETHUSDT"] for crypto; ["EURUSD","GBPUSD"] for forex; ["AAPL","TSLA"] for stocks
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
    "take_profit_pct": 3.0,     // % TP for crypto/stocks; for forex prefer take_profit_pips
    "take_profit_pips": null,   // pip-based TP for forex (e.g. 30); set instead of take_profit_pct
    "take_profit2_pct": null,   // optional second TP, null to disable
    "stop_loss_pct": 1.5,       // % SL for crypto/stocks; for forex prefer stop_loss_pips
    "stop_loss_pips": null,     // pip-based SL for forex (e.g. 15); set instead of stop_loss_pct
    "trailing_stop": false,
    "trailing_stop_pct": null,
    "breakeven_at_pct": null    // move SL to entry when price is up this %
  },
  "risk": {
    "leverage": 10,             // for forex retail: 1–30 (regulatory limit); crypto: up to 100
    "position_size_pct": 5,
    "max_trades_per_day": 3,
    "max_open_positions": 1,
    "cooldown_minutes": 30,
    "daily_loss_limit_pct": 5
  },
  "filters": {
    "time_filter": null,        // null | {"start_hour":8,"end_hour":20}  (UTC)
    "btc_regime": null          // null | "bullish" | "bearish" | "neutral" — crypto only
  }
}
"""

COMPILER_SYSTEM_PROMPT = f"""You are an expert multi-market trading strategy compiler supporting crypto, forex, stocks, and indices.
Your ONLY job: translate a trader's natural-language description into a precise JSON config.

OUTPUT: Return ONLY valid JSON. No markdown fences, no explanation, no comments in JSON.

{STRATEGY_SCHEMA}

{CONDITION_SCHEMA}

=== COMPILATION RULES ===

ASSET CLASS (set "asset_class" field — CRITICAL)
  • Description mentions crypto / coins / BTC / ETH / altcoins / perpetuals → asset_class = "crypto"
  • Description mentions forex / FX / EUR / GBP / USD / JPY / pair / pips / London session / Asian session → asset_class = "forex"
  • Description mentions stocks / equities / shares / AAPL / TSLA / NASDAQ / NYSE → asset_class = "stock"
  • Description mentions indices / S&P / Dow / FTSE / DAX / Nasdaq index → asset_class = "index"
  • Default: "crypto" if no clear indicator

DIRECTION
  • User describes entering on pumps / overbought / shorts → direction = SHORT
  • User describes entering on dips / oversold / longs → direction = LONG
  • User says "both" or RSI-adaptive → direction = BOTH

STYLE PRESETS (override with explicit values if given)
  scalp     → max_trades_per_day 4–8, cooldown 15–30min, tp ≤3%, sl ≤1.5%, leverage 10–20 (crypto) / 5–10 (forex)
  swing     → max_trades_per_day 1–2, cooldown 4h+, tp 5–15%, sl 2–5%, leverage 3–8
  momentum  → max_trades_per_day 3–6, cooldown 20–40min, tp 3–6%, sl 1.5–2%, leverage 10–15
  reversal  → max_trades_per_day 2–4, cooldown 45–90min, tp 3–6%, sl 1.5–2.5%, leverage 8–12
  smc       → max_trades_per_day 2–4, cooldown 60–120min, tp 5–10%, sl 2–3%, leverage 5–10
  sniper    → max_trades_per_day 1–2, cooldown 2h, position_size 2–5%
  custom    → use explicit values from user; apply reasonable defaults for anything unspecified

FOREX-SPECIFIC RULES (apply when asset_class = "forex")
  • Always use take_profit_pips + stop_loss_pips instead of take_profit_pct / stop_loss_pct
    Default pip sizes: scalp 10–20 TP / 8–12 SL, swing 40–80 TP / 20–40 SL
  • Leverage: retail forex max is 30:1 — default to 10, never exceed 30
  • Universe: type="specific", symbols must be valid forex pairs like ["EURUSD","GBPUSD","XAUUSD"]
    Common pairs: EURUSD GBPUSD USDJPY AUDUSD USDCAD EURGBP EURJPY GBPJPY XAUUSD XAGUSD
  • btc_regime filter must be null (irrelevant for forex)
  • Session signals map naturally: London → forex_session_break session=london
    Asian range → forex_session_break session=asian
    NY reversal → forex_session_break session=new_york
  • "Liquidity grab" / "stop hunt" → forex_liquidity_pa
  • "PDH / PDL sweep" / "previous day" → forex_prev_level
  • "Currency strength" → forex_currency_strength
  • "Avoid news" / "NFP filter" → forex_news_avoidance
  • "COT" / "commitment of traders" → forex_cot
  • Instrument: "Gold" / "XAU" → XAUUSD, "Silver" / "XAG" → XAGUSD
  • If no specific pairs mentioned, default to ["EURUSD","GBPUSD"] for major-pair strategies
  ICT / Day-trading signal mappings (forex):
  • "killzone" / "London KZ" / "NY KZ" / "Asian open window" → fx_killzone
  • "OTE" / "optimal trade entry" / "golden zone" / "61.8–78.6 fib" → fx_ote
  • "displacement" / "impulse candle" / "institutional candle" → fx_displacement
  • "equal highs" / "EQH" / "equal lows" / "EQL" / "BSL" / "SSL" → fx_equal_hl
  • "breaker block" / "breaker" / "failed order block" → fx_breaker
  • "CISD" / "change in state of delivery" / "delivery flip" / "change of delivery" / "state of delivery shift" → fx_cisd
  • "premium zone" / "discount zone" / "PD array" / "equilibrium" → fx_pd_array
  • "Judas swing" / "fake move at open" / "manipulation leg" → fx_judas_swing
  • "silver bullet" / "ICT silver bullet" / "3 AM setup" / "10 AM" → fx_silver_bullet
  Cross-asset day-trading signal mappings:
  • "ORB" / "opening range breakout" / "first 30 min high/low" → opening_range_break
  • "VWAP cross" / "cross above VWAP" / "crosses VWAP" → vwap_cross
  • "stochastic" / "stoch cross" / "%K %D" / "stoch oversold/overbought" → stochastic
  • "Power of 3" / "PO3" / "ICT PO3" / "AMD cycle" / "accumulation manipulation distribution" → fx_po3
  • "Wyckoff" / "spring" / "shakeout" / "upthrust" / "markup phase" / "Wyckoff distribution" → wyckoff
  ICT-style forex templates — when the user describes these strategies, compose them:
  • "ICT day trade" / "killzone + OTE" → fx_killzone (primary) + fx_ote + fx_pd_array confirmations
  • "London ICT" → fx_killzone (london_kz) + fx_displacement + fvg confirmations
  • "Silver bullet strategy" → fx_silver_bullet + fvg conditions; tight TP 15–20 pips, SL 10–12 pips
  • "Judas swing fade" → fx_judas_swing + fx_pd_array; BOTH direction, TP 25–35 pips, SL 15 pips
  • "Breaker block entry" → fx_breaker + fx_killzone; TP 30–40 pips, SL 15 pips
  • "Gold ICT CISD" / "XAUUSD liq sweep + iFVG + MSS/CISD" → fx_killzone (london_kz or ny_kz) + forex_liquidity_pa (sweep_eqh/sweep_eql) + market_structure (choch/bos for MSS) + fx_cisd + ifvg (5m) confirmations on XAUUSD; TP 40–80 pips, SL 20–25 pips
  ICT risk profiles for forex:
  • ICT scalp (silver bullet / killzone): TP 15–25 pips, SL 10–15 pips, max 2 trades/day, cooldown 60 min
  • ICT intraday (OTE + displacement): TP 30–50 pips, SL 15–20 pips, max 2 trades/session, cooldown 90 min
  • ICT swing (breaker + PD array): TP 60–100 pips, SL 30–40 pips, max 1–2 trades/day, cooldown 4h

STOCK/INDEX-SPECIFIC RULES (apply when asset_class = "stock" or "index")
  • Use take_profit_pct / stop_loss_pct (not pips)
  • Leverage: stocks 1–5, indices 1–10
  • btc_regime must be null
  • Universe type="specific" with standard tickers for stocks (e.g. AAPL, TSLA, NVDA)
    For indices use: SPX, NDX, DJI, FTSE, DAX, NKY
  • If no specific symbols mentioned for stocks, default to ["AAPL","MSFT","NVDA","TSLA"]
  • If no specific symbols mentioned for indices, default to ["SPX","NDX"]

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
  "IFVG" / "inverted FVG" / "mitigated gap retest" → ifvg
  "hammer" / "pin bar" / "engulfing" / "doji" / "morning star" → candlestick
  "3 red candles" / "consecutive candles" → consecutive_candles
  "BOS" / "break of structure" → market_structure bos_bullish or bos_bearish
  "CHoCH" / "change of character" → market_structure choch_bullish or choch_bearish
  "order block" / "OB" → order_block
  "fib 61.8%" / "golden ratio" / "fibonacci" → fibonacci
  "RSI divergence" / "MACD divergence" → divergence
  "funding rate" → funding_rate (crypto only)
  "open interest" / "OI" → open_interest (crypto only)
  "London session" / "NY session" / "Asian session" → forex_session_break (forex) or session (crypto)
  "price above daily open" / "above session high" → price_relative
  "sentiment" / "social score" → sentiment
  "liquidation cluster" / "liquidity pool" → liquidation (crypto) or forex_liquidity_pa (forex)
  "London breakout" / "Asian range break" → forex_session_break
  "PDH sweep" / "previous day high" → forex_prev_level condition=sweep_pdh
  "stop hunt" / "equal highs sweep" → forex_liquidity_pa
  "currency strength" → forex_currency_strength
  "COT" / "commitment of traders" → forex_cot

RISK/REWARD
  • Never set stop_loss_pct > take_profit_pct (minimum 1:1 R:R)
  • For forex: never set stop_loss_pips > take_profit_pips (minimum 1:1 R:R)
  • Never set leverage > 25 for crypto unless user explicitly requests higher
  • Never set leverage > 30 for forex
  • For scalps with ≤3% TP, keep SL ≤ 2%

OPERATOR GROUPS
  Use AND for confirming setups (most strategies)
  Use OR for breakout/momentum screens (checking multiple coins for any signal)

UNIVERSE
  Crypto — If user mentions specific coins → type="specific", symbols=[...USDT format]
  Crypto — If user mentions "top gainers" / "movers" → min_24h_change=3 (or as specified)
  Crypto — If user mentions "mid-caps" / "altcoins" → exclude_slow_highcap=true (default)
  Crypto — If user mentions "BTC/ETH only" → type="specific", symbols=["BTCUSDT","ETHUSDT"]
  Forex  — Always type="specific", symbols=[valid forex pairs e.g. "EURUSD"]
  Stocks — Always type="specific", symbols=[valid tickers e.g. "AAPL"]
  Index  — Always type="specific", symbols=[index codes e.g. "SPX"]
  Default volume: 500000 USD/24h (crypto only)

FILTERS
  If user mentions time restriction → time_filter with UTC hours
  If user says "only in bull market" → btc_regime="bullish" (crypto only, null for forex/stocks)
  If user says "only in bear market" → btc_regime="bearish" (crypto only)

NEW FIELDS FROM CHAT BUILDER (parse these when present in the description)
  • "TP2: 4%" or "TP2: none" → exit.take_profit2_pct (float or null)
  • "TP2 Pips: 60" or "TP2 Pips: none" → exit.take_profit2_pips (int or null) — forex only
  • "Trailing Stop: true/false" → exit.trailing_stop (boolean)
  • "Breakeven: 70%" or "Breakeven: none" → exit.breakeven_at_pct (int 0–100 or null)
    Meaning: when this % of TP1 distance is covered, move SL to entry (e.g. 70 = move SL to entry after 70% of TP1 hit)
  • "Position Size: 3%" → risk.position_size_pct (float, e.g. 3.0)
  • "Max Trades/Day: 6" → risk.max_trades_per_day (int)
  • "Confirmation 1: ..." and "Confirmation 2: ..." → additional conditions in entry_conditions.conditions

  When Trailing Stop is true and TP2 is set: set trailing_stop_pct equal to half the stop_loss_pct (reasonable default).

COMPLEX / MULTI-CONDITION REQUESTS (decompose carefully — do NOT drop parts)
  • Treat a description as a CHECKLIST. Every distinct concept the user names ("sweep", "FVG",
    "MSS", "killzone", "RSI < 30", "above 200 EMA", "only London", "avoid news") must map to its
    own condition OR an explicit filter/field. Never silently merge two requirements into one, and
    never discard one because it's hard — pick the closest supported type from the lists above.
  • Sequenced/multi-leg setups ("A, THEN B, THEN entry on C") → emit each leg as a separate
    condition in entry_conditions.conditions, ordered as described, joined with AND. The platform
    treats AND-conditions as a confirmation stack, which models the sequence.
  • Honor the user's requested DEPTH. If they explicitly ask for many confirmations (e.g. a full
    ICT stack: killzone + liquidity sweep + MSS + FVG + CISD), include ALL of them rather than
    trimming to 2–3. Only collapse when conditions are genuine duplicates of the same concept.
  • When the user gives a named multi-part playbook that matches a template above (e.g. "Gold ICT
    CISD"), start from that template's composition and then layer on any extra specifics they add.
  • Map qualifiers to the right layer: timing words → time_filter/session, regime words →
    btc_regime (crypto), volatility/volume gates → atr_filter/rvol/volume_spike, news → news
    avoidance. Don't cram a filter into a price condition.
  • Resolve conflicts sensibly: if two requirements contradict (e.g. "scalp" but "swing TPs"),
    prefer the EXPLICIT numbers the user gave over the style preset, and keep R:R ≥ 1:1.

SELF-CHECK BEFORE EMITTING JSON (do this silently, then output only the JSON)
  1. Did I represent EVERY concept the user named (re-read their text, tick each off)?
  2. Is asset_class correct, and are TP/SL in the right unit (pips for forex, % otherwise)?
  3. Is R:R ≥ 1:1 and leverage within caps?
  4. Is there ≥1 entry condition and a plain-English description that matches what I built?
  5. Is the JSON strictly valid (no trailing commas, no comments, no markdown fences)?

ALWAYS INCLUDE
  • Reasonable defaults for any missing fields
  • A clear description field summarising the strategy in plain English
  • At least one entry condition
  • Correct asset_class field
  • If trailing_stop is true and trailing_stop_pct is not specified, default to stop_loss_pct / 2
  • If breakeven_at_pct is not specified and leverage > 8, default to 70 (move SL to entry after 70% of TP hit — protects leveraged trades)
"""


def _parse_json_response(raw: str) -> Optional[Dict]:
    """Strip markdown fences and parse JSON from an AI response."""
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(raw)


async def _compile_with_anthropic(user_description: str) -> Optional[Dict]:
    """Try to compile using Claude (Anthropic). Returns None on any failure."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=3000,
            system=COMPILER_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Compile this trading strategy into the JSON config format:\n\n{user_description}",
            }],
        )
        return _parse_json_response(response.content[0].text)
    except json.JSONDecodeError as e:
        logger.error(f"Anthropic compiler JSON parse error: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "credit balance" in err.lower() or "billing" in err.lower():
            logger.warning("Anthropic credits exhausted — will try fallback compiler")
        else:
            logger.error(f"Anthropic compiler error: {e}")
        return None


async def _compile_with_gemini(user_description: str) -> Optional[Dict]:
    """Fallback compiler using Gemini (free, already integrated). Returns None on any failure."""
    try:
        from google import genai as _genai
        import asyncio as _asyncio

        prompt = (
            f"{COMPILER_SYSTEM_PROMPT}\n\n"
            f"Compile this trading strategy into the JSON config format:\n\n{user_description}"
        )
        client = _genai.Client()
        # genai client is sync — run in executor to stay non-blocking
        loop = _asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            ),
        )
        return _parse_json_response(resp.text)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini compiler JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini compiler error: {e}")
        return None


async def compile_strategy_from_conversation(
    conversation: List[Dict[str, str]],
    user_description: str,
) -> Optional[Dict]:
    """
    Takes user description, returns compiled strategy config dict or None on failure.
    Tries Claude first, falls back to Gemini if Anthropic credits are exhausted.
    """
    result = await _compile_with_anthropic(user_description)
    if result is not None:
        return result
    logger.info("Anthropic unavailable — trying Gemini fallback compiler")
    return await _compile_with_gemini(user_description)


# ─────────────────────────────────────────────────────────────────────────────
# PineScript compiler
# ─────────────────────────────────────────────────────────────────────────────

PINESCRIPT_COMPILER_PROMPT = f"""You are an expert at reading TradingView PineScript code — both strategy() scripts and indicator() scripts — and translating their signal logic into a structured JSON strategy config.

You will receive a PineScript source file. It may be declared with indicator() OR strategy(). Both are valid input.

=== HOW TO HANDLE INDICATOR() SCRIPTS ===
Indicator scripts don't have strategy.entry/strategy.close calls. Instead, find the signals by:
1. Look for alertcondition() calls — these reveal the intended long/short signals. e.g. alertcondition(trendDirection == 1 ...) → LONG entry.
2. Look for ta.crossover / ta.crossunder on key values — these are the natural entry triggers.
3. Look for plotshape / plotarrow calls that mark signal points — these tell you when entries occur.
4. Look for how the script defines bullish vs bearish states (e.g. trendDirection > 0 = bullish).
5. Map the underlying math to the closest supported condition types (CCI, ATR/SuperTrend, EMA crossovers, etc).

Example: "Trend Magic" uses CCI(20) + ATR(5)×2.0 to build a trailing support/resistance line, then detects when price crosses above (bullish) or below (bearish). This maps to:
  - CCI oversold/overbought condition for the CCI component
  - SuperTrend condition (bullish_flip / bearish_flip) for the ATR-trailing-line crossover

=== YOUR TASK ===
1. Identify entry signals — from strategy.entry(), alertcondition(), crossovers, or plotshape() markers.
2. Map recognised indicators to the platform's supported condition types (full schema below).
3. Infer direction (LONG / SHORT / BOTH) from the entry logic.
4. Infer risk defaults (leverage, TP %, SL %) from strategy() params or sensible defaults.
5. Summarise what was mapped and any approximations in "_pine_notes".
6. Flag anything unsupported (e.g. security() multi-timeframe calls) in "_pine_warnings".
7. Ignore visual elements — plots, labels, colors, fills — focus only on signal logic.

{CONDITION_SCHEMA}

{STRATEGY_SCHEMA}

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences, no explanation outside the JSON.
Add two extra top-level fields:
  "_pine_notes": ["string", ...]    — what was mapped and how (plain English)
  "_pine_warnings": ["string", ...]  — anything unsupported or approximated

RULES:
- Always produce at least one entry condition, even if the script uses a fully custom formula. Map to the closest supported type and note the approximation.
- Never set stop_loss_pct > take_profit_pct.
- Default leverage to 10 unless script parameters indicate otherwise.
- Default direction to BOTH unless signals are clearly one-sided.
- Use timeframe from the script if specified; default to 15m if not.
- For custom composite indicators (e.g. Trend Magic, Hull Suite, Lux Algo signals): decompose into their underlying math (ATR, CCI, EMA, etc.) and map each component.
- When PineScript applies an MA (SMA/EMA/SMMA/WMA/VWMA) to CCI before threshold comparison (smoothMagicTrend pattern), set "ma_type" and "ma_period" on the CCI condition — do NOT emit a warning for this. Example: CCI(20) smoothed by EMA(3) → {{"type":"indicator","name":"cci","condition":"bullish","period":20,"ma_type":"EMA","ma_period":3}}
- For Trend Magic specifically: CCI > 0 = bullish trend, CCI < 0 = bearish trend. Use condition "bullish" or "bearish" (not oversold/overbought).
- _pine_notes examples: "Mapped CCI(20) crossover of 0 → cci bullish/bearish condition on 15m", "CCI(20) smoothed with EMA(3) → ma_type=EMA ma_period=3 on cci condition", "ATR trailing line crossover → supertrend bullish_flip/bearish_flip", "alertcondition Bullish Trend → LONG entry signal".
"""


async def _pine_compile_with_anthropic(pine_code: str) -> Optional[Dict]:
    """Try to compile PineScript using Claude. Returns None on any failure."""
    import asyncio as _asyncio
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = await _asyncio.wait_for(
            client.messages.create(
                model="claude-sonnet-4-5",   # faster + cheaper for structured extraction
                max_tokens=2500,
                system=PINESCRIPT_COMPILER_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        "Translate the following PineScript code into the JSON strategy config format. "
                        "Include _pine_notes and _pine_warnings fields.\n\n"
                        f"```pine\n{pine_code}\n```"
                    ),
                }],
            ),
            timeout=50,
        )
        return _parse_json_response(response.content[0].text)
    except _asyncio.TimeoutError:
        logger.warning("Anthropic PineScript compile timed out (50s) — trying Gemini")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Anthropic PineScript compiler JSON parse error: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "credit balance" in err.lower() or "billing" in err.lower():
            logger.warning("Anthropic credits exhausted — will try Gemini for PineScript compile")
        else:
            logger.error(f"Anthropic PineScript compiler error: {e}")
        return None


async def _pine_compile_with_gemini(pine_code: str) -> Optional[Dict]:
    """Fallback PineScript compiler using Gemini. Returns None on any failure."""
    import asyncio as _asyncio
    try:
        from google import genai as _genai

        prompt = (
            f"{PINESCRIPT_COMPILER_PROMPT}\n\n"
            "Translate the following PineScript code into the JSON strategy config format. "
            "Include _pine_notes and _pine_warnings fields.\n\n"
            f"```pine\n{pine_code}\n```"
        )
        client = _genai.Client()
        loop = _asyncio.get_event_loop()
        resp = await _asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                ),
            ),
            timeout=45,
        )
        return _parse_json_response(resp.text)
    except _asyncio.TimeoutError:
        logger.warning("Gemini PineScript compile timed out (45s)")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Gemini PineScript compiler JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini PineScript compiler error: {e}")
        return None


async def compile_from_pinescript(pine_code: str) -> Optional[Dict]:
    """
    Translate a PineScript indicator/strategy into a platform strategy config.
    Tries Claude Haiku first (fast, cheap), falls back to Gemini.
    Both have hard timeouts so the endpoint never hangs indefinitely.
    """
    result = await _pine_compile_with_anthropic(pine_code)
    if result is not None:
        return result
    logger.info("Claude unavailable/timed out — trying Gemini for PineScript compile")
    return await _pine_compile_with_gemini(pine_code)


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
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json_response(response.content[0].text)

    except Exception as e:
        logger.warning(f"Anthropic validation unavailable ({e}) — trying Gemini fallback")
        try:
            from google import genai as _genai
            import asyncio as _asyncio
            client2 = _genai.Client()
            loop = _asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: client2.models.generate_content(model="gemini-2.0-flash", contents=prompt),
            )
            return _parse_json_response(resp.text)
        except Exception as e2:
            logger.error(f"Strategy validation error (all providers): {e2}")
        return {
            "valid": True, "warnings": [], "suggestions": [],
            "summary": config.get("description", "Custom strategy"),
            "risk_rating": "MEDIUM",
        }


async def generate_strategy_summary(config: Dict) -> str:
    """Generate a short human-readable summary for the marketplace listing."""
    summary_prompt = (
        f"Summarise this trading strategy in 2 clear sentences for traders "
        f"browsing a marketplace. Be specific about the signal used:\n"
        f"{json.dumps(config, indent=2)}"
    )
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=200,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        pass
    try:
        from google import genai as _genai
        import asyncio as _asyncio
        client2 = _genai.Client()
        loop = _asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client2.models.generate_content(model="gemini-2.0-flash", contents=summary_prompt),
        )
        return resp.text.strip()
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
