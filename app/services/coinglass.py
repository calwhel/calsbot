"""
CoinGlass API Integration - Derivatives data for enhanced signal quality.
Provides open interest, funding rates, long/short ratios, and liquidation data.
Uses CoinGlass API V4: https://open-api-v4.coinglass.com
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"

_cache: Dict[str, Dict] = {}
CACHE_TTL = 120


def _get_api_key() -> Optional[str]:
    return os.environ.get("COINGLASS_API_KEY")


def _get_cache(key: str) -> Optional[Any]:
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry['ts'] < CACHE_TTL:
            return entry['data']
        del _cache[key]
    return None


def _set_cache(key: str, data: Any) -> None:
    _cache[key] = {'data': data, 'ts': time.time()}


def _to_pair(symbol: str) -> str:
    s = symbol.upper()
    if not s.endswith('USDT'):
        s = s + 'USDT'
    return s


def _strip_usdt(symbol: str) -> str:
    s = symbol.upper()
    for suffix in ['USDT', 'USD', 'PERP']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s


async def _api_request(endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
    api_key = _get_api_key()
    if not api_key:
        logger.warning("CoinGlass API key not configured")
        return None

    url = f"{COINGLASS_BASE_URL}{endpoint}"
    headers = {
        "accept": "application/json",
        "CG-API-KEY": api_key
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=headers, params=params or {})

            if response.status_code == 429:
                logger.warning("CoinGlass rate limit hit, waiting 2s")
                await asyncio.sleep(2)
                response = await client.get(url, headers=headers, params=params or {})

            if response.status_code != 200:
                logger.warning(f"CoinGlass {endpoint} error {response.status_code}")
                return None

            data = response.json()
            code = str(data.get('code', ''))
            if code != '0':
                logger.warning(f"CoinGlass {endpoint} API error code={code}: {data.get('msg', '')}")
                return None

            return data.get('data')

    except Exception as e:
        logger.error(f"CoinGlass {endpoint} request failed: {e}")
        return None


async def get_funding_rate(symbol: str) -> Optional[Dict]:
    coin = _strip_usdt(symbol)
    cache_key = f"funding_{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    data = await _api_request("/api/futures/funding-rate/exchange-list", {
        "symbol": _to_pair(symbol),
    })

    if not data or not isinstance(data, list):
        return None

    try:
        for coin_entry in data:
            entry_symbol = (coin_entry.get('symbol', '') or '').upper()
            if entry_symbol != coin:
                continue

            margin_list = coin_entry.get('stablecoin_margin_list', [])
            if not margin_list:
                continue

            binance_rate = None
            first_rate = None

            for exchange_entry in margin_list:
                ex_name = exchange_entry.get('exchange', '')
                rate = float(exchange_entry.get('funding_rate', 0) or 0)

                if not first_rate:
                    first_rate = {'rate': rate, 'exchange': ex_name}

                if ex_name.lower() == 'binance':
                    binance_rate = {'rate': rate, 'exchange': 'Binance'}
                    break

            chosen = binance_rate or first_rate
            if chosen:
                result = {
                    'symbol': coin,
                    'funding_rate': chosen['rate'],
                    'exchange': chosen['exchange'],
                }
                _set_cache(cache_key, result)
                return result

    except Exception as e:
        logger.error(f"Error parsing funding rate for {coin}: {e}")

    return None


async def get_open_interest_history(symbol: str) -> Optional[Dict]:
    coin = _strip_usdt(symbol)
    pair = _to_pair(symbol)
    cache_key = f"oi_hist_{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    data = await _api_request("/api/futures/open-interest/history", {
        "exchange": "Binance",
        "symbol": pair,
        "interval": "4h",
        "limit": 6,
    })

    if not data or not isinstance(data, list) or len(data) < 2:
        return None

    try:
        oi_values = []
        for item in data:
            close_val = item.get('close', 0)
            if close_val:
                oi_values.append(float(close_val))

        if len(oi_values) >= 2:
            current_oi = oi_values[-1]
            prev_oi = oi_values[0]
            oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0

            result = {
                'symbol': coin,
                'current_oi': current_oi,
                'prev_oi': prev_oi,
                'oi_usd': current_oi,
                'oi_change_pct': round(oi_change_pct, 2),
                'oi_trend': 'RISING' if oi_change_pct > 2 else 'FALLING' if oi_change_pct < -2 else 'STABLE',
            }
            _set_cache(cache_key, result)
            return result

    except Exception as e:
        logger.error(f"Error parsing OI history for {coin}: {e}")

    return None


async def get_long_short_ratio(symbol: str) -> Optional[Dict]:
    coin = _strip_usdt(symbol)
    pair = _to_pair(symbol)
    cache_key = f"lsr_{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    data = await _api_request("/api/futures/global-long-short-account-ratio/history", {
        "exchange": "Binance",
        "symbol": pair,
        "interval": "4h",
        "limit": 3,
    })

    if not data or not isinstance(data, list) or not data:
        data = await _api_request("/api/futures/top-long-short-account-ratio/history", {
            "exchange": "Binance",
            "symbol": pair,
            "interval": "4h",
            "limit": 3,
        })

    if not data or not isinstance(data, list) or not data:
        return None

    try:
        latest = data[-1]
        long_pct = float(latest.get('global_account_long_percent', latest.get('longRate', latest.get('top_account_long_percent', 0))) or 0)
        short_pct = float(latest.get('global_account_short_percent', latest.get('shortRate', latest.get('top_account_short_percent', 0))) or 0)

        if long_pct == 0 and short_pct == 0:
            ratio_val = float(latest.get('global_account_long_short_ratio', latest.get('longShortRatio', 1)) or 1)
            long_pct = round(ratio_val / (1 + ratio_val) * 100, 1)
            short_pct = round(100 - long_pct, 1)

        if long_pct > 0 or short_pct > 0:
            ratio = long_pct / short_pct if short_pct > 0 else 1.0
            result = {
                'symbol': coin,
                'long_pct': round(long_pct, 1),
                'short_pct': round(short_pct, 1),
                'ratio': round(ratio, 2),
                'bias': 'LONG HEAVY' if long_pct > 65 else 'SHORT HEAVY' if short_pct > 65 else 'BALANCED',
            }
            _set_cache(cache_key, result)
            return result

    except Exception as e:
        logger.error(f"Error parsing L/S ratio for {coin}: {e}")

    return None


async def get_liquidation_data(symbol: str) -> Optional[Dict]:
    coin = _strip_usdt(symbol)
    pair = _to_pair(symbol)
    cache_key = f"liq_{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    data = await _api_request("/api/futures/liquidation/coin-list", {
        "symbol": pair,
    })

    if data:
        try:
            if isinstance(data, dict):
                items = [data]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            for item in items:
                item_symbol = (item.get('symbol', '') or '').upper()
                if item_symbol != coin:
                    continue

                long_liq = float(item.get('long_liquidation_usd_24h', item.get('longLiquidationUsd', 0)) or 0)
                short_liq = float(item.get('short_liquidation_usd_24h', item.get('shortLiquidationUsd', 0)) or 0)
                total = long_liq + short_liq
                if total > 0:
                    result = {
                        'symbol': coin,
                        'long_liquidations_usd': long_liq,
                        'short_liquidations_usd': short_liq,
                        'total_liquidations_usd': total,
                        'dominant_side': 'LONGS liquidated' if long_liq > short_liq else 'SHORTS liquidated',
                    }
                    _set_cache(cache_key, result)
                    return result
        except Exception as e:
            logger.error(f"Error parsing liquidation coin-list for {coin}: {e}")

    hist_data = await _api_request("/api/futures/liquidation/history", {
        "exchange": "Binance",
        "symbol": pair,
        "interval": "4h",
        "limit": 6,
    })

    if hist_data and isinstance(hist_data, list):
        try:
            total_long = 0.0
            total_short = 0.0
            for item in hist_data:
                total_long += float(item.get('long_liquidation_usd', 0) or 0)
                total_short += float(item.get('short_liquidation_usd', 0) or 0)

            total = total_long + total_short
            if total > 0:
                result = {
                    'symbol': coin,
                    'long_liquidations_usd': total_long,
                    'short_liquidations_usd': total_short,
                    'total_liquidations_usd': total,
                    'dominant_side': 'LONGS liquidated' if total_long > total_short else 'SHORTS liquidated',
                }
                _set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.error(f"Error parsing liquidation history for {coin}: {e}")

    return None


async def get_derivatives_summary(symbol: str) -> Dict:
    coin = _strip_usdt(symbol)
    logger.info(f"📊 Fetching CoinGlass derivatives data for {coin}...")

    results = await asyncio.gather(
        get_funding_rate(symbol),
        get_open_interest_history(symbol),
        get_long_short_ratio(symbol),
        get_liquidation_data(symbol),
        return_exceptions=True
    )

    funding = results[0] if not isinstance(results[0], Exception) else None
    oi_hist = results[1] if not isinstance(results[1], Exception) else None
    lsr = results[2] if not isinstance(results[2], Exception) else None
    liq = results[3] if not isinstance(results[3], Exception) else None

    names = ["Funding", "OI History", "L/S", "Liquidation"]
    for i, name in enumerate(names):
        if isinstance(results[i], Exception):
            logger.warning(f"CoinGlass {name} error for {coin}: {results[i]}")

    summary: Dict = {
        'symbol': coin,
        'has_data': False,
        'funding_rate': None,
        'funding_rate_pct': None,
        'funding_bias': None,
        'oi_usd': None,
        'oi_change_pct': None,
        'oi_trend': None,
        'long_pct': None,
        'short_pct': None,
        'ls_ratio': None,
        'ls_bias': None,
        'long_liq_usd': None,
        'short_liq_usd': None,
        'liq_dominant': None,
    }

    if funding:
        rate = funding.get('funding_rate', 0)
        summary['funding_rate'] = rate
        rate_pct = rate * 100 if abs(rate) < 1 else rate
        summary['funding_rate_pct'] = round(rate_pct, 4)
        if rate_pct > 0.05:
            summary['funding_bias'] = 'LONGS PAYING (bearish pressure)'
        elif rate_pct > 0.01:
            summary['funding_bias'] = 'Slightly long-heavy'
        elif rate_pct < -0.05:
            summary['funding_bias'] = 'SHORTS PAYING (bullish pressure)'
        elif rate_pct < -0.01:
            summary['funding_bias'] = 'Shorts overleveraged'
        else:
            summary['funding_bias'] = 'Neutral'
        summary['has_data'] = True

    if oi_hist:
        summary['oi_usd'] = oi_hist.get('oi_usd', 0)
        summary['oi_change_pct'] = oi_hist.get('oi_change_pct', 0)
        summary['oi_trend'] = oi_hist.get('oi_trend', 'UNKNOWN')
        summary['has_data'] = True

    if lsr:
        summary['long_pct'] = lsr.get('long_pct')
        summary['short_pct'] = lsr.get('short_pct')
        summary['ls_ratio'] = lsr.get('ratio')
        summary['ls_bias'] = lsr.get('bias')
        summary['has_data'] = True

    if liq:
        summary['long_liq_usd'] = liq.get('long_liquidations_usd', 0)
        summary['short_liq_usd'] = liq.get('short_liquidations_usd', 0)
        summary['liq_dominant'] = liq.get('dominant_side')
        summary['has_data'] = True

    data_points = sum(1 for x in [funding, oi_hist, lsr, liq] if x)
    logger.info(f"📊 CoinGlass {coin}: {data_points}/4 data points retrieved")

    return summary


def format_derivatives_for_ai(deriv: Dict) -> str:
    if not deriv or not deriv.get('has_data'):
        return ""

    lines = [f"CoinGlass Derivatives Data for {deriv['symbol']}:"]

    if deriv.get('funding_rate_pct') is not None:
        lines.append(f"- Funding Rate: {deriv['funding_rate_pct']:.4f}% ({deriv.get('funding_bias', 'N/A')})")

    if deriv.get('oi_usd'):
        oi_display = f"${deriv['oi_usd']/1e6:.1f}M" if deriv['oi_usd'] >= 1e6 else f"${deriv['oi_usd']/1e3:.0f}K"
        lines.append(f"- Open Interest: {oi_display}")

    if deriv.get('oi_change_pct') is not None:
        lines.append(f"- OI Change (24h): {deriv['oi_change_pct']:+.1f}% ({deriv.get('oi_trend', 'N/A')})")

    if deriv.get('long_pct') is not None and deriv.get('short_pct') is not None:
        lines.append(f"- Long/Short: {deriv['long_pct']:.0f}%L / {deriv['short_pct']:.0f}%S (Ratio: {deriv.get('ls_ratio', 0):.2f}) - {deriv.get('ls_bias', 'N/A')}")

    if deriv.get('long_liq_usd') or deriv.get('short_liq_usd'):
        long_liq = deriv.get('long_liq_usd', 0)
        short_liq = deriv.get('short_liq_usd', 0)
        long_d = f"${long_liq/1e6:.1f}M" if long_liq >= 1e6 else f"${long_liq/1e3:.0f}K"
        short_d = f"${short_liq/1e6:.1f}M" if short_liq >= 1e6 else f"${short_liq/1e3:.0f}K"
        lines.append(f"- Recent Liquidations: Longs {long_d} / Shorts {short_d} ({deriv.get('liq_dominant', 'N/A')})")

    return "\n".join(lines)


def format_derivatives_for_message(deriv: Dict) -> str:
    if not deriv or not deriv.get('has_data'):
        return ""

    lines = ["<b>🔗 Derivatives Data (CoinGlass)</b>"]

    if deriv.get('funding_rate_pct') is not None:
        rate = deriv['funding_rate_pct']
        rate_emoji = "🔴" if rate > 0.03 else "🟢" if rate < -0.01 else "⚪"
        lines.append(f"{rate_emoji} Funding <b>{rate:+.4f}%</b> · {deriv.get('funding_bias', '')}")

    if deriv.get('oi_change_pct') is not None:
        oi_chg = deriv['oi_change_pct']
        oi_emoji = "📈" if oi_chg > 0 else "📉"
        oi_usd_display = ""
        if deriv.get('oi_usd'):
            oi_usd_display = f" (${deriv['oi_usd']/1e6:.1f}M)" if deriv['oi_usd'] >= 1e6 else f" (${deriv['oi_usd']/1e3:.0f}K)"
            oi_usd_display = f" · OI{oi_usd_display}"
        lines.append(f"{oi_emoji} OI Change <b>{oi_chg:+.1f}%</b>{oi_usd_display}")

    if deriv.get('long_pct') is not None:
        bias = deriv.get('ls_bias', 'BALANCED')
        bias_emoji = "🐂" if 'LONG' in bias else "🐻" if 'SHORT' in bias else "⚖️"
        lines.append(f"{bias_emoji} L/S Ratio <b>{deriv['long_pct']:.0f}%</b>L / <b>{deriv['short_pct']:.0f}%</b>S · {bias}")

    if deriv.get('long_liq_usd') or deriv.get('short_liq_usd'):
        long_liq = deriv.get('long_liq_usd', 0)
        short_liq = deriv.get('short_liq_usd', 0)
        long_d = f"${long_liq/1e6:.1f}M" if long_liq >= 1e6 else f"${long_liq/1e3:.0f}K"
        short_d = f"${short_liq/1e6:.1f}M" if short_liq >= 1e6 else f"${short_liq/1e3:.0f}K"
        lines.append(f"💥 Liquidations: Longs <b>{long_d}</b> / Shorts <b>{short_d}</b>")

    return "\n".join(lines)


def adjust_tp_sl_from_derivatives(
    direction: str,
    base_tp_pct: float,
    base_sl_pct: float,
    deriv: Dict,
) -> Dict:
    """
    Adjust TP/SL percentages using CoinGlass derivatives data.

    Logic for LONGS:
      - Shorts paying heavy funding → trend has fuel → widen TP
      - OI rising with price → strong momentum → widen TP
      - Crowd heavily long → crowded trade risk → tighten SL
      - Heavy short liquidations → squeeze momentum → widen TP
      - Heavy long liquidations → longs getting rekt → tighten TP, widen SL

    Logic for SHORTS (mirror):
      - Longs paying heavy funding → shorts have fuel → widen TP
      - OI falling → weakening trend → widen TP for shorts
      - Crowd heavily long → reversal potential → widen TP for shorts
      - Heavy long liquidations → cascade potential → widen TP for shorts

    Returns dict with adjusted tp_pct, sl_pct, and reasoning list.
    """
    if not deriv or not deriv.get('has_data'):
        return {
            'tp_pct': base_tp_pct,
            'sl_pct': base_sl_pct,
            'adjustments': [],
            'tp_change': 0.0,
            'sl_change': 0.0,
        }

    tp_mult = 1.0
    sl_mult = 1.0
    adjustments = []
    is_long = direction.upper() == 'LONG'

    funding_pct = deriv.get('funding_rate_pct')
    if funding_pct is not None:
        if is_long:
            if funding_pct < -0.1:
                boost = min(abs(funding_pct) * 0.5, 0.25)
                tp_mult += boost
                adjustments.append(f"Funding {funding_pct:+.3f}% (shorts paying) → TP +{boost*100:.0f}%")
            elif funding_pct > 0.1:
                penalty = min(funding_pct * 0.3, 0.15)
                tp_mult -= penalty
                sl_mult -= min(penalty * 0.5, 0.08)
                adjustments.append(f"Funding {funding_pct:+.3f}% (longs paying) → TP -{penalty*100:.0f}%, SL tighter")
        else:
            if funding_pct > 0.1:
                boost = min(funding_pct * 0.5, 0.25)
                tp_mult += boost
                adjustments.append(f"Funding {funding_pct:+.3f}% (longs paying) → TP +{boost*100:.0f}%")
            elif funding_pct < -0.1:
                penalty = min(abs(funding_pct) * 0.3, 0.15)
                tp_mult -= penalty
                sl_mult -= min(penalty * 0.5, 0.08)
                adjustments.append(f"Funding {funding_pct:+.3f}% (shorts paying) → TP -{penalty*100:.0f}%, SL tighter")

    oi_change = deriv.get('oi_change_pct')
    if oi_change is not None:
        if is_long:
            if oi_change > 5:
                boost = min(oi_change * 0.02, 0.15)
                tp_mult += boost
                adjustments.append(f"OI rising {oi_change:+.1f}% → strong momentum, TP +{boost*100:.0f}%")
            elif oi_change < -5:
                penalty = min(abs(oi_change) * 0.015, 0.10)
                tp_mult -= penalty
                adjustments.append(f"OI falling {oi_change:+.1f}% → weakening, TP -{penalty*100:.0f}%")
        else:
            if oi_change < -5:
                boost = min(abs(oi_change) * 0.02, 0.15)
                tp_mult += boost
                adjustments.append(f"OI falling {oi_change:+.1f}% → unwinding, TP +{boost*100:.0f}%")
            elif oi_change > 5:
                penalty = min(oi_change * 0.015, 0.10)
                sl_mult -= min(penalty, 0.08)
                adjustments.append(f"OI rising {oi_change:+.1f}% → counter-trend risk, SL tighter")

    long_pct = deriv.get('long_pct')
    short_pct = deriv.get('short_pct')
    if long_pct is not None and short_pct is not None:
        if is_long:
            if long_pct > 70:
                penalty = min((long_pct - 70) * 0.01, 0.10)
                sl_mult -= penalty
                adjustments.append(f"Crowd {long_pct:.0f}% long → crowded, SL tighter by {penalty*100:.0f}%")
            elif short_pct > 65:
                boost = min((short_pct - 65) * 0.015, 0.12)
                tp_mult += boost
                adjustments.append(f"Crowd {short_pct:.0f}% short → squeeze potential, TP +{boost*100:.0f}%")
        else:
            if long_pct > 70:
                boost = min((long_pct - 70) * 0.015, 0.12)
                tp_mult += boost
                adjustments.append(f"Crowd {long_pct:.0f}% long → reversal potential, TP +{boost*100:.0f}%")
            elif short_pct > 65:
                penalty = min((short_pct - 65) * 0.01, 0.10)
                sl_mult -= penalty
                adjustments.append(f"Crowd {short_pct:.0f}% short → crowded, SL tighter by {penalty*100:.0f}%")

    long_liq = deriv.get('long_liq_usd', 0) or 0
    short_liq = deriv.get('short_liq_usd', 0) or 0
    total_liq = long_liq + short_liq
    if total_liq > 100_000:
        liq_ratio = short_liq / total_liq if total_liq > 0 else 0.5
        if is_long:
            if liq_ratio > 0.65:
                boost = min((liq_ratio - 0.5) * 0.3, 0.12)
                tp_mult += boost
                adjustments.append(f"Short squeeze ({liq_ratio*100:.0f}% short liqs) → TP +{boost*100:.0f}%")
            elif liq_ratio < 0.35:
                penalty = min((0.5 - liq_ratio) * 0.2, 0.08)
                sl_mult -= penalty
                adjustments.append(f"Long liquidation pressure ({(1-liq_ratio)*100:.0f}% long liqs) → SL tighter")
        else:
            if liq_ratio < 0.35:
                boost = min((0.5 - liq_ratio) * 0.3, 0.12)
                tp_mult += boost
                adjustments.append(f"Long cascade ({(1-liq_ratio)*100:.0f}% long liqs) → TP +{boost*100:.0f}%")
            elif liq_ratio > 0.65:
                penalty = min((liq_ratio - 0.5) * 0.2, 0.08)
                sl_mult -= penalty
                adjustments.append(f"Short squeeze risk ({liq_ratio*100:.0f}% short liqs) → SL tighter")

    tp_mult = max(0.7, min(tp_mult, 1.5))
    sl_mult = max(0.75, min(sl_mult, 1.1))

    adj_tp = round(base_tp_pct * tp_mult, 1)
    adj_sl = round(base_sl_pct * sl_mult, 1)

    adj_tp = max(adj_tp, 3.0)
    adj_tp = min(adj_tp, 150.0)
    adj_sl = max(adj_sl, 2.0)
    adj_sl = min(adj_sl, 15.0)

    tp_change = adj_tp - base_tp_pct
    sl_change = adj_sl - base_sl_pct

    if adjustments:
        logger.info(f"📊 Derivatives TP/SL adjustment for {deriv.get('symbol', '?')} ({direction}): "
                     f"TP {base_tp_pct:.1f}%→{adj_tp:.1f}% | SL {base_sl_pct:.1f}%→{adj_sl:.1f}% | "
                     f"{len(adjustments)} factors")

    return {
        'tp_pct': adj_tp,
        'sl_pct': adj_sl,
        'adjustments': adjustments,
        'tp_change': round(tp_change, 1),
        'sl_change': round(sl_change, 1),
    }


_cascade_alert_cooldowns: Dict[str, float] = {}
CASCADE_COOLDOWN_HOURS = 6

async def detect_liquidation_cascade(symbol: str, social_buzz: Optional[Dict] = None) -> Optional[Dict]:
    """
    Detect potential liquidation cascade zones by combining CoinGlass liquidation/OI data
    with LunarCrush social panic signals.
    
    Triggers when:
    - Large liquidation volume (>$500K total in 24h)
    - Heavy one-sided liquidations (>65% one direction)
    - OI dropping sharply (unwinding)
    - Social buzz FALLING + sentiment DECLINING (panic)
    
    Returns alert dict or None if no cascade detected.
    """
    import time as _time
    
    cooldown_key = f"cascade_{_strip_usdt(symbol)}"
    last_alert = _cascade_alert_cooldowns.get(cooldown_key, 0)
    if _time.time() - last_alert < CASCADE_COOLDOWN_HOURS * 3600:
        return None
    
    deriv = await get_derivatives_summary(symbol)
    if not deriv or not deriv.get('has_data'):
        return None
    
    long_liq = deriv.get('long_liq_usd', 0) or 0
    short_liq = deriv.get('short_liq_usd', 0) or 0
    total_liq = long_liq + short_liq
    oi_change = deriv.get('oi_change_pct', 0) or 0
    funding_pct = deriv.get('funding_rate_pct', 0) or 0
    long_pct = deriv.get('long_pct', 50) or 50
    
    cascade_score = 0
    signals = []
    cascade_direction = None
    
    if total_liq >= 500_000:
        cascade_score += 2
        signals.append(f"Heavy liquidations ${total_liq/1e6:.1f}M")
    elif total_liq >= 200_000:
        cascade_score += 1
        signals.append(f"Elevated liquidations ${total_liq/1e3:.0f}K")
    
    if total_liq > 0:
        long_liq_ratio = long_liq / total_liq
        if long_liq_ratio > 0.65:
            cascade_score += 2
            cascade_direction = 'LONG_CASCADE'
            signals.append(f"Longs getting rekt ({long_liq_ratio*100:.0f}% long liqs)")
        elif long_liq_ratio < 0.35:
            cascade_score += 2
            cascade_direction = 'SHORT_SQUEEZE'
            signals.append(f"Short squeeze ({(1-long_liq_ratio)*100:.0f}% short liqs)")
    
    if oi_change < -8:
        cascade_score += 2
        signals.append(f"OI collapsing {oi_change:+.1f}% (mass unwinding)")
    elif oi_change < -4:
        cascade_score += 1
        signals.append(f"OI declining {oi_change:+.1f}%")
    
    if abs(funding_pct) > 0.08:
        cascade_score += 1
        if funding_pct > 0:
            signals.append(f"Extreme long funding {funding_pct:+.3f}%")
            if not cascade_direction:
                cascade_direction = 'LONG_CASCADE'
        else:
            signals.append(f"Extreme short funding {funding_pct:+.3f}%")
            if not cascade_direction:
                cascade_direction = 'SHORT_SQUEEZE'
    
    if long_pct > 72:
        cascade_score += 1
        signals.append(f"Overcrowded longs {long_pct:.0f}%")
        if not cascade_direction:
            cascade_direction = 'LONG_CASCADE'
    elif long_pct < 28:
        cascade_score += 1
        signals.append(f"Overcrowded shorts {100-long_pct:.0f}%")
        if not cascade_direction:
            cascade_direction = 'SHORT_SQUEEZE'
    
    social_panic = False
    if social_buzz and isinstance(social_buzz, dict):
        buzz_trend = social_buzz.get('trend', '')
        sent_trend = social_buzz.get('sentiment_trend', '')
        buzz_change = social_buzz.get('buzz_change_pct', 0) or 0
        
        if buzz_trend == 'FALLING' and sent_trend == 'DECLINING':
            cascade_score += 3
            social_panic = True
            signals.append(f"Social PANIC (buzz {buzz_change:+.0f}%, sentiment declining)")
        elif buzz_trend == 'FALLING':
            cascade_score += 1
            signals.append(f"Social buzz falling ({buzz_change:+.0f}%)")
        elif sent_trend == 'DECLINING':
            cascade_score += 1
            signals.append(f"Social sentiment declining")
    
    if cascade_score < 4:
        return None
    
    if cascade_score >= 8:
        severity = 'EXTREME'
    elif cascade_score >= 6:
        severity = 'HIGH'
    else:
        severity = 'MODERATE'
    
    if not cascade_direction:
        cascade_direction = 'UNKNOWN'
    
    _cascade_alert_cooldowns[cooldown_key] = _time.time()
    
    coin = _strip_usdt(symbol)
    logger.warning(f"⚠️ LIQUIDATION CASCADE ALERT [{severity}] {coin}: score={cascade_score}, "
                   f"direction={cascade_direction}, signals={len(signals)}")
    
    return {
        'symbol': coin,
        'severity': severity,
        'cascade_score': cascade_score,
        'cascade_direction': cascade_direction,
        'social_panic': social_panic,
        'signals': signals,
        'total_liquidations': total_liq,
        'long_liq_usd': long_liq,
        'short_liq_usd': short_liq,
        'oi_change_pct': oi_change,
        'funding_rate_pct': funding_pct,
        'long_pct': long_pct,
    }


def format_cascade_alert_message(alert: Dict) -> str:
    """Format a liquidation cascade alert for Telegram."""
    severity = alert.get('severity', 'MODERATE')
    symbol = alert.get('symbol', '?')
    direction = alert.get('cascade_direction', 'UNKNOWN')
    score = alert.get('cascade_score', 0)
    signals = alert.get('signals', [])
    social_panic = alert.get('social_panic', False)
    total_liq = alert.get('total_liquidations', 0)
    
    sev_icon = {'EXTREME': '🚨🚨🚨', 'HIGH': '🚨🚨', 'MODERATE': '🚨'}.get(severity, '🚨')
    dir_icon = {'LONG_CASCADE': '📉 LONGS GETTING LIQUIDATED', 'SHORT_SQUEEZE': '📈 SHORT SQUEEZE BUILDING', 'UNKNOWN': '⚠️ LIQUIDATION ACTIVITY'}.get(direction, '⚠️')
    
    liq_display = f"${total_liq/1e6:.1f}M" if total_liq >= 1e6 else f"${total_liq/1e3:.0f}K"
    
    msg = (
        f"{sev_icon} <b>LIQUIDATION CASCADE ALERT</b>\n\n"
        f"<b>${symbol}</b>\n"
        f"{dir_icon}\n\n"
        f"<b>Severity:</b> {severity} ({score}/10)\n"
        f"<b>Total Liquidations:</b> {liq_display}\n\n"
        f"<b>Warning Signals:</b>\n"
    )
    
    for s in signals:
        msg += f"  • {s}\n"
    
    if social_panic:
        msg += f"\n😱 <b>SOCIAL PANIC DETECTED</b> — Retail is fearful\n"
    
    if direction == 'LONG_CASCADE':
        msg += (
            f"\n<b>What this means:</b>\n"
            f"<i>Longs are being liquidated in a cascade. This can create a sharp price drop "
            f"followed by a potential bounce once selling pressure exhausts. "
            f"Consider waiting for reversal confirmation before entering LONG.</i>"
        )
    elif direction == 'SHORT_SQUEEZE':
        msg += (
            f"\n<b>What this means:</b>\n"
            f"<i>Shorts are being squeezed. Price may spike upward as shorts cover. "
            f"Dangerous to SHORT here. Consider waiting for the squeeze to exhaust "
            f"before entering SHORT.</i>"
        )
    else:
        msg += (
            f"\n<b>What this means:</b>\n"
            f"<i>Significant liquidation activity detected. Exercise caution with new positions "
            f"until volatility settles.</i>"
        )
    
    msg += f"\n\n<i>Data: CoinGlass + LunarCrush</i>"
    
    return msg


def calculate_signal_strength(signal_data: Dict) -> Dict:
    """
    Calculate a composite Signal Strength Score (1-10) based on how many
    data sources and confirmations align for a trade signal.
    
    Scoring breakdown:
    - Technical Analysis (RSI, volume, trend): 0-2 pts
    - Social Intelligence (Galaxy, sentiment, buzz): 0-2 pts
    - Influencer Consensus: 0-2 pts
    - Derivatives (funding, OI, L/S ratio): 0-2 pts
    - AI Confidence: 0-2 pts
    
    Returns dict with score, tier label, breakdown, and icon.
    """
    score = 0.0
    breakdown = []
    direction = signal_data.get('direction', 'LONG').upper()
    is_long = direction == 'LONG'
    
    rsi = signal_data.get('rsi', 50)
    vol_ratio = signal_data.get('volume_ratio', 1.0) or 1.0
    change_24h = signal_data.get('24h_change', signal_data.get('change_24h', 0)) or 0
    
    ta_score = 0.0
    if is_long:
        if 35 <= rsi <= 55:
            ta_score += 0.7
        elif 30 <= rsi <= 65:
            ta_score += 0.4
    else:
        if 55 <= rsi <= 75:
            ta_score += 0.7
        elif 45 <= rsi <= 80:
            ta_score += 0.4
    
    if vol_ratio >= 1.5:
        ta_score += 0.8
    elif vol_ratio >= 1.0:
        ta_score += 0.5
    elif vol_ratio >= 0.8:
        ta_score += 0.2
    
    if is_long and change_24h > 3:
        ta_score += 0.5
    elif not is_long and change_24h < -2:
        ta_score += 0.5
    elif is_long and change_24h > 0:
        ta_score += 0.2
    elif not is_long and change_24h < 0:
        ta_score += 0.2
    
    ta_score = min(ta_score, 2.0)
    if ta_score > 0:
        breakdown.append(f"TA {ta_score:.1f}/2")
    score += ta_score
    
    social_score = 0.0
    galaxy = signal_data.get('galaxy_score', 0) or 0
    sentiment = signal_data.get('sentiment', 0) or 0
    social_strength = signal_data.get('social_strength', 0) or 0
    
    if galaxy >= 14:
        social_score += 0.8
    elif galaxy >= 10:
        social_score += 0.5
    elif galaxy >= 7:
        social_score += 0.2
    
    sentiment_pct = sentiment * 100 if sentiment <= 1 else sentiment
    if is_long and sentiment_pct > 65:
        social_score += 0.6
    elif not is_long and sentiment_pct < 40:
        social_score += 0.6
    elif 40 <= sentiment_pct <= 65:
        social_score += 0.3
    
    if social_strength >= 70:
        social_score += 0.6
    elif social_strength >= 45:
        social_score += 0.3
    
    social_score = min(social_score, 2.0)
    if social_score > 0:
        breakdown.append(f"Social {social_score:.1f}/2")
    score += social_score
    
    inf_score = 0.0
    influencer = signal_data.get('influencer_consensus')
    if influencer and isinstance(influencer, dict):
        consensus = influencer.get('consensus', '')
        num_creators = influencer.get('num_creators', 0) or 0
        big_accounts = influencer.get('big_accounts', 0) or 0
        
        if is_long and consensus in ('BULLISH', 'LEAN BULLISH'):
            inf_score += 1.0
        elif not is_long and consensus in ('BEARISH', 'LEAN BEARISH'):
            inf_score += 1.0
        elif consensus == 'MIXED':
            inf_score += 0.3
        
        if num_creators >= 5:
            inf_score += 0.5
        elif num_creators >= 2:
            inf_score += 0.2
        
        if big_accounts >= 2:
            inf_score += 0.5
        elif big_accounts >= 1:
            inf_score += 0.3
    
    inf_score = min(inf_score, 2.0)
    if inf_score > 0:
        breakdown.append(f"Influencers {inf_score:.1f}/2")
    score += inf_score
    
    deriv_score = 0.0
    deriv = signal_data.get('derivatives')
    if deriv and isinstance(deriv, dict) and deriv.get('has_data'):
        funding = deriv.get('funding_rate_pct', 0) or 0
        oi_chg = deriv.get('oi_change_pct', 0) or 0
        l_pct = deriv.get('long_pct', 50) or 50
        
        if is_long and funding < -0.02:
            deriv_score += 0.6
        elif not is_long and funding > 0.02:
            deriv_score += 0.6
        elif abs(funding) < 0.01:
            deriv_score += 0.2
        
        if is_long and oi_chg > 3:
            deriv_score += 0.5
        elif not is_long and oi_chg < -3:
            deriv_score += 0.5
        elif abs(oi_chg) < 2:
            deriv_score += 0.2
        
        if is_long and l_pct < 55:
            deriv_score += 0.5
        elif not is_long and l_pct > 55:
            deriv_score += 0.5
        
        liq_dom = deriv.get('liq_dominant', '')
        if is_long and 'SHORTS' in str(liq_dom):
            deriv_score += 0.4
        elif not is_long and 'LONGS' in str(liq_dom):
            deriv_score += 0.4
    
    deriv_score = min(deriv_score, 2.0)
    if deriv_score > 0:
        breakdown.append(f"Derivatives {deriv_score:.1f}/2")
    score += deriv_score
    
    ai_score = 0.0
    ai_conf = signal_data.get('ai_confidence', 0) or 0
    ai_rec = signal_data.get('ai_recommendation', '')
    
    if ai_conf >= 8:
        ai_score += 1.5
    elif ai_conf >= 6:
        ai_score += 1.0
    elif ai_conf >= 4:
        ai_score += 0.5
    
    if ai_rec in ('STRONG BUY', 'STRONG SELL'):
        ai_score += 0.5
    elif ai_rec in ('BUY', 'SELL'):
        ai_score += 0.3
    
    ai_score = min(ai_score, 2.0)
    if ai_score > 0:
        breakdown.append(f"AI {ai_score:.1f}/2")
    score += ai_score
    
    buzz = signal_data.get('buzz_momentum')
    if buzz and isinstance(buzz, dict):
        buzz_trend = buzz.get('trend', '')
        if is_long and buzz_trend == 'RISING':
            score = min(score + 0.3, 10.0)
        elif not is_long and buzz_trend == 'FALLING':
            score = min(score + 0.3, 10.0)
    
    final_score = max(1, min(10, round(score)))
    
    if final_score >= 9:
        tier = 'ELITE'
        icon = '💎'
    elif final_score >= 7:
        tier = 'STRONG'
        icon = '🟢'
    elif final_score >= 5:
        tier = 'MODERATE'
        icon = '🟡'
    elif final_score >= 3:
        tier = 'WEAK'
        icon = '🟠'
    else:
        tier = 'LOW'
        icon = '🔴'
    
    confirmations = len(breakdown)
    
    return {
        'score': final_score,
        'raw_score': round(score, 1),
        'tier': tier,
        'icon': icon,
        'breakdown': breakdown,
        'confirmations': confirmations,
        'max_confirmations': 5,
    }


def format_signal_strength_line(strength: Dict) -> str:
    """Format signal strength as a single line for signal messages."""
    score = strength.get('score', 0)
    tier = strength.get('tier', 'LOW')
    icon = strength.get('icon', '🔴')
    confirmations = strength.get('confirmations', 0)
    max_conf = strength.get('max_confirmations', 5)
    
    filled = '█' * score
    empty = '░' * (10 - score)
    bar = f"{filled}{empty}"
    
    return f"{icon} Signal Strength <b>{score}/10</b> [{bar}] {tier} ({confirmations}/{max_conf} sources)"


def format_signal_strength_detail(strength: Dict) -> str:
    """Format signal strength with breakdown for signal messages."""
    line = format_signal_strength_line(strength)
    breakdown = strength.get('breakdown', [])
    if breakdown:
        line += "\n" + " · ".join(breakdown)
    return line
