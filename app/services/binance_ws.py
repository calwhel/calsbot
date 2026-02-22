import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://fstream.binance.com/ws/!ticker@arr"
BINANCE_REST_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

_ticker_cache: Dict[str, Dict] = {}
_cache_timestamp: float = 0
_ws_connected: bool = False
_ws_task: Optional[asyncio.Task] = None
_shutting_down: bool = False
_reconnect_delay: float = 1.0
_max_reconnect_delay: float = 30.0
_stats = {
    'messages_received': 0,
    'reconnects': 0,
    'last_message_time': 0,
    'ws_started_at': 0,
    'rest_fallbacks': 0,
}


def get_ws_stats() -> Dict:
    return {
        'connected': _ws_connected,
        'symbols_cached': len(_ticker_cache),
        'messages_received': _stats['messages_received'],
        'reconnects': _stats['reconnects'],
        'rest_fallbacks': _stats['rest_fallbacks'],
        'last_update': _stats['last_message_time'],
        'uptime_seconds': time.time() - _stats['ws_started_at'] if _stats['ws_started_at'] else 0,
    }


def _process_ticker_message(data: list):
    global _cache_timestamp
    now = time.time()

    for ticker in data:
        symbol = ticker.get('s', '')
        if not symbol or not symbol.endswith('USDT'):
            continue

        _ticker_cache[symbol] = {
            'symbol': symbol,
            'lastPrice': ticker.get('c', '0'),
            'priceChangePercent': ticker.get('P', '0'),
            'quoteVolume': ticker.get('q', '0'),
            'highPrice': ticker.get('h', '0'),
            'lowPrice': ticker.get('l', '0'),
            'weightedAvgPrice': ticker.get('w', '0'),
            'openPrice': ticker.get('o', '0'),
            'volume': ticker.get('v', '0'),
            'count': ticker.get('n', 0),
            '_ws_time': now,
        }

    _cache_timestamp = now
    _stats['messages_received'] += 1
    _stats['last_message_time'] = now


async def _ws_listener():
    global _ws_connected, _reconnect_delay, _shutting_down

    _stats['ws_started_at'] = time.time()

    while not _shutting_down:
        try:
            logger.info("Connecting to Binance Futures WebSocket...")
            async with websockets.connect(
                BINANCE_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,
            ) as ws:
                _ws_connected = True
                _reconnect_delay = 1.0
                logger.info(f"Binance WebSocket connected")

                async for message in ws:
                    if _shutting_down:
                        break
                    try:
                        data = json.loads(message)
                        if isinstance(data, list):
                            _process_ticker_message(data)
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.debug(f"WS message processing error: {e}")

        except asyncio.CancelledError:
            logger.info("Binance WebSocket cancelled")
            break
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Binance WebSocket closed: {e}")
        except Exception as e:
            logger.warning(f"Binance WebSocket error: {e}")
        finally:
            _ws_connected = False
            _stats['reconnects'] += 1

        if _shutting_down:
            break

        stale_seconds = time.time() - _stats['last_message_time'] if _stats['last_message_time'] else 0
        if stale_seconds > 30:
            logger.warning(f"WS cache stale for {stale_seconds:.0f}s, scanners using REST fallback")

        logger.info(f"Reconnecting in {_reconnect_delay:.0f}s...")
        await asyncio.sleep(_reconnect_delay)
        _reconnect_delay = min(_reconnect_delay * 2, _max_reconnect_delay)


def start_binance_ws():
    global _ws_task, _shutting_down
    if _ws_task and not _ws_task.done():
        logger.debug("Binance WebSocket already running")
        return

    _shutting_down = False
    try:
        loop = asyncio.get_running_loop()
        _ws_task = loop.create_task(_ws_listener())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        _ws_task = loop.create_task(_ws_listener())
    logger.info("Binance WebSocket task started")


async def stop_binance_ws():
    global _ws_task, _shutting_down
    _shutting_down = True
    if _ws_task and not _ws_task.done():
        _ws_task.cancel()
        try:
            await _ws_task
        except asyncio.CancelledError:
            pass
    _ws_task = None
    logger.info("Binance WebSocket stopped")


def is_cache_fresh(max_age_seconds: float = 5.0) -> bool:
    if not _cache_timestamp:
        return False
    return (time.time() - _cache_timestamp) < max_age_seconds


def get_all_tickers(max_age: float = 10.0) -> Optional[List[Dict]]:
    if not _ticker_cache:
        return None
    if not is_cache_fresh(max_age):
        return None
    return list(_ticker_cache.values())


def get_ticker(symbol: str) -> Optional[Dict]:
    if symbol in _ticker_cache:
        entry = _ticker_cache[symbol]
        age = time.time() - entry.get('_ws_time', 0)
        if age < 10.0:
            return entry
    return None


async def get_all_tickers_with_fallback(http_client=None, max_age: float = 10.0) -> Optional[List[Dict]]:
    cached = get_all_tickers(max_age)
    if cached:
        return cached

    if http_client:
        try:
            _stats['rest_fallbacks'] += 1
            resp = await http_client.get(BINANCE_REST_URL, timeout=8)
            if resp.status_code == 200:
                tickers = resp.json()
                logger.debug(f"REST fallback: got {len(tickers)} tickers")
                return tickers
        except Exception as e:
            logger.error(f"REST fallback failed: {e}")

    return None
