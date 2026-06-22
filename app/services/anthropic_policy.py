"""
Anthropic spend policy — crypto vs forex.

Crypto always-on paths (scanners, auto-trader AI, X poster, AIGen, signal
filter, portal crypto chart reads) are OFF by default. Opt in with
ENABLE_CRYPTO_ANTHROPIC=1. DISABLE_CRYPTO_ANTHROPIC=1 always wins.

Forex / metals paths (gold_strategy_scanner, gold_ai_trader, forex pair
discovery, strategy builder compile for user requests) do NOT use this gate.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_CRYPTO_BLOCK_LOGGED: set[str] = set()

# Major FX / index / metals symbols that may use portal AI read — not crypto-gated.
_FOREX_METALS = frozenset({
    "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "NAS100", "US500", "US30", "UK100", "GER40",
})


def crypto_anthropic_enabled() -> bool:
    """True when crypto-side Anthropic calls are allowed."""
    if os.environ.get("DISABLE_CRYPTO_ANTHROPIC", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("ENABLE_CRYPTO_ANTHROPIC", "").lower() in ("1", "true", "yes")


def is_forex_or_metals_symbol(symbol: str) -> bool:
    """True for tradfi symbols that should not be blocked by the crypto gate."""
    if not symbol:
        return False
    s = re.sub(r"[^A-Z0-9]", "", symbol.upper())
    if s in _FOREX_METALS:
        return True
    if s.endswith("USD") and len(s) == 6 and not s.endswith("USDT"):
        return True
    return False


def portal_chart_ai_read_allowed(symbol: str) -> bool:
    """Portal /trade/ai_read — allow forex/metals; crypto follows crypto policy."""
    if is_forex_or_metals_symbol(symbol):
        return True
    return crypto_anthropic_enabled()


def log_crypto_anthropic_blocked(caller: str) -> None:
    if caller in _CRYPTO_BLOCK_LOGGED:
        logger.debug("[anthropic-policy] blocked %s (crypto Anthropic disabled)", caller)
        return
    _CRYPTO_BLOCK_LOGGED.add(caller)
    logger.info(
        "[anthropic-policy] Crypto Anthropic DISABLED — skipped %s "
        "(set ENABLE_CRYPTO_ANTHROPIC=1 to opt in; forex/gold scanners unaffected)",
        caller,
    )
