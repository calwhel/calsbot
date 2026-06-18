"""
Gold AI Trader — autonomous Claude-driven XAUUSD day-trading module.

Fully isolated from the strategy executor / builder / compiler. Feature-flagged
off by default (GOLD_AI_TRADER_ENABLED=false).
"""

from app.gold_ai_trader.config import gold_ai_trader_enabled

__all__ = ["gold_ai_trader_enabled"]
