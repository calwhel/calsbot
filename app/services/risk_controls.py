import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_consecutive_losses = 0
_loss_circuit_breaker_until: Optional[datetime] = None
MAX_CONSECUTIVE_LOSSES = 3
CIRCUIT_BREAKER_HOURS = 4

def record_trade_result(won: bool):
    global _consecutive_losses, _loss_circuit_breaker_until
    if won:
        _consecutive_losses = 0
        logger.info("✅ Win recorded - loss streak reset")
    else:
        _consecutive_losses += 1
        logger.warning(f"❌ Loss #{_consecutive_losses} recorded")
        if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            _loss_circuit_breaker_until = datetime.now() + timedelta(hours=CIRCUIT_BREAKER_HOURS)
            logger.warning(f"🛑 CIRCUIT BREAKER: {_consecutive_losses} consecutive losses! Pausing signals for {CIRCUIT_BREAKER_HOURS}h until {_loss_circuit_breaker_until}")

def is_circuit_breaker_active() -> bool:
    global _loss_circuit_breaker_until, _consecutive_losses
    if _loss_circuit_breaker_until and datetime.now() < _loss_circuit_breaker_until:
        remaining = (_loss_circuit_breaker_until - datetime.now()).total_seconds() / 60
        logger.info(f"🛑 Circuit breaker active - {remaining:.0f}min remaining ({_consecutive_losses} consecutive losses)")
        return True
    if _loss_circuit_breaker_until and datetime.now() >= _loss_circuit_breaker_until:
        _loss_circuit_breaker_until = None
        _consecutive_losses = 0
        logger.info("🟢 Circuit breaker expired - trading resumed")
    return False
