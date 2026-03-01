import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_consecutive_losses = 0
_loss_circuit_breaker_until: Optional[datetime] = None
MAX_CONSECUTIVE_LOSSES = 3
CIRCUIT_BREAKER_HOURS = 4
CIRCUIT_BREAKER_ENABLED = False

def record_trade_result(won: bool):
    global _consecutive_losses, _loss_circuit_breaker_until
    if won:
        _consecutive_losses = 0
        logger.info("✅ Win recorded - loss streak reset")
    else:
        _consecutive_losses += 1
        logger.warning(f"❌ Loss #{_consecutive_losses} recorded")

def is_circuit_breaker_active() -> bool:
    return False
