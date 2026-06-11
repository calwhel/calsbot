"""cTrader live-order latency instrumentation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderLatency:
    """Monotonic timestamps for one live order attempt."""

    execution_id: int
    signal_mono: float
    queued_mono: Optional[float] = None
    dequeue_mono: Optional[float] = None
    submitted_mono: Optional[float] = None
    broker_ack_mono: Optional[float] = None
    fill_mono: Optional[float] = None

    def mark_queued(self) -> None:
        self.queued_mono = time.monotonic()

    def mark_dequeue(self) -> None:
        self.dequeue_mono = time.monotonic()

    def mark_submitted(self) -> None:
        self.submitted_mono = time.monotonic()

    def mark_broker_ack(self) -> None:
        self.broker_ack_mono = time.monotonic()

    def mark_fill(self) -> None:
        self.fill_mono = time.monotonic()

    @staticmethod
    def _ms(start: Optional[float], end: Optional[float]) -> int:
        if start is None or end is None:
            return -1
        return max(0, int((end - start) * 1000))

    def log_summary(self, *, outcome: str = "done") -> None:
        end = self.fill_mono or self.broker_ack_mono or time.monotonic()
        logger.info(
            "[order-latency] exec=%s signal→queued=%sms queued→submitted=%sms "
            "submitted→broker_ack=%sms ack→fill_event=%sms total=%sms outcome=%s",
            self.execution_id,
            self._ms(self.signal_mono, self.queued_mono),
            self._ms(self.queued_mono or self.dequeue_mono, self.submitted_mono),
            self._ms(self.submitted_mono, self.broker_ack_mono),
            self._ms(self.broker_ack_mono, self.fill_mono),
            self._ms(self.signal_mono, end),
            outcome,
        )


def new_order_latency(execution_id: int, signal_mono: Optional[float] = None) -> OrderLatency:
    return OrderLatency(
        execution_id=int(execution_id),
        signal_mono=float(signal_mono if signal_mono is not None else time.monotonic()),
    )
