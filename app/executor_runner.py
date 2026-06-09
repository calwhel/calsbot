"""
Dedicated strategy-executor process for Railway/production.

Gunicorn workers are killed with WORKER TIMEOUT when a forex+crypto scan cycle
runs longer than GUNICORN_TIMEOUT (even at 300s). That aborts the executor
mid-cycle — no cycle gates, no fires. This module runs the same advisory-lock
claim loop outside gunicorn (same pattern as the Telegram bot companion).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("executor_runner")


async def _run_forever() -> None:
    os.environ.setdefault("FORCE_EXECUTOR", "1")
    os.environ["EXECUTOR_STANDALONE"] = "1"

    try:
        from app.database import init_db_minimal
        init_db_minimal()
    except Exception as exc:
        logger.warning("DB reachability check failed (non-fatal): %s", exc)

    # Drop ghost holders (old Railway container, dev laptop, etc.) before the
    # slow strategy_portal_server import — otherwise we sit in standby for hours.
    from app.executor_lock import reclaim_executor_lock
    reclaim_executor_lock(force=True)

    from strategy_portal_server import _executor_claim_loop

    logger.info("Standalone executor process starting — acquiring advisory lock…")
    await _executor_claim_loop(first_attempt_delay=0)

    # Claim loop returns after tasks are scheduled; keep this process alive.
    while True:
        await asyncio.sleep(3600)


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(_run_forever())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
