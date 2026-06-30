"""
Dedicated Gold AI Trader process for Railway/production.

Gunicorn workers previously each started their own scan loop and fought over a
single Postgres advisory lock. This module runs one loop + watchdog outside
gunicorn (same pattern as app.executor_runner).
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

logger = logging.getLogger("gold_ai_runner")

_RESTART_DELAY_SECS = 5
_HEARTBEAT_INTERVAL_SECS = 30


async def _gold_ai_process_heartbeat_loop() -> None:
    """30s heartbeat log so a silent runner death is visible in deploy logs."""
    while True:
        try:
            from app.gold_ai_trader.loop import scan_heartbeat_age_seconds

            age_s = await asyncio.to_thread(scan_heartbeat_age_seconds)
            if age_s is None:
                logger.info("[gold-ai] heartbeat OK (standalone process, no scan yet)")
            else:
                logger.info(
                    "[gold-ai] heartbeat OK (standalone process, scan_age=%.0fs)",
                    age_s,
                )
        except Exception as exc:
            logger.warning("[gold-ai] heartbeat check failed: %s", exc)
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)


async def _run_gold_ai_session() -> None:
    from app.deploy_stamp import log_deploy_stamp

    log_deploy_stamp("gold_ai_runner")
    os.environ["GOLD_AI_STANDALONE"] = "1"
    os.environ.setdefault("DISABLE_GOLD_AI_IN_GUNICORN", "1")

    from app.database import init_db_minimal

    try:
        init_db_minimal()
    except Exception as exc:
        logger.warning("DB reachability check failed (non-fatal): %s", exc)

    try:
        from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema

        ensure_gold_ai_trader_schema()
    except Exception as exc:
        logger.warning("[gold-ai] schema ensure failed (non-fatal): %s", exc)

    try:
        from app.gold_ai_trader.portal_mount import _maybe_reset_daily_credits_on_startup

        _maybe_reset_daily_credits_on_startup()
    except Exception as exc:
        logger.warning("[gold-ai] startup credits reset failed: %s", exc)

    from app.gold_ai_trader.loop import start_gold_ai_trader_loop

    logger.info("Standalone gold-ai trader starting — scan loop + watchdog")
    asyncio.create_task(_gold_ai_process_heartbeat_loop())
    await start_gold_ai_trader_loop()

    while True:
        await asyncio.sleep(3600)


async def _run_forever() -> None:
    """Never exit silently — CRITICAL log + 5s restart on any fatal error."""
    while True:
        try:
            await _run_gold_ai_session()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.critical(
                "Standalone gold-ai trader exited unexpectedly (%s) — restarting in %ss",
                exc,
                _RESTART_DELAY_SECS,
                exc_info=True,
            )
            await asyncio.sleep(_RESTART_DELAY_SECS)


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
