"""
Dedicated Gemini Vision Gold Trader process for Railway/production.

Runs one scan loop + watchdog outside gunicorn (same pattern as app.gold_ai_runner).
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

logger = logging.getLogger("gemini_gold_runner")

_RESTART_DELAY_SECS = 5
_HEARTBEAT_INTERVAL_SECS = 30


async def _gemini_gold_process_heartbeat_loop() -> None:
    while True:
        try:
            from app.gemini_gold_trader.loop import scan_heartbeat_age_seconds

            age_s = await asyncio.to_thread(scan_heartbeat_age_seconds)
            if age_s is None:
                logger.info("[gemini-gold] heartbeat OK (standalone process, no scan yet)")
            else:
                logger.info(
                    "[gemini-gold] heartbeat OK (standalone process, scan_age=%.0fs)",
                    age_s,
                )
        except Exception as exc:
            logger.warning("[gemini-gold] heartbeat check failed: %s", exc)
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)


async def _run_gemini_gold_session() -> None:
    from app.deploy_stamp import log_deploy_stamp

    log_deploy_stamp("gemini_gold_runner")
    os.environ["GEMINI_GOLD_STANDALONE"] = "1"
    os.environ.setdefault("DISABLE_GEMINI_GOLD_IN_GUNICORN", "1")

    from app.database import init_db_minimal

    try:
        init_db_minimal()
    except Exception as exc:
        logger.warning("DB reachability check failed (non-fatal): %s", exc)

    try:
        from app.gemini_gold_trader.schema import ensure_gemini_gold_trader_schema

        ensure_gemini_gold_trader_schema()
    except Exception as exc:
        logger.warning("[gemini-gold] schema ensure failed (non-fatal): %s", exc)

    from app.gemini_gold_trader.loop import start_gemini_gold_trader_loop

    logger.info("Standalone gemini-gold trader starting — scan loop + watchdog")
    asyncio.create_task(_gemini_gold_process_heartbeat_loop())
    await start_gemini_gold_trader_loop()

    while True:
        await asyncio.sleep(3600)


async def _run_forever() -> None:
    while True:
        try:
            await _run_gemini_gold_session()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.critical(
                "Standalone gemini-gold trader exited unexpectedly (%s) — restarting in %ss",
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
