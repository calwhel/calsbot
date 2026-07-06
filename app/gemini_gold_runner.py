"""
Dedicated Gemini Vision Gold Trader process for Railway/production.

Runs one scan loop + watchdog outside gunicorn (same pattern as app.gold_ai_runner).
A Postgres advisory lock ensures only one replica runs the trader when multiple
portal services share the same database.
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
_LOCK_RETRY_SECS = 30


def _emit_runner_startup_banner() -> None:
    """Early stdout lines before advisory-lock wait (visible through start.sh pipe)."""
    from app.deploy_stamp import get_deploy_commit, log_deploy_stamp

    log_deploy_stamp("gemini_gold_runner")
    print(
        "[gemini-gold] waiting for advisory lock before scan loop "
        f"(commit={get_deploy_commit()})",
        flush=True,
    )


async def _gemini_gold_process_heartbeat_loop() -> None:
    while True:
        try:
            from app.gemini_gold_trader.leadership import holds_gemini_gold_trader_lock
            from app.gemini_gold_trader.loop import scan_heartbeat_age_seconds

            age_s = await asyncio.to_thread(scan_heartbeat_age_seconds)
            lock_ok = holds_gemini_gold_trader_lock()
            if age_s is None:
                logger.info(
                    "[gemini-gold] heartbeat OK (standalone process, no scan yet, lock=%s)",
                    lock_ok,
                )
            else:
                logger.info(
                    "[gemini-gold] heartbeat OK (standalone process, scan_age=%.0fs, lock=%s)",
                    age_s,
                    lock_ok,
                )
        except Exception as exc:
            logger.warning("[gemini-gold] heartbeat check failed: %s", exc)
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)


async def _wait_for_trader_lock() -> None:
    from app.gemini_gold_trader.leadership import (
        lock_holder_hint,
        try_acquire_gemini_gold_trader_lock,
    )

    while True:
        acquired = await asyncio.to_thread(try_acquire_gemini_gold_trader_lock)
        if acquired:
            return
        hint = await asyncio.to_thread(lock_holder_hint)
        retry_msg = (
            f"[gemini-gold] trader lock held elsewhere ({hint or 'unknown'}) "
            f"— retry in {_LOCK_RETRY_SECS}s"
        )
        logger.info("%s", retry_msg)
        print(retry_msg, flush=True)
        await asyncio.sleep(_LOCK_RETRY_SECS)


async def _run_gemini_gold_session() -> None:
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

    try:
        from app.services.ctrader_token_scheduler import (
            start_ctrader_token_scheduler,
            wake_token_scheduler,
        )

        start_ctrader_token_scheduler()
        wake_token_scheduler()
        logger.info("[gemini-gold] cTrader token scheduler started or woken")
    except Exception as exc:
        logger.warning("[gemini-gold] token scheduler start failed (non-fatal): %s", exc)

    from app.gemini_gold_trader.loop import start_gemini_gold_trader_loop

    logger.info("Standalone gemini-gold trader starting — scan loop + watchdog")
    asyncio.create_task(_gemini_gold_process_heartbeat_loop())
    await start_gemini_gold_trader_loop()

    while True:
        await asyncio.sleep(3600)


async def _run_forever() -> None:
    _emit_runner_startup_banner()
    while True:
        try:
            await _wait_for_trader_lock()
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
        finally:
            from app.gemini_gold_trader.leadership import release_gemini_gold_trader_lock

            await asyncio.to_thread(release_gemini_gold_trader_lock)


def main() -> None:
    # Unbuffered stdout before logging/asyncio — visible through start.sh stdbuf|sed pipe.
    print(
        f"runner main() pid={os.getpid()} "
        f"standalone={os.environ.get('GEMINI_GOLD_STANDALONE', '')}",
        flush=True,
    )
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
    print(f"__main__ entry pid={os.getpid()}", flush=True)
    main()
