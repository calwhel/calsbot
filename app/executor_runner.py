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

_RESTART_DELAY_SECS = 5
_HEARTBEAT_INTERVAL_SECS = 30


async def _executor_process_heartbeat_loop() -> None:
    """30s heartbeat log + DB timestamp so a silent death is visible in logs/DB."""
    while True:
        try:
            from app.services.strategy_executor import persist_executor_process_heartbeat
            await persist_executor_process_heartbeat()
            logger.info("[executor] heartbeat OK (standalone process)")
        except Exception as exc:
            logger.warning("[executor] heartbeat persist failed: %s", exc)
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)


async def _run_executor_session() -> None:
    from app.deploy_stamp import log_deploy_stamp
    from app.executor_lock import log_executor_lock_keepalive_config

    log_deploy_stamp("executor_runner")
    # Prove keepalive settings in [executor] logs before advisory-lock acquire.
    log_executor_lock_keepalive_config()

    os.environ.setdefault("FORCE_EXECUTOR", "1")
    os.environ["EXECUTOR_STANDALONE"] = "1"
    if os.environ.get("EXECUTOR_ONLY", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("DISABLE_CRYPTO_EXECUTOR", "1")
        os.environ.setdefault("DISABLE_TELEGRAM_POLL", "1")
        os.environ.setdefault("EXECUTOR_FOREX_SCAN_INTERVAL", "5")
        os.environ.setdefault("EXECUTOR_FOREX_MAX_CONCURRENT", "10")
        logger.info(
            "EXECUTOR_ONLY forex replica — crypto disabled, forex scan=%ss",
            os.environ.get("EXECUTOR_FOREX_SCAN_INTERVAL", "5"),
        )

    from app.database import bg_engine, init_db_minimal

    try:
        init_db_minimal()
    except Exception as exc:
        logger.warning("DB reachability check failed (non-fatal): %s", exc)

    try:
        from app.trade_mgmt_schema import ensure_trade_mgmt_columns
        if not ensure_trade_mgmt_columns(bg_engine, wait_seconds=15.0):
            logger.error(
                "[executor] trade_mgmt columns missing after 15s — "
                "continuing (worklist will retry migration)"
            )
        else:
            logger.info("[executor] trade_mgmt schema verified before advisory lock")
    except Exception as exc:
        logger.error("[executor] trade_mgmt schema check failed: %s", exc)

    from app.executor_lock import reclaim_executor_lock
    reclaim_executor_lock(force=True)

    import strategy_portal_server as _portal
    _portal._initial_executor_reclaim_done = True

    from strategy_portal_server import _executor_claim_loop

    logger.info(
        "Standalone executor process starting — acquiring advisory lock "
        "(cTrader feed starts inside _start_executor_tasks after lock won)"
    )
    asyncio.create_task(_executor_process_heartbeat_loop())
    await _executor_claim_loop(first_attempt_delay=0)

    while True:
        await asyncio.sleep(3600)


async def _run_forever() -> None:
    """Never exit silently — CRITICAL log + 5s restart on any fatal error."""
    while True:
        try:
            await _run_executor_session()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.critical(
                "Standalone executor exited unexpectedly (%s) — restarting in %ss",
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
