"""
Dedicated cTrader spot-feed process for Railway split deployments.

Run with CTRADER_FEED_ONLY=1 on a lightweight service. The main portal sets
CTRADER_REMOTE_FEED=1 (or DISABLE_CTRADER_FEED_IN_EXECUTOR=1) so the executor
does not open a second broker session — it reads ticks from market_spot_ticks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("ctrader_feed_runner")


def _start_ping_server() -> None:
    """Railway healthcheck expects PORT + /ping — feed process has no gunicorn."""
    port = int(os.environ.get("PORT", "8080"))

    def _serve():
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") in ("", "/ping", "/health"):
                    body = b"ok"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_args):
                pass

        HTTPServer(("0.0.0.0", port), _Handler).serve_forever()

    threading.Thread(target=_serve, name="feed-ping", daemon=True).start()
    logger.info("Feed health server on :%s/ping", port)


def _lock_keepalive_thread(conn_holder: list, stop_event: threading.Event) -> None:
    """Ping the advisory-lock session so Neon idle drops do not release the lock."""
    from app.advisory_lock_ids import APP_NAME_CTRADER_FEED, CTRADER_FEED_LOCK_ID
    from app.executor_lock import reconnect_lock_connection

    while not stop_event.is_set():
        for _ in range(30):
            if stop_event.is_set():
                return
            time.sleep(1)
        conn = conn_holder[0] if conn_holder else None
        if conn is None:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as exc:
            new_conn = reconnect_lock_connection(
                conn,
                lock_id=CTRADER_FEED_LOCK_ID,
                max_attempts=1,
                silent=True,
                application_name=APP_NAME_CTRADER_FEED,
            )
            if new_conn:
                conn_holder[0] = new_conn
                logger.debug("cTrader feed lock keepalive: silent reconnect ok")
                continue
            logger.warning("cTrader feed lock keepalive failed: %s", exc)
            break


async def _maintain_feed_lock(conn) -> None:
    holder = [conn]
    stop = threading.Event()
    thread = threading.Thread(
        target=_lock_keepalive_thread,
        args=(holder, stop),
        name="ctrader-feed-lock-keepalive",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            await asyncio.sleep(5)
            c = holder[0]
            if c is None or getattr(c, "closed", 0):
                break
            try:
                with c.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                break
    finally:
        stop.set()
        thread.join(timeout=3.0)


async def _run_feed_holder(conn) -> None:
    from app.ctrader_feed_lock import close_lock_connection
    from app.services.ctrader_price_feed import start as feed_start, stop as feed_stop

    feed_start()
    logger.info("cTrader feed lock acquired — streaming spot ticks to Postgres")

    async def _heartbeat_loop() -> None:
        """Upsert a sentinel tick so consumers know the feed process is alive."""
        await asyncio.sleep(15)
        while True:
            try:
                from app.services.spot_price_store import upsert_tick

                upsert_tick(
                    "_FEED_OK",
                    mid=1.0,
                    bid=1.0,
                    ask=1.0,
                    source="ctrader",
                )
            except Exception:
                pass
            await asyncio.sleep(30)

    hb = asyncio.create_task(_heartbeat_loop())
    try:
        await _maintain_feed_lock(conn)
    finally:
        hb.cancel()
        feed_stop()
        close_lock_connection(conn)


async def _claim_loop() -> None:
    from app.ctrader_feed_lock import (
        reclaim_feed_lock,
        try_acquire_feed_lock,
    )

    reclaim_feed_lock(force=True)
    delay = 0
    while True:
        if delay:
            await asyncio.sleep(delay)
        conn = try_acquire_feed_lock()
        if conn:
            try:
                await _run_feed_holder(conn)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("cTrader feed holder exited: %r", exc)
            delay = 5
            continue
        logger.info("cTrader feed lock held elsewhere — retry in 15s")
        delay = 15


async def _run_forever() -> None:
    os.environ["CTRADER_FEED_ONLY"] = "1"
    _start_ping_server()
    try:
        from app.database import init_db_minimal

        init_db_minimal()
    except Exception as exc:
        logger.warning("DB reachability check failed (non-fatal): %s", exc)

    logger.info("Standalone cTrader feed process starting — acquiring advisory lock…")
    await _claim_loop()


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
