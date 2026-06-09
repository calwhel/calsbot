#!/bin/bash
set -e

# ── Dedicated cTrader feed service (separate Railway replica) ────────────────
# Set CTRADER_FEED_ONLY=1 on a lightweight service that only streams broker
# ticks into Postgres (market_spot_ticks). Main portal sets CTRADER_REMOTE_FEED=1
# so the executor does not open a competing cTrader session.
if [ "${CTRADER_REMOTE_FEED}" = "1" ]; then
  export DISABLE_CTRADER_FEED_IN_EXECUTOR="${DISABLE_CTRADER_FEED_IN_EXECUTOR:-1}"
fi
if [ "${CTRADER_FEED_ONLY}" = "1" ]; then
  echo "[railway] cTrader feed-only service — spot stream + Postgres ticks"
  exec python3 -m app.ctrader_feed_runner
fi

# Railway defaults — free portal features + live strategy executor
export PORTAL_FEATURES_FREE="${PORTAL_FEATURES_FREE:-1}"
export FORCE_EXECUTOR="${FORCE_EXECUTOR:-1}"

# Executor resource profile — set RAILWAY_PRO=1 (or RAILWAY_EXECUTOR_TIER=pro) on Pro plan.
# Hobby/conservative defaults avoid pool exhaustion on 512MB–1GB replicas; Pro raises
# concurrency + DB pool so ~90 forex + ~168 crypto cycles complete faster.
if [ "${RAILWAY_PRO}" = "1" ] || [ "${RAILWAY_EXECUTOR_TIER}" = "pro" ]; then
  _EXEC_PROFILE="pro"
  export BG_POOL_SIZE="${BG_POOL_SIZE:-28}"
  export BG_POOL_OVERFLOW="${BG_POOL_OVERFLOW:-32}"
  export BG_DB_RESERVE="${BG_DB_RESERVE:-6}"
  export BG_DB_MAX_CONCURRENT="${BG_DB_MAX_CONCURRENT:-40}"
  export EXECUTOR_SHARD_COUNT="${EXECUTOR_SHARD_COUNT:-5}"
  export EXECUTOR_SHARD_STAGGER_SECONDS="${EXECUTOR_SHARD_STAGGER_SECONDS:-2}"
  export EXECUTOR_MAX_CONCURRENT="${EXECUTOR_MAX_CONCURRENT:-5}"
  export EXECUTOR_FOREX_MAX_CONCURRENT="${EXECUTOR_FOREX_MAX_CONCURRENT:-6}"
  export EXECUTOR_FOREX_SCAN_INTERVAL="${EXECUTOR_FOREX_SCAN_INTERVAL:-5}"
  export EXECUTOR_SCAN_INTERVAL="${EXECUTOR_SCAN_INTERVAL:-8}"
  export EXECUTOR_CRYPTO_START_DELAY="${EXECUTOR_CRYPTO_START_DELAY:-25}"
  export EXECUTOR_SCAN_BATCH_SIZE="${EXECUTOR_SCAN_BATCH_SIZE:-25}"
  export EXECUTOR_PREFETCH_CONCURRENT="${EXECUTOR_PREFETCH_CONCURRENT:-25}"
  export EXECUTOR_KLINE_BARS="${EXECUTOR_KLINE_BARS:-80}"
  export METAL_KLINE_LIVE_MAX_DRIFT_PCT="${METAL_KLINE_LIVE_MAX_DRIFT_PCT:-0.4}"
  export METAL_SPOT_KLINE_MAX_DRIFT_PCT="${METAL_SPOT_KLINE_MAX_DRIFT_PCT:-0.5}"
  export METAL_KLINE_CACHE_SECONDS="${METAL_KLINE_CACHE_SECONDS:-30}"
  export METAL_KLINE_FETCH_CONCURRENT="${METAL_KLINE_FETCH_CONCURRENT:-1}"
  export KRAKEN_KLINE_TIMEOUT_SECONDS="${KRAKEN_KLINE_TIMEOUT_SECONDS:-8}"
  export CTRADER_PROACTIVE_REFRESH_INTERVAL_S="${CTRADER_PROACTIVE_REFRESH_INTERVAL_S:-2700}"
  export CTRADER_KLINE_TIMEOUT_LIVE_S="${CTRADER_KLINE_TIMEOUT_LIVE_S:-15}"
  export CTRADER_AUTH_TERMINAL_BACKOFF_S="${CTRADER_AUTH_TERMINAL_BACKOFF_S:-300}"
  export METALS_POLL_INTERVAL_SECONDS="${METALS_POLL_INTERVAL_SECONDS:-3}"
  export REALTIME_SPOT_MAX_AGE_METALS_S="${REALTIME_SPOT_MAX_AGE_METALS_S:-5}"
  export REALTIME_SPOT_MAX_AGE_FOREX_S="${REALTIME_SPOT_MAX_AGE_FOREX_S:-5}"
  export SPOT_PRIMER_INTERVAL_SECONDS="${SPOT_PRIMER_INTERVAL_SECONDS:-5}"
  export ENTRY_PRICE_MAX_DRIFT_PCT="${ENTRY_PRICE_MAX_DRIFT_PCT:-0.12}"
  _GUNICORN_WORKERS_DEFAULT=4
else
  _EXEC_PROFILE="hobby"
  export BG_POOL_SIZE="${BG_POOL_SIZE:-8}"
  export BG_POOL_OVERFLOW="${BG_POOL_OVERFLOW:-10}"
  export BG_DB_RESERVE="${BG_DB_RESERVE:-5}"
  export EXECUTOR_MAX_CONCURRENT="${EXECUTOR_MAX_CONCURRENT:-2}"
  export EXECUTOR_FOREX_MAX_CONCURRENT="${EXECUTOR_FOREX_MAX_CONCURRENT:-3}"
  _GUNICORN_WORKERS_DEFAULT=2
fi
# Legacy Telegram social-signals scanner — off on portal/Railway; user strategies only.
export DISABLE_SOCIAL_SCANNING="${DISABLE_SOCIAL_SCANNING:-1}"

# Optional Telegram bot companion (background; does not bind $PORT).
# ONE process only — no inner restart loop (that spawned overlapping pollers →
# TelegramConflictError). Railway's restartPolicy restarts the whole container.
export PYTHONUNBUFFERED=1
_TG_DISABLED="${DISABLE_TELEGRAM_POLL:-}"
# Resolve token via pydantic too — catches Railway vars loaded only into settings.
_TG_HAS_TOKEN="$(python3 -c "
from app.config import settings
import os
t = (getattr(settings, 'TELEGRAM_BOT_TOKEN', None) or os.getenv('TELEGRAM_BOT_TOKEN') or '').strip()
print('yes' if t else 'no')
" 2>/dev/null || echo no)"

if [ "${_TG_DISABLED}" = "1" ]; then
  echo "[railway] Telegram bot SKIPPED — DISABLE_TELEGRAM_POLL=1"
elif [ "${_TG_HAS_TOKEN}" != "yes" ]; then
  echo "[railway] Telegram bot SKIPPED — TELEGRAM_BOT_TOKEN not set (commands like /start will not work)"
else
  export FORCE_BOT_POLL=1
  # Stagger so a rolling deploy's old container releases getUpdates first.
  _tg_delay="${TELEGRAM_POLL_START_DELAY:-10}"
  echo "[railway] Telegram bot companion starts in ${_tg_delay}s (port 8080)..."
  ( sleep "${_tg_delay}"; exec python3 -m uvicorn main:app --host 127.0.0.1 --port 8080 2>&1 | sed 's/^/[tg-bot] /' ) &
fi

# Strategy executor runs in a dedicated process (not gunicorn). Scan cycles for
# 90 forex + 168 crypto strategies exceed gunicorn worker timeout even at 300s,
# causing WORKER TIMEOUT SIGABRT mid-scan and zero trade fires.
if [ "${DISABLE_STANDALONE_EXECUTOR}" != "1" ]; then
  _ex_delay="${EXECUTOR_START_DELAY:-8}"
  echo "[railway] Standalone strategy executor starts in ${_ex_delay}s…"
  ( sleep "${_ex_delay}"; EXECUTOR_STANDALONE=1 exec python3 -m app.executor_runner 2>&1 | sed 's/^/[executor] /' ) &
fi
export DISABLE_EXECUTOR_IN_GUNICORN="${DISABLE_EXECUTOR_IN_GUNICORN:-1}"

# Gunicorn serves HTTP only; executor + feeds run in the standalone process above.
_GUNICORN_WORKERS="${GUNICORN_WORKERS:-${_GUNICORN_WORKERS_DEFAULT}}"
# 90 tradfi + 168 crypto strategies per cycle can exceed 120s (klines + DB +
# per-strategy gate writes). Sub-120s timeouts caused WORKER TIMEOUT SIGABRT
# mid-scan — zero completed forex cycles and no fires. Production script uses 300.
_GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"
echo "[railway] Executor profile=${_EXEC_PROFILE} shards=${EXECUTOR_SHARD_COUNT:-1} bg_pool=${BG_POOL_SIZE}+${BG_POOL_OVERFLOW} forex_conc=${EXECUTOR_FOREX_MAX_CONCURRENT}/shard crypto_conc=${EXECUTOR_MAX_CONCURRENT}/shard"
echo "[railway] Starting Strategy Portal on port ${PORT:-5000} (workers=${_GUNICORN_WORKERS}, timeout=${_GUNICORN_TIMEOUT})..."
exec gunicorn -w "${_GUNICORN_WORKERS}" -k uvicorn.workers.UvicornWorker \
  --max-requests 300 --max-requests-jitter 30 \
  --bind "0.0.0.0:${PORT:-5000}" --timeout "${_GUNICORN_TIMEOUT}" --graceful-timeout 30 \
  --log-level info strategy_portal_server:app
