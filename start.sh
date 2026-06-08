#!/bin/bash
set -e

# Railway defaults — free portal features + live strategy executor
export PORTAL_FEATURES_FREE="${PORTAL_FEATURES_FREE:-1}"
export FORCE_EXECUTOR="${FORCE_EXECUTOR:-1}"

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

# Two gunicorn workers serve HTTP only; executor + feeds run in the process above.
_GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"
# 90 tradfi + 168 crypto strategies per cycle can exceed 120s (klines + DB +
# per-strategy gate writes). Sub-120s timeouts caused WORKER TIMEOUT SIGABRT
# mid-scan — zero completed forex cycles and no fires. Production script uses 300.
_GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"
echo "[railway] Starting Strategy Portal on port ${PORT:-5000} (workers=${_GUNICORN_WORKERS}, timeout=${_GUNICORN_TIMEOUT})..."
exec gunicorn -w "${_GUNICORN_WORKERS}" -k uvicorn.workers.UvicornWorker \
  --max-requests 300 --max-requests-jitter 30 \
  --bind "0.0.0.0:${PORT:-5000}" --timeout "${_GUNICORN_TIMEOUT}" --graceful-timeout 30 \
  --log-level info strategy_portal_server:app
