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

# Default 1 worker on Railway: executor + cTrader/FMP feeds must run in a single
# process (duplicate workers doubled API 429s, cTrader reconnect churn, and
# 30s gunicorn worker kills from stacked kline fetches). Set GUNICORN_WORKERS=2
# only if you have headroom and accept the trade-offs.
_GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
_GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
echo "[railway] Starting Strategy Portal on port ${PORT:-5000} (workers=${_GUNICORN_WORKERS}, timeout=${_GUNICORN_TIMEOUT})..."
exec gunicorn -w "${_GUNICORN_WORKERS}" -k uvicorn.workers.UvicornWorker \
  --max-requests 300 --max-requests-jitter 30 \
  --bind "0.0.0.0:${PORT:-5000}" --timeout "${_GUNICORN_TIMEOUT}" --graceful-timeout 30 \
  --log-level info strategy_portal_server:app
