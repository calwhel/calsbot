#!/bin/bash
set -e

# Railway defaults — free portal features + live strategy executor
export PORTAL_FEATURES_FREE="${PORTAL_FEATURES_FREE:-1}"
export FORCE_EXECUTOR="${FORCE_EXECUTOR:-1}"

# Optional Telegram bot companion (background; does not bind $PORT).
# ONE process only — no inner restart loop (that spawned overlapping pollers →
# TelegramConflictError). Railway's restartPolicy restarts the whole container.
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ "${DISABLE_TELEGRAM_POLL:-}" != "1" ]; then
  export FORCE_BOT_POLL=1
  # Stagger so a rolling deploy's old container releases getUpdates first.
  _tg_delay="${TELEGRAM_POLL_START_DELAY:-25}"
  echo "[railway] Telegram bot companion starts in ${_tg_delay}s (port 8080)..."
  ( sleep "${_tg_delay}"; python -m uvicorn main:app --host 127.0.0.1 --port 8080 2>&1 | sed 's/^/[tg-bot] /' ) &
fi

echo "[railway] Starting Strategy Portal on port ${PORT:-5000}..."
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker --reuse-port \
  --max-requests 300 --max-requests-jitter 30 \
  --bind "0.0.0.0:${PORT:-5000}" --timeout 120 --graceful-timeout 30 \
  --log-level info strategy_portal_server:app
