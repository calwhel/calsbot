#!/bin/bash
# Production startup — runs all three services in parallel.
# Strategy Portal is the primary process (port 5000).
# Telegram Bot (port 8080) and Trade Tracker (port 8000) run as background daemons.

set -e

echo "[startup] Starting Telegram Bot on port 8080..."
while true; do
    python -m uvicorn main:app --host 0.0.0.0 --port 8080 2>&1 | sed 's/^/[tg-bot] /'
    echo "[tg-bot] exited — restarting in 5s..."
    sleep 5
done &
TG_PID=$!

echo "[startup] Starting Trade Tracker on port 8000..."
while true; do
    python3 tracker_server.py 2>&1 | sed 's/^/[tracker] /'
    echo "[tracker] exited — restarting in 5s..."
    sleep 5
done &
TRACKER_PID=$!

echo "[startup] Starting Strategy Portal on port 5000..."
cleanup() {
    echo "[startup] Shutting down background services..."
    kill $TG_PID $TRACKER_PID 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

while true; do
    echo "[portal] starting..."
    gunicorn \
        -w 2 \
        -k uvicorn.workers.UvicornWorker \
        --reuse-port \
        --max-requests 300 \
        --max-requests-jitter 30 \
        --bind 0.0.0.0:5000 \
        --timeout 120 \
        --graceful-timeout 30 \
        --log-level info \
        strategy_portal_server:app
    echo "[portal] exited — restarting in 5s..."
    sleep 5
done
