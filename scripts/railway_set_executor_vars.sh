#!/usr/bin/env bash
# Set optional executor tuning vars on Railway.
# Requires: railway CLI logged in (`npm i -g @railway/cli && railway login`)
#
# These are OPTIONAL — the app already uses these defaults in code.
# Only run if you want to override them without a code change.

set -euo pipefail

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli"
  echo "Then: railway login && railway link"
  exit 1
fi

echo "Setting executor tuning variables on linked Railway service..."
railway variables set \
  EXECUTOR_SCAN_INTERVAL=10 \
  EXECUTOR_FOREX_SCAN_INTERVAL=5 \
  EXECUTOR_MONITOR_INTERVAL=10 \
  EXECUTOR_LIVE_MONITOR_INTERVAL=8 \
  EXECUTOR_TRADE_MONITOR_INTERVAL=15 \
  FMP_POLL_INTERVAL_SECONDS=8 \
  EXECUTOR_MAX_CONCURRENT=3 \
  EXECUTOR_FOREX_MAX_CONCURRENT=6 \
  FOREX_SCANNER_PARALLEL=6

echo "Done. Redeploy or wait for next deploy to pick up changes."
echo "Verify feeds: https://YOUR-APP.up.railway.app/api/ctrader/feed-status"
