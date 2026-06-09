#!/usr/bin/env bash
# Split cTrader spot feed onto a dedicated Railway service (shared Neon Postgres).
#
# Architecture:
#   Feed service  — CTRADER_FEED_ONLY=1  → streams ticks → market_spot_ticks
#   Main service  — CTRADER_REMOTE_FEED=1 → executor reads Postgres, no 2nd socket
#
# Prerequisites:
#   railway CLI logged in, both services linked to the SAME project + DATABASE_URL
#
# Usage:
#   ./scripts/railway_set_ctrader_feed_split.sh main   # run while linked to portal+executor
#   ./scripts/railway_set_ctrader_feed_split.sh feed   # run while linked to feed replica
#
set -euo pipefail

TARGET="${1:-}"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli && railway login && railway link"
  exit 1
fi

case "${TARGET}" in
  main)
    echo "Configuring MAIN service (portal + executor, remote cTrader feed)..."
    railway variables set \
      CTRADER_REMOTE_FEED=1 \
      DISABLE_CTRADER_FEED_IN_EXECUTOR=1
    echo ""
    echo "Main service will NOT open a local cTrader spot stream."
    echo "Ensure a separate feed service runs with CTRADER_FEED_ONLY=1."
    ;;
  feed)
    echo "Configuring FEED service (dedicated cTrader spot stream)..."
    railway variables set \
      CTRADER_FEED_ONLY=1 \
      CTRADER_REMOTE_FEED=0 \
      DISABLE_CTRADER_FEED_IN_EXECUTOR=0 \
      FORCE_EXECUTOR=0 \
      DISABLE_TELEGRAM_POLL=1
    echo ""
    echo "Feed service runs: python3 -m app.ctrader_feed_runner"
    echo "Use a small replica (512MB–1GB). Same DATABASE_URL and CTRADER_CLIENT_* as main."
    ;;
  *)
    echo "Usage: $0 main|feed"
    echo ""
    echo "  main — set on your primary portal/executor Railway service"
    echo "  feed — set on a second lightweight Railway service"
    exit 1
    ;;
esac

echo ""
echo "After both services are configured, redeploy and verify:"
echo "  Feed logs:  'streaming spot ticks to Postgres'"
echo "  Main logs:  'spot stream skipped — remote feed enabled'"
echo "  API:        /api/ctrader/feed-status → remote_feed: true"
