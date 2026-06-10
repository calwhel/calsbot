#!/usr/bin/env bash
# Split crypto (portal) and forex (dedicated executor replica) across Railway services.
#
# Architecture:
#   Portal service  — HTTP + crypto every ~10 min, forex disabled
#   Executor-forex  — EXECUTOR_ONLY=1, forex scan ~5s, crypto disabled
#
# Prerequisites:
#   railway CLI logged in, both services linked to the SAME project + DATABASE_URL
#
# Usage:
#   ./scripts/railway_set_executor_split.sh portal   # main portal service
#   ./scripts/railway_set_executor_split.sh forex    # dedicated forex executor replica
#
set -euo pipefail

TARGET="${1:-}"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli && railway login && railway link"
  exit 1
fi

case "${TARGET}" in
  portal)
    echo "Configuring PORTAL service (HTTP + crypto ~10 min, forex off)..."
    railway variables set \
      DISABLE_FOREX_EXECUTOR=1 \
      EXECUTOR_CRYPTO_SCAN_INTERVAL=600 \
      EXECUTOR_SCAN_INTERVAL=600 \
      EXECUTOR_CRYPTO_START_DELAY=30 \
      DISABLE_STANDALONE_EXECUTOR=0 \
      DISABLE_EXECUTOR_IN_GUNICORN=1 \
      RAILWAY_PRO=1 \
      BG_POOL_SIZE=12 \
      BG_POOL_OVERFLOW=15 \
      EXECUTOR_MAX_CONCURRENT=4 \
      EXECUTOR_KLINE_BARS=80 \
      GUNICORN_WORKERS=3
    echo ""
    echo "Portal runs gunicorn + standalone crypto executor (forex on separate replica)."
    echo "Unset EXECUTOR_ONLY if present — portal is NOT forex-only."
    ;;
  forex)
    echo "Configuring FOREX-ONLY executor replica (fast tradfi, no HTTP)..."
    railway variables set \
      EXECUTOR_ONLY=1 \
      DISABLE_CRYPTO_EXECUTOR=1 \
      DISABLE_FOREX_EXECUTOR=0 \
      EXECUTOR_FOREX_SCAN_INTERVAL=5 \
      EXECUTOR_FOREX_MAX_CONCURRENT=6 \
      EXECUTOR_FOREX_MANAGE_INTERVAL=1 \
      EXECUTOR_FX_WORKLIST_TTL=1 \
      EXECUTOR_FX_RECONCILE_INTERVAL=5 \
      EXECUTOR_SHARD_COUNT=3 \
      EXECUTOR_SHARD_STAGGER_SECONDS=3 \
      DISABLE_TELEGRAM_POLL=1 \
      DISABLE_STANDALONE_EXECUTOR=1 \
      FORCE_EXECUTOR=1 \
      RAILWAY_PRO=1 \
      BG_POOL_SIZE=28 \
      BG_POOL_OVERFLOW=32 \
      BG_DB_MAX_CONCURRENT=40 \
      CTRADER_REMOTE_FEED=1 \
      DISABLE_CTRADER_FEED_IN_EXECUTOR=1 \
      EXECUTOR_KLINE_BARS=80 \
      METAL_KLINE_FETCH_CONCURRENT=2
    echo ""
    echo "Forex replica runs: python3 -m app.executor_runner (via start.sh EXECUTOR_ONLY=1)."
    echo "Use a Pro replica (4–8 GB). Same DATABASE_URL + cTrader OAuth vars as portal."
    ;;
  *)
    echo "Usage: $0 portal|forex"
    echo ""
    echo "  portal — main TradeHub portal (crypto scans every ~10 minutes)"
    echo "  forex  — second Railway service for fast gold/forex execution"
    exit 1
    ;;
esac

echo ""
echo "── Railway UI (forex replica) ───────────────────────────────────────"
echo "  1. Project → + New → GitHub Repo → same repo as portal"
echo "  2. Name it e.g. executor-forex"
echo "  3. Variables → Reference DATABASE_URL from portal"
echo "  4. Copy cTrader + Telegram vars (forex replica skips Telegram poll)"
echo "  5. Run: $0 forex   (while railway link points at executor-forex)"
echo "  6. Run: $0 portal  (while railway link points at main portal)"
echo "  7. Redeploy both services"
echo ""
echo "Optional cTrader feed split: ./scripts/railway_set_ctrader_feed_split.sh feed"
echo ""
echo "Verify: https://tradehubmarkets.com/health/deep"
echo "  portal lock 708110004 + forex lock 708110006 should each show a holder PID"
