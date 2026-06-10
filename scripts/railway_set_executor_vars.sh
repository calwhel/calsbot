#!/usr/bin/env bash
# Set executor tuning vars on Railway.
# Requires: railway CLI logged in (`npm i -g @railway/cli && railway login`)
#
# Usage:
#   ./scripts/railway_set_executor_vars.sh pro    # after upgrading to Railway Pro
#   ./scripts/railway_set_executor_vars.sh hobby  # conservative (small replica)

set -euo pipefail

PROFILE="${1:-pro}"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli"
  echo "Then: railway login && railway link"
  exit 1
fi

if [ "${PROFILE}" = "pro" ]; then
  echo "Setting Railway Pro executor profile..."
  railway variables set \
    RAILWAY_PRO=1 \
    BG_POOL_SIZE=12 \
    BG_POOL_OVERFLOW=15 \
    BG_DB_RESERVE=6 \
    EXECUTOR_MAX_CONCURRENT=6 \
    EXECUTOR_FOREX_MAX_CONCURRENT=8 \
    EXECUTOR_FOREX_SCAN_INTERVAL=5 \
    EXECUTOR_SCAN_INTERVAL=10 \
    EXECUTOR_CRYPTO_START_DELAY=45 \
    EXECUTOR_SCAN_BATCH_SIZE=25 \
    EXECUTOR_PREFETCH_CONCURRENT=30 \
    EXECUTOR_KLINE_BARS=80 \
    GUNICORN_WORKERS=3
else
  echo "Setting hobby/conservative executor profile..."
  railway variables set \
    RAILWAY_PRO=0 \
    BG_POOL_SIZE=8 \
    BG_POOL_OVERFLOW=10 \
    BG_DB_RESERVE=5 \
    EXECUTOR_MAX_CONCURRENT=2 \
    EXECUTOR_FOREX_MAX_CONCURRENT=3 \
    EXECUTOR_FOREX_SCAN_INTERVAL=15 \
    EXECUTOR_SCAN_INTERVAL=20 \
    EXECUTOR_CRYPTO_START_DELAY=120 \
    GUNICORN_WORKERS=2
fi

echo ""
echo "Done. Trigger a redeploy (or push to main) to pick up changes."
echo ""
echo "IMPORTANT (Pro plan): In Railway → your service → Settings → Deploy → Replica Limits"
echo "  set Memory to at least 4096 MB (8 GB recommended for ~250 strategies)."
echo "  Pro plan allows up to 24 GB — the default autoscaler may still start at 512 MB."
echo ""
echo "Executor split (recommended): ./scripts/railway_set_executor_split.sh portal|forex"
echo "  — crypto ~10 min on portal, fast forex on a dedicated replica."
echo "cTrader split feed (optional): ./scripts/railway_set_ctrader_feed_split.sh feed"
echo "  on a second Railway service — see script header."
echo ""
echo "Verify: https://tradehubmarkets.com/health/deep"
