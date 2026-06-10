#!/usr/bin/env bash
# Split crypto (portal) and forex (dedicated executor replica) across Railway services.
#
# Architecture:
#   Portal service  — HTTP + crypto every ~10 min, forex disabled
#   Executor-forex  — EXECUTOR_ONLY=1, forex scan ~5s, crypto disabled, tuned gold prefetch
#
# Prerequisites:
#   railway CLI logged in (`npm i -g @railway/cli && railway login && railway link`)
#   Both services: same project + DATABASE_URL (reference from portal)
#
# Usage:
#   ./scripts/railway_set_executor_split.sh portal   # main portal service
#   ./scripts/railway_set_executor_split.sh forex    # dedicated forex executor replica
#   ./scripts/railway_set_executor_split.sh both     # run portal then forex (switch link between)
#
set -euo pipefail

TARGET="${1:-}"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli"
  echo "Then: railway login && railway link"
  exit 1
fi

_set_portal() {
  echo "Configuring PORTAL service (HTTP + crypto ~10 min, forex OFF)..."
  railway variables set \
    EXECUTOR_ONLY=0 \
    DISABLE_FOREX_EXECUTOR=1 \
    DISABLE_CRYPTO_EXECUTOR=0 \
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
  echo "Portal: gunicorn + standalone crypto executor only."
  echo "Remove EXECUTOR_ONLY from Railway UI if it was set to 1."
}

_set_forex() {
  echo "Configuring FOREX-ONLY executor replica (fast tradfi, tuned gold prefetch)..."
  railway variables set \
    EXECUTOR_ONLY=1 \
    DISABLE_CRYPTO_EXECUTOR=1 \
    DISABLE_FOREX_EXECUTOR=0 \
    EXECUTOR_FOREX_SCAN_INTERVAL=5 \
    EXECUTOR_FOREX_MAX_CONCURRENT=6 \
    EXECUTOR_FOREX_MANAGE_INTERVAL=1 \
    EXECUTOR_FX_WORKLIST_TTL=1 \
    EXECUTOR_FX_RECONCILE_INTERVAL=5 \
    EXECUTOR_SHARD_COUNT=2 \
    EXECUTOR_SHARD_STAGGER_SECONDS=2 \
    EXECUTOR_KLINE_BARS=80 \
    EXECUTOR_PREFETCH_CONCURRENT=10 \
    EXECUTOR_SCAN_BATCH_SIZE=25 \
    METAL_WARM_GLOBAL_TTL_S=180 \
    METAL_KLINE_CACHE_SECONDS=45 \
    METAL_KLINE_FETCH_CONCURRENT=4 \
    KRAKEN_KLINE_TIMEOUT_SECONDS=5 \
    CTRADER_KLINE_TIMEOUT_LIVE_S=6 \
    METAL_KLINE_LIVE_MAX_DRIFT_PCT=0.5 \
    METAL_SPOT_KLINE_MAX_DRIFT_PCT=0.6 \
    DISABLE_TELEGRAM_POLL=1 \
    DISABLE_STANDALONE_EXECUTOR=1 \
    FORCE_EXECUTOR=1 \
    RAILWAY_PRO=1 \
    BG_POOL_SIZE=20 \
    BG_POOL_OVERFLOW=24 \
    BG_DB_MAX_CONCURRENT=32 \
    BG_DB_RESERVE=6 \
    CTRADER_REMOTE_FEED=0 \
    DISABLE_CTRADER_FEED_IN_EXECUTOR=0 \
    EXECUTOR_CRYPTO_START_DELAY=0
  echo ""
  echo "Forex replica: EXECUTOR_ONLY=1 → start.sh runs executor_runner only."
  echo "Local cTrader spot feed on this replica (copy CTRADER_CLIENT_* + OAuth tokens from portal)."
  echo "Use Pro replica 4–8 GB RAM."
}

case "${TARGET}" in
  portal)
    _set_portal
    ;;
  forex)
    _set_forex
    ;;
  both)
    echo "=== PORTAL (link railway to main service first) ==="
    _set_portal
    echo ""
    echo "=== FOREX (now run: railway link → executor-forex service, then re-run: $0 forex) ==="
    ;;
  *)
    echo "Usage: $0 portal|forex|both"
    echo ""
    echo "  portal — main TradeHub portal (crypto ~10 min, forex disabled)"
    echo "  forex  — dedicated forex/gold executor (22 strategies, fast prefetch)"
    echo "  both   — portal vars only; re-link and run 'forex' for second service"
    exit 1
    ;;
esac

echo ""
echo "── Railway setup ────────────────────────────────────────────────────"
echo "  1. Portal:  railway link → main service  →  $0 portal"
echo "  2. New service 'executor-forex' (same GitHub repo)"
echo "  3. Reference DATABASE_URL + copy cTrader vars from portal"
echo "  4. Forex:   railway link → executor-forex  →  $0 forex"
echo "  5. Redeploy BOTH services"
echo ""
echo "Optional later: ./scripts/railway_set_ctrader_feed_split.sh feed + main"
echo ""
echo "Verify: https://tradehubmarkets.com/health/deep"
echo "  locks: portal=708110004  forex_only=708110006 (each needs a holder PID)"
echo "  runtime_profile: portal forex_disabled=true, forex replica executor_only=true"
