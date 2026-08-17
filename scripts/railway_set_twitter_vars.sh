#!/usr/bin/env bash
# Enable X auto-post + mover auto-replies on Railway.
# Requires: railway CLI logged in + linked to the project.
#
# Usage:
#   ./scripts/railway_set_twitter_vars.sh
#
# Does NOT set the OAuth secrets (those should already exist). Only flips
# the enable flags + Bitunix referral URL.

set -euo pipefail

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: npm i -g @railway/cli"
  echo "Then: railway login && railway link"
  exit 1
fi

REFERRAL="${BITUNIX_REFERRAL_URL:-https://www.bitunix.com/activity/basic/ENWeeklyCampaign0817?vipCode=fgq74890}"
CAMPAIGN="${BITUNIX_CAMPAIGN_URL:-https://www.bitunix.com/activity/basic/ENWeeklyCampaign0817?vipCode=fgq74890}"

echo "Enabling X auto-post + Bitunix Aug campaign push on Railway..."
railway variables set \
  TWITTER_ENABLED=1 \
  TWITTER_AUTO_REPLY_ENABLED=1 \
  "BITUNIX_REFERRAL_URL=${REFERRAL}" \
  "BITUNIX_CAMPAIGN_URL=${CAMPAIGN}"

echo "Done. Confirm TWITTER_* OAuth keys already exist, then redeploy."
echo "Look for logs: 'X-poster advisory lock acquired', schedule with bitunix_campaign"
