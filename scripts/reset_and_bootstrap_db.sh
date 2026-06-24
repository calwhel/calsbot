#!/usr/bin/env bash
# Destructive fresh schema rebuild for Railway / Postgres.
# Requires DATABASE_URL in environment (Railway shell injects it automatically).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== DROP public schema + full bootstrap (DESTRUCTIVE) ==="
python -m app.schema_bootstrap --reset-schema --skip-lock --force
echo ""
echo "=== Verify ==="
python -m app.schema_bootstrap --verify-only
