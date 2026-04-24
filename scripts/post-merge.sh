#!/bin/bash
# Post-merge setup for TradeHub Markets.
# Runs automatically after a task is merged into main. Must be idempotent and
# non-interactive (stdin is closed).
#
# This project is a Python app — no JS bundle, no Drizzle migrations. Schema
# changes (CREATE TABLE / ALTER TABLE) are applied lazily by `_ensure_tables()`
# at Strategy Portal startup, so the only mandatory step here is a workflow
# restart, which the platform handles automatically after this script.
set -e

echo "[post-merge] starting"

# Make sure any newly-added Python deps from the merge are installed.
# `uv pip install` is idempotent and skips already-satisfied requirements.
if [ -f "pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  echo "[post-merge] syncing Python deps via uv"
  uv pip install -r <(uv pip compile --quiet pyproject.toml 2>/dev/null) 2>/dev/null || true
fi

echo "[post-merge] done"
