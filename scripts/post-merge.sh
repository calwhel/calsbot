#!/bin/bash
set -e

echo "=== TradeHub post-merge setup ==="

# Install any new Python dependencies
echo "→ Installing Python dependencies..."
pip install -r requirements.txt -q --no-input 2>/dev/null || true

echo "=== Post-merge setup complete ==="
