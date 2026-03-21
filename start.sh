#!/bin/bash

# Strategy Portal — primary web application (port 5000)
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --reuse-port \
  --max-requests 300 --max-requests-jitter 30 \
  --bind 0.0.0.0:5000 --timeout 120 --graceful-timeout 30 \
  --log-level info strategy_portal_server:app
