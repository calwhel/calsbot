#!/bin/bash

export PORT=${PORT:-5000}

python -m uvicorn main:app --host 0.0.0.0 --port $PORT
