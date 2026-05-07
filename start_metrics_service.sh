#!/bin/bash
# Start Metrics Service (Port 5002)
# Metric and chart calculations - handles /orders/*, /revenue/*, /payment/*, etc. endpoints

set -a
source .env 2>/dev/null || true
set +a

HOST="${HOST:-0.0.0.0}"
METRICS_PORT="${METRICS_PORT:-5002}"

echo "Starting Metrics Service on $HOST:$METRICS_PORT..."
exec uvicorn app_metrics:app --host "$HOST" --port "$METRICS_PORT" --reload
