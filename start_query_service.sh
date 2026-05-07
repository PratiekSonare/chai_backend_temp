#!/bin/bash
# Start Query Service (Port 5001)
# Data search workflow agent - handles /plan, /query, /execute endpoints

set -a
source .env 2>/dev/null || true
set +a

HOST="${HOST:-0.0.0.0}"
QUERY_PORT="${QUERY_PORT:-5001}"

echo "Starting Query Service on $HOST:$QUERY_PORT..."
exec uvicorn app_query:app --host "$HOST" --port "$QUERY_PORT" --reload
