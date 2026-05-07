#!/bin/bash
# Start all backend services in parallel
# Query Service: Port 5001
# Metrics Service: Port 5002

set -a
source .env 2>/dev/null || true
set +a

HOST="${HOST:-0.0.0.0}"
QUERY_PORT="${QUERY_PORT:-5001}"
METRICS_PORT="${METRICS_PORT:-5002}"

# Color codes for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting all backend services...${NC}"
echo -e "${BLUE}Query Service:   http://$HOST:$QUERY_PORT${NC}"
echo -e "${BLUE}Metrics Service: http://$HOST:$METRICS_PORT${NC}"
echo ""

# Start Query Service
echo "Launching Query Service (Port $QUERY_PORT)..."
bash start_query_service.sh &
QUERY_PID=$!

# Start Metrics Service
echo "Launching Metrics Service (Port $METRICS_PORT)..."
bash start_metrics_service.sh &
METRICS_PID=$!

echo -e "${GREEN}✓ Services started${NC}"
echo "Query Service PID: $QUERY_PID"
echo "Metrics Service PID: $METRICS_PID"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Trap SIGINT to kill both services
trap "kill $QUERY_PID $METRICS_PID 2>/dev/null; echo 'Services stopped'; exit 0" SIGINT

# Wait for both processes
wait
