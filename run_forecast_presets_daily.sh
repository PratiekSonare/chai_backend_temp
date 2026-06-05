#!/bin/bash

# Generate pre-calculated RTO + Forecast presets
# Wrapper script for generate_rto_forecast_presets.py

set -e

# Source environment variables
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

if [ -f ~/.env ]; then
    export $(cat ~/.env | grep -v '#' | xargs)
fi

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "$BACKEND_DIR/venv" ]; then
    source "$BACKEND_DIR/venv/bin/activate"
fi

# Calculate execution date (yesterday)
EXECUTION_DATE=$(date -d "yesterday" +%Y-%m-%d)

# Get optional parameters from environment or use defaults
AWS_REGION="${AWS_REGION:-ap-south-1}"
METRICS_BUCKET="${METRICS_PRESETS_BUCKET:-chupps-data-portal}"
RTO_PREFIX="${RTO_PRESETS_PREFIX:-rto-presets}"
FORECAST_PREFIX="${FORECAST_PRESETS_PREFIX:-forecast-presets}"

echo "=========================================="
echo "Generating RTO + Forecast presets for $EXECUTION_DATE"
echo "=========================================="
echo "Execution Date: $EXECUTION_DATE"
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $METRICS_BUCKET"
echo "RTO Prefix: $RTO_PREFIX"
echo "Forecast Prefix: $FORECAST_PREFIX"
echo ""

cd "$BACKEND_DIR"

python generate_rto_forecast_presets.py \
    --execution-date "$EXECUTION_DATE" \
    --aws-region "$AWS_REGION" \
    --bucket "$METRICS_BUCKET" \
    --rto-prefix "$RTO_PREFIX" \
    --forecast-prefix "$FORECAST_PREFIX"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ RTO + Forecast presets generation completed successfully"
else
    echo "❌ RTO + Forecast presets generation failed with exit code $exit_code"
fi

exit $exit_code
