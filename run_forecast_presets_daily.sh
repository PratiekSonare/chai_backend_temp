#!/bin/bash

# Generate forecast presets
# Wrapper script for generate_forecast_presets.py

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
FORECAST_PREFIX="${FORECAST_PRESETS_PREFIX:-forecast-presets}"
DYNAMODB_TABLE_NAME="${DYNAMODB_TABLE_NAME:-history-orders-final}"
FORECAST_COLUMNS="${FORECAST_COLUMNS:-order_date}"

echo "=========================================="
echo "Generating Forecast presets for $EXECUTION_DATE"
echo "=========================================="
echo "Execution Date: $EXECUTION_DATE"
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $METRICS_BUCKET"
echo "Forecast Prefix: $FORECAST_PREFIX"
echo "DynamoDB Table: $DYNAMODB_TABLE_NAME"
echo ""

cd "$BACKEND_DIR"

python generate_forecast_presets.py \
    --execution-date "$EXECUTION_DATE" \
    --aws-region "$AWS_REGION" \
    --bucket "$METRICS_BUCKET" \
    --forecast-prefix "$FORECAST_PREFIX" \
    --dynamodb-table-name "$DYNAMODB_TABLE_NAME" \
    --forecast-columns "$FORECAST_COLUMNS"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Forecast presets generation completed successfully"
else
    echo "❌ Forecast presets generation failed with exit code $exit_code"
fi

exit $exit_code
