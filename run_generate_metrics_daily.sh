#!/bin/bash

# Generate pre-calculated metrics presets
# Wrapper script for generate_metrics_presets.py

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
METRICS_PREFIX="${METRICS_PRESETS_PREFIX:-metrics-presets}"
DDB_TABLE="${HISTORY_ORDERS_DYNAMODB_TABLE:-history-orders-dev}"

echo "=========================================="
echo "Generating metrics presets for $EXECUTION_DATE"
echo "=========================================="
echo "Execution Date: $EXECUTION_DATE"
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $METRICS_BUCKET"
echo "S3 Prefix: $METRICS_PREFIX"
echo "DynamoDB Table: $DDB_TABLE"
echo ""

cd "$BACKEND_DIR"

python generate_metrics_presets.py \
    --execution-date "$EXECUTION_DATE" \
    --aws-region "$AWS_REGION" \
    --bucket "$METRICS_BUCKET" \
    --prefix "$METRICS_PREFIX" \
    --ddb-table "$DDB_TABLE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Metrics generation completed successfully"
else
    echo "❌ Metrics generation failed with exit code $exit_code"
fi

exit $exit_code
