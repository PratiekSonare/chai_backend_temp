#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables if present.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Activate local virtualenv if available.
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "[$(date -Iseconds)] Starting product metrics updater"

ARGS=()

# Optional overrides from environment.
if [ -n "${PRODUCT_METRICS_DDB_TABLE:-}" ]; then
  ARGS+=(--ddb-table "$PRODUCT_METRICS_DDB_TABLE")
fi

if [ -n "${PRODUCT_METRICS_S3_BUCKET:-}" ]; then
  ARGS+=(--bucket "$PRODUCT_METRICS_S3_BUCKET")
fi

if [ -n "${PRODUCT_METRICS_S3_PREFIX:-}" ]; then
  ARGS+=(--prefix "$PRODUCT_METRICS_S3_PREFIX")
fi

if [ -n "${PRODUCT_METRICS_AWS_REGION:-}" ]; then
  ARGS+=(--aws-region "$PRODUCT_METRICS_AWS_REGION")
fi

if [ -n "${PRODUCT_METRICS_SINCE_DATE:-}" ]; then
  ARGS+=(--since-date "$PRODUCT_METRICS_SINCE_DATE")
fi

python "$SCRIPT_DIR/product_metrics_updater.py" "${ARGS[@]}"

echo "[$(date -Iseconds)] Product metrics updater completed"
