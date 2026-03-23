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

# Default timezone can be overridden (example: EXTRACT_TZ=Asia/Kolkata).
EXTRACT_TZ="Asia/Kolkata"
RUN_DATE="$(TZ="$EXTRACT_TZ" date -d 'yesterday' +%F)"

echo "[$(date -Iseconds)] Starting previous-day extraction for $RUN_DATE (TZ=$EXTRACT_TZ)"

ARGS=(
  --start-date "$RUN_DATE"
  --end-date "$RUN_DATE"
)

if [ -n "${EXTRACT_BUCKET:-}" ]; then
  ARGS+=(--bucket "$EXTRACT_BUCKET")
fi

if [ -n "${EXTRACT_PREFIX:-}" ]; then
  ARGS+=(--prefix "$EXTRACT_PREFIX")
fi

python "$SCRIPT_DIR/extract_orders.py" "${ARGS[@]}"

echo "[$(date -Iseconds)] Extraction completed for $RUN_DATE"
