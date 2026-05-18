#!/bin/bash
# run_generate_sku_insights.sh
# Wrapper script for generating SKU insights daily via systemd timer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Run the insights generator
python3 "${SCRIPT_DIR}/generate_sku_insights.py" \
    --aws-region ap-south-1 \
    --bucket chupps-data-portal \
    --prefix sku-metrics \
    --log-level INFO

echo "SKU insights generation completed successfully"
