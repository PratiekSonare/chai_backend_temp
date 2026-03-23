#!/bin/bash
# Bash wrapper for uploading orders to Supabase

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/upload_orders_to_supabase.py"

# Load .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check required packages
python3 -c "import supabase" 2>/dev/null || {
    echo "📦 Installing required packages..."
    pip install supabase python-dotenv pandas
}

# Run the upload script
python3 "$PYTHON_SCRIPT"
