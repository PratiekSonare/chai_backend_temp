#!/bin/bash

# Export DynamoDB table to CSV
# Usage: ./export_dynamodb_to_csv.sh [table-name] [aws-region] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
# Or positional: ./export_dynamodb_to_csv.sh [table-name] [aws-region] [start-date] [end-date]
# Dates in YYYY-MM-DD format for optional filtering

TABLE_NAME="${1:-history-orders-final}"
AWS_REGION="${2:-ap-south-1}"
START_DATE=""
END_DATE=""
OUTPUT_FILE="${TABLE_NAME}_export.csv"

# Parse flag-based arguments
shift 2  # Skip table name and region
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-date|--start_date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date|--end_date)
            END_DATE="$2"
            shift 2
            ;;
        *)
            # Assume positional argument (backward compatibility)
            if [ -z "$START_DATE" ]; then
                START_DATE="$1"
            elif [ -z "$END_DATE" ]; then
                END_DATE="$1"
            fi
            shift
            ;;
    esac
done

# Build output filename with date range if provided
if [ -n "$START_DATE" ] && [ -n "$END_DATE" ]; then
    OUTPUT_FILE="${TABLE_NAME}_${START_DATE}_to_${END_DATE}_export.csv"
    echo "Exporting DynamoDB table '$TABLE_NAME' from region '$AWS_REGION' (dates: $START_DATE to $END_DATE)..."
else
    echo "Exporting DynamoDB table '$TABLE_NAME' from region '$AWS_REGION'..."
fi
echo "Output file: $OUTPUT_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if AWS CLI and jq are available
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found. Please install it first."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Please install it first."
    exit 1
fi

# Scan the table and export to CSV
# If dates are provided, filter results by order_date
if [ -n "$START_DATE" ] && [ -n "$END_DATE" ]; then
    echo "Scanning with date filter: $START_DATE to $END_DATE..."
    aws dynamodb scan \
        --table-name "$TABLE_NAME" \
        --region "$AWS_REGION" \
        --output json | \
    jq -r --arg start "$START_DATE" --arg end "$END_DATE" '
      if (.Items | length) == 0 then
        "No items found in table"
      else
        ((.Items[0] | keys_unsorted)) as $keys |
        ($keys | @csv),
        (.Items[] | select(.order_date.S >= $start and .order_date.S <= $end) | [$keys[] as $k | .[$k] | if type == "object" then .S // .N // .BOOL // .NULL else . end] | @csv)
      end
    ' > "$OUTPUT_FILE"
else
    echo "Scanning all records (no date filter)..."
    aws dynamodb scan \
        --table-name "$TABLE_NAME" \
        --region "$AWS_REGION" \
        --output json | \
    jq -r '
      if (.Items | length) == 0 then
        "No items found in table"
      else
        ((.Items[0] | keys_unsorted)) as $keys |
        ($keys | @csv),
        (.Items[] | [$keys[] as $k | .[$k] | if type == "object" then .S // .N // .BOOL // .NULL else . end] | @csv)
      end
    ' > "$OUTPUT_FILE"
fi

if [ $? -eq 0 ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo ""
    echo "✓ Export successful!"
    echo "  Table: $TABLE_NAME"
    echo "  Region: $AWS_REGION"
    if [ -n "$START_DATE" ] && [ -n "$END_DATE" ]; then
        echo "  Date range: $START_DATE to $END_DATE"
    fi
    echo "  Lines: $LINES"
    echo "  File: $OUTPUT_FILE"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "✗ Export failed"
    exit 1
fi
