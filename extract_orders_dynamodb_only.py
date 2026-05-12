"""Extract daily orders from S3 bucket and upsert directly to DynamoDB.

Orders are stored in S3 with structure: {bucket}/{YYYY-MM}/{YYYY-MM-DD}.json
This script scans a date range and uploads all matching orders to DynamoDB.
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import boto3
from boto3.dynamodb.types import TypeSerializer
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATE_FMT = "%Y-%m-%d"
DEFAULT_BUCKET = "chupps-data-portal"
DEFAULT_PREFIX = "orders"
DEFAULT_DYNAMODB_TABLE = "history-orders-final"
DEFAULT_AWS_REGION = "ap-south-1"
PRIMARY_KEY_FIELD = "invoice_id"
DEFAULT_DDB_BATCH_SIZE = 25
REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "total_amount",
    "item_quantity",
    "suborder_quantity",
    "order_quantity",
    "sku",
    "canonical_sku",
    "suborder_sku",
    "suborder_marketplace_sku",
    "suborder_model_no",
    "marketplace_sku",
    "order_status",
    "payment_mode",
    "order_type",
    "marketplace",
    "courier",
    "import_warehouse_name",
    "billing_state",
    "state",
    "city",
    "pin_code",
    "size",
    "suborder_size",
    "suborder_selling_price",
    "suborder_cost",
    "suborder_mrp",
    "suborder_productName",
]


# Load environment variables from backend/.env when present.
load_dotenv()


def parse_date(value: str) -> date:
    """Parse date in YYYY-MM-DD format."""
    return datetime.strptime(value, DATE_FMT).date()


def create_s3_client(region_name: str):
    """Create S3 client."""
    return boto3.client("s3", region_name=region_name)


def create_dynamodb_client(region_name: str):
    """Create DynamoDB low-level client."""
    return boto3.client("dynamodb", region_name=region_name)


def _repair_json(text: str) -> List[Dict]:
    """Repair corrupted or truncated JSON. Returns list of parsed objects."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed at position {e.pos}: {e.msg}. Attempting recovery...")

    for end_char in [']', '}']:
        last_pos = text.rfind(end_char)
        if last_pos > 0:
            try:
                truncated = text[:last_pos + 1]
                parsed = json.loads(truncated)
                if isinstance(parsed, list):
                    logger.info(f"Recovered JSON by truncating to last '{end_char}'")
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

    try:
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(pattern, text)
        if matches:
            results = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, list):
                        results.extend(obj)
                    else:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
            if results:
                logger.info(f"Recovered {len(results)} JSON objects via regex extraction")
                return results
    except Exception as e:
        logger.warning(f"Regex recovery failed: {e}")

    logger.error("Could not recover JSON data")
    return []


def _decode_with_fallback(raw_bytes: bytes) -> str:
    """Decode bytes with encoding fallback strategy."""
    encodings = ["utf-8", "latin-1"]

    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue

    try:
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"All decoding attempts failed: {e}")
        return ""


def read_orders_from_s3_file(
    s3_client,
    bucket: str,
    key: str,
) -> List[Dict]:
    """Read and parse orders from S3 JSON file with corruption recovery."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_bytes = response["Body"].read()

        text = _decode_with_fallback(raw_bytes)
        if not text:
            return []

        orders = _repair_json(text)
        return orders if isinstance(orders, list) else []

    except s3_client.exceptions.NoSuchKey:
        logger.debug(f"File not found: {bucket}/{key}")
        return []
    except Exception as e:
        logger.error(f"Error reading S3 file {bucket}/{key}: {e}")
        return []


def list_s3_files_for_date_range(
    s3_client,
    bucket: str,
    start_date: date,
    end_date: date,
    prefix: str = DEFAULT_PREFIX,
) -> List[str]:
    """List all S3 JSON files matching orders/YYYY-MM/YYYY-MM-DD.json pattern within date range."""
    files = []
    current_date = start_date

    while current_date <= end_date:
        year_month = current_date.strftime("%Y-%m")
        day_file = current_date.strftime("%Y-%m-%d")
        s3_key = f"{prefix}/{year_month}/{day_file}.json"
        files.append(s3_key)
        current_date += timedelta(days=1)

    return files


def fetch_orders_from_s3(
    s3_client,
    bucket: str,
    start_date: date,
    end_date: date,
    prefix: str = DEFAULT_PREFIX,
) -> List[Dict]:
    """Fetch all orders from S3 within date range."""
    all_orders: List[Dict] = []
    s3_keys = list_s3_files_for_date_range(s3_client, bucket, start_date, end_date, prefix=prefix)

    for s3_key in s3_keys:
        logger.info(f"Fetching from S3: {bucket}/{s3_key}")
        orders = read_orders_from_s3_file(s3_client, bucket, s3_key)

        if orders:
            all_orders.extend(orders)
            logger.info(f"Loaded {len(orders)} orders from {s3_key}")
        else:
            logger.debug(f"No orders found in {s3_key}")

    return all_orders


def _normalize_for_dynamodb(value):
    """Convert Python values into DynamoDB-safe native values."""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize_for_dynamodb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_dynamodb(v) for v in value]
    return value


def _serialize_item_for_client(item: Dict, serializer: TypeSerializer) -> Dict:
    normalized_item = {k: _normalize_for_dynamodb(v) for k, v in item.items()}
    return {k: serializer.serialize(v) for k, v in normalized_item.items()}


def _project_suborder_for_dynamodb(
    order: Dict,
    suborder: Dict,
    source_tag: str,
    source_month: str,
    primary_key: str,
    suborder_index: int = 1,
) -> Dict:
    """Project a single suborder as a complete DynamoDB row with unique invoice_id."""
    original_invoice_id = order.get(primary_key)

    if suborder_index > 1:
        unique_invoice_id = f"{original_invoice_id}_{suborder_index}"
    else:
        unique_invoice_id = str(original_invoice_id)

    projected = {
        primary_key: unique_invoice_id,
        "source_file": source_tag,
        "source_month": source_month,
        "order_id": order.get("order_id"),
        "order_date": order.get("order_date"),
        "total_amount": order.get("total_amount"),
        "item_quantity": suborder.get("item_quantity"),
        "suborder_model_no": suborder.get("model_no"),
        "suborder_quantity": suborder.get("suborder_quantity"),
        "order_quantity": order.get("order_quantity"),
        "sku": suborder.get("sku"),
        "suborder_sku": suborder.get("sku"),
        "suborder_marketplace_sku": suborder.get("marketplace_sku"),
        "marketplace_sku": suborder.get("marketplace_sku"),
        "order_status": order.get("order_status"),
        "payment_mode": order.get("payment_mode"),
        "order_type": order.get("order_type"),
        "marketplace": order.get("marketplace"),
        "courier": order.get("courier"),
        "import_warehouse_name": order.get("import_warehouse_name"),
        "billing_state": order.get("billing_state"),
        "state": order.get("state"),
        "city": order.get("city"),
        "pin_code": order.get("pin_code"),
        "size": suborder.get("size"),
        "suborder_size": suborder.get("size"),
        "suborder_selling_price": suborder.get("selling_price"),
        "suborder_cost": suborder.get("cost"),
        "suborder_mrp": suborder.get("mrp"),
        "suborder_productName": suborder.get("productName"),
    }

    projected["canonical_sku"] = suborder.get("sku") or order.get("canonical_sku")

    allowed = set(REQUIRED_COLUMNS + [primary_key, "source_file", "source_month"])
    return {k: v for k, v in projected.items() if k in allowed and v is not None}


def _extract_rows_from_suborders(
    order: Dict,
    source_tag: str,
    source_month: str,
    primary_key: str,
) -> List[Dict]:
    """Extract one row per suborder. If no suborders, use order data with index 1."""
    rows = []
    suborders = order.get("suborders", [])

    if not isinstance(suborders, list):
        suborders = []

    if not suborders:
        row = _project_suborder_for_dynamodb(
            order=order,
            suborder=order,
            source_tag=source_tag,
            source_month=source_month,
            primary_key=primary_key,
            suborder_index=1,
        )
        if row and primary_key in row:
            rows.append(row)
    else:
        for idx, suborder in enumerate(suborders, start=1):
            row = _project_suborder_for_dynamodb(
                order=order,
                suborder=suborder,
                source_tag=source_tag,
                source_month=source_month,
                primary_key=primary_key,
                suborder_index=idx,
            )
            if row and primary_key in row:
                rows.append(row)

    return rows


def prepare_rows_for_dynamodb(
    orders: List[Dict],
    source_tag: str,
    source_month: str,
    primary_key: str = PRIMARY_KEY_FIELD,
) -> List[Dict]:
    """Prepare one row per suborder for DynamoDB upsert, preserving order relationships."""
    prepared_rows: List[Dict] = []

    for order in orders:
        if not isinstance(order, dict):
            continue

        original_key = order.get(primary_key)
        if original_key in (None, ""):
            continue

        rows = _extract_rows_from_suborders(
            order=order,
            source_tag=source_tag,
            source_month=source_month,
            primary_key=primary_key,
        )
        prepared_rows.extend(rows)

    return prepared_rows


def upsert_orders_into_dynamodb(
    dynamodb_client,
    table_name: str,
    rows: List[Dict],
    batch_size: int = DEFAULT_DDB_BATCH_SIZE,
) -> int:
    """Upsert rows into DynamoDB in batches using PutRequest."""
    if not rows:
        return 0

    if batch_size < 1 or batch_size > 25:
        raise ValueError("batch_size must be between 1 and 25 for DynamoDB")

    serializer = TypeSerializer()
    total_upserted = 0

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]

        pending = {
            table_name: [
                {"PutRequest": {"Item": _serialize_item_for_client(row, serializer)}}
                for row in chunk
            ]
        }

        while pending.get(table_name):
            response = dynamodb_client.batch_write_item(RequestItems=pending)
            unprocessed = response.get("UnprocessedItems", {})
            pending = {table_name: unprocessed.get(table_name, [])}
            if pending[table_name]:
                time.sleep(0.5)

        total_upserted += len(chunk)

    return total_upserted


def run_extraction(
    start_day: date,
    end_day: date,
    s3_bucket: str,
    prefix: str,
    table_name: str,
    aws_region: str,
) -> None:
    """Extract orders from S3 and upsert to DynamoDB."""
    if start_day > end_day:
        raise ValueError("start_date must be less than or equal to end_date")

    s3_client = create_s3_client(region_name=aws_region)
    dynamodb_client = create_dynamodb_client(region_name=aws_region)

    logger.info(f"Starting S3 extraction from {start_day} to {end_day}")
    logger.info(f"S3 bucket: {s3_bucket}, DynamoDB table: {table_name}, Region: {aws_region}")

    current_day = start_day
    total_orders = 0
    total_upserted = 0

    while current_day <= end_day:
        logger.info(f"Processing date: {current_day}")

        daily_orders = fetch_orders_from_s3(
            s3_client=s3_client,
            bucket=s3_bucket,
            start_date=current_day,
            end_date=current_day,
            prefix=prefix,
        )

        if not daily_orders:
            logger.warning(f"No orders found for {current_day}")
            current_day += timedelta(days=1)
            continue

        source_month = current_day.strftime("%Y-%m")
        source_tag = f"s3://{s3_bucket}/{prefix}/{source_month}"

        rows_for_dynamodb = prepare_rows_for_dynamodb(
            orders=daily_orders,
            source_tag=source_tag,
            source_month=source_month,
        )

        logger.info(f"Prepared {len(rows_for_dynamodb)} rows for DynamoDB upsert on {current_day}")

        upserted_count = upsert_orders_into_dynamodb(
            dynamodb_client=dynamodb_client,
            table_name=table_name,
            rows=rows_for_dynamodb,
        )

        logger.info(
            f"✓ Completed {current_day}: fetched {len(daily_orders)} orders | upserted {upserted_count} rows"
        )

        total_orders += len(daily_orders)
        total_upserted += upserted_count
        current_day += timedelta(days=1)

    logger.info(f"Extraction complete. Total orders: {total_orders}, Upserted rows: {total_upserted}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract orders from S3 bucket and upsert to DynamoDB"
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_BUCKET", DEFAULT_BUCKET),
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("S3_PREFIX", DEFAULT_PREFIX),
        help=f"S3 prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--ddb-table",
        default=os.getenv("HISTORY_ORDERS_DYNAMODB_TABLE", DEFAULT_DYNAMODB_TABLE),
        help=f"DynamoDB target table (default: {DEFAULT_DYNAMODB_TABLE})",
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
        help=f"AWS region (default: {DEFAULT_AWS_REGION})",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    start_day = parse_date(args.start_date)
    end_day = parse_date(args.end_date)

    run_extraction(
        start_day=start_day,
        end_day=end_day,
        s3_bucket=args.s3_bucket,
        prefix=args.prefix,
        table_name=args.ddb_table,
        aws_region=args.aws_region,
    )


if __name__ == "__main__":
    main()