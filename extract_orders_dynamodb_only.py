"""Extract daily orders from EasyEcom API and upsert directly to DynamoDB.

This variant intentionally skips S3 writes to avoid duplicate storage.
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import boto3
import requests
from boto3.dynamodb.types import TypeSerializer
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.easyecom.io"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_DYNAMODB_TABLE = "history-orders-latest-4"
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
    'marketplace',           
    'courier',               
    'import_warehouse_name', 
    "state",
    "size",
    "suborder_size",
]


# Load environment variables from backend/.env when present.
load_dotenv()


def parse_date(value: str) -> date:
    """Parse date in YYYY-MM-DD format."""
    return datetime.strptime(value, DATE_FMT).date()


def fetch_orders_for_window(
    start_date: str,
    end_date: str,
    api_key: str,
    jwt_token: str,
    base_url: str,
) -> List[Dict]:
    """Fetch all orders for a date window with pagination support."""
    all_orders: List[Dict] = []
    url = f"{base_url}/orders/V2/getAllOrders"

    params = {
        "limit": 250,
        "start_date": start_date,
        "end_date": end_date,
    }

    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    page = 1

    while True:
        print(f"Fetching page {page} for {start_date} to {end_date}")
        response = requests.get(url, params=params, headers=headers, timeout=60)

        # API behavior in existing code: 400 indicates pagination end.
        if response.status_code == 400:
            print(f"Pagination ended with 400 at page {page}")
            break

        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 200 or "data" not in payload:
            print(f"API returned unexpected payload at page {page}: {payload}")
            break

        page_orders = payload["data"].get("orders", [])
        if not page_orders:
            print(f"No orders found on page {page}, stopping")
            break

        all_orders.extend(page_orders)
        print(f"Fetched {len(page_orders)} orders on page {page}")

        next_url = payload["data"].get("nextUrl")
        if not next_url:
            print(f"No nextUrl found at page {page}, stopping")
            break

        if next_url.startswith("/"):
            url = f"{base_url}{next_url}"
        elif next_url.startswith("http"):
            url = next_url
        else:
            url = f"{base_url}/{next_url.lstrip('/')}"

        # nextUrl already includes query params.
        params = {}
        page += 1

    return all_orders


def create_dynamodb_client(region_name: str):
    """Create DynamoDB low-level client."""
    return boto3.client("dynamodb", region_name=region_name)


def _normalize_for_dynamodb(value):
    """Convert Python values into DynamoDB-safe native values."""
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
    """Serialize plain dict item into DynamoDB AttributeValue map."""
    normalized_item = {k: _normalize_for_dynamodb(v) for k, v in item.items()}
    return {k: serializer.serialize(v) for k, v in normalized_item.items()}


def _first_suborder(order: Dict) -> Dict:
    suborders = order.get("suborders")
    if isinstance(suborders, list) and suborders and isinstance(suborders[0], dict):
        return suborders[0]
    return {}


def _pick_first_non_empty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _project_order_for_dynamodb(
    order: Dict,
    source_tag: str,
    source_month: str,
    primary_key: str,
) -> Dict:
    first_sub = _first_suborder(order)

    projected = {
        primary_key: order.get(primary_key),
        "source_file": source_tag,
        "source_month": source_month,
        "order_id": order.get("order_id"),
        "order_date": order.get("order_date"),
        "total_amount": order.get("total_amount"),
        "item_quantity": _pick_first_non_empty(
            order.get("item_quantity"),
            first_sub.get("item_quantity"),
        ),
        "suborder_quantity": _pick_first_non_empty(
            order.get("suborder_quantity"),
            first_sub.get("suborder_quantity"),
        ),
        "order_quantity": order.get("order_quantity"),
        "sku": _pick_first_non_empty(
            order.get("sku"),
            first_sub.get("sku"),
        ),
        "suborder_model_no": _pick_first_non_empty(
            first_sub.get("model_no"),
        ),
        "suborder_sku": _pick_first_non_empty(
            order.get("suborder_sku"),
            first_sub.get("sku"),
        ),
        "suborder_marketplace_sku": _pick_first_non_empty(
            first_sub.get("marketplace_sku"),
        ),
        "marketplace_sku": _pick_first_non_empty(
            order.get("marketplace_sku"),
            first_sub.get("marketplace_sku"),
        ),
        "order_status": order.get("order_status"),
        "payment_mode": order.get("payment_mode"),
        "order_type": order.get("order_type"),
        "marketplace": order.get("marketplace"),
        "courier": order.get("courier"),
        "import_warehouse_name": order.get("import_warehouse_name"),
        "state": order.get("state"),
        "size": _pick_first_non_empty(
            order.get("size"),
            first_sub.get("size"),
        ),
        "suborder_size": _pick_first_non_empty(
            order.get("suborder_size"),
            first_sub.get("size"),
        ),
    }

    projected["canonical_sku"] = _pick_first_non_empty(
        order.get("canonical_sku"),
        projected.get("sku"),
        projected.get("suborder_sku"),
        projected.get("suborder_marketplace_sku"),
        projected.get("marketplace_sku"),
    )

    allowed = set(REQUIRED_COLUMNS + [primary_key, "source_file", "source_month"])
    return {k: v for k, v in projected.items() if k in allowed and v is not None}


def prepare_rows_for_dynamodb(
    orders: List[Dict],
    source_tag: str,
    source_month: str,
    primary_key: str = PRIMARY_KEY_FIELD,
) -> List[Dict]:
    """Prepare projected rows and add lineage metadata for DynamoDB upserts."""
    prepared_rows: List[Dict] = []

    for idx, order in enumerate(orders):
        if not isinstance(order, dict):
            continue

        row = _project_order_for_dynamodb(
            order=order,
            source_tag=source_tag,
            source_month=source_month,
            primary_key=primary_key,
        )

        if row.get(primary_key) in (None, ""):
            raise ValueError(
                f"Order at index {idx} is missing required primary key '{primary_key}'"
            )

        prepared_rows.append(row)

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
    base_url: str,
    table_name: str,
    aws_region: str,
) -> None:
    """Extract orders day-by-day and upsert each day directly to DynamoDB."""
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")

    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in environment")

    if start_day > end_day:
        raise ValueError("start_date must be less than or equal to end_date")

    dynamodb_client = create_dynamodb_client(region_name=aws_region)

    current_day = start_day
    total_orders = 0
    total_upserted = 0

    while current_day <= end_day:
        day_start = datetime.combine(current_day, datetime.min.time()).strftime(DATETIME_FMT)
        day_end = datetime.combine(current_day, datetime.max.time().replace(microsecond=0)).strftime(DATETIME_FMT)

        print(f"\nProcessing {current_day}...")
        daily_orders = fetch_orders_for_window(
            start_date=day_start,
            end_date=day_end,
            api_key=api_key,
            jwt_token=jwt_token,
            base_url=base_url,
        )

        source_tag = f"supabase_only/{current_day.strftime(DATE_FMT)}.json"
        rows_for_dynamodb = prepare_rows_for_dynamodb(
            orders=daily_orders,
            source_tag=source_tag,
            source_month=current_day.strftime("%Y-%m"),
        )

        upserted_count = upsert_orders_into_dynamodb(
            dynamodb_client=dynamodb_client,
            table_name=table_name,
            rows=rows_for_dynamodb,
        )

        day_count = len(daily_orders)
        total_orders += day_count
        total_upserted += upserted_count

        print(f"Fetched {day_count} orders")
        print(f"Upserted {upserted_count} rows into DynamoDB table '{table_name}'")
        current_day += timedelta(days=1)

    print("\nExtraction complete")
    print(f"Total orders fetched: {total_orders}")
    print(f"Total rows upserted: {total_upserted}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract daily EasyEcom orders and upsert directly to DynamoDB"
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EASYECOM_BASE_URL", DEFAULT_BASE_URL),
        help="EasyEcom API base URL",
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
        base_url=args.base_url,
        table_name=args.ddb_table,
        aws_region=args.aws_region,
    )


if __name__ == "__main__":
    main()
