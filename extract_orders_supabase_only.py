"""Extract daily orders from EasyEcom API and upload only to Supabase.

This variant intentionally skips S3 writes to avoid creating/updating bucket objects.
"""

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set

import requests
from dotenv import load_dotenv
from supabase import Client, create_client


DEFAULT_BASE_URL = "https://api.easyecom.io"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
TARGET_TABLE = "history_orders_raw"
DEFAULT_INSERT_CHUNK_SIZE = 500
SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "table_schema.txt")


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


def create_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and one of SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY must be set in environment"
        )

    return create_client(supabase_url, supabase_key)


def normalize_for_supabase(value):
    """Keep primitives as-is and serialize complex values for text-compatible inserts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def load_allowed_columns(schema_file_path: str = SCHEMA_FILE) -> Optional[Set[str]]:
    """Parse column names from local table schema file when available."""
    if not os.path.exists(schema_file_path):
        return None

    with open(schema_file_path, "r", encoding="utf-8") as schema_file:
        lines = schema_file.readlines()

    columns: Set[str] = set()
    pattern = re.compile(r'^\s*(?:"([^"]+)"|([A-Za-z0-9_\.]+))\s+')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("create table"):
            continue
        if stripped.startswith(")"):
            break

        match = pattern.match(stripped)
        if not match:
            continue

        col_name = match.group(1) or match.group(2)
        if col_name:
            columns.add(col_name)

    return columns or None


def prepare_rows_for_supabase(
    orders: List[Dict],
    source_tag: str,
    source_month: str,
    allowed_columns: Optional[Set[str]] = None,
) -> List[Dict]:
    """Prepare rows and add lineage metadata."""
    prepared_rows: List[Dict] = []

    for order in orders:
        if not isinstance(order, dict):
            continue

        if allowed_columns:
            row = {
                key: normalize_for_supabase(value)
                for key, value in order.items()
                if key in allowed_columns
            }
        else:
            row = {key: normalize_for_supabase(value) for key, value in order.items()}

        if not allowed_columns or "source_file" in allowed_columns:
            row["source_file"] = source_tag
        if not allowed_columns or "source_month" in allowed_columns:
            row["source_month"] = source_month

        if not row:
            continue

        prepared_rows.append(row)

    return prepared_rows


def insert_orders_into_supabase(
    supabase: Client,
    table_name: str,
    rows: List[Dict],
    chunk_size: int = DEFAULT_INSERT_CHUNK_SIZE,
) -> int:
    """Insert rows into Supabase in chunks."""
    if not rows:
        return 0

    total_inserted = 0

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        try:
            supabase.table(table_name).insert(chunk).execute()
        except Exception as exc:
            message = str(exc)
            if "42501" in message or "row-level security policy" in message.lower():
                raise RuntimeError(
                    f"RLS blocked insert on table '{table_name}'. "
                    "Use SUPABASE_SERVICE_ROLE_KEY for ETL writes or add an INSERT policy for your role."
                ) from exc
            raise
        total_inserted += len(chunk)

    return total_inserted


def run_extraction(
    start_day: date,
    end_day: date,
    base_url: str,
    table_name: str,
) -> None:
    """Extract orders day-by-day and upload each day only to Supabase."""
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")

    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in environment")

    if start_day > end_day:
        raise ValueError("start_date must be less than or equal to end_date")

    supabase_client = create_supabase_client()
    allowed_columns = load_allowed_columns()

    current_day = start_day
    total_orders = 0

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
        rows_for_supabase = prepare_rows_for_supabase(
            orders=daily_orders,
            source_tag=source_tag,
            source_month=current_day.strftime("%Y-%m"),
            allowed_columns=allowed_columns,
        )

        inserted_count = insert_orders_into_supabase(
            supabase=supabase_client,
            table_name=table_name,
            rows=rows_for_supabase,
        )

        day_count = len(daily_orders)
        total_orders += day_count

        print(f"Fetched {day_count} orders")
        print(f"Inserted {inserted_count} rows into Supabase table '{table_name}'")
        current_day += timedelta(days=1)

    print("\nExtraction complete")
    print(f"Total orders fetched: {total_orders}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract daily EasyEcom orders and upload only to Supabase"
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EASYECOM_BASE_URL", DEFAULT_BASE_URL),
        help="EasyEcom API base URL",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("SUPABASE_TARGET_TABLE", TARGET_TABLE),
        help=f"Supabase target table (default: {TARGET_TABLE})",
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
        table_name=args.table,
    )


if __name__ == "__main__":
    main()
