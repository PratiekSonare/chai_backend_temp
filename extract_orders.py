"""Extract daily orders from EasyEcom API and upload to S3.

S3 output format:
- Bucket: chupps-data-portal
- Prefix: orders/YYYY-MM/YYYY-MM-DD.json
"""

import argparse
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List

import boto3
import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.easyecom.io"
DEFAULT_BUCKET = "chupps-data-portal"
DEFAULT_PREFIX = "orders"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


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


def upload_daily_orders(
    s3_client,
    bucket_name: str,
    prefix: str,
    day: date,
    orders: List[Dict],
) -> str:
    """Upload one day's orders in exact key format orders/YYYY-MM/YYYY-MM-DD.json."""
    month_folder = day.strftime("%Y-%m")
    file_name = f"{day.strftime(DATE_FMT)}.json"
    key = f"{prefix}/{month_folder}/{file_name}"

    body = json.dumps(orders, ensure_ascii=True)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return key


def run_extraction(
    start_day: date,
    end_day: date,
    bucket_name: str,
    prefix: str,
    base_url: str,
) -> None:
    """Extract orders day-by-day and upload each day to S3."""
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")

    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in environment")

    if start_day > end_day:
        raise ValueError("start_date must be less than or equal to end_date")

    s3_client = boto3.client("s3")

    current_day = start_day
    total_orders = 0
    total_files = 0

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

        key = upload_daily_orders(
            s3_client=s3_client,
            bucket_name=bucket_name,
            prefix=prefix,
            day=current_day,
            orders=daily_orders,
        )

        day_count = len(daily_orders)
        total_orders += day_count
        total_files += 1

        print(f"Uploaded {day_count} orders to s3://{bucket_name}/{key}")
        current_day += timedelta(days=1)

    print("\nExtraction complete")
    print(f"Files uploaded: {total_files}")
    print(f"Total orders uploaded: {total_orders}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract daily EasyEcom orders and upload to S3 in month/day JSON structure"
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"S3 prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--base-url", default=os.getenv("EASYECOM_BASE_URL", DEFAULT_BASE_URL), help="EasyEcom API base URL")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    start_day = parse_date(args.start_date)
    end_day = parse_date(args.end_date)

    run_extraction(
        start_day=start_day,
        end_day=end_day,
        bucket_name=args.bucket,
        prefix=args.prefix,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
