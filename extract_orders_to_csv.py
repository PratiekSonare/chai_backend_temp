"""Extract daily orders from EasyEcom API and save to CSV.

CSV output format:
- Filename: orders_YYYY-MM-DD_to_YYYY-MM-DD.csv
- One row per suborder with all extracted fields
"""

import argparse
import csv
import os
from datetime import datetime, timedelta, date
from typing import Dict, List

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.easyecom.io"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
PRIMARY_KEY_FIELD = "invoice_id"
REQUIRED_COLUMNS = [
    "invoice_id",
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
    'billing_state',         
    "state",
    "city",
    "pin_code",
    "size",
    "suborder_size",
    "suborder_selling_price",
    "suborder_cost",
    "suborder_mrp",
    "suborder_productName",
    "source_file",
    "source_month",
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

        # API behavior: 400 indicates pagination end.
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


def _project_suborder_for_csv(
    order: Dict,
    suborder: Dict,
    source_tag: str,
    source_month: str,
    primary_key: str,
    suborder_index: int = 1,
) -> Dict:
    """Project a single suborder as a complete CSV row with unique invoice_id."""
    original_invoice_id = order.get(primary_key)
    
    # Ensure primary key is always a string for consistency
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
        "suborder_model_no": suborder.get("suborder_model_no"),
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
        "state": order.get("state"),
        "billing_state": order.get("billing_state"),
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

    allowed = set(REQUIRED_COLUMNS)
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
    
    # If no suborders, create one row with the order's data
    if not suborders:
        row = _project_suborder_for_csv(
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
        # Create one row per suborder, each with unique invoice_id
        for idx, suborder in enumerate(suborders, start=1):
            row = _project_suborder_for_csv(
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


def prepare_rows_for_csv(
    orders: List[Dict],
    source_key: str,
    source_month: str,
    primary_key: str = PRIMARY_KEY_FIELD,
) -> List[Dict]:
    """Prepare one row per suborder for CSV export."""
    prepared_rows: List[Dict] = []

    for idx, order in enumerate(orders):
        if not isinstance(order, dict):
            continue

        original_key = order.get(primary_key)
        if original_key in (None, ""):
            continue

        rows = _extract_rows_from_suborders(
            order=order,
            source_tag=source_key,
            source_month=source_month,
            primary_key=primary_key,
        )
        prepared_rows.extend(rows)

    return prepared_rows


def save_rows_to_csv(
    filename: str,
    rows: List[Dict],
) -> int:
    """Save rows to CSV file."""
    if not rows:
        print(f"No rows to save to {filename}")
        return 0

    # Use all unique keys from all rows to ensure we capture all fields
    fieldnames = list(REQUIRED_COLUMNS)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def run_extraction(
    start_day: date,
    end_day: date,
    output_file: str,
    base_url: str,
) -> None:
    """Extract orders for date range and save to CSV."""
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")

    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in environment")

    if start_day > end_day:
        raise ValueError("start_date must be less than or equal to end_date")

    current_day = start_day
    total_orders = 0
    total_rows = 0
    all_rows = []

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

        # Use the date string as source_key for tracking
        source_key = f"orders/{current_day.strftime('%Y-%m')}/{current_day.strftime(DATE_FMT)}.json"
        
        rows_for_csv = prepare_rows_for_csv(
            orders=daily_orders,
            source_key=source_key,
            source_month=current_day.strftime("%Y-%m"),
        )

        day_count = len(daily_orders)
        row_count = len(rows_for_csv)
        total_orders += day_count
        total_rows += row_count
        all_rows.extend(rows_for_csv)

        print(f"Fetched {day_count} orders, extracted {row_count} rows for {current_day}")
        current_day += timedelta(days=1)

    # Save all rows to single CSV file
    saved_count = save_rows_to_csv(output_file, all_rows)

    print("\nExtraction complete")
    print(f"Total orders fetched: {total_orders}")
    print(f"Total rows extracted: {total_rows}")
    print(f"Rows saved to CSV: {saved_count}")
    print(f"Output file: {output_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract daily EasyEcom orders and save to CSV"
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename (default: orders_YYYY-MM-DD_to_YYYY-MM-DD.csv)",
    )
    parser.add_argument("--base-url", default=os.getenv("EASYECOM_BASE_URL", DEFAULT_BASE_URL), help="EasyEcom API base URL")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    start_day = parse_date(args.start_date)
    end_day = parse_date(args.end_date)

    # Generate default output filename if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = f"orders_{start_day.strftime(DATE_FMT)}_to_{end_day.strftime(DATE_FMT)}.csv"

    run_extraction(
        start_day=start_day,
        end_day=end_day,
        output_file=output_file,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
