#!/usr/bin/env python3
"""
product_metrics_updater.py
Daily background job: reads new orders from DynamoDB, computes per-SKU metrics,
and incrementally updates JSON profiles in S3 under /metrics/<sku>.json
"""

import json
import re
import logging
import argparse
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DYNAMODB_TABLE = "history-orders-final"
DEFAULT_S3_BUCKET = "chupps-data-portal"
DEFAULT_S3_PREFIX = "sku-metrics"
DEFAULT_AWS_REGION = "ap-south-1"

DYNAMODB_TABLE = os.getenv("PRODUCT_METRICS_DDB_TABLE", DEFAULT_DYNAMODB_TABLE)
S3_BUCKET = os.getenv("PRODUCT_METRICS_S3_BUCKET", DEFAULT_S3_BUCKET)
S3_PREFIX = os.getenv("PRODUCT_METRICS_S3_PREFIX", DEFAULT_S3_PREFIX)
META_KEY = f"{S3_PREFIX}/_meta/last_run.json"

# DEFAULT_BASE_URL = "https://api.easyecom.io"
# DATE_FMT = "%Y-%m-%d"
# DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
# DEFAULT_DYNAMODB_TABLE = "history-orders-dev"
# DEFAULT_AWS_REGION = "ap-south-1"
# PRIMARY_KEY_FIELD = "invoice_id"
# DEFAULT_DDB_BATCH_SIZE = 25

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────
dynamodb = None
s3 = None
table = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-SKU product metrics and incrementally update S3 JSON profiles"
    )
    parser.add_argument(
        "--ddb-table",
        default=DYNAMODB_TABLE,
        help=f"DynamoDB source table (default: {DYNAMODB_TABLE})",
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket for metrics profiles (default: {S3_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=S3_PREFIX,
        help=f"S3 prefix for metrics profiles (default: {S3_PREFIX})",
    )
    parser.add_argument(
        "--meta-key",
        default=None,
        help="S3 key for last-run metadata (default: <prefix>/_meta/last_run.json)",
    )
    parser.add_argument(
        "--since-date",
        default=None,
        help="Override incremental start date in YYYY-MM-DD (skips reading last-run meta)",
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
        help=f"AWS region (default: {DEFAULT_AWS_REGION})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser


# ── SKU Parsing ───────────────────────────────────────────────────────────────
def parse_sku(raw_sku: str) -> tuple[str, str]:
    """
    Extract canonical SKU and size from the raw 'sku' field.

    The format is: [alpha-prefix-]<sku_part1>-<sku_part2>-<size>
    Leading segments containing no digits are treated as ignorable prefixes.

    Examples:
        '10400-120-7'       → sku='10400-120',  size='7'
        'ABC-10421-115-9'   → sku='10421-115',  size='9'
    """
    parts = raw_sku.strip().split("-")

    # Drop leading segments that contain no digits (e.g. 'ABC')
    while parts and not re.search(r"\d", parts[0]):
        parts.pop(0)

    if len(parts) < 2:
        # Can't split cleanly — treat entire value as SKU, size unknown
        return raw_sku, ""

    size = parts[-1]
    sku  = "-".join(parts[:-1])
    return sku, size


# ── DynamoDB helpers ──────────────────────────────────────────────────────────
def get_last_run() -> str:
    """Return the last successful run date (YYYY-MM-DD), or epoch start as fallback."""
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=META_KEY)
        meta = json.loads(obj["Body"].read())
        return meta["last_run_date"]
    except s3.exceptions.NoSuchKey:
        return "1970-01-01"
    except Exception as e:
        log.warning(f"Could not read last_run meta: {e}. Defaulting to full scan.")
        return "1970-01-01"


def save_last_run(date_str: str) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=META_KEY,
        Body=json.dumps({
            "last_run_date": date_str,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }),
        ContentType="application/json",
    )


def scan_orders_since(since_date: str) -> list[dict]:
    """
    Full table scan filtered to order_date >= since_date.
    Consider adding a GSI on order_date for large tables.
    """
    items  = []
    kwargs = {"FilterExpression": Attr("order_date").gte(since_date)}

    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    log.info(f"Fetched {len(items)} records since {since_date}")
    return items


# ── Metric computation ────────────────────────────────────────────────────────
def to_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(Decimal(str(val)))
    except Exception:
        return 0.0


def compute_sku_metrics(df: pd.DataFrame) -> dict:
    """Compute all metrics for a single canonical SKU's dataframe."""
    df = df.copy()

    # Cast numeric columns
    for col in ["suborder_selling_price", "suborder_cost", "suborder_mrp", "suborder_quantity"]:
        df[col] = df[col].apply(to_float)

    df["revenue"]      = df["suborder_selling_price"] * df["suborder_quantity"]
    df["cogs"]         = df["suborder_cost"]          * df["suborder_quantity"]
    df["total_mrp"]    = df["suborder_mrp"]           * df["suborder_quantity"]
    df["gross_profit"] = df["revenue"] - df["cogs"]

    total_revenue    = df["revenue"].sum()
    total_units      = df["suborder_quantity"].sum()
    total_orders     = df["order_id"].nunique()
    avg_order_value  = total_revenue / total_orders if total_orders else 0
    gross_margin_pct = (df["gross_profit"].sum() / total_revenue * 100) if total_revenue else 0

    # ── Order quality ─────────────────────────────────────────────────────────
    status_counts = (
        df.groupby(df["order_status"].str.lower())["order_id"]
          .nunique()
          .to_dict()
    )
    total_status_cnt  = max(sum(status_counts.values()), 1)
    cancellation_rate = round(status_counts.get("cancelled", 0) / total_status_cnt * 100, 2)
    return_rate       = round(status_counts.get("returned",  0) / total_status_cnt * 100, 2)

    # ── Payment mode ──────────────────────────────────────────────────────────
    payment_split = (
        df.groupby("payment_mode")
          .agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"))
          .round(2)
          .to_dict(orient="index")
    )

    # ── Marketplace metrics (key client requirement — tracks price/margin drift) ──
    marketplace_metrics = {}
    for mp, grp in df.groupby("marketplace"):
        mp_rev     = grp["revenue"].sum()
        mp_gp      = grp["gross_profit"].sum()
        mp_units   = grp["suborder_quantity"].sum()
        avg_sp     = grp["suborder_selling_price"].mean()
        avg_mrp    = grp["suborder_mrp"].mean()

        marketplace_metrics[mp] = {
            "revenue":            round(mp_rev, 2),
            "units_sold":         round(mp_units, 2),
            "order_count":        int(grp["order_id"].nunique()),
            "avg_selling_price":  round(avg_sp, 2),
            "avg_mrp":            round(avg_mrp, 2),
            "avg_cost":           round(grp["suborder_cost"].mean(), 2),
            "gross_margin_pct":   round(mp_gp / mp_rev * 100, 2) if mp_rev else 0,
            "mrp_discount_pct":   round((avg_mrp - avg_sp) / avg_mrp * 100, 2) if avg_mrp else 0,
            "revenue_share_pct":  round(mp_rev / total_revenue * 100, 2) if total_revenue else 0,
        }

    # ── Price & margin history — daily per marketplace ────────────────────────
    # Appended each day a SKU sells. Captures CP/SP/MRP and margin per channel
    # so price drift across marketplaces is fully traceable over time.
    price_history_by_marketplace = []
    for (date, mp), grp in df.groupby(["order_date", "marketplace"]):
        rev      = grp["revenue"].sum()
        cogs_sum = grp["cogs"].sum()
        mrp_sum  = grp["total_mrp"].sum()
        units    = grp["suborder_quantity"].sum()
        avg_sp   = rev      / units if units else 0
        avg_cp   = cogs_sum / units if units else 0
        avg_mrp  = mrp_sum  / units if units else 0
        price_history_by_marketplace.append({
            "date":             date,
            "marketplace":      mp,
            "avg_sp":           round(avg_sp, 2),
            "avg_cp":           round(avg_cp, 2),
            "avg_mrp":          round(avg_mrp, 2),
            "gross_margin_pct": round((rev - cogs_sum) / rev * 100, 2) if rev else 0,
            "mrp_discount_pct": round((avg_mrp - avg_sp) / avg_mrp * 100, 2) if avg_mrp else 0,
        })
    price_history_by_marketplace.sort(key=lambda x: x["date"])

    # ── Geographic ────────────────────────────────────────────────────────────
    state_df = (
        df.groupby("state")
          .agg(revenue=("revenue", "sum"), units_sold=("suborder_quantity", "sum"))
          .round(2)
          .sort_values("revenue", ascending=False)
    )
    state_metrics = state_df.to_dict(orient="index")
    top_states    = list(state_df.index[:5])

    # ── Size breakdown ────────────────────────────────────────────────────────
    size_df = (
        df.groupby("size")
          .agg(units_sold=("suborder_quantity", "sum"), revenue=("revenue", "sum"))
          .round(2)
    )
    size_df["revenue_share_pct"] = (size_df["revenue"] / total_revenue * 100).round(2)
    size_metrics = size_df.to_dict(orient="index")

    # ── Fulfillment ───────────────────────────────────────────────────────────
    courier_dist   = df.groupby("courier")["order_id"].nunique().to_dict()
    warehouse_dist = df.groupby("import_warehouse_name")["suborder_quantity"].sum().round(2).to_dict()

    # ── Daily series ──────────────────────────────────────────────────────────
    daily = (
        df.groupby("order_date")
          .agg(
              revenue      =("revenue",           "sum"),
              units_sold   =("suborder_quantity", "sum"),
              orders       =("order_id",          "nunique"),
              gross_profit =("gross_profit",      "sum"),
              cogs         =("cogs",              "sum"),
              total_mrp    =("total_mrp",         "sum"),
          )
          .reset_index()
          .sort_values("order_date")
    )
    daily["avg_sp"]           = (daily["revenue"]      / daily["units_sold"]).round(2)
    daily["avg_cp"]           = (daily["cogs"]         / daily["units_sold"]).round(2)
    daily["avg_mrp"]          = (daily["total_mrp"]    / daily["units_sold"]).round(2)
    daily["gross_margin_pct"] = (daily["gross_profit"] / daily["revenue"] * 100).round(2)
    daily["mrp_discount_pct"] = ((daily["total_mrp"] - daily["revenue"]) / daily["total_mrp"] * 100).round(2)
    daily_series = daily.drop(columns=["total_mrp"]).round(2).to_dict(orient="records")

    return {
        "cumulative": {
            "total_revenue":          round(total_revenue, 2),
            "total_units_sold":       round(total_units, 2),
            "total_orders":           int(total_orders),
            "avg_order_value":        round(avg_order_value, 2),
            "total_cogs":             round(df["cogs"].sum(), 2),
            "gross_margin_pct":       round(gross_margin_pct, 2),
            "cancellation_rate":      cancellation_rate,
            "return_rate":            return_rate,
            "order_status_breakdown": status_counts,
            "payment_split":          payment_split,
        },
        "by_marketplace":              marketplace_metrics,
        "price_history_by_marketplace": price_history_by_marketplace,
        "by_state":                    state_metrics,
        "top_states":                  top_states,
        "by_size":                     size_metrics,
        "courier_distribution":        courier_dist,
        "warehouse_distribution":      warehouse_dist,
        "daily_series":                daily_series,
    }


# ── Rolling windows ───────────────────────────────────────────────────────────
def compute_rolling(daily_series: list[dict]) -> dict:
    """Compute 7d and 30d rolling windows from the full daily_series."""
    if not daily_series:
        return {}

    df = pd.DataFrame(daily_series).sort_values("order_date")
    result = {}

    for window in [7, 30]:
        tail = df.tail(window)
        rev  = tail["revenue"].sum()
        result[f"{window}d"] = {
            "revenue":          round(rev, 2),
            "units_sold":       round(tail["units_sold"].sum(), 2),
            "orders":           int(tail["orders"].sum()),
            "gross_margin_pct": round(
                tail["gross_profit"].sum() / rev * 100, 2
            ) if rev else 0,
        }

    return result


# ── Merge logic ───────────────────────────────────────────────────────────────
def merge_metrics(existing: dict, new_metrics: dict) -> dict:
    """
    Merge new metrics into the existing S3 profile.
    - daily_series: deduped by date, new data wins for the same date.
    - All breakdowns (marketplace, state, size) are replaced with the latest
      full-dataset values from this run (they're always recomputed from scratch
      per SKU so they're already cumulative for all fetched data).
    """
    if not existing:
        merged = dict(new_metrics)
        merged["rolling"] = compute_rolling(merged.get("daily_series", []))
        return merged

    # Merge daily_series — deduped by date, new data wins
    existing_daily = {d["order_date"]: d for d in existing.get("daily_series", [])}
    for day in new_metrics.get("daily_series", []):
        existing_daily[day["order_date"]] = day

    # Merge price_history_by_marketplace — deduped by (date, marketplace), new data wins
    existing_ph = {
        (d["date"], d["marketplace"]): d
        for d in existing.get("price_history_by_marketplace", [])
    }
    for entry in new_metrics.get("price_history_by_marketplace", []):
        existing_ph[(entry["date"], entry["marketplace"])] = entry

    merged = dict(new_metrics)
    merged["daily_series"] = sorted(existing_daily.values(), key=lambda x: x["order_date"])
    merged["price_history_by_marketplace"] = sorted(
        existing_ph.values(), key=lambda x: (x["date"], x["marketplace"])
    )
    merged["rolling"] = compute_rolling(merged["daily_series"])
    return merged


# ── S3 read / write ───────────────────────────────────────────────────────────
def read_sku_profile(sku: str) -> dict:
    key = f"{S3_PREFIX}/{sku}.json"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception as e:
        log.warning(f"Could not read existing profile for {sku}: {e}")
        return {}


def write_sku_profile(sku: str, profile: dict) -> None:
    key = f"{S3_PREFIX}/{sku}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(profile, default=str),
        ContentType="application/json",
    )
    log.info(f"Written s3://{S3_BUCKET}/{key}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global DYNAMODB_TABLE, S3_BUCKET, S3_PREFIX, META_KEY, dynamodb, s3, table

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.since_date:
        try:
            datetime.strptime(args.since_date, "%Y-%m-%d")
        except ValueError:
            parser.error("--since-date must be in YYYY-MM-DD format")

    log.setLevel(getattr(logging, args.log_level))

    DYNAMODB_TABLE = args.ddb_table
    S3_BUCKET = args.bucket
    S3_PREFIX = args.prefix
    META_KEY = args.meta_key or f"{S3_PREFIX}/_meta/last_run.json"

    dynamodb = boto3.resource("dynamodb", region_name=args.aws_region)
    s3 = boto3.client("s3", region_name=args.aws_region)
    table = dynamodb.Table(DYNAMODB_TABLE)

    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_run = args.since_date or get_last_run()
    log.info(f"Run date: {today} | Processing orders since: {last_run}")

    # 1. Fetch new/updated orders from DynamoDB
    raw_items = scan_orders_since(last_run)
    if not raw_items:
        log.info("No new orders found. Updating last_run and exiting.")
        save_last_run(today)
        return

    # 2. Build DataFrame and parse canonical SKU + size from the 'sku' field
    df = pd.DataFrame(raw_items)
    parsed          = df["sku"].apply(lambda x: pd.Series(parse_sku(str(x)), index=["canonical_sku", "size"]))
    df["canonical_sku"] = parsed["canonical_sku"]
    df["size"]          = parsed["size"]

    # 3. Process each affected SKU
    affected_skus = df["canonical_sku"].unique()
    log.info(f"Updating metrics for {len(affected_skus)} SKU(s)")

    for sku in affected_skus:
        try:
            sku_df      = df[df["canonical_sku"] == sku]
            new_metrics = compute_sku_metrics(sku_df)

            # 4. Fetch existing profile and merge incrementally
            existing_profile = read_sku_profile(sku)
            merged_profile   = merge_metrics(existing_profile, new_metrics)

            merged_profile["sku"]          = sku
            merged_profile["last_updated"] = datetime.now(timezone.utc).isoformat()

            # 5. Write back to S3
            write_sku_profile(sku, merged_profile)

        except Exception as e:
            log.error(f"Failed to process SKU {sku}: {e}", exc_info=True)
            # Continue processing remaining SKUs

    # 6. Stamp last_run so next run is incremental
    save_last_run(today)
    log.info("Done.")


if __name__ == "__main__":
    main()
