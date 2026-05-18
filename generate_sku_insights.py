#!/usr/bin/env python3
"""
generate_sku_insights.py
Daily background job (runs at 00:15 UTC after SKU metrics generation):
Scans all SKU metric JSONs from S3, aggregates insights across key metrics,
and generates 7 ranked insight cards for homepage display.

Insight Cards Generated:
1. Best Selling — SKUs by total units sold
2. Trending — 7d revenue momentum vs 30d average
3. Margin Leaders — SKUs by gross margin %
4. Growth Accelerators — 30d vs 7d revenue growth rate
5. Price Volatility — Price variance across marketplaces
6. Quality Issues — SKUs by combined cancellation/return rate
7. Fulfillment Performance — Courier/marketplace efficiency

Output:
- Latest: s3://chupps-data-portal/sku-metrics/insights-master.json
- Archive: s3://chupps-data-portal/sku-metrics/insights-archive/YYYY-MM-DD/insights.json
- Metadata: s3://chupps-data-portal/sku-metrics/_meta/insights_last_run.json
- Retention: Snapshots > 90 days automatically deleted
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_S3_BUCKET = "chupps-data-portal"
DEFAULT_S3_PREFIX = "sku-metrics"
DEFAULT_AWS_REGION = "ap-south-1"
INSIGHTS_KEY = f"{DEFAULT_S3_PREFIX}/insights-master.json"
INSIGHTS_ARCHIVE_PREFIX = f"{DEFAULT_S3_PREFIX}/insights-archive"
INSIGHTS_META_KEY = f"{DEFAULT_S3_PREFIX}/_meta/insights_last_run.json"

TOP_N_PER_CARD = 5
RETENTION_DAYS = 90

# Thresholds for filtering noise
MIN_REVENUE_MARGIN_LEADERS = 5000  # ₹5000
MIN_30D_REVENUE_GROWTH_ACCEL = 3000  # ₹3000
MIN_ORDERS_QUALITY = 5
MIN_MARKETPLACES_VOLATILITY = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────
s3 = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate SKU metrics and generate ranked insight cards for homepage"
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_S3_BUCKET,
        help=f"S3 bucket for metrics (default: {DEFAULT_S3_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_S3_PREFIX,
        help=f"S3 prefix for metrics (default: {DEFAULT_S3_PREFIX})",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute insights but don't write to S3 (useful for testing)",
    )
    return parser


# ── S3 Operations ─────────────────────────────────────────────────────────────
def list_sku_metrics(bucket: str, prefix: str) -> list[str]:
    """List all SKU metric JSON files in S3."""
    sku_metrics = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/")

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Match pattern: sku-metrics/{sku}.json (not _meta or insights-archive)
            if key.endswith(".json") and "_meta" not in key and "insights" not in key:
                sku_metrics.append(key)

    log.info(f"Found {len(sku_metrics)} SKU metric files in S3")
    return sku_metrics


def read_sku_profile(bucket: str, key: str) -> dict:
    """Read a single SKU metric JSON from S3."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        log.warning(f"Failed to read {key}: {e}")
        return {}
    except json.JSONDecodeError as e:
        log.warning(f"Malformed JSON in {key}: {e}")
        return {}


def write_insights_to_s3(
    bucket: str, insights: dict, dry_run: bool = False
) -> bool:
    """Write insights to S3 (both latest and daily archive)."""
    if dry_run:
        log.info("[DRY-RUN] Would write insights-master.json to S3")
        return True

    try:
        # Write latest insights
        s3.put_object(
            Bucket=bucket,
            Key=INSIGHTS_KEY,
            Body=json.dumps(insights, default=str, indent=2),
            ContentType="application/json",
        )
        log.info(f"Written s3://{bucket}/{INSIGHTS_KEY}")

        # Write daily archive snapshot
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_key = f"{INSIGHTS_ARCHIVE_PREFIX}/{today}/insights.json"
        s3.put_object(
            Bucket=bucket,
            Key=archive_key,
            Body=json.dumps(insights, default=str, indent=2),
            ContentType="application/json",
        )
        log.info(f"Written s3://{bucket}/{archive_key}")

        # Update metadata
        metadata = {
            "last_run_date": today,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "insights_master_key": INSIGHTS_KEY,
            "archive_key": archive_key,
        }
        s3.put_object(
            Bucket=bucket,
            Key=INSIGHTS_META_KEY,
            Body=json.dumps(metadata, default=str, indent=2),
            ContentType="application/json",
        )
        log.info(f"Updated metadata at s3://{bucket}/{INSIGHTS_META_KEY}")

        return True
    except ClientError as e:
        log.error(f"Failed to write insights to S3: {e}")
        return False


def cleanup_old_archives(bucket: str, dry_run: bool = False) -> None:
    """Delete archive snapshots older than RETENTION_DAYS."""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{INSIGHTS_ARCHIVE_PREFIX}/")

        to_delete = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract date from path: insights-archive/YYYY-MM-DD/insights.json
                parts = key.split("/")
                if len(parts) >= 3 and parts[-2].count("-") == 2:
                    try:
                        snapshot_date = datetime.strptime(
                            parts[-2], "%Y-%m-%d"
                        ).replace(tzinfo=timezone.utc)
                        if snapshot_date < cutoff_date:
                            to_delete.append(key)
                    except ValueError:
                        pass

        if to_delete:
            if not dry_run:
                for key in to_delete:
                    s3.delete_object(Bucket=bucket, Key=key)
                    log.info(f"Deleted old archive: s3://{bucket}/{key}")
            else:
                log.info(
                    f"[DRY-RUN] Would delete {len(to_delete)} old archive snapshots"
                )
    except ClientError as e:
        log.warning(f"Failed to cleanup old archives: {e}")


# ── Insight Card Calculations ────────────────────────────────────────────────
def card_best_selling(profiles: dict) -> dict:
    """Card: Best Selling SKUs by total units sold (all-time)."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue
        cum = profile.get("cumulative", {})
        units = float(cum.get("total_units_sold", 0))
        if units > 0:
            model_no = (profile.get("suborder_model_no") or [""])[0]
            product_name = (profile.get("suborder_productName") or [""])[0]
            data.append(
                {
                    "sku": sku,
                    "model_no": model_no,
                    "product_name": product_name,
                    "units_sold": round(units, 2),
                    "revenue": round(float(cum.get("total_revenue", 0)), 2),
                    "order_count": int(cum.get("total_orders", 0)),
                    "avg_order_value": round(float(cum.get("avg_order_value", 0)), 2),
                    "score": units,
                }
            )

    data.sort(key=lambda x: x["score"], reverse=True)
    return {
        "title": "Best Selling SKUs",
        "description": "Top SKUs by total units sold (all-time)",
        "time_window": "all-time",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {"total_skus": len(data), "shown": min(len(data), TOP_N_PER_CARD)},
    }


def card_trending(profiles: dict) -> dict:
    """Card: Trending SKUs based on 7d revenue momentum vs 30d average."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue
        rolling = profile.get("rolling", {})
        rev_7d = float(rolling.get("7d", {}).get("revenue", 0))
        rev_30d = float(rolling.get("30d", {}).get("revenue", 0))

        if rev_7d > 0 and rev_30d > 0:
            # Trending score: (7d_rev / (30d_rev / 4.3)) * 100
            # Captures if 7d is outperforming daily average
            daily_avg = rev_30d / 4.3
            trending_score = (rev_7d / daily_avg) * 100 if daily_avg > 0 else 0

            model_no = (profile.get("suborder_model_no") or [""])[0]
            product_name = (profile.get("suborder_productName") or [""])[0]
            data.append(
                {
                    "sku": sku,
                    "model_no": model_no,
                    "product_name": product_name,
                    "trending_score": round(trending_score, 2),
                    "revenue_7d": round(rev_7d, 2),
                    "revenue_30d": round(rev_30d, 2),
                    "units_sold_7d": round(float(rolling.get("7d", {}).get("units_sold", 0)), 2),
                    "momentum_pct": round(((rev_7d - daily_avg) / daily_avg * 100), 2),
                    "score": trending_score,
                }
            )

    data.sort(key=lambda x: x["score"], reverse=True)
    return {
        "title": "Trending SKUs",
        "description": "7d revenue momentum vs 30d daily average (>100 = outperforming)",
        "time_window": "7d vs 30d",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {"total_skus": len(data), "shown": min(len(data), TOP_N_PER_CARD)},
    }


def card_margin_leaders(profiles: dict) -> dict:
    """Card: Margin Leaders by gross margin % (with minimum revenue threshold)."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue
        cum = profile.get("cumulative", {})
        revenue = float(cum.get("total_revenue", 0))

        # Filter: minimum revenue threshold to avoid noise
        if revenue < MIN_REVENUE_MARGIN_LEADERS:
            continue

        margin_pct = float(cum.get("gross_margin_pct", 0))
        total_cogs = float(cum.get("total_cogs", 0))

        model_no = (profile.get("suborder_model_no") or [""])[0]
        product_name = (profile.get("suborder_productName") or [""])[0]
        data.append(
            {
                "sku": sku,
                "model_no": model_no,
                "product_name": product_name,
                "gross_margin_pct": round(margin_pct, 2),
                "total_revenue": round(revenue, 2),
                "total_cogs": round(total_cogs, 2),
                "gross_profit": round(revenue - total_cogs, 2),
                "order_count": int(cum.get("total_orders", 0)),
                "avg_margin_per_order": round((revenue - total_cogs) / max(cum.get("total_orders", 1), 1), 2),
                "score": margin_pct,
            }
        )

    # Separate profitable and at-risk
    profitable = [x for x in data if x["gross_margin_pct"] >= 0]
    profitable.sort(key=lambda x: x["score"], reverse=True)

    at_risk = [x for x in data if x["gross_margin_pct"] < 0]
    at_risk.sort(key=lambda x: x["score"])

    all_data = profitable + at_risk

    return {
        "title": "Margin Leaders",
        "description": f"Most profitable SKUs (min ₹{MIN_REVENUE_MARGIN_LEADERS} revenue)",
        "time_window": "all-time",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in all_data[:TOP_N_PER_CARD]
        ],
        "metadata": {
            "total_skus": len(data),
            "shown": min(len(data), TOP_N_PER_CARD),
            "profitable_count": len(profitable),
            "at_risk_count": len(at_risk),
        },
    }


def card_growth_accelerators(profiles: dict) -> dict:
    """Card: Growth Accelerators based on 30d vs 7d revenue growth rate."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue
        rolling = profile.get("rolling", {})
        rev_30d = float(rolling.get("30d", {}).get("revenue", 0))

        # Filter: minimum 30d revenue threshold
        if rev_30d < MIN_30D_REVENUE_GROWTH_ACCEL:
            continue

        rev_7d = float(rolling.get("7d", {}).get("revenue", 0))

        if rev_30d > 0:
            # Growth formula: ((30d_rev/7.2 days) - 7d_rev) / (30d_rev/7.2) * 100
            daily_avg_30d = rev_30d / 7.2
            growth_pct = (
                ((daily_avg_30d - rev_7d) / daily_avg_30d * 100)
                if daily_avg_30d > 0
                else 0
            )

            model_no = (profile.get("suborder_model_no") or [""])[0]
            product_name = (profile.get("suborder_productName") or [""])[0]
            data.append(
                {
                    "sku": sku,
                    "model_no": model_no,
                    "product_name": product_name,
                    "growth_pct": round(growth_pct, 2),
                    "revenue_7d": round(rev_7d, 2),
                    "revenue_30d": round(rev_30d, 2),
                    "daily_avg_30d": round(daily_avg_30d, 2),
                    "order_velocity_7d": int(rolling.get("7d", {}).get("orders", 0)),
                    "order_velocity_30d": int(rolling.get("30d", {}).get("orders", 0)),
                    "score": growth_pct,
                }
            )

    data.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": "Growth Accelerators",
        "description": f"Fastest growing SKUs (min ₹{MIN_30D_REVENUE_GROWTH_ACCEL} 30d revenue)",
        "time_window": "30d vs 7d",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {
            "total_skus": len(data),
            "shown": min(len(data), TOP_N_PER_CARD),
            "min_threshold": MIN_30D_REVENUE_GROWTH_ACCEL,
        },
    }


def card_price_volatility(profiles: dict) -> dict:
    """Card: Price Volatility across marketplaces."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue

        by_mp = profile.get("by_marketplace", {})

        # Only consider SKUs sold on 2+ marketplaces
        if len(by_mp) < MIN_MARKETPLACES_VOLATILITY:
            continue

        # Extract average selling prices per marketplace
        prices_per_mp = {}
        for mp, mp_data in by_mp.items():
            prices_per_mp[mp] = float(mp_data.get("avg_selling_price", 0))

        if not prices_per_mp:
            continue

        # Calculate volatility: (max_price - min_price) / avg_price * 100
        prices = list(prices_per_mp.values())
        max_price = max(prices)
        min_price = min(prices)
        avg_price = sum(prices) / len(prices)

        volatility_score = (
            ((max_price - min_price) / avg_price * 100) if avg_price > 0 else 0
        )

        # Extract discount variance
        discount_by_mp = {}
        for mp, mp_data in by_mp.items():
            discount_by_mp[mp] = float(mp_data.get("mrp_discount_pct", 0))

        model_no = (profile.get("suborder_model_no") or [""])[0]
        product_name = (profile.get("suborder_productName") or [""])[0]
        data.append(
            {
                "sku": sku,
                "model_no": model_no,
                "product_name": product_name,
                "volatility_score": round(volatility_score, 2),
                "marketplace_count": len(prices_per_mp),
                "marketplace_list": list(prices_per_mp.keys()),
                "price_range": f"₹{round(min_price, 2)} - ₹{round(max_price, 2)}",
                "avg_price": round(avg_price, 2),
                "price_variance_pct": round((max_price - min_price) / avg_price * 100, 2),
                "discount_variance_pct": round(
                    max(discount_by_mp.values()) - min(discount_by_mp.values()), 2
                ),
                "score": volatility_score,
            }
        )

    data.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": "Price Volatility",
        "description": "SKUs with highest price variance across marketplaces",
        "time_window": "all-time",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {
            "total_skus": len(data),
            "shown": min(len(data), TOP_N_PER_CARD),
            "min_marketplace_threshold": MIN_MARKETPLACES_VOLATILITY,
        },
    }


def card_quality_issues(profiles: dict) -> dict:
    """Card: Quality Issues by combined cancellation & return rate."""
    data = []
    for sku, profile in profiles.items():
        if not profile:
            continue
        cum = profile.get("cumulative", {})
        total_orders = int(cum.get("total_orders", 0))

        # Filter: minimum order count threshold
        if total_orders < MIN_ORDERS_QUALITY:
            continue

        cancellation_rate = float(cum.get("cancellation_rate", 0))
        return_rate = float(cum.get("return_rate", 0))
        quality_score = (cancellation_rate + return_rate) / 2

        # Estimate at-risk orders
        at_risk_order_count = max(
            int((cancellation_rate / 100) * total_orders),
            int((return_rate / 100) * total_orders),
        )

        model_no = (profile.get("suborder_model_no") or [""])[0]
        product_name = (profile.get("suborder_productName") or [""])[0]
        data.append(
            {
                "sku": sku,
                "model_no": model_no,
                "product_name": product_name,
                "quality_score": round(quality_score, 2),
                "cancellation_rate": round(cancellation_rate, 2),
                "return_rate": round(return_rate, 2),
                "total_orders": total_orders,
                "at_risk_order_count": at_risk_order_count,
                "revenue_at_risk": round(float(cum.get("avg_order_value", 0)) * at_risk_order_count, 2),
                "score": quality_score,
            }
        )

    data.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": "Quality Issues",
        "description": "SKUs with highest combined cancellation & return rates",
        "time_window": "all-time",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {
            "total_skus": len(data),
            "shown": min(len(data), TOP_N_PER_CARD),
            "min_orders_threshold": MIN_ORDERS_QUALITY,
        },
    }


def card_fulfillment_performance(profiles: dict) -> dict:
    """Card: Fulfillment Performance based on courier efficiency & marketplace reliability."""
    data = []

    # Heuristic: Preferred couriers (in order of reliability)
    PREFERRED_COURIERS = ["Amazon Easyship", "Easyship", "Delhivery", "BlueDart"]

    for sku, profile in profiles.items():
        if not profile:
            continue

        cum = profile.get("cumulative", {})
        total_orders = int(cum.get("total_orders", 0))
        total_revenue = float(cum.get("total_revenue", 0))

        if total_orders == 0:
            continue

        courier_dist = profile.get("courier_distribution", {})
        by_mp = profile.get("by_marketplace", {})

        # Calculate fulfillment score: preferred courier order % + marketplace diversity
        preferred_orders = 0
        for courier in PREFERRED_COURIERS:
            preferred_orders += courier_dist.get(courier, 0)

        fulfillment_score = (preferred_orders / total_orders * 100) if total_orders > 0 else 0

        # Marketplace split
        marketplace_split = {mp: mp_data.get("order_count", 0) for mp, mp_data in by_mp.items()}
        top_marketplace = max(marketplace_split.items(), key=lambda x: x[1])[0] if marketplace_split else "Unknown"

        # Primary courier
        primary_courier = (
            max(courier_dist.items(), key=lambda x: x[1])[0]
            if courier_dist
            else "Unknown"
        )

        model_no = (profile.get("suborder_model_no") or [""])[0]
        product_name = (profile.get("suborder_productName") or [""])[0]
        data.append(
            {
                "sku": sku,
                "model_no": model_no,
                "product_name": product_name,
                "fulfillment_score": round(fulfillment_score, 2),
                "primary_courier": primary_courier,
                "preferred_courier_pct": round(fulfillment_score, 2),
                "total_orders": total_orders,
                "marketplace_count": len(by_mp),
                "top_marketplace": top_marketplace,
                "courier_count": len(courier_dist),
                "score": fulfillment_score,
            }
        )

    data.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": "Fulfillment Performance",
        "description": "SKUs with best courier & marketplace fulfillment efficiency",
        "time_window": "all-time",
        "data": [
            {k: v for k, v in item.items() if k != "score"} for item in data[:TOP_N_PER_CARD]
        ],
        "metadata": {
            "total_skus": len(data),
            "shown": min(len(data), TOP_N_PER_CARD),
            "preferred_couriers": PREFERRED_COURIERS,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global s3

    parser = build_arg_parser()
    args = parser.parse_args()

    log.setLevel(getattr(logging, args.log_level))

    s3 = boto3.client("s3", region_name=args.aws_region)

    start_time = datetime.now(timezone.utc)
    log.info(
        f"Starting SKU insights aggregation | Bucket: {args.bucket} | Prefix: {args.prefix}"
    )

    # 1. List all SKU metric files in S3
    sku_metric_keys = list_sku_metrics(args.bucket, args.prefix)
    if not sku_metric_keys:
        log.warning("No SKU metric files found in S3. Exiting.")
        return

    # 2. Read all SKU profiles
    log.info(f"Reading {len(sku_metric_keys)} SKU profiles from S3...")
    profiles = {}
    for key in sku_metric_keys:
        profile = read_sku_profile(args.bucket, key)
        if profile:
            # Extract SKU from key: sku-metrics/{sku}.json
            sku = key.split("/")[-1].replace(".json", "")
            profiles[sku] = profile

    if not profiles:
        log.error("No valid SKU profiles could be loaded. Exiting.")
        return

    log.info(f"Successfully loaded {len(profiles)} SKU profiles")

    # 3. Generate insight cards
    log.info("Generating insight cards...")
    cards = {
        "best_selling": card_best_selling(profiles),
        "trending": card_trending(profiles),
        "margin_leaders": card_margin_leaders(profiles),
        "growth_accelerators": card_growth_accelerators(profiles),
        "price_volatility": card_price_volatility(profiles),
        "quality_issues": card_quality_issues(profiles),
        "fulfillment_performance": card_fulfillment_performance(profiles),
    }

    # Validate cards have minimum data
    for card_name, card in cards.items():
        card_data_count = len(card.get("data", []))
        if card_data_count == 0:
            log.warning(f"Card '{card_name}' has no data after filtering")
        else:
            log.info(f"Card '{card_name}': {card_data_count} SKUs")

    # 4. Build insights master object
    insights = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_skus_processed": len(profiles),
        "cards": cards,
    }

    # 5. Write to S3
    write_success = write_insights_to_s3(args.bucket, insights, dry_run=args.dry_run)

    # 6. Cleanup old archives
    if write_success:
        cleanup_old_archives(args.bucket, dry_run=args.dry_run)

    # 7. Summary log
    execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    log.info(
        f"Completed in {execution_time:.2f}s | {len(profiles)} SKUs processed | 7 insight cards generated"
    )


if __name__ == "__main__":
    main()
