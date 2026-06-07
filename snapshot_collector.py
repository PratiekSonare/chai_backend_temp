#!/usr/bin/env python3
"""
Weekly Inventory Snapshot Collector

Fetches the inventory snapshot every Wednesday from EasyEcom API
and stores it to S3 for historical delta computation.

Run weekly via cron or systemd timer:
    # Every Wednesday at 00:15 UTC
    15 0 * * 3 cd /path/to/backend && python snapshot_collector.py
"""

import os
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configuration
S3_BUCKET = os.getenv("INVENTORY_SNAPSHOT_S3_BUCKET", "chupps-data-portal")
S3_PREFIX = os.getenv("INVENTORY_SNAPSHOT_S3_PREFIX", "inventory-snapshots")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


def collect_weekly_snapshot(target_date: date = None) -> dict:
    """
    Collect inventory snapshot for a given Wednesday.
    Also refreshes the forecastable SKUs JSON on S3.
    """
    from inventory_delta import collect_and_store_snapshot, load_snapshots_from_s3, save_forecastable_skus_to_s3

    if target_date is None:
        target_date = _last_wednesday()

    start_str = f"{target_date.isoformat()} 00:00:00"
    end_str = f"{target_date.isoformat()} 23:59:59"

    try:
        df = collect_and_store_snapshot(
            start_date=start_str,
            end_date=end_str,
            s3_bucket=S3_BUCKET,
            s3_prefix=S3_PREFIX,
        )

        # Refresh forecastable SKUs after new snapshot
        forecastable_count = 0
        try:
            snapshots = load_snapshots_from_s3(S3_BUCKET, S3_PREFIX)
            save_forecastable_skus_to_s3(S3_BUCKET, S3_PREFIX, snapshots)
            from inventory_delta import compute_forecastable_skus
            forecastable_count = len(compute_forecastable_skus(snapshots))
        except Exception as e:
            print(f"[WARNING] Could not refresh forecastable SKUs: {e}")

        return {
            "status": "success",
            "date": target_date.isoformat(),
            "weekday": target_date.strftime("%A"),
            "sku_count": len(df),
            "forecastable_skus": forecastable_count,
            "s3_path": f"s3://{S3_BUCKET}/{S3_PREFIX}/{target_date.isoformat()}.csv",
        }

    except Exception as e:
        return {
            "status": "error",
            "date": target_date.isoformat(),
            "error": str(e),
        }


def backfill_weekly(start_date: date, end_date: date) -> list:
    """
    Backfill weekly snapshots for a date range (Wednesdays only).
    Refreshes forecastable SKUs once at the end.
    """
    from inventory_delta import load_snapshots_from_s3, save_forecastable_skus_to_s3

    results = []
    current = start_date

    while current.weekday() != 2:
        current += timedelta(days=1)

    while current <= end_date:
        print(f"[BACKFILL] Collecting snapshot for Wednesday {current}...")
        result = collect_weekly_snapshot(current)
        results.append(result)
        print(f"[BACKFILL] {result['status']}: {current} ({result.get('sku_count', 0)} SKUs)")
        current += timedelta(days=7)

    # Refresh forecastable SKUs after full backfill
    try:
        snapshots = load_snapshots_from_s3(S3_BUCKET, S3_PREFIX)
        save_forecastable_skus_to_s3(S3_BUCKET, S3_PREFIX, snapshots)
    except Exception as e:
        print(f"[WARNING] Could not refresh forecastable SKUs: {e}")

    return results


def _last_wednesday() -> date:
    """Get the most recent Wednesday (including today if it's Wednesday)."""
    today = date.today()
    days_since_wed = (today.weekday() - 2) % 7
    if days_since_wed == 0 and today.hour < 0:
        # If it's Wednesday but before midnight, use previous Wednesday
        return today - timedelta(days=7)
    return today - timedelta(days=days_since_wed)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect weekly inventory snapshots (Wednesdays)")
    parser.add_argument("--date", help="Wednesday date to collect (YYYY-MM-DD). Default: last Wednesday")
    parser.add_argument("--backfill-start", help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--backfill-end", help="Backfill end date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.backfill_start and args.backfill_end:
        start = datetime.strptime(args.backfill_start, "%Y-%m-%d").date()
        end = datetime.strptime(args.backfill_end, "%Y-%m-%d").date()
        results = backfill_weekly(start, end)

        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        print(f"\n[BACKFILL COMPLETE] {success} succeeded, {failed} failed")

        if failed > 0:
            for r in results:
                if r["status"] == "error":
                    print(f"  FAILED: {r['date']} - {r.get('error', 'unknown')}")

    else:
        target = None
        if args.date:
            target = datetime.strptime(args.date, "%Y-%m-%d").date()

        result = collect_weekly_snapshot(target)
        print(json.dumps(result, indent=2))

        if result["status"] == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()
