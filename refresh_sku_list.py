#!/usr/bin/env python3
"""Recompute forecastable SKUs from S3 snapshots and update _forecastable-skus.json."""

import os
from dotenv import load_dotenv
from inventory_delta import load_snapshots_from_s3, compute_forecastable_skus, save_forecastable_skus_to_s3

load_dotenv()

BUCKET = os.getenv("INVENTORY_SNAPSHOT_S3_BUCKET", "chupps-data-portal")
PREFIX = os.getenv("INVENTORY_SNAPSHOT_S3_PREFIX", "inventory-snapshots")


def main():
    snapshots = load_snapshots_from_s3(BUCKET, PREFIX)
    print(f"Loaded {len(snapshots)} snapshots")

    if len(snapshots) < 2:
        print("Need at least 2 snapshots.")
        return

    skus = compute_forecastable_skus(snapshots)
    save_forecastable_skus_to_s3(BUCKET, PREFIX, snapshots)

    dates = sorted(snapshots.keys())
    print(f"Forecastable SKUs: {len(skus)}")
    print(f"Snapshot range: {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
