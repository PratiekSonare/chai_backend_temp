#!/usr/bin/env python3
"""Download all weekly snapshot CSVs from S3 and merge into one file."""

import os
import io
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("INVENTORY_SNAPSHOT_S3_BUCKET", "chupps-data-portal")
PREFIX = os.getenv("INVENTORY_SNAPSHOT_S3_PREFIX", "inventory-snapshots")
REGION = os.getenv("AWS_REGION", "ap-south-1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "inventory-snapshots")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    s3 = boto3.client("s3", region_name=REGION)

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    all_dfs = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            print(f"Reading {key}")
            response = s3.get_object(Bucket=BUCKET, Key=key)
            csv_text = response["Body"].read().decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(csv_text))
            # Extract date from filename for the snapshot_date column
            filename = os.path.basename(key).replace(".csv", "")
            df["snapshot_file"] = filename
            all_dfs.append(df)

    if not all_dfs:
        print("No CSVs found.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "ALL_snapshots_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nMerged {len(all_dfs)} files → {out_path}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
