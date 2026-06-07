"""
Inventory Delta Computation Module (Weekly)

Loads consecutive weekly inventory snapshots, computes per-SKU deltas,
and aggregates into weekly time series suitable for Prophet forecasting.

Snapshots are taken every Wednesday at 00:00 UTC. Deltas between
consecutive Wednesdays reveal weekly inflow, outflow, and loss.

Usage:
    from inventory_delta import load_snapshots_from_s3, compute_weekly_timeseries

    snapshots = load_snapshots_from_s3("chupps-data-portal", "inventory-snapshots")
    ts = compute_weekly_timeseries(snapshots)
"""

import os
import io
import glob as globmod
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

from inventory_tools import INVENTORY_COLUMNS, NUMERIC_COLUMNS, _to_numeric_safe


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def _parse_snapshot_csv(csv_text: str, snapshot_date: date) -> pd.DataFrame:
    """Parse a raw CSV string into a cleaned inventory DataFrame with snapshot_date."""
    df = pd.read_csv(io.StringIO(csv_text))

    rename_map = {col: INVENTORY_COLUMNS.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    df["snapshot_date"] = snapshot_date
    return df


def _extract_date_from_filename(filename: str) -> Optional[date]:
    """Extract a date from filenames like 2026-06-03.csv or snapshot_20260603.csv."""
    import re

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    return None


def _dedup_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by SKU within a single snapshot, keeping the latest report_date."""
    if "sku" not in df.columns:
        return df
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.sort_values("report_date", ascending=False)
    return df.drop_duplicates(subset=["sku"], keep="first")


def load_snapshots_from_dir(directory: str) -> Dict[date, pd.DataFrame]:
    """
    Load all weekly snapshot CSVs from a local directory.

    Expected filenames: YYYY-MM-DD.csv (the Wednesday date).
    Returns: {date: DataFrame} sorted by date.
    """
    csv_files = sorted(globmod.glob(os.path.join(directory, "*.csv")))
    snapshots: Dict[date, pd.DataFrame] = {}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        snapshot_date = _extract_date_from_filename(filename)
        if snapshot_date is None:
            mtime = os.path.getmtime(filepath)
            snapshot_date = datetime.fromtimestamp(mtime).date()

        try:
            df = pd.read_csv(filepath)
            rename_map = {col: INVENTORY_COLUMNS.get(col, col) for col in df.columns}
            df = df.rename(columns=rename_map)
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = _to_numeric_safe(df[col])
            df["snapshot_date"] = snapshot_date

            if "sku" not in df.columns:
                print(f"[DELTA] Skipping {filename}: no SKU column")
                continue

            df = _dedup_sku(df)
            snapshots[snapshot_date] = df
            print(f"[DELTA] Loaded {len(df)} SKUs from {filename} (week={snapshot_date})")

        except Exception as e:
            print(f"[DELTA] Error loading {filename}: {e}")

    return dict(sorted(snapshots.items()))


def load_snapshots_from_s3(bucket: str, prefix: str, region: str = "ap-south-1") -> Dict[date, pd.DataFrame]:
    """
    Load all weekly snapshot CSVs from S3.

    Expected key structure: inventory-snapshots/YYYY-MM-DD.csv
    Returns: {date: DataFrame} sorted by date.
    """
    import boto3

    s3 = boto3.client("s3", region_name=region)
    snapshots: Dict[date, pd.DataFrame] = {}

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue

            filename = os.path.basename(key)
            snapshot_date = _extract_date_from_filename(filename)
            if snapshot_date is None:
                continue

            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                csv_text = response["Body"].read().decode("utf-8-sig")
                df = _parse_snapshot_csv(csv_text, snapshot_date)

                if "sku" not in df.columns:
                    continue

                df = _dedup_sku(df)
                snapshots[snapshot_date] = df
                print(f"[DELTA] Loaded {len(df)} SKUs from s3://{bucket}/{key}")

            except Exception as e:
                print(f"[DELTA] Error loading s3://{bucket}/{key}: {e}")

    return dict(sorted(snapshots.items()))


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

INFLOW_COLUMNS = ["received", "return_available", "to_receive", "qc_passed"]
LOSS_COLUMNS = ["damaged", "discard_fraud", "total_lost", "lost_cycle_count", "quarantine", "questionable"]
RESERVE_COLUMNS = ["reserved_picked", "reserved_not_picked", "amazon_reserved"]
STOCK_COLUMNS = ["available_qty", "available_bin_locked", "marketplace_available", "website_inventory", "ecom_inventory", "retail_inventory"]


def _compute_sku_deltas(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """Compute per-SKU deltas between two consecutive weekly snapshots."""
    merged = prev.merge(curr, on="sku", suffixes=("_prev", "_curr"), how="inner")

    result = pd.DataFrame()
    result["sku"] = merged["sku"]

    # Stock level change
    if "available_qty_prev" in merged.columns and "available_qty_curr" in merged.columns:
        result["stock_change"] = merged["available_qty_curr"] - merged["available_qty_prev"]
    else:
        result["stock_change"] = 0

    # Inflow: positive delta in received/return columns
    result["inflow"] = 0
    for col in [c for c in INFLOW_COLUMNS if f"{c}_prev" in merged.columns]:
        result["inflow"] += (merged[f"{col}_curr"] - merged[f"{col}_prev"]).clip(lower=0)

    # Loss: positive delta in damage/loss columns
    result["loss"] = 0
    for col in [c for c in LOSS_COLUMNS if f"{c}_prev" in merged.columns]:
        result["loss"] += (merged[f"{col}_curr"] - merged[f"{col}_prev"]).clip(lower=0)

    # Damage change
    if "damaged_prev" in merged.columns:
        result["damage_change"] = (merged["damaged_curr"] - merged["damaged_prev"]).clip(lower=0)
    else:
        result["damage_change"] = 0

    # Reserved change
    result["reserve_change"] = 0
    for col in [c for c in RESERVE_COLUMNS if f"{c}_prev" in merged.columns]:
        result["reserve_change"] += merged[f"{col}_curr"] - merged[f"{col}_prev"]

    # Outflow: decrease in stock not explained by inflow/loss
    result["outflow"] = (-result["stock_change"] - result["inflow"] + result["loss"]).clip(lower=0)

    # Current stock levels
    for col in STOCK_COLUMNS:
        curr_col = f"{col}_curr"
        if curr_col in merged.columns:
            result[col] = merged[curr_col]

    return result


def compute_weekly_timeseries(
    snapshots: Dict[date, pd.DataFrame],
    aggregate: bool = True,
) -> pd.DataFrame:
    """
    Compute weekly time series from consecutive inventory snapshots.

    Each row represents one week's delta from the previous Wednesday snapshot.

    Args:
        snapshots: {date: DataFrame} sorted by date (one per Wednesday)
        aggregate: True = aggregate across all SKUs; False = per-SKU deltas

    Returns:
        DataFrame with columns [ds, y, inflow, outflow, loss, ...]
        where ds = week-start Wednesday, y = total available stock that week.
    """
    if len(snapshots) < 2:
        print("[DELTA] Need at least 2 weekly snapshots to compute deltas")
        return pd.DataFrame()

    dates = sorted(snapshots.keys())
    all_sku_deltas = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        deltas = _compute_sku_deltas(snapshots[prev_date], snapshots[curr_date])
        deltas["snapshot_date"] = curr_date
        all_sku_deltas.append(deltas)

    if not all_sku_deltas:
        return pd.DataFrame()

    combined = pd.concat(all_sku_deltas, ignore_index=True)

    if not aggregate:
        return combined

    # Aggregate across all SKUs per week
    agg = combined.groupby("snapshot_date").agg({
        "stock_change": "sum",
        "inflow": "sum",
        "outflow": "sum",
        "loss": "sum",
        "damage_change": "sum",
        "reserve_change": "sum",
        "sku": "count",
    }).reset_index()
    agg = agg.rename(columns={"sku": "sku_count"})

    # Add total stock level from each snapshot
    stock_levels = []
    for d in dates:
        df = snapshots[d]
        total_stock = int(df["available_qty"].sum()) if "available_qty" in df.columns else 0
        stock_levels.append({"snapshot_date": d, "total_available": total_stock})

    stock_df = pd.DataFrame(stock_levels)
    agg = agg.merge(stock_df, on="snapshot_date", how="left")

    # Prophet format: ds = datetime, y = stock level
    agg = agg.rename(columns={"snapshot_date": "ds"})
    agg["ds"] = pd.to_datetime(agg["ds"])
    agg["y"] = agg["total_available"].astype(float)
    agg = agg.sort_values("ds").reset_index(drop=True)

    return agg


def compute_sku_level_timeseries(
    snapshots: Dict[date, pd.DataFrame],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute per-SKU weekly time series for top N SKUs by available quantity.
    Returns long format: [ds, sku, y, inflow, outflow, loss]
    """
    if len(snapshots) < 2:
        return pd.DataFrame()

    latest_date = max(snapshots.keys())
    latest_df = snapshots[latest_date]
    if "available_qty" not in latest_df.columns:
        return pd.DataFrame()

    top_skus = (
        latest_df.groupby("sku")["available_qty"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    all_deltas = compute_weekly_timeseries(snapshots, aggregate=False)
    if all_deltas.empty:
        return pd.DataFrame()

    sku_deltas = all_deltas[all_deltas["sku"].isin(top_skus)].copy()
    sku_deltas = sku_deltas.rename(columns={"snapshot_date": "ds"})
    sku_deltas["ds"] = pd.to_datetime(sku_deltas["ds"])

    if "available_qty" in sku_deltas.columns:
        sku_deltas["y"] = sku_deltas["available_qty"].astype(float)
    else:
        sku_deltas["y"] = sku_deltas["stock_change"].cumsum()

    return sku_deltas.sort_values(["sku", "ds"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Snapshot collection (weekly, called from cron)
# ---------------------------------------------------------------------------

def collect_and_store_snapshot(
    start_date: str,
    end_date: str,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "inventory-snapshots",
) -> pd.DataFrame:
    """
    Fetch a snapshot from EasyEcom API and store to S3.
    Returns the parsed DataFrame.

    This is called weekly by the snapshot collector.
    """
    from inventory_tools import get_inventory_snapshot

    df = get_inventory_snapshot(start_date, end_date)

    # Determine the snapshot date from the data
    if "report_date" in df.columns:
        snap_date = df["report_date"].iloc[0]
        if pd.notna(snap_date):
            snap_date = pd.to_datetime(snap_date).date()
        else:
            snap_date = date.today()
    else:
        snap_date = date.today()

    date_str = snap_date.isoformat()

    if s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        key = f"{s3_prefix}/{date_str}.csv"

        reverse_map = {v: k for k, v in INVENTORY_COLUMNS.items()}
        export_df = df.rename(columns={c: reverse_map.get(c, c) for c in df.columns})
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        s3.put_object(
            Bucket=s3_bucket,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )
        print(f"[DELTA] Stored weekly snapshot to s3://{s3_bucket}/{key}")

    return df


# ---------------------------------------------------------------------------
# Prophet integration
# ---------------------------------------------------------------------------

def prepare_prophet_dataframe(
    snapshots: Dict[date, pd.DataFrame],
    target: str = "aggregate",
    extra_regressors: bool = True,
) -> pd.DataFrame:
    """
    Prepare a Prophet-ready DataFrame from weekly inventory snapshots.

    For "aggregate": uses total stock across all SKUs per week.
    For a specific SKU: reads stock level directly from each snapshot
    (no inner join, no delta — handles gaps gracefully).

    Returns DataFrame with 'ds' and 'y' columns, plus optional regressors.
    """
    if not snapshots or len(snapshots) < 2:
        return pd.DataFrame(columns=["ds", "y"])

    if target == "aggregate":
        ts = compute_weekly_timeseries(snapshots, aggregate=True)
        if ts.empty:
            return pd.DataFrame(columns=["ds", "y"])
        prophet_df = ts[["ds", "y"]].copy()
        if extra_regressors:
            for col in ["inflow", "outflow", "loss", "damage_change"]:
                if col in ts.columns:
                    prophet_df[col] = ts[col].values
    else:
        # Single SKU: read stock level from each snapshot directly
        dates = sorted(snapshots.keys())
        rows = []
        for d in dates:
            df = snapshots[d]
            if "sku" not in df.columns or "available_qty" not in df.columns:
                continue
            sku_rows = df[df["sku"] == target]
            if sku_rows.empty:
                continue
            stock = float(sku_rows["available_qty"].sum())
            rows.append({"ds": pd.Timestamp(d), "y": stock})

        if len(rows) < 2:
            return pd.DataFrame(columns=["ds", "y"])

        prophet_df = pd.DataFrame(rows)

        if extra_regressors:
            # Compute inflow/outflow/loss per week for this SKU from deltas
            deltas = []
            for i in range(1, len(dates)):
                prev_df = snapshots[dates[i - 1]]
                curr_df = snapshots[dates[i]]
                sku_prev = prev_df[prev_df["sku"] == target] if "sku" in prev_df.columns else pd.DataFrame()
                sku_curr = curr_df[curr_df["sku"] == target] if "sku" in curr_df.columns else pd.DataFrame()

                inflow = 0
                loss = 0
                outflow = 0

                if not sku_prev.empty and not sku_curr.empty:
                    prev_stock = float(sku_prev["available_qty"].sum()) if "available_qty" in sku_prev.columns else 0
                    curr_stock = float(sku_curr["available_qty"].sum()) if "available_qty" in sku_curr.columns else 0
                    stock_change = curr_stock - prev_stock

                    for col in INFLOW_COLUMNS:
                        if col in sku_curr.columns and col in sku_prev.columns:
                            delta = float(sku_curr[col].sum()) - float(sku_prev[col].sum())
                            inflow += max(0, delta)

                    for col in LOSS_COLUMNS:
                        if col in sku_curr.columns and col in sku_prev.columns:
                            delta = float(sku_curr[col].sum()) - float(sku_prev[col].sum())
                            loss += max(0, delta)

                    outflow = max(0, -stock_change - inflow + loss)

                deltas.append({
                    "ds": pd.Timestamp(dates[i]),
                    "inflow": inflow,
                    "outflow": outflow,
                    "loss": loss,
                })

            if deltas:
                delta_df = pd.DataFrame(deltas)
                prophet_df = prophet_df.merge(delta_df, on="ds", how="left")
                prophet_df[["inflow", "outflow", "loss"]] = prophet_df[["inflow", "outflow", "loss"]].fillna(0)

    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"] = prophet_df["y"].astype(float)
    prophet_df = prophet_df.dropna(subset=["ds", "y"])

    return prophet_df


# ---------------------------------------------------------------------------
# Forecastable SKU computation
# ---------------------------------------------------------------------------

def compute_forecastable_skus(
    snapshots: Dict[date, pd.DataFrame],
    min_snapshots: int = 2,
) -> list:
    """
    Find SKUs present in at least `min_snapshots` snapshots
    AND with non-zero stock in at least one snapshot.

    Returns sorted list of SKU strings worth forecasting.
    """
    if len(snapshots) < min_snapshots:
        return []

    dates = sorted(snapshots.keys())
    sku_presence: Dict[str, int] = {}
    sku_total_stock: Dict[str, float] = {}

    for d in dates:
        df = snapshots[d]
        if "sku" not in df.columns:
            continue
        for sku in df["sku"].dropna().unique():
            sku_presence[sku] = sku_presence.get(sku, 0) + 1
            if "available_qty" in df.columns:
                stock = float(df[df["sku"] == sku]["available_qty"].sum())
                sku_total_stock[sku] = sku_total_stock.get(sku, 0) + stock

    forecastable = [
        sku for sku, count in sku_presence.items()
        if count >= min_snapshots and sku_total_stock.get(sku, 0) > 0
    ]
    return sorted(forecastable)


def save_forecastable_skus_to_s3(
    bucket: str,
    prefix: str,
    snapshots: Dict[date, pd.DataFrame],
    region: str = "ap-south-1",
) -> str:
    """
    Compute forecastable SKUs and save as JSON to S3.

    Key: {prefix}/_forecastable-skus.json
    Returns: S3 key
    """
    import json
    import boto3

    skus = compute_forecastable_skus(snapshots)
    dates = sorted(snapshots.keys())

    payload = {
        "skus": skus,
        "count": len(skus),
        "snapshot_count": len(dates),
        "snapshot_range": {
            "first": dates[0].isoformat() if dates else None,
            "last": dates[-1].isoformat() if dates else None,
        },
    }

    s3 = boto3.client("s3", region_name=region)
    key = f"{prefix}/_forecastable-skus.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    print(f"[DELTA] Saved {len(skus)} forecastable SKUs to s3://{bucket}/{key}")
    return key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inventory_delta.py <snapshot_dir>              # Weekly aggregate time series")
        print("  python inventory_delta.py <snapshot_dir> --sku SKU    # Per-SKU weekly deltas")
        sys.exit(1)

    snapshot_dir = sys.argv[1]
    print(f"Loading weekly snapshots from {snapshot_dir}...")
    snapshots = load_snapshots_from_dir(snapshot_dir)
    print(f"Loaded {len(snapshots)} weekly snapshots")

    if len(snapshots) < 2:
        print("Need at least 2 weekly snapshots for delta computation.")
        sys.exit(1)

    if "--sku" in sys.argv:
        sku = sys.argv[sys.argv.index("--sku") + 1]
        ts = compute_sku_level_timeseries(snapshots, top_n=999)
        ts = ts[ts["sku"] == sku]
        print(f"\nSKU-level weekly time series for {sku}:")
        print(ts.to_string(index=False))
    else:
        ts = compute_weekly_timeseries(snapshots, aggregate=True)
        print(f"\nAggregate weekly time series ({len(ts)} weeks):")
        print(ts.to_string(index=False))
