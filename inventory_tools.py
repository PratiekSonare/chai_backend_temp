"""
Inventory Intelligence Tool Functions
Fetches CSV snapshots from EasyEcom API, parses to DataFrame,
and provides metric calculation functions for inventory analytics.
"""
import os
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from utils.type_converters import convert_numpy_types

# Column mapping for clean internal names
INVENTORY_COLUMNS = {
    'Report Generated Date': 'report_date',
    'Company Token': 'company_token',
    'Location': 'location',
    'Product Name': 'product_name',
    'Description': 'description',
    'SKU': 'sku',
    'EAN': 'ean',
    'Model No': 'model_no',
    'Category': 'category',
    'Brand': 'brand',
    'Weight(gm)': 'weight_gm',
    'Length(cm)': 'length_cm',
    'Height(cm)': 'height_cm',
    'Width(cm)': 'width_cm',
    'Received': 'received',
    'Reserved (Not Picked)': 'reserved_not_picked',
    'Reserved (Picked)': 'reserved_picked',
    'Damaged': 'damaged',
    'Discard/Fraud': 'discard_fraud',
    'Repair': 'repair',
    'To Receive': 'to_receive',
    'Return Available': 'return_available',
    'Available Quantity': 'available_qty',
    'Available Quantity (Bin Locked)': 'available_bin_locked',
    'Quarantine': 'quarantine',
    'Marketplace_Available': 'marketplace_available',
    'Undispatched Unassigned Quantity': 'undispatched_unassigned',
    'QC Passed': 'qc_passed',
    'QC Failed': 'qc_failed',
    'QC Pending': 'qc_pending',
    'NearExpiry': 'near_expiry',
    'Expiry': 'expiry',
    'Total Lost': 'total_lost',
    'Questionable': 'questionable',
    'Website Inventory': 'website_inventory',
    'E-Com Inventory': 'ecom_inventory',
    'Retail Inventory': 'retail_inventory',
    'IIA Inventory': 'iia_inventory',
    'Amazon Reserved': 'amazon_reserved',
    'Lost In Cycle Count': 'lost_cycle_count',
    'Used In Lite Kitting In-Progress': 'kitting_in_progress',
}

NUMERIC_COLUMNS = [
    'weight_gm', 'length_cm', 'height_cm', 'width_cm',
    'received', 'reserved_not_picked', 'reserved_picked',
    'damaged', 'discard_fraud', 'repair', 'to_receive',
    'return_available', 'available_qty', 'available_bin_locked',
    'quarantine', 'marketplace_available', 'undispatched_unassigned',
    'qc_passed', 'qc_failed', 'qc_pending',
    'near_expiry', 'expiry', 'total_lost', 'questionable',
    'website_inventory', 'ecom_inventory', 'retail_inventory',
    'iia_inventory', 'amazon_reserved', 'lost_cycle_count',
    'kitting_in_progress',
]


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing errors to 0."""
    return pd.to_numeric(series, errors='coerce').fillna(0)


def _ensure_inventory_df(table) -> pd.DataFrame:
    """Ensure input is a clean inventory DataFrame."""
    if isinstance(table, pd.DataFrame):
        df = table.copy()
    elif isinstance(table, list):
        df = pd.DataFrame(table)
    elif isinstance(table, str):
        import json
        df = pd.DataFrame(json.loads(table))
    else:
        return pd.DataFrame()

    # Clean numeric columns if present
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    return df


# ===================================================================
# DATA FETCHING
# ===================================================================
def get_inventory_snapshot(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch inventory snapshot CSV from EasyEcom API and return as DataFrame.

    Args:
        start_date: Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_date: End date in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Parsed inventory DataFrame with cleaned column names
    """
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")
    base_url = os.getenv("EASYECOM_BASE_URL", "https://api.easyecom.io")

    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in .env")

    print(f"[INVENTORY] Fetching snapshot from EasyEcom: {start_date} to {end_date}")

    # Step 1: Get snapshot metadata (file URLs)
    snapshot_url = f"{base_url}/inventory/getInventorySnapshotApi"
    params = {"start_date": start_date, "end_date": end_date}
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(snapshot_url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != 200 or not data.get("data"):
        raise ValueError(f"Inventory snapshot API returned: {data.get('message', 'No data')}")

    snapshots = data["data"]
    print(f"[INVENTORY] Found {len(snapshots)} snapshot(s)")

    # Step 2: Download and parse each CSV
    all_dfs = []
    for snapshot in snapshots:
        file_url = snapshot.get("file_url")
        if not file_url:
            continue

        print(f"[INVENTORY] Downloading CSV: {file_url[:80]}...")
        csv_response = requests.get(file_url)
        csv_response.raise_for_status()

        # Parse CSV - handle BOM and encoding issues
        csv_text = csv_response.content.decode('utf-8-sig')
        df = pd.read_csv(io.StringIO(csv_text))

        # Map column names
        rename_map = {col: INVENTORY_COLUMNS.get(col, col) for col in df.columns}
        df = df.rename(columns=rename_map)

        # Carry entry_date from snapshot metadata into each row
        entry_date = snapshot.get("entry_date")
        if entry_date:
            df["entry_date"] = entry_date

        # Clean numeric columns
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = _to_numeric_safe(df[col])

        all_dfs.append(df)
        print(f"[INVENTORY] Parsed {len(df)} rows from snapshot")

    if not all_dfs:
        raise ValueError("No inventory data found for the given date range")

    # Combine all snapshots (keep latest if duplicates)
    result_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by SKU (keep latest report_date)
    if 'report_date' in result_df.columns and 'sku' in result_df.columns:
        result_df['report_date'] = pd.to_datetime(result_df['report_date'], errors='coerce')
        result_df = result_df.sort_values('report_date', ascending=False)
        result_df = result_df.drop_duplicates(subset=['sku'], keep='first')

    print(f"[INVENTORY] Final dataset: {len(result_df)} SKUs")
    return result_df


# ===================================================================
# STOCK HEALTH TOOLS
# ===================================================================
def get_stock_health(table) -> dict:
    """Get stock health breakdown: Available vs Reserved vs Damaged vs Lost."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    total_skus = len(df)
    total_received = int(df['received'].sum()) if 'received' in df.columns else 0
    total_available = int(df['available_qty'].sum()) if 'available_qty' in df.columns else 0
    total_bin_locked = int(df['available_bin_locked'].sum()) if 'available_bin_locked' in df.columns else 0
    total_reserved_picked = int(df['reserved_picked'].sum()) if 'reserved_picked' in df.columns else 0
    total_reserved_not_picked = int(df['reserved_not_picked'].sum()) if 'reserved_not_picked' in df.columns else 0
    total_amazon_reserved = int(df['amazon_reserved'].sum()) if 'amazon_reserved' in df.columns else 0
    total_damaged = int(df['damaged'].sum()) if 'damaged' in df.columns else 0
    total_discard = int(df['discard_fraud'].sum()) if 'discard_fraud' in df.columns else 0
    total_lost = int(df['total_lost'].sum()) if 'total_lost' in df.columns else 0
    total_quarantine = int(df['quarantine'].sum()) if 'quarantine' in df.columns else 0
    total_questionable = int(df['questionable'].sum()) if 'questionable' in df.columns else 0
    total_repair = int(df['repair'].sum()) if 'repair' in df.columns else 0
    total_return_available = int(df['return_available'].sum()) if 'return_available' in df.columns else 0
    total_to_receive = int(df['to_receive'].sum()) if 'to_receive' in df.columns else 0

    total_reserved = total_reserved_picked + total_reserved_not_picked + total_amazon_reserved
    total_problem = total_damaged + total_discard + total_lost + total_quarantine + total_questionable
    total_usable = total_available + total_bin_locked

    health_score = round((total_usable / total_received * 100), 2) if total_received > 0 else 0

    return convert_numpy_types({
        "total_skus": total_skus,
        "total_received": total_received,
        "total_available": total_available,
        "total_bin_locked": total_bin_locked,
        "total_usable": total_usable,
        "total_reserved": total_reserved,
        "reserved_picked": total_reserved_picked,
        "reserved_not_picked": total_reserved_not_picked,
        "amazon_reserved": total_amazon_reserved,
        "total_damaged": total_damaged,
        "total_discard_fraud": total_discard,
        "total_lost": total_lost,
        "total_quarantine": total_quarantine,
        "total_questionable": total_questionable,
        "total_repair": total_repair,
        "total_problem_stock": total_problem,
        "total_return_available": total_return_available,
        "total_to_receive": total_to_receive,
        "health_score_pct": health_score,
        "problem_stock_pct": round(total_problem / total_received * 100, 2) if total_received > 0 else 0,
    })


def get_damage_rate(table) -> dict:
    """Calculate damage, loss, fraud, and return rates."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    total_received = int(df['received'].sum()) if 'received' in df.columns else 0
    if total_received == 0:
        return {"error": "No received inventory data"}

    total_damaged = int(df['damaged'].sum()) if 'damaged' in df.columns else 0
    total_discard = int(df['discard_fraud'].sum()) if 'discard_fraud' in df.columns else 0
    total_lost = int(df['total_lost'].sum()) if 'total_lost' in df.columns else 0
    total_lost_cycle = int(df['lost_cycle_count'].sum()) if 'lost_cycle_count' in df.columns else 0
    total_return = int(df['return_available'].sum()) if 'return_available' in df.columns else 0
    total_repair = int(df['repair'].sum()) if 'repair' in df.columns else 0

    damage_rate = round(total_damaged / total_received * 100, 2)
    fraud_rate = round(total_discard / total_received * 100, 2)
    loss_rate = round(total_lost / total_received * 100, 2)
    return_rate = round(total_return / total_received * 100, 2)
    repair_rate = round(total_repair / total_received * 100, 2)
    combined_loss = round((total_damaged + total_discard + total_lost) / total_received * 100, 2)

    # Top damaged SKUs
    top_damaged = []
    if 'damaged' in df.columns and 'sku' in df.columns:
        damaged_df = df[df['damaged'] > 0].nlargest(5, 'damaged')
        top_damaged = damaged_df[['sku', 'product_name', 'damaged', 'received']].to_dict('records') if 'product_name' in df.columns else damaged_df[['sku', 'damaged', 'received']].to_dict('records')

    return convert_numpy_types({
        "total_received": total_received,
        "damage_rate_pct": damage_rate,
        "fraud_rate_pct": fraud_rate,
        "loss_rate_pct": loss_rate,
        "return_rate_pct": return_rate,
        "repair_rate_pct": repair_rate,
        "combined_loss_pct": combined_loss,
        "top_damaged_skus": top_damaged,
    })


def get_dead_stock(table) -> dict:
    """Identify dead stock: SKUs with available inventory but zero marketplace/website availability."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    # Dead stock = available > 0 but all channels are 0
    channel_cols = ['marketplace_available', 'website_inventory', 'ecom_inventory']
    available_col = 'available_qty'

    if available_col not in df.columns:
        return {"error": "Available Quantity column not found"}

    mask = df[available_col] > 0
    for col in channel_cols:
        if col in df.columns:
            mask = mask & (df[col] == 0)

    dead_stock_df = df[mask].copy()
    total_dead_units = int(dead_stock_df[available_col].sum()) if not dead_stock_df.empty else 0
    total_dead_skus = len(dead_stock_df)

    total_available = int(df[available_col].sum())
    dead_stock_pct = round(total_dead_units / total_available * 100, 2) if total_available > 0 else 0

    # Top dead stock SKUs
    top_dead = []
    if not dead_stock_df.empty:
        sort_col = 'weight_gm' if 'weight_gm' in dead_stock_df.columns else available_col
        top_df = dead_stock_df.nlargest(10, available_col)
        cols = ['sku', 'product_name', 'category', 'brand', 'location'] if 'product_name' in dead_stock_df.columns else ['sku', 'category', 'brand', 'location']
        cols = [c for c in cols if c in top_df.columns]
        top_dead = top_df[cols + [available_col]].to_dict('records')

    return convert_numpy_types({
        "total_dead_skus": total_dead_skus,
        "total_dead_units": total_dead_units,
        "dead_stock_pct": dead_stock_pct,
        "total_available": total_available,
        "top_dead_stock": top_dead,
    })


def get_dead_score(table) -> dict:
    """
    Calculate dead score for each SKU:
        base = 0.4 * log(available_qty) + 0.6 * log(age_days)
        dead_score = base × channel_multiplier × expiry_bonus

    channel_multiplier = 1.5 if no sales channel has availability, else 1.0
    expiry_bonus = 1.3 if near_expiry or expiry > 0, else 1.0

    Age is computed from the snapshot's entry_date to today.
    Returns top 10 SKUs with the highest dead score.
    """
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    if 'available_qty' not in df.columns:
        return {"error": "Available Quantity column not found"}

    df = df.copy()
    df['available_qty'] = _to_numeric_safe(df['available_qty'])

    # Filter to SKUs with positive available qty
    df = df[df['available_qty'] > 0].copy()
    if df.empty:
        return {"error": "No SKUs with available quantity"}

    # Parse entry_date and compute age in days
    # Fall back to report_date if entry_date is not available (e.g. stale cache)
    if 'entry_date' not in df.columns or df['entry_date'].isna().all():
        if 'report_date' in df.columns:
            df['entry_date'] = df['report_date']
        else:
            return {"error": "No date column found (entry_date or report_date)"}

    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df = df.dropna(subset=['entry_date'])
    if df.empty:
        return {"error": "No valid entry dates found"}

    now = pd.Timestamp.now()
    df['age_days'] = (now - df['entry_date']).dt.total_seconds() / 86400
    df['age_days'] = df['age_days'].clip(lower=1)  # avoid log(0)

    # Channel multiplier: 1.5x if no sales channel has availability
    channel_cols = ['marketplace_available', 'website_inventory', 'ecom_inventory']
    for col in channel_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = _to_numeric_safe(df[col])

    df['total_channel'] = df['marketplace_available'] + df['website_inventory'] + df['ecom_inventory']
    df['channel_multiplier'] = np.where(df['total_channel'] == 0, 1.5, 1.0)

    # Expiry bonus: 1.3x if item is near expiry or expired
    for col in ['near_expiry', 'expiry']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = _to_numeric_safe(df[col])

    df['expiry_bonus'] = np.where((df['near_expiry'] > 0) | (df['expiry'] > 0), 1.3, 1.0)

    # Calculate dead score
    df['dead_score'] = (
        (0.4 * np.log(df['available_qty']) + 0.6 * np.log(df['age_days']))
        * df['channel_multiplier']
        * df['expiry_bonus']
    )

    # Top 10 by dead score
    top_df = df.nlargest(10, 'dead_score')
    cols = ['sku', 'product_name', 'category', 'brand', 'available_qty', 'age_days', 'dead_score', 'entry_date', 'total_channel']
    cols = [c for c in cols if c in top_df.columns]
    top_skus = top_df[cols].to_dict('records')

    # Round numeric fields for readability
    for rec in top_skus:
        if 'age_days' in rec:
            rec['age_days'] = round(rec['age_days'], 1)
        if 'dead_score' in rec:
            rec['dead_score'] = round(rec['dead_score'], 4)
        if 'entry_date' in rec:
            rec['entry_date'] = str(rec['entry_date'])

    return convert_numpy_types({
        "total_skus_scored": len(df),
        "top_10_dead_score": top_skus,
    })


def get_overstock_risk(table, threshold_pct: float = 80.0) -> dict:
    """Identify overstock risk: SKUs where available is much higher than channel availability."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    if 'available_qty' not in df.columns:
        return {"error": "Available Quantity column not found"}

    # Calculate total channel availability
    channel_cols = ['marketplace_available', 'website_inventory', 'ecom_inventory', 'retail_inventory']
    available_cols = [c for c in channel_cols if c in df.columns]

    if not available_cols:
        return {"error": "No channel availability columns found"}

    df = df.copy()
    df['total_channel_avail'] = df[available_cols].sum(axis=1)
    df['channel_pct'] = np.where(
        df['available_qty'] > 0,
        (df['total_channel_avail'] / df['available_qty'] * 100).round(2),
        0
    )

    # Overstock = high available but very low channel availability percentage
    overstock_mask = (df['available_qty'] > 0) & (df['channel_pct'] < (100 - threshold_pct))
    overstock_df = df[overstock_mask].copy()

    total_overstock_units = int(overstock_df['available_qty'].sum()) if not overstock_df.empty else 0
    total_overstock_skus = len(overstock_df)

    top_overstock = []
    if not overstock_df.empty:
        top_df = overstock_df.nlargest(10, 'available_qty')
        cols = ['sku', 'product_name', 'category', 'available_qty', 'channel_pct']
        cols = [c for c in cols if c in top_df.columns]
        top_overstock = top_df[cols].to_dict('records')

    return convert_numpy_types({
        "total_overstock_skus": total_overstock_skus,
        "total_overstock_units": total_overstock_units,
        "threshold_pct": threshold_pct,
        "top_overstock": top_overstock,
    })


def get_understock_risk(table, safety_stock: int = 5) -> dict:
    """Identify understock risk: SKUs where available is critically low."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    if 'available_qty' not in df.columns:
        return {"error": "Available Quantity column not found"}

    # Understock = available <= safety stock AND has channel presence
    channel_cols = ['marketplace_available', 'website_inventory', 'ecom_inventory']
    available_cols = [c for c in channel_cols if c in df.columns]

    df = df.copy()
    if available_cols:
        df['total_channel'] = df[available_cols].sum(axis=1)
        understock_mask = (df['available_qty'] <= safety_stock) & (df['total_channel'] > 0)
    else:
        understock_mask = df['available_qty'] <= safety_stock

    understock_df = df[understock_mask].copy()

    top_understock = []
    if not understock_df.empty:
        top_df = understock_df.nsmallest(10, 'available_qty')
        cols = ['sku', 'product_name', 'category', 'available_qty', 'to_receive']
        cols = [c for c in cols if c in top_df.columns]
        top_understock = top_df[cols].to_dict('records')

    return convert_numpy_types({
        "total_understock_skus": len(understock_df),
        "safety_stock_threshold": safety_stock,
        "top_understock": top_understock,
    })


def get_qc_performance(table) -> dict:
    """Calculate QC pass/fail/pending rates."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    qc_cols = ['qc_passed', 'qc_failed', 'qc_pending']
    if not all(c in df.columns for c in qc_cols):
        return {"error": "QC columns not found"}

    total_passed = int(df['qc_passed'].sum())
    total_failed = int(df['qc_failed'].sum())
    total_pending = int(df['qc_pending'].sum())
    total_qc = total_passed + total_failed + total_pending

    pass_rate = round(total_passed / total_qc * 100, 2) if total_qc > 0 else 0
    fail_rate = round(total_failed / total_qc * 100, 2) if total_qc > 0 else 0
    pending_rate = round(total_pending / total_qc * 100, 2) if total_qc > 0 else 0

    # Worst performing SKUs (highest fail rate)
    worst_qc = []
    if 'sku' in df.columns:
        df_with_rate = df[df['qc_failed'] > 0].copy()
        if not df_with_rate.empty:
            df_with_rate['qc_total'] = df_with_rate['qc_passed'] + df_with_rate['qc_failed'] + df_with_rate['qc_pending']
            df_with_rate['fail_rate'] = np.where(
                df_with_rate['qc_total'] > 0,
                (df_with_rate['qc_failed'] / df_with_rate['qc_total'] * 100).round(2),
                0
            )
            top_df = df_with_rate.nlargest(5, 'fail_rate')
            cols = ['sku', 'product_name', 'qc_passed', 'qc_failed', 'qc_pending', 'fail_rate']
            cols = [c for c in cols if c in top_df.columns]
            worst_qc = top_df[cols].to_dict('records')

    return convert_numpy_types({
        "total_qc_units": total_qc,
        "qc_passed": total_passed,
        "qc_failed": total_failed,
        "qc_pending": total_pending,
        "pass_rate_pct": pass_rate,
        "fail_rate_pct": fail_rate,
        "pending_rate_pct": pending_rate,
        "worst_performing_skus": worst_qc,
    })


def get_expiry_risk(table) -> dict:
    """Identify SKUs at expiry risk."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    near_expiry = int(df['near_expiry'].sum()) if 'near_expiry' in df.columns else 0
    expired = int(df['expiry'].sum()) if 'expiry' in df.columns else 0
    total_available = int(df['available_qty'].sum()) if 'available_qty' in df.columns else 0

    expiry_risk_pct = round((near_expiry + expired) / total_available * 100, 2) if total_available > 0 else 0

    # Top expiry risk SKUs
    top_expiry = []
    if 'sku' in df.columns:
        if 'near_expiry' in df.columns and 'expiry' in df.columns:
            df = df.copy()
            df['expiry_total'] = df['near_expiry'] + df['expiry']
            expiry_df = df[df['expiry_total'] > 0].nlargest(10, 'expiry_total')
            cols = ['sku', 'product_name', 'category', 'near_expiry', 'expiry', 'available_qty']
            cols = [c for c in cols if c in expiry_df.columns]
            top_expiry = expiry_df[cols].to_dict('records')

    return convert_numpy_types({
        "total_near_expiry": near_expiry,
        "total_expired": expired,
        "expiry_risk_pct": expiry_risk_pct,
        "total_available": total_available,
        "top_expiry_risk_skus": top_expiry,
    })


def get_channel_distribution(table) -> dict:
    """Get inventory distribution across sales channels."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    channels = {}
    channel_map = {
        'marketplace_available': 'Marketplace',
        'website_inventory': 'Website',
        'ecom_inventory': 'E-Commerce',
        'retail_inventory': 'Retail',
        'iia_inventory': 'IIA',
        'amazon_reserved': 'Amazon Reserved',
    }

    total_channel_units = 0
    for col, name in channel_map.items():
        if col in df.columns:
            val = int(df[col].sum())
            channels[name] = val
            total_channel_units += val

    # Add non-channel inventory
    non_channel = {
        'bin_locked': int(df['available_bin_locked'].sum()) if 'available_bin_locked' in df.columns else 0,
        'quarantine': int(df['quarantine'].sum()) if 'quarantine' in df.columns else 0,
        'repair': int(df['repair'].sum()) if 'repair' in df.columns else 0,
    }

    # Calculate percentages
    channel_pcts = {}
    for name, val in channels.items():
        channel_pcts[name] = round(val / total_channel_units * 100, 2) if total_channel_units > 0 else 0

    return convert_numpy_types({
        "channels": channels,
        "channel_percentages": channel_pcts,
        "non_channel_inventory": non_channel,
        "total_channel_units": total_channel_units,
    })


def get_category_breakdown(table) -> dict:
    """Get inventory breakdown by product category."""
    df = _ensure_inventory_df(table)
    if df.empty or 'category' not in df.columns:
        return {"error": "No category data"}

    agg_cols = {}
    for col in ['available_qty', 'received', 'damaged', 'total_lost']:
        if col in df.columns:
            agg_cols[col] = 'sum'

    if not agg_cols:
        return {"error": "No numeric columns to aggregate"}

    category_df = df.groupby('category').agg(agg_cols).reset_index()
    category_df = category_df.sort_values('available_qty' if 'available_qty' in category_df.columns else list(agg_cols.keys())[0], ascending=False)

    return convert_numpy_types({
        "categories": category_df.to_dict('records'),
        "total_categories": len(category_df),
    })


def get_brand_breakdown(table) -> dict:
    """Get inventory breakdown by brand."""
    df = _ensure_inventory_df(table)
    if df.empty or 'brand' not in df.columns:
        return {"error": "No brand data"}

    agg_cols = {}
    for col in ['available_qty', 'received', 'damaged', 'total_lost']:
        if col in df.columns:
            agg_cols[col] = 'sum'

    if not agg_cols:
        return {"error": "No numeric columns to aggregate"}

    brand_df = df.groupby('brand').agg(agg_cols).reset_index()
    brand_df = brand_df.sort_values('available_qty' if 'available_qty' in brand_df.columns else list(agg_cols.keys())[0], ascending=False)

    return convert_numpy_types({
        "brands": brand_df.to_dict('records'),
        "total_brands": len(brand_df),
    })


def get_location_breakdown(table) -> dict:
    """Get inventory breakdown by warehouse location."""
    df = _ensure_inventory_df(table)
    if df.empty or 'location' not in df.columns:
        return {"error": "No location data"}

    agg_cols = {}
    for col in ['available_qty', 'received', 'damaged', 'total_lost']:
        if col in df.columns:
            agg_cols[col] = 'sum'

    if not agg_cols:
        return {"error": "No numeric columns to aggregate"}

    location_df = df.groupby('location').agg(agg_cols).reset_index()
    location_df = location_df.sort_values('available_qty' if 'available_qty' in location_df.columns else list(agg_cols.keys())[0], ascending=False)

    return convert_numpy_types({
        "locations": location_df.to_dict('records'),
        "total_locations": len(location_df),
    })


def get_inventory_summary(table) -> dict:
    """Get comprehensive inventory summary with all key metrics."""
    df = _ensure_inventory_df(table)
    if df.empty:
        return {"error": "No inventory data"}

    result = {
        "stock_health": get_stock_health(df),
        "damage_analysis": get_damage_rate(df),
        "dead_stock": get_dead_stock(df),
        "dead_score": get_dead_score(df),
        "qc_performance": get_qc_performance(df),
        "expiry_risk": get_expiry_risk(df),
        "channel_distribution": get_channel_distribution(df),
    }

    return convert_numpy_types(result)


def apply_inventory_filters(table, filters: List[Dict]) -> pd.DataFrame:
    """
    Apply filters to inventory DataFrame.

    Args:
        table: Inventory DataFrame or list of dicts
        filters: List of filter dicts with structure:
            [{"field": "category", "operator": "eq", "value": "Footwear"}, ...]

    Returns:
        Filtered DataFrame
    """
    df = _ensure_inventory_df(table)
    if df.empty or not filters:
        return df

    for filter_spec in filters:
        field = filter_spec.get("field")
        operator = filter_spec.get("operator", "eq")
        value = filter_spec.get("value")

        if field not in df.columns:
            print(f"[INVENTORY FILTER] Field '{field}' not found, skipping")
            continue

        if operator == "eq":
            if isinstance(value, str):
                df = df[df[field].astype(str).str.lower() == value.lower()]
            else:
                df = df[df[field] == value]
        elif operator == "ne":
            if isinstance(value, str):
                df = df[df[field].astype(str).str.lower() != value.lower()]
            else:
                df = df[df[field] != value]
        elif operator == "gt":
            df = df[pd.to_numeric(df[field], errors='coerce') > float(value)]
        elif operator == "lt":
            df = df[pd.to_numeric(df[field], errors='coerce') < float(value)]
        elif operator == "gte":
            df = df[pd.to_numeric(df[field], errors='coerce') >= float(value)]
        elif operator == "lte":
            df = df[pd.to_numeric(df[field], errors='coerce') <= float(value)]
        elif operator == "contains":
            df = df[df[field].astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == "in":
            if isinstance(value, list):
                df = df[df[field].isin(value)]

    return df
