"""Generate pre-calculated metrics presets and upload to S3.

This script pre-calculates metrics for 7d, 30d, and all-time presets,
then uploads the results to S3 for direct frontend consumption.

Usage:
    python generate_metrics_presets.py --execution-date 2026-05-07

S3 output format:
- Bucket: chupps-data-portal (configurable via METRICS_PRESETS_BUCKET)
- Prefix: metrics-presets (configurable via METRICS_PRESETS_PREFIX)
- Key: metrics-presets/YYYY-MM-DD/all.json

Schema:
{
    "_execution_timestamp": "2026-05-07T00:10:00Z",
    "_execution_date": "2026-05-07",
    "_fallback_date": false,  # true if using previous day's metrics on failure
    "7d": { ...full metrics response... },
    "30d": { ...full metrics response... },
    "all": { ...full metrics response... }
}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple

import boto3
import pandas as pd
from dotenv import load_dotenv


# Constants
DEFAULT_BUCKET = "chupps-data-portal"
DEFAULT_PREFIX = "metrics-presets"
DEFAULT_AWS_REGION = "ap-south-1"
DEFAULT_DYNAMODB_TABLE = "history-orders-final"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

PRESET_7D = "7d"
PRESET_30D = "30d"
PRESET_ALL = "all"
SUPPORTED_PRESETS = (PRESET_7D, PRESET_30D, PRESET_ALL)

DEFAULT_ALL_TIME_START = os.getenv("HISTORY_CACHE_ALL_TIME_START", "2025-09-01 00:00:00")

# Enum for order status
class OrderStatus:
    DELIVERED = "Delivered"
    CANCELLED = "Cancelled"
    RETURNED = "Returned"
    RTO = "RTO"
    
    @staticmethod
    def normalize(value):
        if not value:
            return None
        value_str = str(value).strip()
        if "Delivered" in value_str or "Pickup" in value_str:
            return OrderStatus.DELIVERED
        elif "Cancelled" in value_str:
            return OrderStatus.CANCELLED
        elif "RTO" in value_str or "Return" in value_str:
            return OrderStatus.RETURNED
        return None


# Load environment variables
load_dotenv()


def parse_date(value: str) -> date:
    """Parse date in YYYY-MM-DD format."""
    return datetime.strptime(value, DATE_FMT).date()


def create_dynamodb_client(region_name: str):
    """Create DynamoDB low-level client."""
    return boto3.client("dynamodb", region_name=region_name)


def create_s3_client(region_name: str):
    """Create S3 client."""
    return boto3.client("s3", region_name=region_name)


def _normalize_for_dynamodb(value):
    """Convert Python values to DynamoDB-serializable primitives."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize_for_dynamodb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_dynamodb(v) for v in value]
    return value


def fetch_historical_orders_from_dynamodb(
    dynamodb_client,
    table_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch orders from DynamoDB between start_date and end_date.
    Returns a pandas DataFrame.
    """
    print(f"Fetching orders from DynamoDB table '{table_name}' between {start_date} and {end_date}")
    
    all_items = []
    scan_kwargs = {}
    
    try:
        paginator = dynamodb_client.get_paginator("scan")
        page_iterator = paginator.paginate(TableName=table_name, **scan_kwargs)
        
        for page in page_iterator:
            items = page.get("Items", [])
            for item in items:
                # Convert DynamoDB format to Python dict
                row = {}
                for key, value in item.items():
                    if "S" in value:
                        row[key] = value["S"]
                    elif "N" in value:
                        try:
                            row[key] = float(value["N"])
                        except ValueError:
                            row[key] = value["N"]
                    elif "BOOL" in value:
                        row[key] = value["BOOL"]
                    elif "NULL" in value:
                        row[key] = None
                    else:
                        row[key] = value
                all_items.append(row)
        
        print(f"Fetched {len(all_items)} items from DynamoDB")
        
        if not all_items:
            print("No items found, returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_items)
        
        # Convert order_date to datetime and filter by date range
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            
            mask = (df["order_date"] >= start_dt) & (df["order_date"] <= end_dt)
            df = df[mask]
            print(f"After date filtering: {len(df)} rows")
        
        return df
    
    except Exception as e:
        print(f"Error fetching orders from DynamoDB: {str(e)}")
        return pd.DataFrame()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif hasattr(obj, "item"):  # NumPy types
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _ensure_numeric(df, columns):
    """Ensure specified columns are numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _order_count(df):
    """Get count of distinct orders."""
    if "order_id" in df.columns:
        return int(df["order_id"].nunique())
    return len(df)


def _status_order_count(df, status_set):
    """Count orders with specified status."""
    if "order_status" not in df.columns:
        return 0
    return len(df[df["order_status"].isin(status_set)])


def _safe_pct(numerator, denominator):
    """Safely calculate percentage."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def _build_time_groups(df):
    """Build time groups for chart data."""
    if "order_date" not in df.columns:
        return "daily", [], df, None
    
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    date_range = (df["order_date"].max() - df["order_date"].min()).days
    
    if date_range <= 7:
        grouped_df = df.copy()
        grouped_df["date_group"] = grouped_df["order_date"].dt.date
        group_col = "date_group"
        labels = [str(d) for d in sorted(grouped_df["date_group"].unique())]
        return "daily", labels, grouped_df, group_col
    elif date_range <= 30:
        grouped_df = df.copy()
        grouped_df["date_group"] = grouped_df["order_date"].dt.to_period("D")
        group_col = "date_group"
        labels = [str(d) for d in sorted(grouped_df["date_group"].unique())]
        return "daily", labels, grouped_df, group_col
    else:
        grouped_df = df.copy()
        grouped_df["date_group"] = grouped_df["order_date"].dt.to_period("W")
        group_col = "date_group"
        labels = [str(d) for d in sorted(grouped_df["date_group"].unique())]
        return "weekly", labels, grouped_df, group_col


def _calculate_growth_rates(values):
    """Calculate growth rates for chart data."""
    growth_rates = []
    for i, val in enumerate(values):
        if i == 0:
            growth_rates.append(0.0)
        else:
            prev_val = values[i - 1]
            if prev_val == 0:
                growth_rates.append(0.0 if val == 0 else 100.0)
            else:
                growth_rate = ((val - prev_val) / prev_val) * 100
                growth_rates.append(round(growth_rate, 2))
    return growth_rates


def calculate_metrics_for_preset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all metrics from a DataFrame.
    This mirrors the batch_all_metrics endpoint logic.
    """
    print(f"Calculating metrics for {len(df)} rows")
    
    if df.empty:
        return {
            "primaryKpis": {},
            "productMetrics": {},
            "performanceMetrics": {},
            "geographicMetrics": {},
            "channelPaymentMetrics": {},
            "orderTypeMetrics": {},
            "qualityRiskMetrics": {},
            "advancedMetrics": {}
        }
    
    # Ensure numeric columns
    df = _ensure_numeric(df, ["total_amount", "item_quantity", "suborder_quantity", "order_quantity"])
    
    def _get_units_col():
        if "item_quantity" in df.columns and df["item_quantity"].sum() > 0:
            return "item_quantity"
        elif "suborder_quantity" in df.columns and df["suborder_quantity"].sum() > 0:
            return "suborder_quantity"
        elif "order_quantity" in df.columns:
            return "order_quantity"
        return None
    
    units_col = _get_units_col()
    
    # === PRIMARY KPIs ===
    primaryKpis = {}
    
    total_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)
    primaryKpis["totalOrders"] = {
        "success": True,
        "data": total_orders,
        "unit": "orders"
    }
    
    units_sold = 0
    if units_col:
        units_sold = int(df[units_col].sum())
    primaryKpis["unitsSold"] = {
        "success": True,
        "data": units_sold,
        "unit": "units"
    }
    
    gross_revenue = float(df["total_amount"].sum()) if "total_amount" in df.columns else 0.0
    primaryKpis["grossRevenue"] = {
        "success": True,
        "data": gross_revenue,
        "currency": "INR"
    }
    
    # === BUILD CHART DATA FOR LINE CHARTS ===
    # Generate time-series data for totalOrders and grossRevenue
    try:
        chart_type, labels, grouped_df, group_col = _build_time_groups(df)
        
        # Total Orders Chart Data
        if not grouped_df.empty and 'order_id' in grouped_df.columns:
            orders_by_date = grouped_df.groupby(group_col)['order_id'].nunique().sort_index()
            orders_list = orders_by_date.tolist()
            growth_rates = _calculate_growth_rates(orders_list)
            chart_data_orders = [
                {
                    "date": label, 
                    "totalOrders": int(value),
                    "growth": growth,
                    "count": int(value)
                }
                for label, value, growth in zip(labels, orders_list, growth_rates)
            ]
            primaryKpis['totalOrders']['chart'] = chart_data_orders
        
        # Gross Revenue Chart Data
        if not grouped_df.empty and 'total_amount' in grouped_df.columns:
            revenue_by_date = grouped_df.groupby(group_col)['total_amount'].sum().sort_index()
            revenue_list = revenue_by_date.tolist()
            growth_rates = _calculate_growth_rates(revenue_list)
            chart_data_revenue = [
                {
                    "date": label, 
                    "grossRevenue": float(value),
                    "growth": growth,
                    "revenue": float(value)
                }
                for label, value, growth in zip(labels, revenue_list, growth_rates)
            ]
            primaryKpis['grossRevenue']['chart'] = chart_data_revenue
    except Exception as e:
        print(f"Warning: Could not generate chart data: {str(e)}", flush=True)
    
    aov = float(df["total_amount"].mean()) if "total_amount" in df.columns else 0.0
    primaryKpis["aov"] = {
        "success": True,
        "data": aov,
        "currency": "INR"
    }
    
    unique_skus = int(df["sku"].nunique()) if "sku" in df.columns else 0
    primaryKpis["uniqueSkus"] = {
        "success": True,
        "data": unique_skus,
        "unit": "skus"
    }
    
    total = _order_count(df)
    cancelled = _status_order_count(df, {"Cancelled"})
    cancellation_rate = round(_safe_pct(float(cancelled), float(total)), 2)
    primaryKpis["cancellationRate"] = {
        "success": True,
        "data": cancellation_rate,
        "unit": "%"
    }
    
    rto = _status_order_count(df, {"RTO"})
    rto_rate = round(_safe_pct(float(rto), float(total)), 2)
    primaryKpis["rtoRate"] = {
        "success": True,
        "data": rto_rate,
        "unit": "%"
    }
    
    cod_share = 0.0
    if "payment_mode" in df.columns:
        total_items = len(df)
        cod_items = len(df[df["payment_mode"] == "COD"])
        cod_share = round((cod_items / total_items * 100) if total_items > 0 else 0.0, 2)
    primaryKpis["codShare"] = {
        "success": True,
        "data": cod_share,
        "unit": "%"
    }
    
    delivered_rate = 0.0
    if "order_status" in df.columns:
        total_items = len(df)
        delivered = len(df[df["order_status"].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.DELIVERED])
        delivered_rate = round((delivered / total_items * 100) if total_items > 0 else 0.0, 2)
    primaryKpis["deliveredRate"] = {
        "success": True,
        "data": delivered_rate,
        "unit": "%"
    }
    
    # === PRODUCT METRICS ===
    productMetrics = {}
    
    diversity_index = float(unique_skus / total_orders) if total_orders > 0 else 0.0
    productMetrics["skuDiversityIndex"] = {
        "success": True,
        "data": round(diversity_index, 4)
    }
    
    # Top SKUs by Revenue
    if "sku" in df.columns:
        df_sku = df.copy()
        df_sku["sku_clean"] = df_sku["sku"].astype(str).str.strip()
        agg_dict = {
            "order_id": "nunique",
            "total_amount": "sum",
            "item_quantity": "sum",
            "suborder_quantity": "sum",
            "order_quantity": "sum"
        }
        if "suborder_model_no" in df_sku.columns:
            agg_dict["suborder_model_no"] = "first"
        sku_metrics = df_sku.groupby("sku_clean", dropna=False).agg(agg_dict).rename(columns={"order_id": "order_count", "total_amount": "revenue"})
        sku_metrics = sku_metrics.sort_values("revenue", ascending=False).head(5)
        
        top_skus_revenue = []
        for sku, row in sku_metrics.iterrows():
            units = row[units_col] if units_col and pd.notna(row[units_col]) else 0
            aov_sku = float(row["revenue"] / row["order_count"]) if row["order_count"] > 0 else 0.0
            style_name = str(row["suborder_model_no"]) if "suborder_model_no" in sku_metrics.columns and pd.notna(row["suborder_model_no"]) else ""
            top_skus_revenue.append({
                "sku": str(sku),
                "style_name": style_name,
                "revenue": float(row["revenue"]),
                "order_count": int(row["order_count"]),
                "units": int(units),
                "aov": float(aov_sku)
            })
        productMetrics["topSkusByRevenue"] = {
            "success": True,
            "data": top_skus_revenue
        }
    else:
        productMetrics["topSkusByRevenue"] = {"success": True, "data": []}
    
    # Top SKUs by Units
    if "sku" in df.columns and units_col:
        df_sku = df.copy()
        df_sku["sku_clean"] = df_sku["sku"].astype(str).str.strip()
        agg_dict = {
            "order_id": "nunique",
            "total_amount": "sum",
            units_col: "sum"
        }
        if "suborder_model_no" in df_sku.columns:
            agg_dict["suborder_model_no"] = "first"
        sku_units = df_sku.groupby("sku_clean", dropna=False).agg(agg_dict).rename(columns={"order_id": "order_count", "total_amount": "revenue", units_col: "units"})
        sku_units = sku_units.sort_values("units", ascending=False).head(5)
        
        top_skus_units = []
        for sku, row in sku_units.iterrows():
            aov_sku = float(row["revenue"] / row["order_count"]) if row["order_count"] > 0 else 0.0
            style_name = str(row["suborder_model_no"]) if "suborder_model_no" in sku_units.columns and pd.notna(row["suborder_model_no"]) else ""
            top_skus_units.append({
                "sku": str(sku),
                "style_name": style_name,
                "units": int(row["units"]),
                "order_count": int(row["order_count"]),
                "revenue": float(row["revenue"]),
                "aov": float(aov_sku)
            })
        productMetrics["topSkusByUnits"] = {
            "success": True,
            "data": top_skus_units
        }
    else:
        productMetrics["topSkusByUnits"] = {"success": True, "data": []}
    
    # Avg Units per Order
    avg_units_per_order = 0.0
    if units_col and total_orders > 0:
        avg_units_per_order = float(df[units_col].sum() / total_orders)
    productMetrics["avgUnitsPerOrder"] = {
        "success": True,
        "data": round(avg_units_per_order, 2)
    }
    
    # Size Mix Distribution
    size_mix = []
    if "size" in df.columns:
        size_dist = df["size"].value_counts()
        total_size = size_dist.sum()
        for size, count in size_dist.items():
            pct = round((count / total_size * 100), 2) if total_size > 0 else 0.0
            size_mix.append({
                "size": str(size),
                "count": int(count),
                "percentage": float(pct)
            })
    productMetrics["sizeMixDistribution"] = {
        "success": True,
        "data": size_mix
    }
    
    # SKU Performance Matrix
    sku_perf = []
    if "sku" in df.columns and units_col:
        df_perf = df.copy()
        df_perf["sku_clean"] = df_perf["sku"].astype(str).str.strip()
        agg_dict = {
            "order_id": "nunique",
            "total_amount": "sum",
            units_col: "sum"
        }
        if "suborder_model_no" in df_perf.columns:
            agg_dict["suborder_model_no"] = "first"
        sku_perf_agg = df_perf.groupby("sku_clean", dropna=False).agg(agg_dict).rename(columns={"order_id": "orders", "total_amount": "revenue", units_col: "units"})
        sku_perf_agg = sku_perf_agg.sort_values("revenue", ascending=False).head(10)
        
        for sku, row in sku_perf_agg.iterrows():
            style_name = str(row["suborder_model_no"]) if "suborder_model_no" in sku_perf_agg.columns and pd.notna(row["suborder_model_no"]) else ""
            sku_perf.append({
                "sku": str(sku),
                "style_name": style_name,
                "units": int(row["units"]),
                "revenue": float(row["revenue"]),
                "orders": int(row["orders"])
            })
    productMetrics["skuPerformanceMatrix"] = {
        "success": True,
        "data": sku_perf
    }
    
    # === PERFORMANCE METRICS ===
    performanceMetrics = {}
    
    fulfillment_rate = 0.0
    if "order_status" in df.columns:
        total_items = len(df)
        fulfilled = len(df[df["order_status"].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.DELIVERED, OrderStatus.RETURNED])])
        fulfillment_rate = round((fulfilled / total_items * 100) if total_items > 0 else 0.0, 2)
    performanceMetrics["fulfillmentRate"] = {
        "success": True,
        "data": fulfillment_rate
    }
    
    order_value_dist = {}
    if "total_amount" in df.columns:
        order_value_dist = {
            "min": float(df["total_amount"].min()),
            "max": float(df["total_amount"].max()),
            "mean": float(df["total_amount"].mean()),
            "median": float(df["total_amount"].median()),
            "std": float(df["total_amount"].std())
        }
    performanceMetrics["orderValueDist"] = {
        "success": True,
        "data": order_value_dist
    }
    
    order_velocity = {}
    if "order_date" in df.columns:
        df_copy = df.copy()
        df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], errors="coerce")
        days = (df_copy["order_date"].max() - df_copy["order_date"].min()).days + 1
        daily_orders = total_orders / days if days > 0 else 0
        order_velocity = {
            "total_orders": int(total_orders),
            "days": int(days),
            "daily_average": float(round(daily_orders, 2))
        }
    performanceMetrics["orderVelocity"] = {
        "success": True,
        "data": order_velocity
    }
    
    units_velocity = 0.0
    if units_col and "order_date" in df.columns:
        df_copy = df.copy()
        df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], errors="coerce")
        days = (df_copy["order_date"].max() - df_copy["order_date"].min()).days + 1
        units_velocity = units_sold / days if days > 0 else 0
    performanceMetrics["unitsVelocity"] = {
        "success": True,
        "data": round(units_velocity, 2)
    }
    
    # === GEOGRAPHIC METRICS ===
    geographicMetrics = {}
    
    top_states_revenue = []
    if "state" in df.columns:
        state_revenue = df.groupby("state", dropna=False)["total_amount"].agg(["sum", "count"]).rename(columns={"sum": "revenue", "count": "orders"})
        state_revenue = state_revenue.sort_values("revenue", ascending=False).head(10)
        total_rev = state_revenue["revenue"].sum()
        
        for state, row in state_revenue.iterrows():
            pct = round((row["revenue"] / total_rev * 100), 2) if total_rev > 0 else 0.0
            top_states_revenue.append({
                "state": str(state),
                "revenue": float(row["revenue"]),
                "orders": int(row["orders"]),
                "percentage": float(pct)
            })
    geographicMetrics["topStatesByRevenue"] = {
        "success": True,
        "data": top_states_revenue
    }
    
    top_states_orders = []
    if "state" in df.columns:
        state_orders = df.groupby("state", dropna=False).size().sort_values(ascending=False).head(10)
        total_state_orders = state_orders.sum()
        
        for state, count in state_orders.items():
            pct = round((count / total_state_orders * 100), 2) if total_state_orders > 0 else 0.0
            top_states_orders.append({
                "state": str(state),
                "orders": int(count),
                "percentage": float(pct)
            })
    geographicMetrics["topStatesByOrders"] = {
        "success": True,
        "data": top_states_orders
    }
    
    geo_concentration = 0.0
    if "state" in df.columns:
        state_revenue = df.groupby("state", dropna=False)["total_amount"].sum().sort_values(ascending=False)
        top_3_rev = state_revenue.head(3).sum()
        total_revenue = state_revenue.sum()
        geo_concentration = round((top_3_rev / total_revenue * 100), 2) if total_revenue > 0 else 0.0
    geographicMetrics["geoConcentration"] = {
        "success": True,
        "data": geo_concentration
    }
    
    state_cancel_rates = []
    if "state" in df.columns:
        state_cancel = df.groupby("state", dropna=False).apply(
            lambda x: {
                "total": int(x["order_id"].nunique()),
                "cancelled": int(_status_order_count(x, {"Cancelled"}))
            }
        ).reset_index()
        state_cancel.columns = ["state", "metrics"]
        
        for _, row in state_cancel.iterrows():
            cancel_rate = round((row["metrics"]["cancelled"] / row["metrics"]["total"] * 100), 2) if row["metrics"]["total"] > 0 else 0.0
            if cancel_rate > 0:
                state_cancel_rates.append({
                    "state": str(row["state"]),
                    "cancellation_rate": float(cancel_rate),
                    "total_orders": int(row["metrics"]["total"])
                })
        state_cancel_rates = sorted(state_cancel_rates, key=lambda x: x["cancellation_rate"], reverse=True)[:10]
    geographicMetrics["stateCancellationRates"] = {
        "success": True,
        "data": state_cancel_rates
    }
    
    # === CHANNEL & PAYMENT METRICS ===
    channelPaymentMetrics = {}
    
    marketplace_perf = []
    if "marketplace" in df.columns:
        mp_metrics = df.groupby("marketplace", dropna=False).agg({
            "order_id": "nunique",
            "total_amount": "sum",
            units_col: "sum" if units_col else "size"
        }).rename(columns={"order_id": "orders", "total_amount": "revenue"})
        mp_metrics = mp_metrics.sort_values("revenue", ascending=False)
        
        for mp, row in mp_metrics.iterrows():
            aov_mp = float(row["revenue"] / row["orders"]) if row["orders"] > 0 else 0.0
            marketplace_perf.append({
                "marketplace": str(mp),
                "orders": int(row["orders"]),
                "revenue": float(row["revenue"]),
                "aov": float(aov_mp)
            })
    channelPaymentMetrics["marketplacePerf"] = {
        "success": True,
        "data": marketplace_perf
    }
    
    courier_perf = []
    if "courier" in df.columns:
        courier_metrics = df.groupby("courier", dropna=False).agg({
            "order_id": "nunique",
            "total_amount": "sum",
            units_col: "sum" if units_col else "size"
        }).rename(columns={"order_id": "orders", "total_amount": "revenue"})
        courier_metrics = courier_metrics.sort_values("revenue", ascending=False)
        
        for courier, row in courier_metrics.iterrows():
            aov_courier = float(row["revenue"] / row["orders"]) if row["orders"] > 0 else 0.0
            courier_perf.append({
                "courier": str(courier),
                "orders": int(row["orders"]),
                "revenue": float(row["revenue"]),
                "aov": float(aov_courier)
            })
    channelPaymentMetrics["courierPerf"] = {
        "success": True,
        "data": courier_perf
    }
    
    warehouse_eff = []
    if "import_warehouse_name" in df.columns:
        wh_metrics = df.groupby("import_warehouse_name", dropna=False).agg({
            "order_id": "nunique",
            "total_amount": "sum",
            units_col: "sum" if units_col else "size"
        }).rename(columns={"order_id": "orders", "total_amount": "revenue"})
        wh_metrics = wh_metrics.sort_values("revenue", ascending=False)
        
        for wh, row in wh_metrics.iterrows():
            aov_wh = float(row["revenue"] / row["orders"]) if row["orders"] > 0 else 0.0
            warehouse_eff.append({
                "warehouse": str(wh),
                "orders": int(row["orders"]),
                "revenue": float(row["revenue"]),
                "aov": float(aov_wh)
            })
    channelPaymentMetrics["warehouseEff"] = {
        "success": True,
        "data": warehouse_eff
    }
    
    payment_breakdown = {}
    if "payment_mode" in df.columns:
        payment_dist = df["payment_mode"].value_counts()
        total_payment = payment_dist.sum()
        for mode, count in payment_dist.items():
            pct = round((count / total_payment * 100), 2) if total_payment > 0 else 0.0
            payment_breakdown[str(mode)] = float(pct)
    channelPaymentMetrics["paymentModeBreakdown"] = {
        "success": True,
        "data": payment_breakdown
    }
    
    # === ORDER TYPE METRICS ===
    orderTypeMetrics = {}
    
    b2b_b2c = {}
    if "order_type" in df.columns:
        type_dist = df["order_type"].value_counts()
        total_type = type_dist.sum()
        for otype, count in type_dist.items():
            pct = round((count / total_type * 100), 2) if total_type > 0 else 0.0
            b2b_b2c[str(otype)] = float(pct)
    orderTypeMetrics["b2bVsB2c"] = {
        "success": True,
        "data": b2b_b2c
    }
    
    # === QUALITY & RISK METRICS ===
    qualityRiskMetrics = {}
    
    overall_fulfillment = 0.0
    if "order_status" in df.columns:
        total_items = len(df)
        fulfilled = len(df[df["order_status"].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.DELIVERED, OrderStatus.RETURNED])])
        overall_fulfillment = round((fulfilled / total_items * 100), 2) if total_items > 0 else 0.0
    qualityRiskMetrics["overallFulfillment"] = {
        "success": True,
        "data": overall_fulfillment
    }
    
    overall_issue_rate = 0.0
    if "order_status" in df.columns:
        total_items = len(df)
        issues = len(df[df["order_status"].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.CANCELLED, OrderStatus.RETURNED])])
        overall_issue_rate = round((issues / total_items * 100), 2) if total_items > 0 else 0.0
    qualityRiskMetrics["overallIssueRate"] = {
        "success": True,
        "data": overall_issue_rate
    }
    
    qualityRiskMetrics["paymentRiskAnalysis"] = {
        "success": True,
        "data": {"low_risk": 95.0, "medium_risk": 4.0, "high_risk": 1.0}
    }
    
    marketplace_risk = []
    if "marketplace" in df.columns:
        mp_risk = df.groupby("marketplace", dropna=False).apply(
            lambda x: {
                "total": int(x["order_id"].nunique()),
                "issues": int(_status_order_count(x, {"Cancelled", "RTO"}))
            }
        ).reset_index()
        mp_risk.columns = ["marketplace", "metrics"]
        
        for _, row in mp_risk.iterrows():
            risk_score = round((row["metrics"]["issues"] / row["metrics"]["total"] * 100), 2) if row["metrics"]["total"] > 0 else 0.0
            marketplace_risk.append({
                "marketplace": str(row["marketplace"]),
                "risk_score": float(risk_score),
                "total_orders": int(row["metrics"]["total"])
            })
    qualityRiskMetrics["marketplaceRiskScore"] = {
        "success": True,
        "data": marketplace_risk
    }
    
    # === ADVANCED METRICS ===
    advancedMetrics = {}
    
    revenue_per_channel = {}
    if "marketplace" in df.columns:
        channel_revenue = df.groupby("marketplace", dropna=False)["total_amount"].sum()
        for channel, rev in channel_revenue.items():
            revenue_per_channel[str(channel)] = float(rev)
    advancedMetrics["revenuePerChannel"] = {
        "success": True,
        "data": revenue_per_channel
    }
    
    seasonal_trends = {}
    if "order_date" in df.columns:
        df_copy = df.copy()
        df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], errors="coerce")
        df_copy["month"] = df_copy["order_date"].dt.to_period("M")
        monthly_data = df_copy.groupby("month", dropna=False).agg({
            "order_id": "nunique",
            "total_amount": "sum"
        }).rename(columns={"order_id": "orders", "total_amount": "revenue"})
        
        for month, row in monthly_data.iterrows():
            seasonal_trends[str(month)] = {
                "orders": int(row["orders"]),
                "revenue": float(row["revenue"])
            }
    advancedMetrics["seasonalTrends"] = {
        "success": True,
        "data": seasonal_trends
    }
    
    advancedMetrics["productPaymentCorr"] = {
        "success": True,
        "data": {}
    }
    
    return {
        "primaryKpis": primaryKpis,
        "productMetrics": productMetrics,
        "performanceMetrics": performanceMetrics,
        "geographicMetrics": geographicMetrics,
        "channelPaymentMetrics": channelPaymentMetrics,
        "orderTypeMetrics": orderTypeMetrics,
        "qualityRiskMetrics": qualityRiskMetrics,
        "advancedMetrics": advancedMetrics
    }


def build_preset_window(preset: str) -> Tuple[str, str]:
    """Build date window for a preset."""
    now = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    
    if preset == PRESET_7D:
        start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif preset == PRESET_30D:
        start = (now - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif preset == PRESET_ALL:
        try:
            start = datetime.fromisoformat(DEFAULT_ALL_TIME_START)
        except:
            start = datetime(2025, 9, 1, 0, 0, 0)
    else:
        raise ValueError(f"Unsupported preset: {preset}")
    
    return (
        start.isoformat(sep=" ", timespec="seconds"),
        now.isoformat(sep=" ", timespec="seconds")
    )


def generate_metrics_for_date(
    execution_date: date,
    dynamodb_client,
    ddb_table: str,
) -> Dict[str, Any]:
    """Generate all preset metrics for a given execution date."""
    print(f"\n=== Generating metrics for {execution_date} ===")
    
    result = {
        "_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "_execution_date": execution_date.isoformat(),
        "_fallback_date": False
    }
    
    for preset in SUPPORTED_PRESETS:
        print(f"\nProcessing preset: {preset}")
        start_date, end_date = build_preset_window(preset)
        print(f"Date range: {start_date} to {end_date}")
        
        try:
            df = fetch_historical_orders_from_dynamodb(
                dynamodb_client,
                ddb_table,
                start_date,
                end_date
            )
            
            if df.empty:
                print(f"No data for preset {preset}")
                result[preset] = {
                    "success": True,
                    "data": {
                        "primaryKpis": {},
                        "productMetrics": {},
                        "performanceMetrics": {},
                        "geographicMetrics": {},
                        "channelPaymentMetrics": {},
                        "orderTypeMetrics": {},
                        "qualityRiskMetrics": {},
                        "advancedMetrics": {}
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                }
            else:
                metrics = calculate_metrics_for_preset(df)
                result[preset] = {
                    "success": True,
                    "data": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                }
        
        except Exception as e:
            print(f"Error processing preset {preset}: {str(e)}")
            result[preset] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
    
    return result


def upload_metrics_to_s3(
    s3_client,
    bucket_name: str,
    prefix: str,
    execution_date: date,
    metrics_payload: Dict[str, Any],
) -> str:
    """Upload metrics payload to S3."""
    folder = execution_date.strftime("%Y-%m-%d")
    key = f"{prefix}/{folder}/all.json"
    
    body = json.dumps(convert_numpy_types(metrics_payload), ensure_ascii=True, indent=2)
    
    print(f"Uploading to s3://{bucket_name}/{key}")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body,
        ContentType="application/json"
    )
    print(f"✅ Uploaded successfully to s3://{bucket_name}/{key}")
    return key


def fetch_previous_day_metrics(
    s3_client,
    bucket_name: str,
    prefix: str,
    execution_date: date,
) -> Optional[Dict[str, Any]]:
    """Fetch metrics from previous day as fallback."""
    previous_day = execution_date - timedelta(days=1)
    folder = previous_day.strftime("%Y-%m-%d")
    key = f"{prefix}/{folder}/all.json"
    
    try:
        print(f"Fetching fallback metrics from s3://{bucket_name}/{key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read().decode("utf-8")
        fallback_metrics = json.loads(content)
        fallback_metrics["_fallback_date"] = True
        fallback_metrics["_fallback_from_date"] = previous_day.isoformat()
        print(f"✅ Loaded fallback metrics from {previous_day}")
        return fallback_metrics
    except Exception as e:
        print(f"⚠️  Could not load fallback metrics: {str(e)}")
        return None


def run_generation(
    execution_date: date,
    bucket_name: str,
    prefix: str,
    aws_region: str,
    ddb_table: str,
) -> None:
    """Main execution flow."""
    start_time = time.time()
    print(f"Starting metrics generation for {execution_date}")
    print(f"DynamoDB table: {ddb_table}")
    print(f"S3 bucket: {bucket_name}")
    print(f"S3 prefix: {prefix}")
    
    dynamodb_client = create_dynamodb_client(region_name=aws_region)
    s3_client = create_s3_client(region_name=aws_region)
    
    try:
        # Generate metrics
        metrics_payload = generate_metrics_for_date(
            execution_date,
            dynamodb_client,
            ddb_table,
        )
        
        # Check if all presets succeeded
        all_successful = all(
            metrics_payload.get(preset, {}).get("success", False)
            for preset in SUPPORTED_PRESETS
        )
        
        if not all_successful:
            print("⚠️  Some presets failed, attempting fallback")
            fallback = fetch_previous_day_metrics(s3_client, bucket_name, prefix, execution_date)
            if fallback:
                metrics_payload = fallback
        
        # Upload to S3
        key = upload_metrics_to_s3(
            s3_client,
            bucket_name,
            prefix,
            execution_date,
            metrics_payload,
        )
        
        duration = time.time() - start_time
        print(f"\n✅ Metrics generation complete in {duration:.2f} seconds")
        print(f"Output: s3://{bucket_name}/{key}")
    
    except Exception as e:
        print(f"❌ Fatal error during metrics generation: {str(e)}")
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate pre-calculated metrics presets and upload to S3"
    )
    parser.add_argument(
        "--execution-date",
        required=True,
        help="Execution date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("METRICS_PRESETS_BUCKET", DEFAULT_BUCKET),
        help=f"S3 bucket (default: {DEFAULT_BUCKET})"
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("METRICS_PRESETS_PREFIX", DEFAULT_PREFIX),
        help=f"S3 prefix (default: {DEFAULT_PREFIX})"
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
        help=f"AWS region (default: {DEFAULT_AWS_REGION})"
    )
    parser.add_argument(
        "--ddb-table",
        default=os.getenv("HISTORY_ORDERS_DYNAMODB_TABLE", DEFAULT_DYNAMODB_TABLE),
        help=f"DynamoDB table (default: {DEFAULT_DYNAMODB_TABLE})"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    
    execution_date = parse_date(args.execution_date)
    
    run_generation(
        execution_date=execution_date,
        bucket_name=args.bucket,
        prefix=args.prefix,
        aws_region=args.aws_region,
        ddb_table=args.ddb_table,
    )


if __name__ == "__main__":
    main()
