"""
Inventory Intelligence Routes
FastAPI endpoints for inventory snapshot fetching, metrics, and forecasting.
"""
import os
import logging
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from inventory_tools import (
    get_inventory_snapshot,
    get_stock_health,
    get_damage_rate,
    get_dead_stock,
    get_dead_score,
    get_overstock_risk,
    get_understock_risk,
    get_qc_performance,
    get_expiry_risk,
    get_channel_distribution,
    get_category_breakdown,
    get_brand_breakdown,
    get_location_breakdown,
    get_inventory_summary,
    apply_inventory_filters,
    _ensure_inventory_df,
)
from inventory_delta import (
    load_snapshots_from_s3,
    compute_weekly_timeseries,
    prepare_prophet_dataframe,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory cache for inventory data (per-session)
_inventory_cache: Dict[str, Any] = {}

# S3 storage for weekly inventory snapshots
SNAPSHOT_S3_BUCKET = os.getenv("INVENTORY_SNAPSHOT_S3_BUCKET", "chupps-data-portal")
SNAPSHOT_S3_PREFIX = os.getenv("INVENTORY_SNAPSHOT_S3_PREFIX", "inventory-snapshots")


def _get_cache_key(start_date: str, end_date: str) -> str:
    return f"inventory_{start_date}_{end_date}"


class InventorySnapshotRequest(BaseModel):
    start_date: str = Field(..., description="Start date YYYY-MM-DD HH:MM:SS")
    end_date: str = Field(..., description="End date YYYY-MM-DD HH:MM:SS")


class InventoryFilterRequest(BaseModel):
    filters: List[Dict[str, Any]]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class InventoryForecastRequest(BaseModel):
    sku: Optional[str] = None
    forecast_months: int = Field(default=3, ge=1, le=12)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class InventoryForecastCompareRequest(BaseModel):
    sku: Optional[str] = None
    forecast_months: int = Field(default=3, ge=1, le=12)
    methods: List[str] = Field(
        default=["prophet", "naive", "moving_avg", "exp_smoothing", "holt", "croston"],
        description="Forecasting methods to run",
    )


@router.get("/inventory/snapshot")
def fetch_inventory_snapshot(
    start_date: str = Query(..., description="Start date YYYY-MM-DD HH:MM:SS"),
    end_date: str = Query(..., description="End date YYYY-MM-DD HH:MM:SS"),
):
    """
    Fetch inventory snapshot CSV from EasyEcom API.
    Returns parsed data + summary metrics.
    """
    try:
        cache_key = _get_cache_key(start_date, end_date)

        # Check cache
        if cache_key in _inventory_cache:
            cached = _inventory_cache[cache_key]
            df = cached["data"]
        else:
            df = get_inventory_snapshot(start_date, end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        # Compute metrics
        summary = get_inventory_summary(df)
        records = df.to_dict(orient='records')

        # Convert numpy types for JSON serialization
        from utils.type_converters import convert_numpy_types
        records = convert_numpy_types(records)

        return {
            "success": True,
            "total_skus": len(df),
            "date_range": {"start": start_date, "end": end_date},
            "summary": summary,
            "data": records,
        }

    except Exception as e:
        logger.error(f"Inventory snapshot error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch inventory: {str(e)}")


@router.post("/inventory/snapshot")
def fetch_inventory_snapshot_post(request: InventorySnapshotRequest):
    """POST variant for inventory snapshot fetch."""
    return fetch_inventory_snapshot(request.start_date, request.end_date)


@router.get("/inventory/summary")
def inventory_summary(
    start_date: str = Query(..., description="Start date YYYY-MM-DD HH:MM:SS"),
    end_date: str = Query(..., description="End date YYYY-MM-DD HH:MM:SS"),
):
    """Get inventory summary metrics only (lighter payload)."""
    try:
        cache_key = _get_cache_key(start_date, end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(start_date, end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        summary = get_inventory_summary(df)
        return {"success": True, "summary": summary}

    except Exception as e:
        logger.error(f"Inventory summary error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.post("/inventory/metrics/stock-health")
def inventory_stock_health(request: InventorySnapshotRequest):
    """Get stock health breakdown."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_stock_health(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/damage")
def inventory_damage_rate(request: InventorySnapshotRequest):
    """Get damage and loss rates."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_damage_rate(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/dead-stock")
def inventory_dead_stock(request: InventorySnapshotRequest):
    """Get dead stock analysis."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_dead_stock(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/dead-score")
def inventory_dead_score(request: InventorySnapshotRequest):
    """Get dead score analysis: top 10 SKUs."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_dead_score(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/qc")
def inventory_qc_performance(request: InventorySnapshotRequest):
    """Get QC performance metrics."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_qc_performance(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/expiry")
def inventory_expiry_risk(request: InventorySnapshotRequest):
    """Get expiry risk analysis."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_expiry_risk(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/channels")
def inventory_channels(request: InventorySnapshotRequest):
    """Get channel distribution metrics."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_channel_distribution(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/metrics/categories")
def inventory_categories(request: InventorySnapshotRequest):
    """Get category breakdown."""
    try:
        cache_key = _get_cache_key(request.start_date, request.end_date)
        if cache_key in _inventory_cache:
            df = _inventory_cache[cache_key]["data"]
        else:
            df = get_inventory_snapshot(request.start_date, request.end_date)
            _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}

        return {"success": True, "metrics": get_category_breakdown(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/filter")
def inventory_filter(request: InventoryFilterRequest):
    """Apply filters to inventory data."""
    try:
        if request.start_date and request.end_date:
            cache_key = _get_cache_key(request.start_date, request.end_date)
            if cache_key in _inventory_cache:
                df = _inventory_cache[cache_key]["data"]
            else:
                df = get_inventory_snapshot(request.start_date, request.end_date)
                _inventory_cache[cache_key] = {"data": df, "timestamp": datetime.now()}
        else:
            # Try to use any cached data
            if _inventory_cache:
                latest_key = list(_inventory_cache.keys())[-1]
                df = _inventory_cache[latest_key]["data"]
            else:
                raise HTTPException(status_code=400, detail="No inventory data loaded. Provide start_date and end_date.")

        filtered = apply_inventory_filters(df, request.filters)
        from utils.type_converters import convert_numpy_types
        records = convert_numpy_types(filtered.to_dict(orient='records'))

        return {
            "success": True,
            "total_records": len(filtered),
            "data": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inventory/sku-list")
def inventory_sku_list():
    """
    Get forecastable SKUs from pre-computed JSON on S3.
    Only SKUs present in >=2 snapshots are listed.
    """
    try:
        import boto3
        import json as _json

        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "ap-south-1"))
        key = f"{SNAPSHOT_S3_PREFIX}/_forecastable-skus.json"

        response = s3.get_object(Bucket=SNAPSHOT_S3_BUCKET, Key=key)
        data = _json.loads(response["Body"].read().decode("utf-8"))

        return {
            "success": True,
            "skus": data.get("skus", []),
            "count": data.get("count", 0),
            "snapshot_count": data.get("snapshot_count", 0),
            "snapshot_range": data.get("snapshot_range", {}),
        }

    except Exception as e:
        logger.warning(f"Could not load forecastable SKUs JSON: {e}")
        return {"success": True, "skus": [], "count": 0, "error": str(e)}


@router.post("/inventory/forecast")
def inventory_forecast(request: InventoryForecastRequest):
    """
    Generate weekly inventory forecast using Prophet.

    Loads historical Wednesday snapshots from S3, computes deltas
    between consecutive weeks, and forecasts future stock levels.

    If SKU is provided, forecasts for that specific SKU.
    Otherwise, forecasts aggregate inventory levels.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="prophet package not installed. Run: pip install prophet"
        )

    try:
        # Load weekly snapshots from S3
        snapshots = {}
        try:
            snapshots = load_snapshots_from_s3(SNAPSHOT_S3_BUCKET, SNAPSHOT_S3_PREFIX)
            logger.info(f"Loaded {len(snapshots)} weekly snapshots from S3")
        except Exception as e:
            logger.warning(f"Could not load snapshots from S3: {e}")

        if len(snapshots) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 weekly snapshots for forecasting. Found {len(snapshots)}. Run snapshot_collector.py to backfill.",
            )

        # Build Prophet dataframe from real weekly deltas
        target = request.sku if request.sku else "aggregate"
        logger.info(f"Forecast request: target={target}, snapshots={len(snapshots)}")

        # Debug: check if SKU exists in snapshots
        if target != "aggregate":
            sku_dates = []
            for d, df in sorted(snapshots.items()):
                if "sku" in df.columns:
                    match = df[df["sku"] == target]
                    if not match.empty:
                        sku_dates.append(d)
            logger.info(f"SKU '{target}' found in {len(sku_dates)} snapshots: {sku_dates}")

        prophet_df = prepare_prophet_dataframe(snapshots, target=target, extra_regressors=True)

        logger.info(f"Prophet DataFrame: {len(prophet_df)} rows, columns={list(prophet_df.columns)}")
        if not prophet_df.empty:
            logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

        if prophet_df.empty or len(prophet_df) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data points after delta computation")

        # --- Prophet model (weekly frequency) ---
        has_extra_regressors = "inflow" in prophet_df.columns

        # Use logistic growth to prevent negative forecasts
        prophet_df["floor"] = 0
        max_stock = float(prophet_df["y"].max()) * 1.5
        prophet_df["cap"] = max_stock

        model = Prophet(
            growth="logistic",
            yearly_seasonality=False,  # Need >=1 year of data for this
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.01,  # Stiffer trend, less overfitting
            seasonality_prior_scale=10.0,
            interval_width=0.95,
        )

        if has_extra_regressors:
            for regressor in ["inflow", "outflow", "loss"]:
                if regressor in prophet_df.columns:
                    model.add_regressor(regressor, prior_scale=0.1)

        model.fit(prophet_df)

        # Forecast N weeks into the future
        weeks_to_forecast = request.forecast_months * 4
        future = model.make_future_dataframe(periods=weeks_to_forecast, freq="W")
        future["floor"] = 0
        future["cap"] = max_stock

        if has_extra_regressors:
            # Fill future regressor values with recent weekly averages
            recent = prophet_df.tail(4)
            for regressor in ["inflow", "outflow", "loss"]:
                if regressor in prophet_df.columns:
                    avg_val = recent[regressor].mean() if not recent.empty else 0
                    future[regressor] = avg_val

        forecast = model.predict(future)

        # --- Build response ---
        hist_data = [
            {"date": row["ds"].strftime("%Y-%m-%d"), "inventory": int(row["y"])}
            for _, row in prophet_df.iterrows()
        ]

        forecast_data = []
        for _, row in forecast.iloc[len(prophet_df) :].iterrows():
            forecast_data.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "predicted": max(0, int(round(row["yhat"]))),
                "lower": max(0, int(round(row["yhat_lower"]))),
                "upper": int(round(row["yhat_upper"])),
            })

        # Summary stats from latest snapshot
        latest_date = max(snapshots.keys())
        latest_df = snapshots[latest_date]
        if target != "aggregate":
            latest_df = latest_df[latest_df["sku"] == target]
        total_available = int(latest_df["available_qty"].sum()) if "available_qty" in latest_df.columns else 0
        total_marketplace = int(latest_df["marketplace_available"].sum()) if "marketplace_available" in latest_df.columns else 0
        total_received = int(latest_df["received"].sum()) if "received" in latest_df.columns else total_available

        avg_weekly_stock = round(prophet_df["y"].mean(), 0) if not prophet_df.empty else 0
        avg_weekly_inflow = round(prophet_df["inflow"].mean(), 0) if "inflow" in prophet_df.columns else 0
        avg_weekly_outflow = round(prophet_df["outflow"].mean(), 0) if "outflow" in prophet_df.columns else 0

        return {
            "success": True,
            "sku": request.sku or "ALL",
            "forecast_months": request.forecast_months,
            "granularity": "weekly",
            "data_source": "snapshots",
            "snapshots_used": len(snapshots),
            "summary": {
                "current_inventory": total_available,
                "marketplace_available": total_marketplace,
                "total_received": total_received,
                "avg_weekly_stock": avg_weekly_stock,
                "avg_weekly_inflow": avg_weekly_inflow,
                "avg_weekly_outflow": avg_weekly_outflow,
                "weeks_of_stock_remaining": round(total_available / avg_weekly_outflow, 1) if avg_weekly_outflow > 0 else None,
            },
            "historical": hist_data,
            "forecast": forecast_data,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inventory forecast error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@router.post("/inventory/forecast/compare")
def inventory_forecast_compare(request: InventoryForecastCompareRequest):
    """
    Run multiple forecasting methods on the same inventory data.
    Returns results from Prophet + simpler methods (naive, moving avg, etc.)
    for side-by-side comparison.
    """
    try:
        from forecasting_methods import run_forecast_comparison
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="forecasting_methods module not found"
        )

    try:
        # Load weekly snapshots from S3
        snapshots = {}
        try:
            snapshots = load_snapshots_from_s3(SNAPSHOT_S3_BUCKET, SNAPSHOT_S3_PREFIX)
            logger.info(f"Loaded {len(snapshots)} weekly snapshots from S3")
        except Exception as e:
            logger.warning(f"Could not load snapshots from S3: {e}")

        if len(snapshots) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 weekly snapshots. Found {len(snapshots)}.",
            )

        target = request.sku if request.sku else "aggregate"
        logger.info(f"Forecast compare request: target={target}, methods={request.methods}")

        # Build base dataframe from real weekly data
        prophet_df = prepare_prophet_dataframe(snapshots, target=target, extra_regressors=True)
        if prophet_df.empty or len(prophet_df) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data points")

        # Summary stats from latest snapshot
        latest_date = max(snapshots.keys())
        latest_df = snapshots[latest_date]
        if target != "aggregate":
            latest_df = latest_df[latest_df["sku"] == target]
        total_available = int(latest_df["available_qty"].sum()) if "available_qty" in latest_df.columns else 0
        total_marketplace = int(latest_df["marketplace_available"].sum()) if "marketplace_available" in latest_df.columns else 0
        total_received = int(latest_df["received"].sum()) if "received" in latest_df.columns else total_available

        avg_weekly_stock = round(prophet_df["y"].mean(), 0) if not prophet_df.empty else 0
        avg_weekly_inflow = round(prophet_df["inflow"].mean(), 0) if "inflow" in prophet_df.columns else 0
        avg_weekly_outflow = round(prophet_df["outflow"].mean(), 0) if "outflow" in prophet_df.columns else 0

        summary = {
            "current_inventory": total_available,
            "marketplace_available": total_marketplace,
            "total_received": total_received,
            "avg_weekly_stock": avg_weekly_stock,
            "avg_weekly_inflow": avg_weekly_inflow,
            "avg_weekly_outflow": avg_weekly_outflow,
            "weeks_of_stock_remaining": round(total_available / avg_weekly_outflow, 1) if avg_weekly_outflow > 0 else None,
        }

        # Historical data (shared across all methods)
        historical = [
            {"date": row["ds"].strftime("%Y-%m-%d"), "inventory": int(row["y"])}
            for _, row in prophet_df.iterrows()
        ]

        # Run Prophet separately (needs its own setup)
        results = {}
        if "prophet" in request.methods:
            try:
                from prophet import Prophet as ProphetModel

                prophet_df_cap = prophet_df.copy()
                prophet_df_cap["floor"] = 0
                max_stock = float(prophet_df_cap["y"].max()) * 1.5
                prophet_df_cap["cap"] = max_stock

                model = ProphetModel(
                    growth="logistic",
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.01,
                    interval_width=0.95,
                )

                has_regressors = "inflow" in prophet_df_cap.columns
                if has_regressors:
                    for reg in ["inflow", "outflow", "loss"]:
                        if reg in prophet_df_cap.columns:
                            model.add_regressor(reg, prior_scale=0.1)

                model.fit(prophet_df_cap)

                weeks = request.forecast_months * 4
                future = model.make_future_dataframe(periods=weeks, freq="W")
                future["floor"] = 0
                future["cap"] = max_stock

                if has_regressors:
                    recent = prophet_df_cap.tail(4)
                    for reg in ["inflow", "outflow", "loss"]:
                        if reg in prophet_df_cap.columns:
                            future[reg] = float(recent[reg].mean())

                fc = model.predict(future)
                fc_future = fc.iloc[len(prophet_df_cap):]

                results["prophet"] = {
                    "method": "Prophet",
                    "description": "Facebook Prophet with logistic growth. Best for data with clear seasonality patterns.",
                    "forecast": [
                        {
                            "date": row["ds"].strftime("%Y-%m-%d"),
                            "predicted": max(0, int(round(row["yhat"]))),
                            "lower": max(0, int(round(row["yhat_lower"]))),
                            "upper": int(round(row["yhat_upper"])),
                        }
                        for _, row in fc_future.iterrows()
                    ],
                }
            except Exception as e:
                logger.error(f"Prophet failed: {e}")
                results["prophet"] = {
                    "method": "Prophet",
                    "description": "Facebook Prophet",
                    "error": str(e),
                    "forecast": [],
                }

        # Run simpler methods
        simple_methods = [m for m in request.methods if m != "prophet"]
        if simple_methods:
            simple_results = run_forecast_comparison(prophet_df, simple_methods, request.forecast_months)
            results.update(simple_results)

        return {
            "success": True,
            "sku": request.sku or "ALL",
            "forecast_months": request.forecast_months,
            "granularity": "weekly",
            "data_source": "snapshots",
            "snapshots_used": len(snapshots),
            "summary": summary,
            "historical": historical,
            "methods": results,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast compare error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast compare failed: {str(e)}")
