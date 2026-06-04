import os
import logging
import pandas as pd
import numpy as np
import boto3
from decimal import Decimal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter()

DYNAMODB_REGION = os.getenv("AWS_REGION", "ap-south-1")
DYNAMODB_TABLE_NAME = os.getenv("HISTORY_ORDERS_DYNAMODB_TABLE", "history-orders-final")

try:
    dynamodb = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
except Exception as e:
    logger.warning(f"DynamoDB initialization warning: {str(e)}")
    dynamodb = None

FORECAST_COLUMNS = [
    'order_id',
    'order_date',
    'order_status',
    'item_quantity',
    'suborder_quantity',
    'order_quantity',
]

PROJECTION_ALIASES = {
    "state": "#state",
    "size": "#size",
}


def _build_projection(columns: list[str]) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    aliases: dict[str, str] = {}
    for col in columns:
        if col in PROJECTION_ALIASES:
            parts.append(PROJECTION_ALIASES[col])
            aliases[PROJECTION_ALIASES[col]] = col
        else:
            parts.append(col)
    return ", ".join(parts), aliases


def _decimal_to_native(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, dict):
        return {k: _decimal_to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_to_native(v) for v in value]
    return value


class ForecastRequest(BaseModel):
    start_date: Optional[str] = "2025-09-01"
    end_date: Optional[str] = None
    forecast_months: int = 3
    granularity: str = "weekly"


@router.post("/forecast/demand")
def forecast_demand(request: ForecastRequest):
    """
    Generate order demand forecast using Meta's Prophet model.

    Fetches historical orders from DynamoDB, aggregates by date,
    fits a Prophet model with seasonal parameters suited for footwear,
    and returns historical + forecasted order counts.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="prophet package not installed. Run: pip install prophet"
        )

    if not dynamodb:
        raise HTTPException(status_code=500, detail="DynamoDB client not initialized")

    if request.granularity not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="granularity must be 'daily' or 'weekly'")

    try:
        end_date = request.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = request.start_date or "2025-09-01"

        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        projection, expr_names = _build_projection(FORECAST_COLUMNS)

        scan_kwargs: Dict[str, Any] = {"ProjectionExpression": projection}
        if expr_names:
            scan_kwargs["ExpressionAttributeNames"] = expr_names

        items: list[dict] = []
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            response = table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"], **scan_kwargs
            )
            items.extend(response.get("Items", []))

        if not items:
            raise HTTPException(status_code=404, detail="No orders found in DynamoDB")

        normalized = [_decimal_to_native(item) for item in items]
        df = pd.DataFrame(normalized)

        if "order_date" not in df.columns:
            raise HTTPException(status_code=500, detail="order_date column missing from data")

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", utc=True)
        df["order_date"] = df["order_date"].dt.tz_convert(None)
        df = df.dropna(subset=["order_date"])

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
        df = df[(df["order_date"] >= start_ts) & (df["order_date"] <= end_ts)]

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No orders found between {start_date} and {end_date}",
            )

        df["order_date"] = df["order_date"].dt.normalize()

        if request.granularity == "weekly":
            df["period"] = df["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
        else:
            df["period"] = df["order_date"]

        order_counts = df.groupby("period").size().reset_index(name="order_count")
        order_counts = order_counts.sort_values("period").reset_index(drop=True)

        if len(order_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data points for forecasting (need at least 2 periods)",
            )

        prophet_df = pd.DataFrame({
            "ds": order_counts["period"],
            "y": order_counts["order_count"].astype(float),
        })

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=request.granularity == "daily",
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            interval_width=0.95,
        )

        if request.granularity == "daily":
            model.add_seasonality(
                name="weekly",
                period=7,
                fourier_order=3,
                prior_scale=0.1,
            )

        model.fit(prophet_df)

        last_date = prophet_df["ds"].max()
        future_date = last_date + timedelta(days=request.forecast_months * 30)
        future = model.make_future_dataframe(
            periods=(future_date - last_date).days,
            freq="D" if request.granularity == "daily" else "W",
        )

        forecast = model.predict(future)

        hist_data = []
        for _, row in prophet_df.iterrows():
            hist_data.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "order_count": int(row["y"]),
            })

        cutoff_idx = len(prophet_df)
        forecast_data = []
        for _, row in forecast.iloc[cutoff_idx:].iterrows():

                forecast_data.append({
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "predicted": max(0, int(round(row["yhat"]))) ,
                    "lower": max(0, int(round(row["yhat_lower"]))) ,
                    "upper": int(round(row["yhat_upper"])),
                })


        total_historical = int(prophet_df["y"].sum())
        total_predicted = sum(f["predicted"] for f in forecast_data)
        avg_weekly_actual = round(prophet_df["y"].mean(), 1)

        return {
            "success": True,
            "granularity": request.granularity,
            "forecast_months": request.forecast_months,
            "date_range": {
                "start": start_date,
                "end": end_date,
            },
            "model_params": {
                "yearly_seasonality": True,
                "weekly_seasonality": request.granularity == "daily",
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
                "interval_width": 0.95,
            },
            "summary": {
                "total_historical_orders": total_historical,
                "total_forecasted_orders": total_predicted,
                "avg_orders_per_period": avg_weekly_actual,
                "data_points_used": len(prophet_df),
                "forecast_periods": len(forecast_data),
            },
            "historical": hist_data,
            "forecast": forecast_data,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
