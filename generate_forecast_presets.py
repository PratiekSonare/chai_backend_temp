# backend/generate_rto_forecast_presets.py
# Replaced entire logic to generate forecasts and save to S3 as per user request.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv
import boto3
import decimal # For Decimal conversion
from typing import Dict, Any, List # For type hints

# Try to import Prophet, handle if not installed
try:
    from prophet import Prophet
except ImportError:
    print("Prophet package not installed. Please install it: pip install prophet")
    # Assign None if import fails. This will cause errors if Prophet is used.
    Prophet = None

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# S3 Configuration
DEFAULT_BUCKET = os.getenv("METRICS_PRESETS_BUCKET", "chupps-data-portal")
DEFAULT_FORECAST_PREFIX = os.getenv("FORECAST_PRESETS_PREFIX", "forecast-presets") # As per instructions
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# DynamoDB Configuration
# These should be set in environment variables or passed as arguments.
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "history-orders-final") # Placeholder
FORECAST_COLUMNS = os.getenv("FORECAST_COLUMNS", "order_date").split(",") # Placeholder

# Default start date for historical data fetching
DEFAULT_ALL_TIME_START = os.getenv("HISTORY_CACHE_ALL_TIME_START", "2025-09-01")

# --- Mock objects/clients for standalone script execution ---

class ForecastRequest:
    """Mock class to mimic Pydantic model for forecast requests."""
    def __init__(self, start_date: str, end_date: str, forecast_months: int, granularity: str):
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_months = forecast_months
        self.granularity = granularity

# Global variable for DynamoDB resource
dynamodb_resource = None

def get_dynamodb_resource(region_name: str):
    """Initializes and returns a DynamoDB resource object."""
    global dynamodb_resource
    if dynamodb_resource is None:
        try:
            # Use boto3.resource for Table object access
            dynamodb_resource = boto3.resource("dynamodb", region_name=region_name)
            print(f"DynamoDB resource initialized for region: {region_name}")
        except Exception as e:
            print(f"Error initializing DynamoDB resource: {e}")
            dynamodb_resource = None # Ensure it's None if initialization fails
    return dynamodb_resource

# Helper function to convert DynamoDB Decimal types to native Python types
def _decimal_to_native(obj):
    """Converts Decimal types from DynamoDB to native Python types."""
    if isinstance(obj, dict):
        return {k: _decimal_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decimal_to_native(item) for item in obj]
    elif isinstance(obj, decimal.Decimal):
        # Convert Decimal to float if it's not an integer, otherwise to int
        if obj % 1 == 0:
            return int(obj)
        else:
            return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif hasattr(obj, "item"): # For objects with an 'item' method (like numpy scalars)
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj) # Fallback for unknown types

# Helper function to build DynamoDB ProjectionExpression and ExpressionAttributeNames
def _build_projection(columns: list[str]) -> tuple[str, dict]:
    """Builds ProjectionExpression and ExpressionAttributeNames for DynamoDB scan."""
    projection_parts = []
    expression_names = {}
    for i, col in enumerate(columns):
        # DynamoDB attribute names cannot start with a number or contain certain characters.
        # We use ExpressionAttributeNames to map safe names to original names.
        placeholder = f"#col{i}"
        projection_parts.append(placeholder)
        expression_names[placeholder] = col
    return ", ".join(projection_parts), expression_names

# --- Existing Utility Functions (modified/kept from original script) ---

def create_s3_client(region_name: str):
    """Creates an S3 client."""
    return boto3.client("s3", region_name=region_name)

def convert_numpy_types(obj):
    """Handles numpy types for JSON serialization. (Copied from original file)"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif hasattr(obj, "item"):
        try:
            return obj.item()
        except AttributeError: # Handle cases where .item() might not exist or is not applicable
            return obj
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def build_preset_windows(today: date):
    """
    Defines time windows for historical data.
    'all' window starts from DEFAULT_ALL_TIME_START.
    """
    today_str = today.strftime("%Y-%m-%d")
    d7_ago = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    d30_ago = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    all_start = DEFAULT_ALL_TIME_START.split(" ")[0] # Take only the date part

    return {
        "7d": (d7_ago, today_str),
        "30d": (d30_ago, today_str),
        "all": (all_start, today_str),
    }

def upload_to_s3(s3_client, bucket: str, prefix: str, execution_date: date, payload: dict, granularity: str, months: int) -> str:
    """Uploads payload to S3 with a specific key structure: prefix/YYYY-MM-DD/granularity_Xm.json."""
    folder = execution_date.strftime("%Y-%m-%d")
    key = f"{prefix}/{folder}/{granularity}_{months}m.json"
    try:
        body = json.dumps(convert_numpy_types(payload), ensure_ascii=True, indent=2)
    except Exception as e:
        print(f"  ❌ Error serializing payload to JSON for s3://{bucket}/{key}: {e}")
        return ""

    print(f"  Uploading to s3://{bucket}/{key}")
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        print(f"  ✅ Uploaded s3://{bucket}/{key}")
        return key
    except Exception as e:
        print(f"  ❌ Upload failed for s3://{bucket}/{key}: {e}")
        return ""

# --- New Core Function for Forecast Generation ---
def generate_forecast_data(
    execution_date: date,
    dynamodb_table_name: str,
    dynamodb_resource, # Expecting boto3.resource object
    forecast_columns_to_fetch: list[str],
    bucket: str,
    forecast_prefix: str,
    aws_region: str,
    forecast_periods_in_months: list[int],
    granularities: list[str]
) -> list[dict]:
    """
    Generates forecast data for specified periods and granularities, fetches from DynamoDB,
    uses Prophet for forecasting, and returns the structured forecast data.
    """
    print(f"--- Generating Forecast Data for {execution_date.isoformat()} ---")
    generation_start_time = time.time()

    if Prophet is None:
        print("Error: Prophet library is not installed. Cannot generate forecasts.")
        return []

    if dynamodb_resource is None:
        print("Error: DynamoDB resource not initialized. Cannot fetch historical data.")
        return []
    
    try:
        table = dynamodb_resource.Table(dynamodb_table_name)
    except Exception as e:
        print(f"Error accessing DynamoDB table '{dynamodb_table_name}': {e}")
        return []

    # Define historical data date range for fetching
    # The 'end_date' for historical data is the execution date itself.
    historical_end_date_str = execution_date.strftime("%Y-%m-%d")
    # Use DEFAULT_ALL_TIME_START for the historical data start
    historical_start_date_str = DEFAULT_ALL_TIME_START.split(" ")[0]

    # Fetch historical orders from DynamoDB
    print(f"Fetching historical orders from DynamoDB table '{dynamodb_table_name}' between {historical_start_date_str} and {historical_end_date_str}")
    try:
        projection, expr_names = _build_projection(forecast_columns_to_fetch)
        
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
            print(f"  ⚠️ No orders found in DynamoDB for the date range {historical_start_date_str} to {historical_end_date_str}.")
            return []

        normalized_items = [_decimal_to_native(item) for item in items]
        df_raw = pd.DataFrame(normalized_items)

        # Data validation and processing
        if "order_date" not in df_raw.columns:
            print(f"  ❌ 'order_date' column missing from DynamoDB data. Required columns: {forecast_columns_to_fetch}")
            return []

        # Ensure order_date is timezone-naive datetime
        df_raw["order_date"] = pd.to_datetime(df_raw["order_date"], errors="coerce", utc=True)
        df_raw["order_date"] = df_raw["order_date"].dt.tz_convert(None) # Convert to naive datetime
        df_raw = df_raw.dropna(subset=["order_date"])

        start_ts = pd.to_datetime(historical_start_date_str)
        end_ts = pd.to_datetime(historical_end_date_str).replace(hour=23, minute=59, second=59)
        df_historical = df_raw[(df_raw["order_date"] >= start_ts) & (df_raw["order_date"] <= end_ts)]

        if df_historical.empty:
            print(f"  ⚠️ No orders found within the filtered historical range {historical_start_date_str} to {historical_end_date_str}.")
            return []

        # Normalize date to midnight for consistent daily aggregation
        df_historical["order_date"] = df_historical["order_date"].dt.normalize()

        all_generated_results = [] # To store results for all granularity/month combinations

        # Generate forecasts for each granularity and month combination
        for granularity in granularities:
            print(f"\nProcessing granularity: '{granularity}'...")

            # Aggregate historical data based on granularity
            if granularity == "weekly":
                # Use period to group by week, then get the start time of the week
                df_historical["period"] = df_historical["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
            else: # daily
                df_historical["period"] = df_historical["order_date"]
            
            # Group by the period and count orders
            order_counts = df_historical.groupby("period").size().reset_index(name="order_count")
            order_counts = order_counts.sort_values("period").reset_index(drop=True)

            if len(order_counts) < 2:
                print(f"  ⚠️ Insufficient historical data points ({len(order_counts)}) for '{granularity}' granularity (need at least 2 periods). Skipping.")
                continue

            # Prepare DataFrame for Prophet
            prophet_df = pd.DataFrame({
                "ds": order_counts["period"],
                "y": order_counts["order_count"].astype(float),
            })

            # Generate forecasts for different month periods
            for months in forecast_periods_in_months:
                forecast_key_suffix = f"{granularity}_{months}m" # For S3 key and output
                print(f"  Generating forecast for {months} months ({granularity})...")

                try:
                    # Prophet model configuration
                    # Adjusting seasonality based on granularity
                    model = Prophet(
                        yearly_seasonality=False, # Usually not needed for monthly forecasts
                        weekly_seasonality=True if granularity == "daily" else False, # Weekly seasonality is relevant for daily data
                        daily_seasonality=False if granularity == "weekly" else True, # Daily seasonality for daily data if granular
                        seasonality_mode="additive",
                        changepoint_prior_scale=0.01,
                        seasonality_prior_scale=10.0,
                        holidays_prior_scale=10.0,
                        interval_width=0.95,
                    )

                    # Add weekly seasonality explicitly for daily granularity if not set by default
                    if granularity == "daily" and not model.weekly_seasonality: # Check if Prophet itself didn't enable it based on freq
                         model.add_seasonality(name="weekly", period=7, fourier_order=3, prior_scale=0.1)
                    
                    # Fit the model with the historical data
                    model.fit(prophet_df)

                    # Create future dataframe for prediction
                    freq = "D" if granularity == "daily" else "W"
                    periods_to_forecast = 0
                    if freq == 'D':
                        # Approximate days in N months
                        periods_to_forecast = months * 30
                    elif freq == 'W':
                        # Approximate weeks in N months
                        periods_to_forecast = months * 4
                    
                    if periods_to_forecast == 0:
                        print(f"  ⚠️ Invalid periods_to_forecast calculated for granularity '{granularity}' and months '{months}'. Skipping.")
                        continue

                    future = model.make_future_dataframe(periods=periods_to_forecast, freq=freq)

                    # Make predictions
                    forecast = model.predict(future)

                    # --- Structure the output data ---
                    hist_data = []
                    # Use prophet_df for historical data
                    for _, row in prophet_df.iterrows():
                        hist_data.append({
                            "date": row["ds"].strftime("%Y-%m-%d"),
                            "order_count": int(row["y"]),
                        })

                    # Extract forecast data from the prediction DataFrame
                    cutoff_idx = len(prophet_df)
                    forecast_data = []
                    # Ensure we only take rows that are actually in the future
                    forecast_future_rows = forecast.iloc[cutoff_idx:]

                    for _, row in forecast_future_rows.iterrows():
                        # Ensure date is formatted correctly and values are integers, non-negative
                        date_str = row["ds"].strftime("%Y-%m-%d")
                        predicted_val = max(0, int(round(row["yhat"])))
                        lower_val = max(0, int(round(row["yhat_lower"])))
                        upper_val = max(0, int(round(row["yhat_upper"]))) # Ensure upper bound is also non-negative

                        forecast_data.append({
                            "date": date_str,
                            "predicted": predicted_val,
                            "lower": lower_val,
                            "upper": upper_val,
                        })

                    # Calculate summary statistics
                    total_historical = int(prophet_df["y"].sum())
                    total_predicted = sum(f["predicted"] for f in forecast_data)
                    # Average orders per period (daily or weekly)
                    avg_period_actual = round(prophet_df["y"].mean(), 1) if not prophet_df.empty else 0

                    # Construct the full result for this specific forecast run
                    result_payload = {
                        "_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "_execution_date": execution_date.isoformat(),
                        "granularity": granularity,
                        "forecast_months": months,
                        "date_range": { # Date range of historical data used for training
                            "start": historical_start_date_str,
                            "end": historical_end_date_str,
                        },
                        "summary": {
                            "total_historical_orders": total_historical,
                            "total_forecasted_orders": total_predicted,
                            "avg_orders_per_period": avg_period_actual,
                            "data_points_used": len(prophet_df),
                            "forecast_periods_generated": len(forecast_data),
                        },
                        "historical": hist_data,
                        "forecast": forecast_data,
                    }
                    all_generated_results.append(result_payload)

                except Exception as e:
                    print(f"  ⚠️  Forecast generation for {forecast_key_suffix} failed: {e}")
                    # Optionally add an error entry to all_generated_results or log it
                    # For now, just print and continue to the next combination

        generation_duration = time.time() - generation_start_time
        print(f"\n--- Forecast data generation completed in {generation_duration:.1f}s ---")
        return all_generated_results

    except Exception as e:
        print(f"❌ An unhandled error occurred during forecast data generation: {e}")
        return []

# --- Modified run_generation function ---
def run_generation(execution_date: date, bucket: str, forecast_prefix: str, aws_region: str, dynamodb_table_name: str, forecast_columns_to_fetch: list[str]):
    """
    Orchestrates forecast data generation and uploads to S3.
    """
    print(f"=== Starting Forecast preset generation for {execution_date} ===")
    overall_start_time = time.time()

    # Define parameters for forecast generation
    forecast_periods_in_months = [1, 2, 3, 6] # As per user requirement for frontend graph
    granularities = ["daily", "weekly"]

    # Initialize DynamoDB resource
    dynamodb_res = get_dynamodb_resource(aws_region)

    # Generate all forecast data
    all_generated_forecasts = generate_forecast_data(
        execution_date=execution_date,
        dynamodb_table_name=dynamodb_table_name,
        dynamodb_resource=dynamodb_res,
        forecast_columns_to_fetch=forecast_columns_to_fetch,
        bucket=bucket,
        forecast_prefix=forecast_prefix,
        aws_region=aws_region,
        forecast_periods_in_months=forecast_periods_in_months,
        granularities=granularities,
    )

    # Upload generated forecasts to S3
    if all_generated_forecasts:
        print("\nUploading generated forecasts to S3...")
        s3_client = create_s3_client(region_name=aws_region)
        uploaded_keys = []
        for forecast_result in all_generated_forecasts:
            months = forecast_result["forecast_months"]
            granularity = forecast_result["granularity"]
            key = upload_to_s3(
                s3_client,
                bucket,
                forecast_prefix,
                execution_date,
                forecast_result,
                granularity,
                months
            )
            if key:
                uploaded_keys.append(key)
        
        if uploaded_keys:
            print(f"\n✅ All generated forecasts uploaded successfully.")
            print("   Uploaded files:")
            for key in uploaded_keys:
                print(f"   - s3://{bucket}/{key}")
        else:
            print("\n❌ No forecasts were successfully uploaded.")
    else:
        print("\nNo forecast data was generated, so nothing to upload.")

    overall_duration = time.time() - overall_start_time
    print(f"\nTotal execution time: {overall_duration:.1f}s")

# --- Modified main function ---
def main():
    parser = argparse.ArgumentParser(description="Generate forecast presets and upload to S3.")
    parser.add_argument(
        "--execution-date",
        required=False, # Made optional to default to today
        help="The date for which to generate forecasts (YYYY-MM-DD). Defaults to today if not provided.",
        default=datetime.now().strftime("%Y-%m-%d")
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("METRICS_PRESETS_BUCKET", DEFAULT_BUCKET),
        help="S3 bucket name to upload forecasts to."
    )
    parser.add_argument(
        "--forecast-prefix",
        default=os.getenv("FORECAST_PRESETS_PREFIX", DEFAULT_FORECAST_PREFIX),
        help="S3 prefix for forecast files."
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
        help="AWS region for S3 and DynamoDB."
    )
    parser.add_argument(
        "--dynamodb-table-name",
        default=os.getenv("DYNAMODB_TABLE_NAME", DYNAMODB_TABLE_NAME), # Use global default if env var not set
        help="Name of the DynamoDB table containing order data."
    )
    parser.add_argument(
        "--forecast-columns",
        default=os.getenv("FORECAST_COLUMNS", ",".join(FORECAST_COLUMNS)), # Use global default if env var not set
        help="Comma-separated list of columns to fetch from DynamoDB for forecasting (e.g., 'order_date,order_count')."
    )

    args = parser.parse_args()

    # Parse execution date
    try:
        execution_date = datetime.strptime(args.execution_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Invalid execution date format '{args.execution_date}'. Please use YYYY-MM-DD.")
        sys.exit(1)
    
    # Parse forecast columns
    forecast_columns_to_fetch = [col.strip() for col in args.forecast_columns.split(",") if col.strip()] # Ensure no empty strings

    run_generation(
        execution_date=execution_date,
        bucket=args.bucket,
        forecast_prefix=args.forecast_prefix,
        aws_region=args.aws_region,
        dynamodb_table_name=args.dynamodb_table_name,
        forecast_columns_to_fetch=forecast_columns_to_fetch,
    )

if __name__ == "__main__":
    main()
