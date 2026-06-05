"""Generate pre-calculated RTO dashboard and forecast demand presets, upload to S3.

Runs daily via systemd timer. Calls route functions directly (no HTTP server needed).
Outputs JSON files to S3 for frontend consumption.

S3 output:
- Bucket: chupps-data-portal (configurable via METRICS_PRESETS_BUCKET)
- RTO:    rto-presets/YYYY-MM-DD/all.json
- Forecast: forecast-presets/YYYY-MM-DD/all.json

RTO schema:
{
    "_execution_date": "2026-06-05",
    "yesterday": { ...rto_dashboard response... },
    "7d": { ... },
    "30d": { ... },
    "all": { ... }
}

Forecast schema:
{
    "_execution_date": "2026-06-05",
    "7d_daily_1m": { ...forecast_demand response... },
    "7d_daily_2m": { ... },
    ...
    "all_daily_4m": { ... }
}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BUCKET = "chupps-data-portal"
DEFAULT_RTO_PREFIX = "rto-presets"
DEFAULT_FORECAST_PREFIX = "forecast-presets"
DEFAULT_AWS_REGION = "ap-south-1"
DEFAULT_ALL_TIME_START = os.getenv("HISTORY_CACHE_ALL_TIME_START", "2025-09-01")


def create_s3_client(region_name: str):
    return boto3.client("s3", region_name=region_name)


def convert_numpy_types(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def build_preset_windows(today: date):
    today_str = today.strftime("%Y-%m-%d")
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    d7_ago = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    d30_ago = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    all_start = DEFAULT_ALL_TIME_START.split(" ")[0]

    return {
        "yesterday": (yesterday, yesterday),
        "7d": (d7_ago, today_str),
        "30d": (d30_ago, today_str),
        "all": (all_start, today_str),
    }


def generate_rto_presets(today: date) -> dict:
    from models import DateRangeOrdersRequest
    from routes.cancellation import rto_dashboard

    windows = build_preset_windows(today)
    result = {
        "_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "_execution_date": today.isoformat(),
    }

    for label, (start_date, end_date) in windows.items():
        print(f"  RTO preset '{label}': {start_date} to {end_date}")
        try:
            req = DateRangeOrdersRequest(start_date=start_date, end_date=end_date)
            payload = rto_dashboard(req)
            result[label] = payload
        except Exception as e:
            print(f"  ⚠️  RTO preset '{label}' failed: {e}")
            result[label] = {"success": False, "error": str(e)}

    return result


def generate_forecast_presets(today: date) -> dict:
    from routes.forecast import ForecastRequest, forecast_demand

    windows = build_preset_windows(today)
    result = {
        "_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "_execution_date": today.isoformat(),
    }

    for label, (start_date, end_date) in windows.items():
        for months in [1, 2, 3, 4]:
            key = f"{label}_daily_{months}m"
            print(f"  Forecast preset '{key}': {start_date} to {end_date}, {months}M daily")
            try:
                req = ForecastRequest(
                    start_date=start_date,
                    end_date=end_date,
                    forecast_months=months,
                    granularity="daily",
                )
                payload = forecast_demand(req)
                result[key] = payload
            except Exception as e:
                print(f"  ⚠️  Forecast preset '{key}' failed: {e}")
                result[key] = {"success": False, "error": str(e)}

    return result


def upload_to_s3(s3_client, bucket: str, prefix: str, execution_date: date, payload: dict) -> str:
    folder = execution_date.strftime("%Y-%m-%d")
    key = f"{prefix}/{folder}/all.json"
    body = json.dumps(convert_numpy_types(payload), ensure_ascii=True, indent=2)

    print(f"  Uploading to s3://{bucket}/{key}")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    print(f"  ✅ Uploaded s3://{bucket}/{key}")
    return key


def run_generation(execution_date: date, bucket: str, rto_prefix: str, forecast_prefix: str, aws_region: str):
    start_time = time.time()
    print(f"=== RTO + Forecast preset generation for {execution_date} ===\n")

    s3_client = create_s3_client(region_name=aws_region)

    print("Generating RTO presets...")
    rto_payload = generate_rto_presets(execution_date)

    print("\nGenerating Forecast presets...")
    forecast_payload = generate_forecast_presets(execution_date)

    print("\nUploading to S3...")
    rto_key = upload_to_s3(s3_client, bucket, rto_prefix, execution_date, rto_payload)
    forecast_key = upload_to_s3(s3_client, bucket, forecast_prefix, execution_date, forecast_payload)

    duration = time.time() - start_time
    print(f"\n✅ Complete in {duration:.1f}s")
    print(f"   RTO:      s3://{bucket}/{rto_key}")
    print(f"   Forecast: s3://{bucket}/{forecast_key}")


def main():
    parser = argparse.ArgumentParser(description="Generate RTO + Forecast presets and upload to S3")
    parser.add_argument("--execution-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--bucket", default=os.getenv("METRICS_PRESETS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--rto-prefix", default=os.getenv("RTO_PRESETS_PREFIX", DEFAULT_RTO_PREFIX))
    parser.add_argument("--forecast-prefix", default=os.getenv("FORECAST_PRESETS_PREFIX", DEFAULT_FORECAST_PREFIX))
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", DEFAULT_AWS_REGION))
    args = parser.parse_args()

    execution_date = datetime.strptime(args.execution_date, "%Y-%m-%d").date()

    run_generation(
        execution_date=execution_date,
        bucket=args.bucket,
        rto_prefix=args.rto_prefix,
        forecast_prefix=args.forecast_prefix,
        aws_region=args.aws_region,
    )


if __name__ == "__main__":
    main()
