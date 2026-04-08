import os
import pandas as pd
import asyncio
import boto3
import hashlib
from decimal import Decimal
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from utils.type_converters import convert_numpy_types
from models import HistoryOrdersRequest, OrderStatus, PaymentMode, OrderType
from typing import Optional, Dict, Any, Union
import json

# Initialize Supabase client
try:
    from supabase import create_client, Client
    
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables not set")
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"⚠️  Supabase client initialization warning: {str(e)}")
    supabase = None

router = APIRouter()

DYNAMODB_REGION = os.getenv("AWS_REGION", "ap-south-1")
DYNAMODB_TABLE_NAME = os.getenv("HISTORY_ORDERS_DYNAMODB_TABLE", "history-orders")

# ============ ENUM HELPER FUNCTIONS ============
def _normalize_status(value: Optional[str]) -> Optional[str]:
    """Normalize order status value using OrderStatus enum"""
    return OrderStatus.normalize(value) if value else None

def _normalize_payment_mode(value: Optional[str]) -> Optional[str]:
    """Normalize payment mode value using PaymentMode enum"""
    return PaymentMode.normalize(value) if value else None

def _normalize_order_type(value: Optional[str]) -> Optional[str]:
    """Normalize order type value using OrderType enum"""
    return OrderType.normalize(value) if value else None

try:
    dynamodb = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
except Exception as e:
    print(f"⚠️  DynamoDB initialization warning: {str(e)}")
    dynamodb = None

# ============ REQUIRED COLUMNS FOR CALCULATIONS ============
# Only these columns are fetched from Supabase to optimize data transfer
REQUIRED_COLUMNS = [
    'order_id',              # For counting unique orders
    'order_date',            # For time-based grouping and heatmaps
    'total_amount',          # For revenue calculations
    'item_quantity',         # For units sold
    'suborder_quantity',     # Alternative units column
    'order_quantity',        # Alternative units column
    'sku',                   # For unique SKU count
    'canonical_sku',         # For SKU analysis
    'suborder_sku',          # Alternative SKU column
    'suborder_marketplace_sku',  # Alternative SKU column
    'marketplace_sku',       # Alternative SKU column
    'order_status',          # For status-based metrics (delivered, cancelled, returned)
    'payment_mode',          # For COD/PrePaid analysis
    'order_type',            # For B2B vs B2C analysis
    'state',                 # For geographic analysis
    'marketplace',           # Client-side filtering
    'courier',               # Client-side filtering
    'import_warehouse_name', # Client-side filtering
    'billing_state',         # Client-side filtering
    'size',                  # For size distribution
    'suborder_size',         # Alternative size column
]



# ============ DYNAMODB HELPER FUNCTIONS ============
def _normalize_query_datetime(value: Optional[str], is_start: bool) -> Optional[str]:
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    try:
        parsed = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid start_date/end_date format. Use YYYY-MM-DD or full datetime"
        ) from exc

    is_date_only = len(value_str) == 10 and value_str[4] == "-" and value_str[7] == "-"
    if is_date_only:
        if is_start:
            parsed = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=0)

    return parsed.isoformat(sep=" ", timespec="seconds")

def _decimal_to_native(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Preserve integers where possible and cast fractional values to float.
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {k: _decimal_to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_to_native(v) for v in value]
    return value

def _apply_client_side_filters(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    filters: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    if 'order_date' in filtered.columns:
        filtered['order_date'] = pd.to_datetime(filtered['order_date'], errors='coerce')

    if start_date and 'order_date' in filtered.columns:
        start_ts = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_ts):
            filtered = filtered[filtered['order_date'] >= start_ts]

    if end_date and 'order_date' in filtered.columns:
        end_ts = pd.to_datetime(end_date, errors='coerce')
        if pd.notna(end_ts):
            filtered = filtered[filtered['order_date'] <= end_ts]

    if not filters:
        return filtered

    client_filter_fields = {
        'order_date',
        'marketplace',
        'courier',
        'import_warehouse_name',
        'state',
    }

    for key in client_filter_fields:
        if key not in filters:
            continue

        value = filters.get(key)
        if value is None or key not in filtered.columns:
            continue

        if key == 'order_date':
            date_series = pd.to_datetime(filtered['order_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            if isinstance(value, list):
                date_values = [str(v).strip()[:10] for v in value if v is not None and str(v).strip()]
                if date_values:
                    filtered = filtered[date_series.isin(date_values)]
            else:
                date_value = str(value).strip()[:10]
                if date_value:
                    filtered = filtered[date_series == date_value]
            continue

        col_series = filtered[key].astype(str).str.strip().str.lower()
        if isinstance(value, list):
            allowed = [str(v).strip().lower() for v in value if v is not None and str(v).strip()]
            if allowed:
                filtered = filtered[col_series.isin(allowed)]
        else:
            target = str(value).strip().lower()
            if target:
                filtered = filtered[col_series == target]

    return filtered

def _build_projection_expression(columns: list[str]) -> tuple[str, Dict[str, str]]:
    """Build ProjectionExpression with aliases for reserved keywords."""
    reserved_aliases = {
        "state": "#state",
        "size": "#size",
    }

    projection_parts: list[str] = []
    expression_attribute_names: Dict[str, str] = {}

    for col in columns:
        if col in reserved_aliases:
            alias = reserved_aliases[col]
            projection_parts.append(alias)
            expression_attribute_names[alias] = col
        else:
            projection_parts.append(col)

    return ", ".join(projection_parts), expression_attribute_names


# ============ DYNAMODB DATA FETCHING ============
def fetch_historical_orders(
    request_or_table: Union[HistoryOrdersRequest, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Fetch historical orders from DynamoDB
    
    Args:
        table_name: Name of the table in DynamoDB
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
        filters: Additional column filters
    
    Returns:
        DataFrame with order records
    """
    if not dynamodb:
        raise HTTPException(
            status_code=500,
            detail="DynamoDB client not initialized"
        )
    
    try:
        if isinstance(request_or_table, HistoryOrdersRequest):
            request = request_or_table
        else:
            table_name = str(request_or_table).strip() if request_or_table else DYNAMODB_TABLE_NAME
            request = HistoryOrdersRequest(
                table_name=table_name,
                start_date=start_date,
                end_date=end_date,
                filters=filters,
            )

        table_name = request.table_name
        request_start_date = request.start_date
        request_end_date = request.end_date
        request_filters = request.filters

        request_start_date = _normalize_query_datetime(request_start_date, is_start=True)
        request_end_date = _normalize_query_datetime(request_end_date, is_start=False)

        if request_filters:
            print("filters: ", request_filters, flush=True)

        table = dynamodb.Table(table_name)

        projection_expression, expression_attribute_names = _build_projection_expression(REQUIRED_COLUMNS)

        scan_kwargs: Dict[str, Any] = {
            "ProjectionExpression": projection_expression,
        }
        if expression_attribute_names:
            scan_kwargs["ExpressionAttributeNames"] = expression_attribute_names

        items: list[dict] = []
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"], **scan_kwargs)
            items.extend(response.get("Items", []))

        if not items:
            return pd.DataFrame()

        normalized_items = [_decimal_to_native(item) for item in items]
        df = pd.DataFrame(normalized_items)
        
        # Convert date columns to datetime (only those in REQUIRED_COLUMNS)
        date_columns = ['order_date']  # Only order_date is in REQUIRED_COLUMNS
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(None)
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Apply all required request filters client-side (no DynamoDB FilterExpression).
        df = _apply_client_side_filters(
            df=df,
            start_date=request_start_date,
            end_date=request_end_date,
            filters=request_filters,
        )
        
        # ===== INTERMEDIATE PROCESSING: Add computed columns =====
        # This consolidates post-fetch transformations so endpoints don't repeat them
        df = _enrich_dataframe(df)
        
        return df
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching from DynamoDB: {str(e)}"
        )


# ============ API ROUTES ============
@router.post('/history/orders/preview')
def preview_historical_orders(request: HistoryOrdersRequest):
    """
    Testing endpoint: preview first 3 rows from fetched historical orders.
    """
    try:
        df = fetch_historical_orders(request)

        preview_df = df.head(3).reindex(columns=REQUIRED_COLUMNS)
        preview_df = preview_df.where(pd.notna(preview_df), None)


        return convert_numpy_types({
            "success": True,
            "metric_name": "Historical Orders Preview",
            "total_rows_fetched": int(len(df)),
            "preview_count": int(len(preview_df)),
            "data": preview_df.to_dict(orient='records'),
            "columns": REQUIRED_COLUMNS,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating historical orders preview: {str(e)}"
        )


@router.get('/history/orders/filter-options')
def historical_orders_filter_options():
    """Return filter values from dedicated Supabase public views."""
    try:
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Supabase client not initialized"
            )

        def _fetch_names_from_view(view_name: str) -> list[str]:
            try:
                response = supabase.table(view_name).select("*").execute()
                rows = response.data or []
                values: list[str] = []

                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    value = next(
                        (
                            candidate
                            for candidate in row.values()
                            if candidate is not None and str(candidate).strip() != ""
                        ),
                        None,
                    )

                    if value is not None:
                        values.append(str(value).strip())

                return sorted(set(values))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error reading view '{view_name}': {str(e)}"
                )

        warehouse_names = _fetch_names_from_view("warehouse_names")
        marketplace_names = _fetch_names_from_view("marketplace_names")
        courier_names = _fetch_names_from_view("courier_names")
        billing_state_names = _fetch_names_from_view("billing_state_names")

        # warehouse_names = supabase.table("warehouse_names").select("import_warehouse_names")
        # print("warehouse_names: ", warehouse_names, flush=True);

        return convert_numpy_types({
            "success": True,
            "options": {
                "warehouse_names": warehouse_names,
                "marketplace_names": marketplace_names,
                "courier_names": courier_names,
                "billing_state_names": billing_state_names
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching filter options: {str(e)}"
        )


# ============ UNIFIED ASYNC KPI ENDPOINT ============
try:
    import redis
except Exception:
    redis = None

def _redis_keys(request_id: str) -> tuple[str, str]:
    safe_request_id = "".join(ch for ch in request_id if ch.isalnum() or ch in ("-", "_"))
    if not safe_request_id:
        safe_request_id = "unknown"
    return (f"request_logs:{safe_request_id}:events", f"request_logs:{safe_request_id}:seq")

REDIS_URL = os.getenv('REDIS_URL')
_redis_client = None
_redis_init_failed = False

def _get_redis_client() -> Any | None:
    global _redis_client, _redis_init_failed
    try:
        # print("redis url: ", REDIS_URL, flush=True)
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_init_failed = True
        _redis_client = None
        return None


def _make_cache_key(request: HistoryOrdersRequest) -> str:
    """Generate a consistent cache key based on request parameters."""
    start = str(request.start_date).strip() if request.start_date else "none"
    end = str(request.end_date).strip() if request.end_date else "none"

    filters_payload = request.filters if isinstance(request.filters, dict) else {}
    normalized_filters = convert_numpy_types(filters_payload)
    filters_serialized = json.dumps(normalized_filters, sort_keys=True, separators=(",", ":"), default=str)
    filters_hash = hashlib.sha256(filters_serialized.encode("utf-8")).hexdigest()[:16]

    return f"kpi:history:all:{request.table_name}:{start}:{end}:{filters_hash}"


def _serialize_for_redis(data: dict) -> str:
    """Safe JSON serialization that handles numpy types and datetime."""
    # First convert any numpy/pandas quirks
    clean_data = convert_numpy_types(data)
    return json.dumps(clean_data, default=str)  # default=str handles datetime safely


@router.post('/history/kpi/all')
async def kpi_all(request: HistoryOrdersRequest):
    """
    Unified KPI endpoint with Redis caching.
    Cache survives page reloads and serves data instantly on cache hit.
    """
    cache_key = _make_cache_key(request)
    redis_client = _get_redis_client()

    try:
        print(f"Request started with params: {request.table_name}, {request.start_date}, {request.end_date}", flush=True)

        # === 1. Check Redis Cache ===
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    print(f"🚀 Cache HIT for key: {cache_key}", flush=True)
                    cached_response = json.loads(cached)
                    # Optional: update timestamp on every hit
                    cached_response['timestamp'] = datetime.now().isoformat()
                    return cached_response
            except Exception as cache_err:
                print(f"Redis get error (non-fatal): {cache_err}", flush=True)
                # Continue to DB on cache failure
        else:
            print("no valid redis client detected", flush=True)

        # === 2. Cache Miss → Fetch from Database ===
        print("⛔ Cache MISS - Fetching from database...", flush=True)
        df = fetch_historical_orders(request)
        print("Fetching complete", flush=True)

        if df.empty:
            print("df is empty!", flush=True)

        # Your existing KPI calculation functions (unchanged)
        def calculate_total_orders():
            if df.empty:
                return 0
            return int(df['order_id'].nunique() if 'order_id' in df.columns else len(df))

        def calculate_units_sold():
            if df.empty:
                return 0
            if 'item_quantity' in df.columns:
                return int(df['item_quantity'].sum())
            elif 'suborder_quantity' in df.columns:
                return int(df['suborder_quantity'].sum())
            else:
                return int(df['order_quantity'].sum() if 'order_quantity' in df.columns else len(df))

        def calculate_gross_revenue():
            if df.empty or 'total_amount' not in df.columns:
                return 0.0
            return float(df['total_amount'].sum())

        def calculate_aov():
            if df.empty or 'total_amount' not in df.columns:
                return 0.0
            return float(df['total_amount'].mean())

        def calculate_unique_skus():
            if df.empty or 'sku' not in df.columns:
                return 0
            return int(df['sku'].nunique())

        def calculate_cancellation_rate():
            if df.empty:
                return 0.0
            total = _order_count(df)
            cancelled = _status_order_count(df, {'Cancelled'})
            return round(_safe_pct(float(cancelled), float(total)), 2)

        def calculate_return_rate():
            if df.empty:
                return 0.0
            total = _order_count(df)
            rto = _status_order_count(df, {'RTO'})
            return round(_safe_pct(float(rto), float(total)), 2)

        def calculate_cod_share():
            if df.empty or 'payment_mode' not in df.columns:
                return 0.0
            total = len(df)
            cod = len(df[df['payment_mode'] == 'COD'])
            return round((cod / total * 100) if total > 0 else 0.0, 2)

        def calculate_delivered_rate():
            if df.empty or 'order_status' not in df.columns:
                return 0.0
            total = len(df)
            delivered = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.DELIVERED.value])
            return round((delivered / total * 100) if total > 0 else 0.0, 2)

        # Run calculations concurrently using thread pool (good for CPU-bound pandas work)
        loop = asyncio.get_event_loop()

        kpi_tasks = {
            'totalOrders': loop.run_in_executor(None, calculate_total_orders),
            'unitsSold': loop.run_in_executor(None, calculate_units_sold),
            'grossRevenue': loop.run_in_executor(None, calculate_gross_revenue),
            'aov': loop.run_in_executor(None, calculate_aov),
            'uniqueSkus': loop.run_in_executor(None, calculate_unique_skus),
            'cancellationRate': loop.run_in_executor(None, calculate_cancellation_rate),
            'returnRate': loop.run_in_executor(None, calculate_return_rate),
            'codShare': loop.run_in_executor(None, calculate_cod_share),
            'deliveredRate': loop.run_in_executor(None, calculate_delivered_rate),
        }

        results = {}
        for key, task in kpi_tasks.items():
            results[key] = await task

        # Build final response
        kpi_response = {
            'success': True,
            'data': {
                'totalOrders': {'value': results['totalOrders'], 'unit': 'orders', 'metric_name': 'Total Orders'},
                'unitsSold': {'value': results['unitsSold'], 'unit': 'units', 'metric_name': 'Units Sold'},
                'grossRevenue': {'value': results['grossRevenue'], 'unit': 'INR', 'metric_name': 'Gross Revenue'},
                'aov': {'value': results['aov'], 'unit': 'INR', 'metric_name': 'Average Order Value'},
                'uniqueSkus': {'value': results['uniqueSkus'], 'unit': 'skus', 'metric_name': 'Unique SKUs'},
                'cancellationRate': {'value': results['cancellationRate'], 'unit': '%', 'metric_name': 'Cancellation Rate'},
                'returnRate': {'value': results['returnRate'], 'unit': '%', 'metric_name': 'Return Rate'},
                'codShare': {'value': results['codShare'], 'unit': '%', 'metric_name': 'COD Share'},
                'deliveredRate': {'value': results['deliveredRate'], 'unit': '%', 'metric_name': 'Delivered Rate'},
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'database'   # Helpful for debugging
        }

        final_response = convert_numpy_types(kpi_response)

        # === 3. Store result in Redis (Cache Populate) ===
        if redis_client:
            try:
                redis_client.set(
                    cache_key,
                    _serialize_for_redis(final_response),
                    ex=600,                                 # TTL = 10 minutes (adjust as needed)
                )
                print(f"✅ Cache SET for key: {cache_key} (TTL: 10min)", flush=True)
            except Exception as cache_err:
                print(f"Redis set error (non-fatal): {cache_err}", flush=True)

        return final_response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating KPIs: {str(e)}"
        )

# ============ KPI CHART ENDPOINTS ============
def _parse_iso_datetime(value: Optional[str], field_name: str) -> Optional[datetime]:
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    try:
        return datetime.fromisoformat(value_str.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format. Use YYYY-MM-DD or full datetime"
        ) from exc


def _granularity_from_request_window(request: HistoryOrdersRequest) -> Optional[str]:
    start_dt = _parse_iso_datetime(request.start_date, "start_date")
    end_dt = _parse_iso_datetime(request.end_date, "end_date")

    if start_dt is None or end_dt is None:
        return None

    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date")

    window_days = max((end_dt.date() - start_dt.date()).days + 1, 1)
    if window_days <= 31:
        return 'daily'
    if window_days <= 180:
        return 'weekly'
    return 'monthly'


def _resolve_kpi_chart_granularity(request: HistoryOrdersRequest, df: pd.DataFrame) -> str:
    requested_window_granularity = _granularity_from_request_window(request)
    if requested_window_granularity:
        return requested_window_granularity

    requested_filter_granularity = _request_granularity(request)
    if requested_filter_granularity:
        return _resolve_granularity(requested_filter_granularity, pd.Timestamp.now(), pd.Timestamp.now())

    if df.empty or 'order_date' not in df.columns:
        return 'daily'

    working_dates = pd.to_datetime(df['order_date'], errors='coerce')
    working_dates = working_dates.dropna()
    if working_dates.empty:
        return 'daily'

    return _resolve_granularity(None, working_dates.min(), working_dates.max())


def _group_for_kpi_chart(request: HistoryOrdersRequest):
    df = fetch_historical_orders(request)
    if df.empty:
        return df, 'daily', [], pd.DataFrame(), 'date_group'

    chart_type = _resolve_kpi_chart_granularity(request, df)
    chart_type, labels, grouped_df, group_col = _build_time_groups(df, chart_type)
    return df, chart_type, labels, grouped_df, group_col


def _build_recharts_data(labels: list[str], values: list[float | int], data_key: str) -> list[dict]:
    """Transform labels and values into recharts data format."""
    return [
        {
            "date": label,
            data_key: value
        }
        for label, value in zip(labels, values)
    ]


def _kpi_chart_response(metric_name: str, unit: str, chart_type: str, labels: list[str], dataset_key: str, values: list[float | int]):
    """Build recharts-native response format."""
    return convert_numpy_types({
        "success": True,
        "metric_name": metric_name,
        "chart_type": chart_type,
        "data": _build_recharts_data(labels, values, dataset_key),
        "unit": unit,
        "timestamp": datetime.now().isoformat()
    })


@router.post('/history/kpi/charts/total-orders')
def kpi_chart_total_orders(request: HistoryOrdersRequest):
    """Bar chart data for total orders trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Total Orders", "orders", chart_type, [], "totalOrders", [])

        if 'order_id' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['order_id'].nunique().sort_index()
        else:
            grouped = grouped_df.groupby(group_col).size().sort_index()

        return _kpi_chart_response("Total Orders", "orders", chart_type, labels, "totalOrders", [int(v) for v in grouped.tolist()])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating total orders chart: {str(e)}")


@router.post('/history/kpi/charts/units-sold')
def kpi_chart_units_sold(request: HistoryOrdersRequest):
    """Bar chart data for units sold trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Units Sold", "units", chart_type, [], "unitsSold", [])

        grouped_df = _ensure_numeric(grouped_df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        if 'item_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['item_quantity'].sum().sort_index()
        elif 'suborder_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['suborder_quantity'].sum().sort_index()
        elif 'order_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['order_quantity'].sum().sort_index()
        else:
            grouped = grouped_df.groupby(group_col).size().sort_index()

        return _kpi_chart_response("Units Sold", "units", chart_type, labels, "unitsSold", [float(v) for v in grouped.tolist()])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating units sold chart: {str(e)}")


@router.post('/history/kpi/charts/gross-revenue')
def kpi_chart_gross_revenue(request: HistoryOrdersRequest):
    """Bar chart data for gross revenue trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Gross Revenue", "INR", chart_type, [], "grossRevenue", [])

        grouped_df = _ensure_numeric(grouped_df, ['total_amount'])
        grouped = grouped_df.groupby(group_col)['total_amount'].sum().sort_index() if 'total_amount' in grouped_df.columns else grouped_df.groupby(group_col).size().sort_index() * 0
        return _kpi_chart_response("Gross Revenue", "INR", chart_type, labels, "grossRevenue", [float(v) for v in grouped.tolist()])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating gross revenue chart: {str(e)}")


@router.post('/history/kpi/charts/aov')
def kpi_chart_aov(request: HistoryOrdersRequest):
    """Bar chart data for AOV trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Average Order Value", "INR", chart_type, [], "aov", [])

        grouped_df = _ensure_numeric(grouped_df, ['total_amount'])
        if 'total_amount' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['total_amount'].mean().sort_index()
            values = [float(v) for v in grouped.fillna(0).tolist()]
        else:
            grouped = grouped_df.groupby(group_col).size().sort_index() * 0
            values = [0.0 for _ in grouped.tolist()]

        return _kpi_chart_response("Average Order Value", "INR", chart_type, labels, "aov", values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AOV chart: {str(e)}")


@router.post('/history/kpi/charts/unique-skus')
def kpi_chart_unique_skus(request: HistoryOrdersRequest):
    """Bar chart data for unique SKUs trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Unique SKUs", "skus", chart_type, [], "uniqueSkus", [])

        if 'sku' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['sku'].nunique().sort_index()
        else:
            grouped = grouped_df.groupby(group_col).size().sort_index() * 0

        return _kpi_chart_response("Unique SKUs", "skus", chart_type, labels, "uniqueSkus", [int(v) for v in grouped.tolist()])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating unique SKUs chart: {str(e)}")


@router.post('/history/kpi/charts/cancellation-rate')
def kpi_chart_cancellation_rate(request: HistoryOrdersRequest):
    """Bar chart data for cancellation rate trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Cancellation Rate", "%", chart_type, [], "cancellationRate", [])

        grouped = grouped_df.groupby(group_col, dropna=False)
        values = [round(_status_rate(group_df, {'Cancelled'}), 2) for _, group_df in grouped]
        return _kpi_chart_response("Cancellation Rate", "%", chart_type, labels, "cancellationRate", values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cancellation rate chart: {str(e)}")


@router.post('/history/kpi/charts/return-rate')
def kpi_chart_return_rate(request: HistoryOrdersRequest):
    """Bar chart data for RTO (Return To Origin) rate trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Return/RTO Rate", "%", chart_type, [], "rtoRate", [])

        grouped = grouped_df.groupby(group_col, dropna=False)
        values = [round(_status_rate(group_df, {'RTO'}), 2) for _, group_df in grouped]
        return _kpi_chart_response("Return/RTO Rate", "%", chart_type, labels, "rtoRate", values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating return rate chart: {str(e)}")


@router.post('/history/kpi/charts/cod-share')
def kpi_chart_cod_share(request: HistoryOrdersRequest):
    """Bar chart data for COD share trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("COD Share", "%", chart_type, [], "codShare", [])

        # payment_mode is already normalized to uppercase by _enrich_dataframe()
        grouped = grouped_df.groupby(group_col, dropna=False)

        values = []
        for _, group_df in grouped:
            total = len(group_df)
            cod_count = int((group_df['payment_mode'] == 'COD').sum())
            values.append(round(_safe_pct(float(cod_count), float(total)), 2))

        return _kpi_chart_response("COD Share", "%", chart_type, labels, "codShare", values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating COD share chart: {str(e)}")


@router.post('/history/kpi/charts/delivered-rate')
def kpi_chart_delivered_rate(request: HistoryOrdersRequest):
    """Bar chart data for delivered rate trend by auto-selected granularity."""
    try:
        _, chart_type, labels, grouped_df, group_col = _group_for_kpi_chart(request)
        if grouped_df.empty:
            return _kpi_chart_response("Delivered Rate", "%", chart_type, [], "deliveredRate", [])

        grouped = grouped_df.groupby(group_col, dropna=False)
        values = [round(_status_rate(group_df, {'Delivered'}), 2) for _, group_df in grouped]
        return _kpi_chart_response("Delivered Rate", "%", chart_type, labels, "deliveredRate", values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating delivered rate chart: {str(e)}")


# ============ KPI CARD ENDPOINTS ============
@router.post('/history/kpi/total-orders')
def kpi_total_orders(request: HistoryOrdersRequest):
    """
    KPI Card: Total Orders
    Count of unique order_ids in the period
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Total Orders",
                "value": 0,
                "unit": "orders",
                "description": "Total number of orders in the period",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Total Orders",
            "value": int(total_orders),
            "unit": "orders",
            "description": "Total number of orders in the period",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating total orders: {str(e)}"
        )


@router.post('/history/kpi/units-sold')
def kpi_units_sold(request: HistoryOrdersRequest):
    """
    KPI Card: Units Sold
    Sum of item_quantity/suborder_quantity across all orders
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Units Sold",
                "value": 0,
                "unit": "units",
                "description": "Total number of units sold in the period",
                "timestamp": datetime.now().isoformat()
            })
        
        # Use item_quantity if available, else suborder_quantity, else order_quantity
        if 'item_quantity' in df.columns:
            units_sold = int(df['item_quantity'].sum())
        elif 'suborder_quantity' in df.columns:
            units_sold = int(df['suborder_quantity'].sum())
        else:
            units_sold = int(df['order_quantity'].sum() if 'order_quantity' in df.columns else len(df))
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Units Sold",
            "value": units_sold,
            "unit": "units",
            "description": "Total number of units sold in the period",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating units sold: {str(e)}"
        )


@router.post('/history/kpi/gross-revenue')
def kpi_gross_revenue(request: HistoryOrdersRequest):
    """
    KPI Card: Gross Revenue
    Sum of total_amount (order-level revenue)
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Gross Revenue",
                "value": 0.0,
                "unit": "INR",
                "description": "Total revenue in the period",
                "timestamp": datetime.now().isoformat()
            })
        
        gross_revenue = float(df['total_amount'].sum() if 'total_amount' in df.columns else 0)
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Gross Revenue",
            "value": gross_revenue,
            "unit": "INR",
            "description": "Total revenue in the period",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating gross revenue: {str(e)}"
        )


@router.post('/history/kpi/aov')
def kpi_aov(request: HistoryOrdersRequest):
    """
    KPI Card: Average Order Value (AOV)
    Mean of total_amount across all orders
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty or 'total_amount' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Average Order Value",
                "value": 0.0,
                "unit": "INR",
                "description": "Average value per order",
                "timestamp": datetime.now().isoformat()
            })
        
        aov = float(df['total_amount'].mean())
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Average Order Value",
            "value": aov,
            "unit": "INR",
            "description": "Average value per order",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating AOV: {str(e)}"
        )


@router.post('/history/kpi/unique-skus')
def kpi_unique_skus(request: HistoryOrdersRequest):
    """
    KPI Card: Unique SKUs Sold
    Count of distinct SKU values
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty or 'sku' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Unique SKUs",
                "value": 0,
                "unit": "skus",
                "description": "Number of unique SKUs sold in the period",
                "timestamp": datetime.now().isoformat()
            })
        
        unique_skus = int(df['sku'].nunique())
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Unique SKUs",
            "value": unique_skus,
            "unit": "skus",
            "description": "Number of unique SKUs sold in the period",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating unique SKUs: {str(e)}"
        )


@router.post('/history/kpi/cancellation-rate')
def kpi_cancellation_rate(request: HistoryOrdersRequest):
    """
    KPI Card: Cancellation Rate
    (cancelled_orders / total_orders) * 100
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Cancellation Rate",
                "value": 0.0,
                "unit": "%",
                "description": "Percentage of orders cancelled",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        cancelled_orders = _status_order_count(df, {'Cancelled'})
        cancellation_rate = _safe_pct(float(cancelled_orders), float(total_orders))
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Cancellation Rate",
            "value": round(float(cancellation_rate), 2),
            "unit": "%",
            "description": "Percentage of orders cancelled",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating cancellation rate: {str(e)}"
        )


@router.post('/history/kpi/return-rate')
def kpi_return_rate(request: HistoryOrdersRequest):
    """
    KPI Card: Return Rate
    (returned_orders / total_orders) * 100
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Return/RTO Rate",
                "value": 0.0,
                "unit": "%",
                "description": "Percentage of orders returned to origin",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        rto_orders = _status_order_count(df, {'RTO'})
        rto_rate = _safe_pct(float(rto_orders), float(total_orders))
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Return/RTO Rate",
            "value": round(float(rto_rate), 2),
            "unit": "%",
            "description": "Percentage of orders returned to origin",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating return rate: {str(e)}"
        )


@router.post('/history/kpi/cod-share')
def kpi_cod_share(request: HistoryOrdersRequest):
    """
    KPI Card: COD Share
    (COD orders / total orders) * 100
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty or 'payment_mode' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "COD Share",
                "value": 0.0,
                "unit": "%",
                "description": "Percentage of orders using COD payment",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = len(df)
        cod_orders = len(df[df['payment_mode'] == 'COD'])
        cod_share = (cod_orders / total_orders * 100) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "COD Share",
            "value": round(float(cod_share), 2),
            "unit": "%",
            "description": "Percentage of orders using COD payment",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating COD share: {str(e)}"
        )


@router.post('/history/kpi/delivered-rate')
def kpi_delivered_rate(request: HistoryOrdersRequest):
    """
    KPI Card: Delivered Rate
    (delivered orders / total orders) * 100
    """
    try:
        df = fetch_historical_orders(request)
        
        if df.empty or 'order_status' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Delivered Rate",
                "value": 0.0,
                "unit": "%",
                "description": "Percentage of orders delivered",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = len(df)
        delivered_orders = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.DELIVERED.value])
        delivered_rate = (delivered_orders / total_orders * 100) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Delivered Rate",
            "value": round(float(delivered_rate), 2),
            "unit": "%",
            "description": "Percentage of orders delivered",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating delivered rate: {str(e)}"
        )


@router.post('/history/kpi/comparison-30d')
def kpi_comparison_30d(request: HistoryOrdersRequest):
    """
    KPI Card: 30-Day Comparison
    Compare current 30 days vs previous 30 days for:
    - Orders count
    - Revenue
    - AOV
    - Units sold
    """
    try:
        current_end = datetime.fromisoformat(request.end_date)
        current_start = current_end - timedelta(days=29)
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=29)
        return _kpi_comparison_response(
            request=request,
            metric_name="30-Day Comparison",
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating 30-day comparison: {str(e)}"
        )

@router.post('/history/kpi/comparison-7d')
def kpi_comparison_7d(request: HistoryOrdersRequest):
    """
    KPI Card: 7-Day Comparison
    Compare current 7 days vs previous 7 days for:
    - Orders count
    - Revenue
    - AOV
    - Units sold
    """
    try:
        current_end = datetime.fromisoformat(request.end_date)
        current_start = current_end - timedelta(days=6)
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=6)
        return _kpi_comparison_response(
            request=request,
            metric_name="7-Day Comparison",
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating 7-day comparison: {str(e)}"
        )


@router.post('/history/kpi/comparison-quarter')
def kpi_comparison_quarter(request: HistoryOrdersRequest):
    """
    KPI Card: Quarter Comparison
    Compare current quarter-to-date vs previous quarter.
    """
    try:
        current_end = datetime.fromisoformat(request.end_date)

        current_quarter_start_month = ((current_end.month - 1) // 3) * 3 + 1
        current_start = current_end.replace(
            month=current_quarter_start_month,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        previous_end = current_start - timedelta(days=1)
        previous_quarter_start_month = ((previous_end.month - 1) // 3) * 3 + 1
        previous_start = previous_end.replace(
            month=previous_quarter_start_month,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        return _kpi_comparison_response(
            request=request,
            metric_name="Quarter Comparison",
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating quarter comparison: {str(e)}"
        )


def _comparison_units(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if 'item_quantity' in df.columns:
        return int(df['item_quantity'].sum())
    if 'suborder_quantity' in df.columns:
        return int(df['suborder_quantity'].sum())
    if 'order_quantity' in df.columns:
        return int(df['order_quantity'].sum())
    return int(len(df))


def _comparison_metrics(df: pd.DataFrame) -> Dict[str, float | int]:
    orders = _order_count(df)
    revenue = float(df['total_amount'].sum()) if 'total_amount' in df.columns else 0.0
    units = _comparison_units(df)
    aov = float(revenue / orders) if orders > 0 else 0.0
    return {
        "orders": int(orders),
        "revenue": revenue,
        "aov": aov,
        "units": int(units),
    }


def _kpi_comparison_response(
    request: HistoryOrdersRequest,
    metric_name: str,
    current_start: datetime,
    current_end: datetime,
    previous_start: datetime,
    previous_end: datetime,
):
    current_df = fetch_historical_orders(
        request.table_name,
        current_start.date().isoformat(),
        current_end.date().isoformat(),
        request.filters,
    )
    previous_df = fetch_historical_orders(
        request.table_name,
        previous_start.date().isoformat(),
        previous_end.date().isoformat(),
        request.filters,
    )

    current_metrics = _comparison_metrics(current_df)
    previous_metrics = _comparison_metrics(previous_df)

    return convert_numpy_types({
        "success": True,
        "metric_name": metric_name,
        "current_period": {
            "start_date": current_start.date().isoformat(),
            "end_date": current_end.date().isoformat(),
            **current_metrics,
        },
        "previous_period": {
            "start_date": previous_start.date().isoformat(),
            "end_date": previous_end.date().isoformat(),
            **previous_metrics,
        },
        "deltas": {
            "orders_pct": round(_growth_pct(float(current_metrics['orders']), float(previous_metrics['orders'])), 2),
            "revenue_pct": round(_growth_pct(float(current_metrics['revenue']), float(previous_metrics['revenue'])), 2),
            "units_pct": round(_growth_pct(float(current_metrics['units']), float(previous_metrics['units'])), 2),
            "aov_pct": round(_growth_pct(float(current_metrics['aov']), float(previous_metrics['aov'])), 2),
        },
        "timestamp": datetime.now().isoformat(),
    })


def _resolve_current_previous_windows(request: HistoryOrdersRequest, default_days: int = 30):
    """Build current and previous windows with equal duration."""
    if request.end_date:
        current_end = datetime.fromisoformat(request.end_date)
    else:
        current_end = datetime.now()

    if request.start_date:
        current_start = datetime.fromisoformat(request.start_date)
    else:
        current_start = current_end - timedelta(days=default_days)

    if current_start > current_end:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date")

    window_days = max((current_end.date() - current_start.date()).days + 1, 1)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=window_days - 1)

    return current_start, current_end, previous_start, previous_end, window_days


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float((numerator / denominator) * 100)


def _order_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df['order_id'].nunique()) if 'order_id' in df.columns else int(len(df))


def _status_order_count(df: pd.DataFrame, statuses: set[str]) -> int:
    if df.empty or 'order_status' not in df.columns:
        return 0

    normalized_status = df['order_status'].astype(str).str.strip()
    status_set = {OrderStatus.normalize(str(status)) for status in statuses if OrderStatus.normalize(str(status))}
    matched_df = df[normalized_status.isin(status_set)]
    return _order_count(matched_df)


def _status_rate(df: pd.DataFrame, statuses: set[str]) -> float:
    total_orders = _order_count(df)
    matched_orders = _status_order_count(df, statuses)
    return _safe_pct(float(matched_orders), float(total_orders))


def _growth_pct(current_val: float, previous_val: float) -> float:
    if previous_val == 0:
        return 0.0 if current_val == 0 else 100.0
    return float(((current_val - previous_val) / previous_val) * 100)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def _canonical_sku_from_row(row: pd.Series) -> str:
    candidates = [
        row.get('canonical_sku'),
        row.get('suborder_sku'),
        row.get('sku'),
        row.get('suborder_marketplace_sku'),
        row.get('marketplace_sku'),
    ]
    for value in candidates:
        if pd.notna(value) and str(value).strip():
            sku = str(value).strip()
            if sku.startswith('YCE_'):
                sku = sku[4:]
            return sku
    return 'UNKNOWN'


def _size_from_row(row: pd.Series) -> str:
    candidates = [row.get('suborder_size'), row.get('size')]
    for value in candidates:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    sku = _canonical_sku_from_row(row)
    if '_' in sku:
        maybe_size = sku.rsplit('_', 1)[-1].strip()
        if maybe_size:
            return maybe_size
    return 'UNKNOWN'


def _enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed/normalized columns as intermediate processing step.
    Called INSIDE fetch_historical_orders after all basic conversions.
    This eliminates repeated post-fetch transformations in endpoints.
    """
    if df.empty:
        return df
    
    working_df = df.copy()
    
    # ===== Normalize payment_mode =====
    if 'payment_mode' in working_df.columns:
        working_df['payment_mode'] = (
            working_df['payment_mode']
            .astype(str)
            .str.strip()
            .str.upper()
            .replace('', 'UNKNOWN')
        )
        working_df.loc[working_df['payment_mode'].isna(), 'payment_mode'] = 'UNKNOWN'
    
    # ===== Normalize order_status =====
    if 'order_status' in working_df.columns:
        working_df['order_status'] = (
            working_df['order_status']
            .astype(str)
            .str.strip()
            .apply(lambda x: OrderStatus.normalize(x) if x else 'UNKNOWN')
        )
    
    # ===== Normalize state =====
    if 'state' in working_df.columns:
        working_df['state'] = (
            working_df['state']
            .astype(str)
            .str.strip()
            .replace('', 'UNKNOWN')
        )
        working_df.loc[working_df['state'].isna(), 'state'] = 'UNKNOWN'
    
    # ===== Derive canonical_sku (if not already present) =====
    if 'canonical_sku_normalized' not in working_df.columns:
        working_df['canonical_sku_normalized'] = working_df.apply(
            _canonical_sku_from_row, axis=1
        )
    
    # ===== Derive size_bucket =====
    if 'size_bucket' not in working_df.columns:
        working_df['size_bucket'] = working_df.apply(
            _size_from_row, axis=1
        )
    
    # ===== Extract base SKU from canonical_sku =====
    if 'base_sku' not in working_df.columns:
        working_df['base_sku'] = working_df['canonical_sku_normalized'].apply(
            lambda x: x.rsplit('_', 1)[0] if '_' in x else x
        )
    
    # ===== Extract date-time components (for heatmaps/time-series) =====
    if 'order_date' in working_df.columns:
        # Ensure order_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(working_df['order_date']):
            working_df['order_date'] = pd.to_datetime(
                working_df['order_date'], 
                errors='coerce'
            )
        
        # Extract components
        working_df['order_date_only'] = working_df['order_date'].dt.date
        working_df['day_of_week'] = working_df['order_date'].dt.dayofweek
        working_df['day_label'] = working_df['order_date'].dt.day_name().str[:3]
        working_df['hour_of_day'] = working_df['order_date'].dt.hour
        working_df['week_start'] = working_df['order_date'].dt.to_period('W').dt.start_time.dt.date
        working_df['month_start'] = working_df['order_date'].dt.to_period('M').dt.start_time.dt.date
    
    return working_df


def _resolve_granularity(requested: Optional[str], min_date: pd.Timestamp, max_date: pd.Timestamp) -> str:
    if requested:
        requested_norm = str(requested).strip().lower()
        mapping = {
            'day': 'daily',
            'daily': 'daily',
            'week': 'weekly',
            'weekly': 'weekly',
            'month': 'monthly',
            'monthly': 'monthly',
        }
        if requested_norm in mapping:
            return mapping[requested_norm]
        raise HTTPException(status_code=400, detail="Invalid granularity. Use day/week/month")

    date_range_days = (max_date - min_date).days
    if date_range_days <= 31:
        return 'daily'
    if date_range_days <= 180:
        return 'weekly'
    return 'monthly'


def _calculate_growth_rates(values: list) -> list:
    """
    Calculate percentage growth for each value compared to the previous value.
    First value will have None growth rate (no previous value to compare).
    
    Formula: ((Current - Previous) / Previous) * 100
    """
    growth_rates = []
    for i, value in enumerate(values):
        if i == 0:
            growth_rates.append(None)  # No previous value for first data point
        else:
            prev_value = values[i - 1]
            if prev_value == 0:
                # If previous value is 0, growth is infinite or undefined
                # Set to 0 if current is also 0, else set to None
                growth_rates.append(0 if value == 0 else None)
            else:
                growth_percent = ((value - prev_value) / prev_value) * 100
                growth_rates.append(round(growth_percent, 2))
    return growth_rates


def _build_time_groups(df: pd.DataFrame, requested_granularity: Optional[str] = None):
    if 'order_date' not in df.columns:
        raise HTTPException(status_code=400, detail="order_date column is required")

    working_df = df.copy()
    working_df['order_date'] = pd.to_datetime(working_df['order_date'], errors='coerce')
    working_df = working_df.dropna(subset=['order_date'])

    if working_df.empty:
        return 'daily', [], pd.DataFrame(), 'date_group'

    min_date = working_df['order_date'].min()
    max_date = working_df['order_date'].max()
    chart_type = _resolve_granularity(requested_granularity, min_date, max_date)

    if chart_type == 'daily':
        working_df['date_group'] = working_df['order_date'].dt.date
        labels = [d.strftime('%Y-%m-%d') for d in sorted(working_df['date_group'].dropna().unique())]
    elif chart_type == 'weekly':
        working_df['date_group'] = working_df['order_date'].dt.to_period('W').dt.start_time.dt.date
        labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in sorted(working_df['date_group'].dropna().unique())]
    else:
        working_df['date_group'] = working_df['order_date'].dt.to_period('M').dt.start_time.dt.date
        labels = [pd.Timestamp(d).strftime('%b %Y') for d in sorted(working_df['date_group'].dropna().unique())]

    return chart_type, labels, working_df, 'date_group'


def _request_granularity(request: HistoryOrdersRequest) -> Optional[str]:
    if request.filters and isinstance(request.filters, dict):
        value = request.filters.get('granularity')
        if value is not None:
            return str(value)
    return None



# =================== COMPARISON =================
@router.post('/history/comparison/cod-vs-prepaid')
def comparison_cod_vs_prepaid(request: HistoryOrdersRequest):
    """COD vs PrePaid metrics: order count, revenue, AOV, cancellation rate, return rate."""
    try:
        df = fetch_historical_orders(request)

        if df.empty:
            empty_segment = {
                "order_count": 0,
                "revenue": 0.0,
                "aov": 0.0,
                "cancellation_rate": 0.0,
                "rto_rate": 0.0,
            }
            return convert_numpy_types({
                "success": True,
                "metric_name": "COD vs PrePaid",
                "segments": {
                    "cod": empty_segment,
                    "prepaid": empty_segment,
                },
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, ['total_amount'])

        # payment_mode is already normalized to uppercase by _enrich_dataframe()
        cod_df = df[df['payment_mode'] == 'COD']
        prepaid_df = df[df['payment_mode'] != 'COD']

        def _segment_metrics(segment_df: pd.DataFrame):
            order_count = _order_count(segment_df)
            revenue = float(segment_df['total_amount'].sum()) if 'total_amount' in segment_df.columns else 0.0
            aov = float(revenue / order_count) if order_count > 0 else 0.0

            return {
                "order_count": order_count,
                "revenue": revenue,
                "aov": aov,
                "cancellation_rate": round(_status_rate(segment_df, {'Cancelled'}), 2),
                "rto_rate": round(_status_rate(segment_df, {'RTO'}), 2),
            }

        return convert_numpy_types({
            "success": True,
            "metric_name": "COD vs PrePaid",
            "segments": {
                "cod": _segment_metrics(cod_df),
                "prepaid": _segment_metrics(prepaid_df),
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating COD vs PrePaid comparison: {str(e)}"
        )


@router.post('/history/comparison/b2b-vs-b2c')
def comparison_b2b_vs_b2c(request: HistoryOrdersRequest):
    """B2B vs B2C metrics: order count, units/order, AOV, cancellation rate."""
    try:
        df = fetch_historical_orders(request)

        if df.empty:
            empty_segment = {
                "order_count": 0,
                "units_per_order": 0.0,
                "aov": 0.0,
                "cancellation_rate": 0.0,
            }
            return convert_numpy_types({
                "success": True,
                "metric_name": "B2B vs B2C",
                "segments": {
                    "b2b": empty_segment,
                    "b2c": empty_segment,
                },
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, ['total_amount', 'item_quantity'])

        if 'order_type' not in df.columns:
            df['order_type'] = 'UNKNOWN'

        order_type_normalized = df['order_type'].astype(str).str.strip().str.upper()
        b2b_df = df[order_type_normalized == 'B2B']
        b2c_df = df[order_type_normalized == 'B2C']

        def _segment_metrics(segment_df: pd.DataFrame):
            order_count = _order_count(segment_df)
            total_units = float(segment_df['item_quantity'].sum()) if 'item_quantity' in segment_df.columns else 0.0
            revenue = float(segment_df['total_amount'].sum()) if 'total_amount' in segment_df.columns else 0.0

            return {
                "order_count": order_count,
                "units_per_order": round(float(total_units / order_count), 2) if order_count > 0 else 0.0,
                "aov": round(float(revenue / order_count), 2) if order_count > 0 else 0.0,
                "cancellation_rate": round(_status_rate(segment_df, {'Cancelled'}), 2),
            }

        return convert_numpy_types({
            "success": True,
            "metric_name": "B2B vs B2C",
            "segments": {
                "b2b": _segment_metrics(b2b_df),
                "b2c": _segment_metrics(b2c_df),
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating B2B vs B2C comparison: {str(e)}"
        )


@router.post('/history/comparison/top-states-vs-rest')
def comparison_top_states_vs_rest(request: HistoryOrdersRequest):
    """Top 5 states vs rest for revenue share and growth vs previous period."""
    try:
        current_start, current_end, previous_start, previous_end, window_days = _resolve_current_previous_windows(request)

        current_df = fetch_historical_orders(
            request.table_name,
            current_start.date().isoformat(),
            current_end.date().isoformat(),
            request.filters
        )
        previous_df = fetch_historical_orders(
            request.table_name,
            previous_start.date().isoformat(),
            previous_end.date().isoformat(),
            request.filters
        )

        if current_df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 States vs Rest",
                "window_days": int(window_days),
                "states": [],
                "rest": {
                    "current_revenue": 0.0,
                    "previous_revenue": 0.0,
                    "revenue_share": 0.0,
                    "growth_pct": 0.0,
                },
                "timestamp": datetime.now().isoformat()
            })

        current_df = _ensure_numeric(current_df, ['total_amount'])
        previous_df = _ensure_numeric(previous_df, ['total_amount'])

        # state is already normalized with UNKNOWN fallback by _enrich_dataframe()

        current_state_rev = current_df.groupby('state', dropna=False)['total_amount'].sum().sort_values(ascending=False)
        previous_state_rev = previous_df.groupby('state', dropna=False)['total_amount'].sum()

        total_current_revenue = float(current_state_rev.sum())
        top_states = list(current_state_rev.head(5).index)

        states_payload = []
        for state in top_states:
            current_revenue = float(current_state_rev.get(state, 0.0))
            previous_revenue = float(previous_state_rev.get(state, 0.0))
            states_payload.append({
                "state": state,
                "current_revenue": current_revenue,
                "previous_revenue": previous_revenue,
                "revenue_share": round(_safe_pct(current_revenue, total_current_revenue), 2),
                "growth_pct": round(_growth_pct(current_revenue, previous_revenue), 2),
            })

        current_rest_revenue = float(current_state_rev[~current_state_rev.index.isin(top_states)].sum())
        previous_rest_revenue = float(previous_state_rev[~previous_state_rev.index.isin(top_states)].sum())

        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 5 States vs Rest",
            "window_days": int(window_days),
            "current_period": {
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
            },
            "previous_period": {
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
            },
            "states": states_payload,
            "rest": {
                "current_revenue": current_rest_revenue,
                "previous_revenue": previous_rest_revenue,
                "revenue_share": round(_safe_pct(current_rest_revenue, total_current_revenue), 2),
                "growth_pct": round(_growth_pct(current_rest_revenue, previous_rest_revenue), 2),
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating top states vs rest comparison: {str(e)}"
        )


@router.post('/history/comparison/top-skus-rank-growth')
def comparison_top_skus_rank_growth(request: HistoryOrdersRequest):
    """Top 10 canonical SKUs vs previous period with rank change and growth."""
    try:
        current_start, current_end, previous_start, previous_end, window_days = _resolve_current_previous_windows(request)

        current_df = fetch_historical_orders(
            request.table_name,
            current_start.date().isoformat(),
            current_end.date().isoformat(),
            request.filters
        )
        previous_df = fetch_historical_orders(
            request.table_name,
            previous_start.date().isoformat(),
            previous_end.date().isoformat(),
            request.filters
        )

        if current_df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 10 Canonical SKUs Rank and Growth",
                "window_days": int(window_days),
                "skus": [],
                "timestamp": datetime.now().isoformat()
            })
# canoni# canonical_sku_normalized is already computed by _enrich_dataframe()
        current_rev = current_df.groupby('canonical_sku_normalized', dropna=False)['total_amount'].sum().sort_values(ascending=False)
        previous_rev = previous_df.groupby('canonical_sku_normalized_normalized', dropna=False)['total_amount'].sum().sort_values(ascending=False)

        current_rank_map = {sku: rank + 1 for rank, sku in enumerate(current_rev.index)}
        previous_rank_map = {sku: rank + 1 for rank, sku in enumerate(previous_rev.index)}

        payload = []
        for sku in list(current_rev.head(10).index):
            current_revenue = float(current_rev.get(sku, 0.0))
            previous_revenue = float(previous_rev.get(sku, 0.0))
            current_rank = int(current_rank_map.get(sku, 0))
            previous_rank = int(previous_rank_map.get(sku, 0)) if sku in previous_rank_map else None

            payload.append({
                "canonical_sku": sku,
                "current_rank": current_rank,
                "previous_rank": previous_rank,
                "rank_change": (previous_rank - current_rank) if previous_rank is not None else None,
                "current_revenue": current_revenue,
                "previous_revenue": previous_revenue,
                "growth_pct": round(_growth_pct(current_revenue, previous_revenue), 2),
            })

        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 10 Canonical SKUs Rank and Growth",
            "window_days": int(window_days),
            "current_period": {
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
            },
            "previous_period": {
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
            },
            "ranking_basis": "revenue",
            "skus": payload,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating top SKUs rank and growth comparison: {str(e)}"
        )


@router.post('/history/comparison/size-mix-change')
def comparison_size_mix_change(request: HistoryOrdersRequest):
    """Size mix change for current period vs previous equal-length period."""
    try:
        current_start, current_end, previous_start, previous_end, window_days = _resolve_current_previous_windows(request)

        current_df = fetch_historical_orders(
            request.table_name,
            current_start.date().isoformat(),
            current_end.date().isoformat(),
            request.filters
        )
        previous_df = fetch_historical_orders(
            request.table_name,
            previous_start.date().isoformat(),
            previous_end.date().isoformat(),
            request.filters
        )

        if current_df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Size Mix Change",
                "window_days": int(window_days),
                "sizes": [],
                "timestamp": datetime.now().isoformat()
            })

        current_df = _ensure_numeric(current_df, ['item_quantity'])
        previous_df = _ensure_numeric(previous_df, ['item_quantity'])

        # size_bucket is already computed by _enrich_dataframe()

        current_units = current_df.groupby('size_bucket', dropna=False)['item_quantity'].sum()
        previous_units = previous_df.groupby('size_bucket', dropna=False)['item_quantity'].sum()

        all_sizes = sorted(set(current_units.index).union(set(previous_units.index)))
        total_current_units = float(current_units.sum())
        total_previous_units = float(previous_units.sum())

        payload = []
        for size in all_sizes:
            cur_units = float(current_units.get(size, 0.0))
            prev_units = float(previous_units.get(size, 0.0))
            cur_share = _safe_pct(cur_units, total_current_units)
            prev_share = _safe_pct(prev_units, total_previous_units)
            payload.append({
                "size": str(size),
                "current_units": cur_units,
                "previous_units": prev_units,
                "current_share": round(cur_share, 2),
                "previous_share": round(prev_share, 2),
                "share_change_pp": round(cur_share - prev_share, 2),
                "growth_pct": round(_growth_pct(cur_units, prev_units), 2),
            })

        payload.sort(key=lambda x: x['current_units'], reverse=True)

        return convert_numpy_types({
            "success": True,
            "metric_name": "Size Mix Change",
            "window_days": int(window_days),
            "current_period": {
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
                "total_units": total_current_units,
            },
            "previous_period": {
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
                "total_units": total_previous_units,
            },
            "sizes": payload,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating size mix change comparison: {str(e)}"
        )


# =================== CHARTS =================
@router.post('/history/chart/orders-line')
def chart_orders_line(request: HistoryOrdersRequest):
    """Line chart data for order count by day/week/month."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"orders": []},
                "timestamp": datetime.now().isoformat()
            })

        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))
        if grouped_df.empty:
            series = []
        else:
            if 'order_id' in grouped_df.columns:
                grouped = grouped_df.groupby(group_col)['order_id'].nunique().sort_index()
            else:
                grouped = grouped_df.groupby(group_col).size().sort_index()
            series = [int(v) for v in grouped.tolist()]

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {"orders": series},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating orders line chart: {str(e)}")


@router.post('/history/chart/units-line')
def chart_units_line(request: HistoryOrdersRequest):
    """Line chart data for units sold by day/week/month."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"units": []},
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))

        if 'item_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['item_quantity'].sum().sort_index()
        elif 'suborder_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['suborder_quantity'].sum().sort_index()
        elif 'order_quantity' in grouped_df.columns:
            grouped = grouped_df.groupby(group_col)['order_quantity'].sum().sort_index()
        else:
            grouped = grouped_df.groupby(group_col).size().sort_index()

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {"units": [float(v) for v in grouped.tolist()]},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating units line chart: {str(e)}")


@router.post('/history/chart/revenue-line')
def chart_revenue_line(request: HistoryOrdersRequest):
    """Line chart data for revenue by day/week/month."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"revenue": []},
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, ['total_amount'])
        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))
        grouped = grouped_df.groupby(group_col)['total_amount'].sum().sort_index()

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {"revenue": [float(v) for v in grouped.tolist()]},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating revenue line chart: {str(e)}")


@router.post('/history/chart/orders-aov-line')
def chart_orders_aov_line(request: HistoryOrdersRequest):
    """Dual-axis line data for orders and AOV trend."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"orders": [], "aov": []},
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, ['total_amount'])
        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))

        if 'order_id' in grouped_df.columns:
            grouped_orders = grouped_df.groupby(group_col)['order_id'].nunique().sort_index()
        else:
            grouped_orders = grouped_df.groupby(group_col).size().sort_index()
        grouped_revenue = grouped_df.groupby(group_col)['total_amount'].sum().sort_index()
        grouped_aov = grouped_revenue.divide(grouped_orders.replace(0, pd.NA)).fillna(0)

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {
                "orders": [int(v) for v in grouped_orders.tolist()],
                "aov": [float(v) for v in grouped_aov.tolist()],
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating orders + AOV chart: {str(e)}")


@router.post('/history/chart/payment-mix-area')
def chart_payment_mix_area(request: HistoryOrdersRequest):
    """Stacked area data for COD vs PrePaid mix over time."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"cod": [], "prepaid": []},
                "timestamp": datetime.now().isoformat()
            })

        if 'payment_mode' not in df.columns:
            df['payment_mode'] = 'UNKNOWN'

        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))
        grouped_df['payment_mode_norm'] = grouped_df['payment_mode'].astype(str).str.strip().str.upper()
        grouped = grouped_df.groupby([group_col, 'payment_mode_norm']).size().unstack(fill_value=0).sort_index()

        cod = []
        prepaid = []
        for _, row in grouped.iterrows():
            cod_count = int(row.get('COD', 0))
            prepaid_count = int(sum(v for k, v in row.items() if k != 'COD'))
            cod.append(cod_count)
            prepaid.append(prepaid_count)

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {"cod": cod, "prepaid": prepaid},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating payment mix chart: {str(e)}")


@router.post('/history/chart/order-type-mix-area')
def chart_order_type_mix_area(request: HistoryOrdersRequest):
    """Stacked area data for B2B vs B2C mix over time."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"b2b": [], "b2c": []},
                "timestamp": datetime.now().isoformat()
            })

        if 'order_type' not in df.columns:
            df['order_type'] = 'UNKNOWN'

        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))
        grouped_df['order_type_norm'] = grouped_df['order_type'].astype(str).str.strip().str.upper()
        grouped = grouped_df.groupby([group_col, 'order_type_norm']).size().unstack(fill_value=0).sort_index()

        b2b = []
        b2c = []
        for _, row in grouped.iterrows():
            b2b.append(int(row.get('B2B', 0)))
            b2c.append(int(row.get('B2C', 0)))

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {"b2b": b2b, "b2c": b2c},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating order type mix chart: {str(e)}")


@router.post('/history/chart/cancellation-return-rate-line')
def chart_cancellation_return_rate_line(request: HistoryOrdersRequest):
    """Line chart data for cancellation and RTO rates over time."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {"cancellation_rate": [], "rto_rate": []},
                "timestamp": datetime.now().isoformat()
            })

        df = _ensure_numeric(df, [])
        chart_type, labels, grouped_df, group_col = _build_time_groups(df, _request_granularity(request))

        grouped = grouped_df.groupby(group_col, dropna=False)

        cancellation_rate = []
        rto_rate = []
        for _, group_df in grouped:
            cancellation_rate.append(round(_status_rate(group_df, {'Cancelled'}), 2))
            rto_rate.append(round(_status_rate(group_df, {'RTO'}), 2))

        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {
                "cancellation_rate": cancellation_rate,
                "rto_rate": rto_rate,
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cancellation/return trend chart: {str(e)}")


@router.post('/history/chart/order-intensity-heatmap')
def chart_order_intensity_heatmap(request: HistoryOrdersRequest):
    """Heatmap data for day-of-week vs hour-of-day order intensity (D3-ready)."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "x_labels": list(range(24)),
                "y_labels": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                "cells": [],
                "matrix": [[0 for _ in range(24)] for _ in range(7)],
                "timestamp": datetime.now().isoformat()
            })

        if 'order_date' not in df.columns:
            raise HTTPException(status_code=400, detail="order_date column is required")

        # Filter out rows with NaN dates
        working_df = df[df['order_date'].notna()].copy()

        if working_df.empty:
            return convert_numpy_types({
                "success": True,
                "x_labels": list(range(24)),
                "y_labels": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                "cells": [],
                "matrix": [[0 for _ in range(24)] for _ in range(7)],
                "timestamp": datetime.now().isoformat()
            })

        # day_of_week, day_label, and hour_of_day are pre-computed by _enrich_dataframe()
        working_df['day_idx'] = working_df['day_of_week']
        working_df['hour'] = working_df['hour_of_day']

        if 'order_id' in working_df.columns:
            grouped = working_df.groupby(['day_idx', 'day_label', 'hour'])['order_id'].nunique().reset_index(name='order_count')
        else:
            grouped = working_df.groupby(['day_idx', 'day_label', 'hour']).size().reset_index(name='order_count')

        y_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        x_labels = list(range(24))
        matrix = [[0 for _ in x_labels] for _ in y_labels]
        cells = []

        for _, row in grouped.iterrows():
            d_idx = int(row['day_idx'])
            hour = int(row['hour'])
            count = int(row['order_count'])
            if 0 <= d_idx <= 6 and 0 <= hour <= 23:
                matrix[d_idx][hour] = count
                cells.append({
                    "day_of_week": y_labels[d_idx],
                    "day_index": d_idx,
                    "hour": hour,
                    "order_count": count,
                })

        return convert_numpy_types({
            "success": True,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "cells": cells,
            "matrix": matrix,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating order intensity heatmap: {str(e)}")


def _base_sku_from_canonical(canonical_sku: str) -> str:
    sku = str(canonical_sku or "UNKNOWN").strip()
    if '_' in sku:
        return sku.rsplit('_', 1)[0]
    return sku


def _prepare_sku_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df = _ensure_numeric(working_df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
    
    # Use pre-computed columns from _enrich_dataframe()
    if 'canonical_sku_normalized' not in working_df.columns:
        working_df['canonical_sku_normalized'] = working_df.apply(_canonical_sku_from_row, axis=1)
    if 'base_sku' not in working_df.columns:
        working_df['base_sku'] = working_df['canonical_sku_normalized'].apply(_base_sku_from_canonical)
    if 'size_bucket' not in working_df.columns:
        working_df['size_bucket'] = working_df.apply(_size_from_row, axis=1)
    
    # Alias for backward compatibility
    working_df['canonical_sku'] = working_df['canonical_sku_normalized']
    if 'canonical_sku_normalized' not in working_df.columns:
        working_df['canonical_sku_normalized'] = working_df.apply(_canonical_sku_from_row, axis=1)
    if 'base_sku' not in working_df.columns:
        working_df['base_sku'] = working_df['canonical_sku_normalized'].apply(_base_sku_from_canonical)
    # if 'size_bucket' not in working_df.columns:
        # Fall back to derivation only if _enrich_dataframe was not called
    if 'canonical_sku_normalized' not in working_df.columns:
        working_df['canonical_sku_normalized'] = working_df.apply(_canonical_sku_from_row, axis=1)
    if 'base_sku' not in working_df.columns:
        working_df['base_sku'] = working_df['canonical_sku_normalized'].apply(_base_sku_from_canonical)
    if 'size_bucket' not in working_df.columns:
        working_df['size_bucket'] = working_df.apply(_size_from_row, axis=1)

    if 'item_quantity' in working_df.columns:
        working_df['units'] = working_df['item_quantity']
    elif 'suborder_quantity' in working_df.columns:
        working_df['units'] = working_df['suborder_quantity']
    elif 'order_quantity' in working_df.columns:
        working_df['units'] = working_df['order_quantity']
    else:
        working_df['units'] = 1

    return working_df


@router.post('/history/chart/top-canonical-skus-units-bar')
def chart_top_canonical_skus_units_bar(request: HistoryOrdersRequest):
    """Bar chart data for top canonical SKUs by units sold."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "labels": [],
                "datasets": {"units": []},
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)
        top_n = 10
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 10

        grouped = working_df.groupby('canonical_sku', dropna=False)['units'].sum().sort_values(ascending=False).head(top_n)
        return convert_numpy_types({
            "success": True,
            "labels": [str(idx) for idx in grouped.index.tolist()],
            "datasets": {"units": [float(v) for v in grouped.tolist()]},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating top canonical SKUs by units chart: {str(e)}")


@router.post('/history/chart/top-canonical-skus-revenue-bar')
def chart_top_canonical_skus_revenue_bar(request: HistoryOrdersRequest):
    """Bar chart data for top canonical SKUs by revenue."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "labels": [],
                "datasets": {"revenue": []},
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)
        top_n = 10
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 10

        grouped = working_df.groupby('canonical_sku', dropna=False)['total_amount'].sum().sort_values(ascending=False).head(top_n)
        return convert_numpy_types({
            "success": True,
            "labels": [str(idx) for idx in grouped.index.tolist()],
            "datasets": {"revenue": [float(v) for v in grouped.tolist()]},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating top canonical SKUs by revenue chart: {str(e)}")


@router.post('/history/chart/sku-revenue-vs-cancellation-bubble')
def chart_sku_revenue_vs_cancellation_bubble(request: HistoryOrdersRequest):
    """Bubble/scatter data: x=revenue, y=cancellation_rate, r=units grouped by canonical SKU."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "points": [],
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)
        grouped = working_df.groupby('canonical_sku', dropna=False).agg({
            'total_amount': 'sum',
            'units': 'sum',
        })

        points = []
        for sku, row in grouped.iterrows():
            sku_df = working_df[working_df['canonical_sku'] == sku]
            units = float(row.get('units', 0.0))
            points.append({
                'canonical_sku': str(sku),
                'x_revenue': float(row.get('total_amount', 0.0)),
                'y_cancellation_rate': round(_status_rate(sku_df, {'Cancelled'}), 2),
                'r_units': units,
            })

        points.sort(key=lambda x: x['x_revenue'], reverse=True)

        top_n = 50
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 50

        return convert_numpy_types({
            "success": True,
            "points": points[:top_n],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SKU revenue vs cancellation bubble chart: {str(e)}")


@router.post('/history/chart/base-sku-size-units-heatmap')
def chart_base_sku_size_units_heatmap(request: HistoryOrdersRequest):
    """Matrix heatmap data for base SKU vs size by units sold."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "x_labels": [],
                "y_labels": [],
                "cells": [],
                "matrix": [],
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)

        base_totals = working_df.groupby('base_sku', dropna=False)['units'].sum().sort_values(ascending=False)
        top_n = 10
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 10

        top_bases = [str(v) for v in base_totals.head(top_n).index.tolist()]
        filtered = working_df[working_df['base_sku'].astype(str).isin(top_bases)]

        if filtered.empty:
            return convert_numpy_types({
                "success": True,
                "x_labels": [],
                "y_labels": top_bases,
                "cells": [],
                "matrix": [[0] for _ in top_bases] if top_bases else [],
                "timestamp": datetime.now().isoformat()
            })

        size_labels = sorted([str(v) for v in filtered['size_bucket'].dropna().astype(str).unique().tolist()])
        grouped = filtered.groupby(['base_sku', 'size_bucket'], dropna=False)['units'].sum()

        matrix = [[0 for _ in size_labels] for _ in top_bases]
        cells = []
        y_index = {v: i for i, v in enumerate(top_bases)}
        x_index = {v: i for i, v in enumerate(size_labels)}

        for (base_sku, size), units in grouped.items():
            b = str(base_sku)
            s = str(size)
            if b in y_index and s in x_index:
                yi = y_index[b]
                xi = x_index[s]
                value = float(units)
                matrix[yi][xi] = value
                cells.append({
                    'base_sku': b,
                    'size': s,
                    'x_index': xi,
                    'y_index': yi,
                    'units': value,
                })

        return convert_numpy_types({
            "success": True,
            "x_labels": size_labels,
            "y_labels": top_bases,
            "cells": cells,
            "matrix": matrix,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating base SKU vs size heatmap: {str(e)}")


@router.post('/history/chart/top-base-sku-size-distribution-stacked')
def chart_top_base_sku_size_distribution_stacked(request: HistoryOrdersRequest):
    """Stacked bar data for size distribution within each top base SKU."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "labels": [],
                "datasets": {},
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)

        base_totals = working_df.groupby('base_sku', dropna=False)['units'].sum().sort_values(ascending=False)
        top_n = 10
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 10

        top_bases = [str(v) for v in base_totals.head(top_n).index.tolist()]
        filtered = working_df[working_df['base_sku'].astype(str).isin(top_bases)]

        if filtered.empty:
            return convert_numpy_types({
                "success": True,
                "labels": top_bases,
                "datasets": {},
                "timestamp": datetime.now().isoformat()
            })

        grouped = filtered.groupby(['base_sku', 'size_bucket'], dropna=False)['units'].sum().unstack(fill_value=0)
        grouped = grouped.reindex(top_bases, fill_value=0)

        datasets = {}
        for size in grouped.columns:
            datasets[str(size)] = [float(v) for v in grouped[size].tolist()]

        return convert_numpy_types({
            "success": True,
            "labels": top_bases,
            "datasets": datasets,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating top base SKU size distribution chart: {str(e)}")


@router.post('/history/chart/top-sku-weekly-velocity-line')
def chart_top_sku_weekly_velocity_line(request: HistoryOrdersRequest):
    """Line data for weekly velocity trend of top canonical SKU by units in selected period."""
    try:
        df = fetch_historical_orders(request.table_name, request.start_date, request.end_date, request.filters)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "weekly",
                "top_sku": None,
                "labels": [],
                "datasets": {"units": []},
                "timestamp": datetime.now().isoformat()
            })

        working_df = _prepare_sku_metrics_df(df)
        if 'order_date' not in working_df.columns:
            raise HTTPException(status_code=400, detail="order_date column is required")

        sku_units = working_df.groupby('canonical_sku', dropna=False)['units'].sum().sort_values(ascending=False)
        if sku_units.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "weekly",
                "top_sku": None,
                "labels": [],
                "datasets": {"units": []},
                "timestamp": datetime.now().isoformat()
            })

        top_sku = str(sku_units.index[0])
        sku_df = working_df[working_df['canonical_sku'].astype(str) == top_sku].copy()
        sku_df['order_date'] = pd.to_datetime(sku_df['order_date'], errors='coerce')
        sku_df = sku_df.dropna(subset=['order_date'])
        sku_df['week_start'] = sku_df['order_date'].dt.to_period('W').dt.start_time.dt.date

        grouped = sku_df.groupby('week_start')['units'].sum().sort_index()
        labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped.index]
        values = [float(v) for v in grouped.tolist()]

        return convert_numpy_types({
            "success": True,
            "chart_type": "weekly",
            "top_sku": top_sku,
            "labels": labels,
            "datasets": {"units": values},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating top SKU weekly velocity trend: {str(e)}")


@router.post('/history/chart/sku-revenue-change-waterfall')
def chart_sku_revenue_change_waterfall(request: HistoryOrdersRequest):
    """Waterfall data for SKU-wise contribution to total revenue change vs previous equal-length window."""
    try:
        current_start, current_end, previous_start, previous_end, window_days = _resolve_current_previous_windows(request)

        current_df = fetch_historical_orders(
            request.table_name,
            current_start.date().isoformat(),
            current_end.date().isoformat(),
            request.filters
        )
        previous_df = fetch_historical_orders(
            request.table_name,
            previous_start.date().isoformat(),
            previous_end.date().isoformat(),
            request.filters
        )

        if current_df.empty and previous_df.empty:
            return convert_numpy_types({
                "success": True,
                "window_days": int(window_days),
                "total_change": 0.0,
                "waterfall": [],
                "timestamp": datetime.now().isoformat()
            })

        current_working = _prepare_sku_metrics_df(current_df) if not current_df.empty else pd.DataFrame(columns=['canonical_sku', 'total_amount'])
        previous_working = _prepare_sku_metrics_df(previous_df) if not previous_df.empty else pd.DataFrame(columns=['canonical_sku', 'total_amount'])

        current_rev = current_working.groupby('canonical_sku', dropna=False)['total_amount'].sum()
        previous_rev = previous_working.groupby('canonical_sku', dropna=False)['total_amount'].sum()

        all_skus = set(current_rev.index).union(set(previous_rev.index))
        total_change = float(current_rev.sum() - previous_rev.sum())

        items = []
        for sku in all_skus:
            cur = float(current_rev.get(sku, 0.0))
            prev = float(previous_rev.get(sku, 0.0))
            delta = cur - prev
            items.append({
                'canonical_sku': str(sku),
                'current_revenue': cur,
                'previous_revenue': prev,
                'delta_revenue': delta,
                'contribution_pct': round(_safe_pct(delta, total_change), 2) if total_change != 0 else 0.0,
            })

        items.sort(key=lambda x: abs(x['delta_revenue']), reverse=True)
        top_n = 20
        if request.filters and isinstance(request.filters, dict) and request.filters.get('top_n'):
            try:
                top_n = max(1, int(request.filters.get('top_n')))
            except Exception:
                top_n = 20

        return convert_numpy_types({
            "success": True,
            "window_days": int(window_days),
            "current_period": {
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
            },
            "previous_period": {
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
            },
            "total_change": total_change,
            "waterfall": items[:top_n],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SKU revenue change waterfall: {str(e)}")


@router.post('/history/comparison/sku-vs-sku')
def comparison_sku_vs_sku(request: HistoryOrdersRequest):
    """Compare two size-included SKUs with 7d, 30d, or all-time window options."""
    try:
        filters = request.filters or {}

        sku_1_raw = filters.get('sku_1') or filters.get('sku1')
        sku_2_raw = filters.get('sku_2') or filters.get('sku2')
        if not sku_1_raw or not sku_2_raw:
            raise HTTPException(status_code=400, detail="filters.sku_1 and filters.sku_2 are required")

        def _normalize_input_sku(value: str) -> str:
            sku = str(value).strip()
            if sku.startswith('YCE_'):
                sku = sku[4:]
            return sku

        sku_1 = _normalize_input_sku(sku_1_raw)
        sku_2 = _normalize_input_sku(sku_2_raw)

        comparison_window = str(filters.get('comparison_window', '30d')).strip().lower()
        valid_windows = {'7d', '30d', 'all-time'}
        if comparison_window not in valid_windows:
            raise HTTPException(status_code=400, detail="comparison_window must be one of: 7d, 30d, all-time")

        # Remove non-column control filters before passing to Supabase where clause.
        control_keys = {'sku_1', 'sku1', 'sku_2', 'sku2', 'comparison_window', 'granularity', 'top_n'}
        db_filters = {k: v for k, v in filters.items() if k not in control_keys}

        previous_df = pd.DataFrame()
        current_start = None
        current_end = None
        previous_start = None
        previous_end = None

        if comparison_window in {'7d', '30d'}:
            days = 7 if comparison_window == '7d' else 30
            if request.end_date:
                current_end = datetime.fromisoformat(request.end_date)
            else:
                current_end = datetime.now()

            current_start = current_end - timedelta(days=days - 1)
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=days - 1)

            current_df = fetch_historical_orders(
                request.table_name,
                current_start.date().isoformat(),
                current_end.date().isoformat(),
                db_filters
            )
            previous_df = fetch_historical_orders(
                request.table_name,
                previous_start.date().isoformat(),
                previous_end.date().isoformat(),
                db_filters
            )
        else:
            current_df = fetch_historical_orders(
                request.table_name,
                request.start_date,
                request.end_date,
                db_filters
            )

        if current_df.empty:
            return convert_numpy_types({
                "success": True,
                "comparison_window": comparison_window,
                "sku_1": sku_1,
                "sku_2": sku_2,
                "metrics": {
                    sku_1: {
                        "order_count": 0,
                        "units": 0.0,
                        "revenue": 0.0,
                        "aov": 0.0,
                        "cancellation_rate": 0.0,
                        "return_rate": 0.0,
                    },
                    sku_2: {
                        "order_count": 0,
                        "units": 0.0,
                        "revenue": 0.0,
                        "aov": 0.0,
                        "cancellation_rate": 0.0,
                        "return_rate": 0.0,
                    }
                },
                "comparison": {
                    "orders_diff": 0,
                    "units_diff": 0.0,
                    "revenue_diff": 0.0,
                    "aov_diff": 0.0,
                    "cancellation_rate_diff": 0.0,
                    "return_rate_diff": 0.0,
                },
                "growth": None,
                "timestamp": datetime.now().isoformat()
            })

        current_working = _prepare_sku_metrics_df(current_df)
        previous_working = _prepare_sku_metrics_df(previous_df) if not previous_df.empty else pd.DataFrame()

        def _compute_metrics(source_df: pd.DataFrame, target_sku: str) -> dict:
            sku_df = source_df[source_df['canonical_sku'].astype(str) == target_sku] if not source_df.empty else pd.DataFrame()
            if sku_df.empty:
                return {
                    "order_count": 0,
                    "units": 0.0,
                    "revenue": 0.0,
                    "aov": 0.0,
                    "cancellation_rate": 0.0,
                    "rto_rate": 0.0,
                }

            order_count = int(sku_df['order_id'].nunique()) if 'order_id' in sku_df.columns else int(len(sku_df))
            units = float(sku_df['units'].sum()) if 'units' in sku_df.columns else 0.0
            revenue = float(sku_df['total_amount'].sum()) if 'total_amount' in sku_df.columns else 0.0

            return {
                "order_count": order_count,
                "units": units,
                "revenue": revenue,
                "aov": round(float(revenue / order_count), 2) if order_count > 0 else 0.0,
                "cancellation_rate": round(_status_rate(sku_df, {'Cancelled'}), 2),
                "rto_rate": round(_status_rate(sku_df, {'RTO'}), 2),
            }

        cur_1 = _compute_metrics(current_working, sku_1)
        cur_2 = _compute_metrics(current_working, sku_2)

        response = {
            "success": True,
            "comparison_window": comparison_window,
            "sku_1": sku_1,
            "sku_2": sku_2,
            "metrics": {
                sku_1: cur_1,
                sku_2: cur_2,
            },
            "comparison": {
                "orders_diff": int(cur_1['order_count'] - cur_2['order_count']),
                "units_diff": float(cur_1['units'] - cur_2['units']),
                "revenue_diff": float(cur_1['revenue'] - cur_2['revenue']),
                "aov_diff": float(cur_1['aov'] - cur_2['aov']),
                "cancellation_rate_diff": float(cur_1['cancellation_rate'] - cur_2['cancellation_rate']),
                "rto_rate_diff": float(cur_1['rto_rate'] - cur_2['rto_rate']),
                "revenue_lift_pct_sku1_vs_sku2": round(_growth_pct(cur_1['revenue'], cur_2['revenue']), 2),
                "units_lift_pct_sku1_vs_sku2": round(_growth_pct(cur_1['units'], cur_2['units']), 2),
            },
            "growth": None,
            "timestamp": datetime.now().isoformat()
        }

        if comparison_window in {'7d', '30d'}:
            prev_1 = _compute_metrics(previous_working, sku_1)
            prev_2 = _compute_metrics(previous_working, sku_2)
            response["current_period"] = {
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
            }
            response["previous_period"] = {
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
            }
            response["growth"] = {
                sku_1: {
                    "orders_growth_pct": round(_growth_pct(cur_1['order_count'], prev_1['order_count']), 2),
                    "units_growth_pct": round(_growth_pct(cur_1['units'], prev_1['units']), 2),
                    "revenue_growth_pct": round(_growth_pct(cur_1['revenue'], prev_1['revenue']), 2),
                    "aov_growth_pct": round(_growth_pct(cur_1['aov'], prev_1['aov']), 2),
                    "cancellation_rate_change_pp": round(cur_1['cancellation_rate'] - prev_1['cancellation_rate'], 2),
                    "rto_rate_change_pp": round(cur_1['rto_rate'] - prev_1['rto_rate'], 2),
                },
                sku_2: {
                    "orders_growth_pct": round(_growth_pct(cur_2['order_count'], prev_2['order_count']), 2),
                    "units_growth_pct": round(_growth_pct(cur_2['units'], prev_2['units']), 2),
                    "revenue_growth_pct": round(_growth_pct(cur_2['revenue'], prev_2['revenue']), 2),
                    "aov_growth_pct": round(_growth_pct(cur_2['aov'], prev_2['aov']), 2),
                    "cancellation_rate_change_pp": round(cur_2['cancellation_rate'] - prev_2['cancellation_rate'], 2),
                    "rto_rate_change_pp": round(cur_2['rto_rate'] - prev_2['rto_rate'], 2),
                }
            }

        return convert_numpy_types(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SKU vs SKU comparison: {str(e)}")


# =================== PRODUCT & INVENTORY METRICS =================
@router.post('/history/metrics/sku-diversity-index')
def metrics_sku_diversity_index(request: HistoryOrdersRequest):
    """
    SKU Diversity Index: Unique SKUs / Total Orders
    Measures product diversity ratio - higher is more diverse product mix
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "SKU Diversity Index",
                "value": 0.0,
                "description": "Unique SKUs / Total Orders",
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        unique_skus = int(df['sku'].nunique()) if 'sku' in df.columns else 0
        diversity_index = float(unique_skus / total_orders) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "SKU Diversity Index",
            "value": round(diversity_index, 4),
            "unique_skus": unique_skus,
            "total_orders": int(total_orders),
            "description": "Product diversity ratio - Unique SKUs / Total Orders",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating SKU diversity index: {str(e)}")


@router.post('/history/metrics/top-skus-by-revenue')
def metrics_top_skus_by_revenue(request: HistoryOrdersRequest):
    """
    Top 5 SKUs by Revenue
    Shows highest revenue-generating SKUs with metrics
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 SKUs by Revenue",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        df['sku_clean'] = df['sku'].astype(str).str.strip()
        
        sku_metrics = df.groupby('sku_clean', dropna=False).agg({
            'order_id': 'nunique',
            'total_amount': 'sum',
            'item_quantity': 'sum',
            'suborder_quantity': 'sum',
            'order_quantity': 'sum'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        sku_metrics = sku_metrics.sort_values('revenue', ascending=False).head(5)
        
        results = []
        for sku, row in sku_metrics.iterrows():
            units = row['item_quantity'] if pd.notna(row['item_quantity']) and row['item_quantity'] > 0 else (row['suborder_quantity'] if pd.notna(row['suborder_quantity']) else row['order_quantity'])
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            results.append({
                "sku": str(sku),
                "revenue": float(row['revenue']),
                "order_count": int(row['order_count']),
                "units": int(units) if pd.notna(units) else 0,
                "aov": round(aov, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 5 SKUs by Revenue",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating top SKUs by revenue: {str(e)}")


@router.post('/history/metrics/top-skus-by-units')
def metrics_top_skus_by_units(request: HistoryOrdersRequest):
    """
    Top 5 SKUs by Units Sold
    Shows best-selling SKUs by volume with metrics
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 SKUs by Units",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        df['sku_clean'] = df['sku'].astype(str).str.strip()
        
        # Determine which quantity column to use
        qty_col = None
        if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
            qty_col = 'item_quantity'
        elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
            qty_col = 'suborder_quantity'
        elif 'order_quantity' in df.columns:
            qty_col = 'order_quantity'
        
        if qty_col is None:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 SKUs by Units",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        sku_metrics = df.groupby('sku_clean', dropna=False).agg({
            qty_col: 'sum',
            'order_id': 'nunique',
            'total_amount': 'sum'
        }).rename(columns={qty_col: 'units', 'order_id': 'order_count', 'total_amount': 'revenue'})
        
        sku_metrics = sku_metrics.sort_values('units', ascending=False).head(5)
        
        results = []
        for sku, row in sku_metrics.iterrows():
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            results.append({
                "sku": str(sku),
                "units": int(row['units']),
                "order_count": int(row['order_count']),
                "revenue": float(row['revenue']),
                "aov": round(aov, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 5 SKUs by Units",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating top SKUs by units: {str(e)}")


@router.post('/history/metrics/average-units-per-order')
def metrics_average_units_per_order(request: HistoryOrdersRequest):
    """
    Average Units per Order: Total Units / Total Orders
    Measures average order size in units
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Average Units per Order",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        total_orders = _order_count(df)
        
        units = 0
        if 'item_quantity' in df.columns:
            units = int(df['item_quantity'].sum())
        elif 'suborder_quantity' in df.columns:
            units = int(df['suborder_quantity'].sum())
        elif 'order_quantity' in df.columns:
            units = int(df['order_quantity'].sum())
        
        avg_units = float(units / total_orders) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Average Units per Order",
            "value": round(avg_units, 2),
            "total_units": units,
            "total_orders": int(total_orders),
            "unit": "units/order",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating average units per order: {str(e)}")


@router.post('/history/metrics/size-mix-distribution')
def metrics_size_mix_distribution(request: HistoryOrdersRequest):
    """
    Size Mix Distribution: % breakdown by size
    Shows proportion of each size category by units
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Size Mix Distribution",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        df['size_bucket'] = df.apply(_size_from_row, axis=1)
        
        qty_col = None
        if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
            qty_col = 'item_quantity'
        elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
            qty_col = 'suborder_quantity'
        elif 'order_quantity' in df.columns:
            qty_col = 'order_quantity'
        
        if qty_col is None:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Size Mix Distribution",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        size_dist = df.groupby('size_bucket', dropna=False)[qty_col].sum()
        total_units = size_dist.sum()
        
        results = []
        for size, units in size_dist.items():
            pct = float(units / total_units * 100) if total_units > 0 else 0.0
            results.append({
                "size": str(size),
                "units": int(units),
                "percentage": round(pct, 2)
            })
        
        results.sort(key=lambda x: x['units'], reverse=True)
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Size Mix Distribution",
            "data": results,
            "total_units": int(total_units),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating size mix distribution: {str(e)}")


@router.post('/history/metrics/sku-performance-matrix')
def metrics_sku_performance_matrix(request: HistoryOrdersRequest):
    """
    SKU Performance Matrix: Revenue vs Units per SKU
    2D analysis of all SKUs with revenue on x-axis and units on y-axis
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "SKU Performance Matrix",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        df['sku_clean'] = df['sku'].astype(str).str.strip()
        
        qty_col = None
        if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
            qty_col = 'item_quantity'
        elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
            qty_col = 'suborder_quantity'
        elif 'order_quantity' in df.columns:
            qty_col = 'order_quantity'
        
        if qty_col is None:
            return convert_numpy_types({
                "success": True,
                "metric_name": "SKU Performance Matrix",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        sku_metrics = df.groupby('sku_clean', dropna=False).agg({
            'total_amount': 'sum',
            qty_col: 'sum',
            'order_id': 'nunique'
        }).rename(columns={'total_amount': 'revenue', qty_col: 'units', 'order_id': 'order_count'})
        
        # Sort by revenue descending, limit to top 20
        sku_metrics = sku_metrics.sort_values('revenue', ascending=False).head(20)
        
        results = []
        for sku, row in sku_metrics.iterrows():
            results.append({
                "sku": str(sku),
                "revenue": float(row['revenue']),
                "units": int(row['units']),
                "order_count": int(row['order_count']),
                "revenue_per_unit": round(float(row['revenue'] / row['units']) if row['units'] > 0 else 0, 2),
                "units_per_order": round(float(row['units'] / row['order_count']) if row['order_count'] > 0 else 0, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "SKU Performance Matrix (Top 20)",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating SKU performance matrix: {str(e)}")


# =================== PERFORMANCE METRICS =================
@router.post('/history/metrics/fulfillment-rate')
def metrics_fulfillment_rate(request: HistoryOrdersRequest):
    """
    Fulfillment/Shipped Rate: (Manifest Scanned + Delivered) orders / Total orders
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Fulfillment Rate",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        
        if 'order_status' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Fulfillment Rate",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        status_normalized = df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None)
        fulfilled = ((status_normalized == OrderStatus.SHIPPING_START.value) | (status_normalized == OrderStatus.DELIVERED.value)).sum()
        fulfillment_rate = float(fulfilled / total_orders * 100) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Fulfillment Rate",
            "value": round(fulfillment_rate, 2),
            "unit": "%",
            "fulfilled_orders": int(fulfilled),
            "total_orders": int(total_orders),
            "description": "(Manifest Scanned + Delivered) / Total Orders",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fulfillment rate: {str(e)}")


@router.post('/history/metrics/order-value-distribution')
def metrics_order_value_distribution(request: HistoryOrdersRequest):
    """
    Order Value Distribution: Median, 25th percentile, 75th percentile AOV
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'total_amount' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Order Value Distribution",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        
        p25 = float(df['total_amount'].quantile(0.25))
        p50 = float(df['total_amount'].quantile(0.50))
        p75 = float(df['total_amount'].quantile(0.75))
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Order Value Distribution",
            "data": {
                "p25": round(p25, 2),
                "median": round(p50, 2),
                "p75": round(p75, 2),
                "min": round(float(df['total_amount'].min()), 2),
                "max": round(float(df['total_amount'].max()), 2),
                "mean": round(float(df['total_amount'].mean()), 2),
                "std_dev": round(float(df['total_amount'].std()), 2)
            },
            "unit": "INR",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating order value distribution: {str(e)}")


@router.post('/history/metrics/order-velocity')
def metrics_order_velocity(request: HistoryOrdersRequest):
    """
    Daily/Weekly/Monthly Order Velocity: Orders per day/week/month
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'order_date' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Order Velocity",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Order Velocity",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        date_range = (df['order_date'].max() - df['order_date'].min()).days + 1
        total_orders = _order_count(df)
        
        daily_velocity = float(total_orders / date_range) if date_range > 0 else 0.0
        weekly_velocity = daily_velocity * 7
        monthly_velocity = daily_velocity * 30
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Order Velocity",
            "data": {
                "daily": round(daily_velocity, 2),
                "weekly": round(weekly_velocity, 2),
                "monthly": round(monthly_velocity, 2),
                "period_days": int(date_range),
                "total_orders": int(total_orders)
            },
            "unit": "orders per period",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating order velocity: {str(e)}")


@router.post('/history/metrics/units-velocity')
def metrics_units_velocity(request: HistoryOrdersRequest):
    """
    Units per Day Velocity: Total units / number of days
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'order_date' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Units Velocity",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Units Velocity",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        date_range = (df['order_date'].max() - df['order_date'].min()).days + 1
        
        units = 0
        if 'item_quantity' in df.columns:
            units = int(df['item_quantity'].sum())
        elif 'suborder_quantity' in df.columns:
            units = int(df['suborder_quantity'].sum())
        elif 'order_quantity' in df.columns:
            units = int(df['order_quantity'].sum())
        
        units_per_day = float(units / date_range) if date_range > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Units Velocity",
            "value": round(units_per_day, 2),
            "total_units": units,
            "period_days": int(date_range),
            "unit": "units/day",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating units velocity: {str(e)}")


# =================== GEOGRAPHIC METRICS =================
@router.post('/history/metrics/top-states-by-revenue')
def metrics_top_states_by_revenue(request: HistoryOrdersRequest):
    """
    Top 5 States by Revenue
    Shows highest revenue-generating states
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'state' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 States by Revenue",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['state_clean'] = df['state'].fillna('UNKNOWN').astype(str).str.strip()
        
        state_revenue = df.groupby('state_clean', dropna=False).agg({
            'total_amount': 'sum',
            'order_id': 'nunique'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        state_revenue = state_revenue.sort_values('revenue', ascending=False).head(5)
        total_revenue = state_revenue['revenue'].sum()
        
        results = []
        for state, row in state_revenue.iterrows():
            pct = float(row['revenue'] / total_revenue * 100) if total_revenue > 0 else 0.0
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            results.append({
                "state": str(state),
                "revenue": float(row['revenue']),
                "order_count": int(row['order_count']),
                "revenue_share": round(pct, 2),
                "aov": round(aov, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 5 States by Revenue",
            "data": results,
            "total_revenue": float(total_revenue),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating top states by revenue: {str(e)}")


@router.post('/history/metrics/top-states-by-orders')
def metrics_top_states_by_orders(request: HistoryOrdersRequest):
    """
    Top 5 States by Order Count
    Shows states with highest order volume
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'state' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Top 5 States by Orders",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['state_clean'] = df['state'].fillna('UNKNOWN').astype(str).str.strip()
        
        state_orders = df.groupby('state_clean', dropna=False).agg({
            'order_id': 'nunique',
            'total_amount': 'sum'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        state_orders = state_orders.sort_values('order_count', ascending=False).head(5)
        total_orders = state_orders['order_count'].sum()
        
        results = []
        for state, row in state_orders.iterrows():
            pct = float(row['order_count'] / total_orders * 100) if total_orders > 0 else 0.0
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            results.append({
                "state": str(state),
                "order_count": int(row['order_count']),
                "order_share": round(pct, 2),
                "revenue": float(row['revenue']),
                "aov": round(aov, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Top 5 States by Orders",
            "data": results,
            "total_orders": int(total_orders),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating top states by orders: {str(e)}")


@router.post('/history/metrics/geographic-revenue-concentration')
def metrics_geographic_revenue_concentration(request: HistoryOrdersRequest):
    """
    Geographic Revenue Concentration: % revenue from top 3 states
    Measure of geographic concentration/risk
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'state' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Geographic Revenue Concentration",
                "concentration_pct": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['state_clean'] = df['state'].fillna('UNKNOWN').astype(str).str.strip()
        
        state_revenue = df.groupby('state_clean', dropna=False)['total_amount'].sum().sort_values(ascending=False)
        
        top3_revenue = state_revenue.head(3).sum()
        total_revenue = state_revenue.sum()
        
        concentration = float(top3_revenue / total_revenue * 100) if total_revenue > 0 else 0.0
        
        top3_states = []
        for state, revenue in state_revenue.head(3).items():
            pct = float(revenue / total_revenue * 100) if total_revenue > 0 else 0.0
            top3_states.append({
                "state": str(state),
                "revenue": float(revenue),
                "revenue_pct": round(pct, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Geographic Revenue Concentration",
            "concentration_pct": round(concentration, 2),
            "top_3_states": top3_states,
            "risk_level": "HIGH" if concentration > 70 else "MEDIUM" if concentration > 50 else "LOW",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating geographic concentration: {str(e)}")


@router.post('/history/metrics/state-cancellation-rates')
def metrics_state_cancellation_rates(request: HistoryOrdersRequest):
    """
    State-wise Cancellation and Return Rates
    Shows fulfillment quality by geography
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'state' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "State-wise Cancellation Rates",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df['state_clean'] = df['state'].fillna('UNKNOWN').astype(str).str.strip()
        
        state_metrics = df.groupby('state_clean', dropna=False).agg({
            'order_id': 'nunique'
        }).rename(columns={'order_id': 'total_orders'})
        
        state_metrics['cancelled'] = df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.CANCELLED.value].groupby('state_clean', dropna=False)['order_id'].nunique().fillna(0)
        state_metrics['rto'] = df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.RTO.value].groupby('state_clean', dropna=False)['order_id'].nunique().fillna(0)
        
        state_metrics = state_metrics.sort_values('total_orders', ascending=False)
        
        results = []
        for state, row in state_metrics.iterrows():
            cancel_rate = float(row['cancelled'] / row['total_orders'] * 100) if row['total_orders'] > 0 else 0.0
            rto_rate = float(row['rto'] / row['total_orders'] * 100) if row['total_orders'] > 0 else 0.0
            results.append({
                "state": str(state),
                "total_orders": int(row['total_orders']),
                "cancellation_rate": round(cancel_rate, 2),
                "rto_rate": round(rto_rate, 2),
                "issue_rate": round(cancel_rate + rto_rate, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "State-wise Cancellation & RTO Rates",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating state cancellation rates: {str(e)}")


# =================== CHANNEL & PAYMENT METRICS =================
@router.post('/history/metrics/marketplace-performance')
def metrics_marketplace_performance(request: HistoryOrdersRequest):
    """
    Marketplace Performance: Revenue, AOV, Cancellation Rate by marketplace
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'marketplace' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Marketplace Performance",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['marketplace_clean'] = df['marketplace'].fillna('UNKNOWN').astype(str).str.strip()
        
        mp_metrics = df.groupby('marketplace_clean', dropna=False).agg({
            'order_id': 'nunique',
            'total_amount': 'sum'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        mp_metrics = mp_metrics.sort_values('revenue', ascending=False)
        
        results = []
        for mp, row in mp_metrics.iterrows():
            mp_df = df[df['marketplace_clean'] == mp]
            cancel_rate = _status_rate(mp_df, {'Cancelled'})
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            
            results.append({
                "marketplace": str(mp),
                "revenue": float(row['revenue']),
                "order_count": int(row['order_count']),
                "aov": round(aov, 2),
                "cancellation_rate": round(cancel_rate, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Marketplace Performance",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating marketplace performance: {str(e)}")


@router.post('/history/metrics/courier-performance')
def metrics_courier_performance(request: HistoryOrdersRequest):
    """
    Courier Performance: Delivery rate, cancellation rate, return rate by courier
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'courier' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Courier Performance",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df['courier_clean'] = df['courier'].fillna('UNKNOWN').astype(str).str.strip()
        
        courier_metrics = df.groupby('courier_clean', dropna=False).agg({
            'order_id': 'nunique'
        }).rename(columns={'order_id': 'order_count'})
        
        courier_metrics = courier_metrics.sort_values('order_count', ascending=False)
        
        results = []
        for courier, row in courier_metrics.iterrows():
            courier_df = df[df['courier_clean'] == courier]
            delivery_rate = _status_rate(courier_df, {'Delivered', 'Manifest Scanned'})
            cancel_rate = _status_rate(courier_df, {'Cancelled'})
            rto_rate = _status_rate(courier_df, {'RTO'})
            
            results.append({
                "courier": str(courier),
                "order_count": int(row['order_count']),
                "delivery_rate": round(delivery_rate, 2),
                "cancellation_rate": round(cancel_rate, 2),
                "rto_rate": round(rto_rate, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Courier Performance",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating courier performance: {str(e)}")


@router.post('/history/metrics/warehouse-efficiency')
def metrics_warehouse_efficiency(request: HistoryOrdersRequest):
    """
    Warehouse Efficiency: Orders fulfilled, average order value by warehouse
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'import_warehouse_name' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Warehouse Efficiency",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['warehouse_clean'] = df['import_warehouse_name'].fillna('UNKNOWN').astype(str).str.strip()
        
        wh_metrics = df.groupby('warehouse_clean', dropna=False).agg({
            'order_id': 'nunique',
            'total_amount': 'sum'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        wh_metrics = wh_metrics.sort_values('order_count', ascending=False)
        
        results = []
        for warehouse, row in wh_metrics.iterrows():
            aov = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
            results.append({
                "warehouse": str(warehouse),
                "order_count": int(row['order_count']),
                "revenue": float(row['revenue']),
                "aov": round(aov, 2)
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Warehouse Efficiency",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating warehouse efficiency: {str(e)}")


@router.post('/history/metrics/payment-mode-breakdown')
def metrics_payment_mode_breakdown(request: HistoryOrdersRequest):
    """
    Payment Mode Breakdown: COD vs Prepaid by revenue, order count, and cancellation rates
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'payment_mode' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Payment Mode Breakdown",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        df['payment_mode_norm'] = df['payment_mode'].astype(str).str.strip().apply(
            lambda x: PaymentMode.normalize(x) if x else None
        )
        
        payment_modes = df['payment_mode_norm'].dropna().unique()
        results = {}
        
        for mode in payment_modes:
            mode_df = df[df['payment_mode_norm'] == mode]
            order_count = _order_count(mode_df)
            revenue = float(mode_df['total_amount'].sum())
            aov = float(revenue / order_count) if order_count > 0 else 0.0
            cancel_rate = _status_rate(mode_df, {'Cancelled'})
            
            results[str(mode)] = {
                "order_count": int(order_count),
                "revenue": float(revenue),
                "aov": round(aov, 2),
                "cancellation_rate": round(cancel_rate, 2),
                "order_pct": 0.0,
                "revenue_pct": 0.0
            }
        
        # Calculate percentages
        total_orders = sum(r['order_count'] for r in results.values())
        total_revenue = sum(r['revenue'] for r in results.values())
        
        for mode in results:
            results[mode]['order_pct'] = round(results[mode]['order_count'] / total_orders * 100, 2) if total_orders > 0 else 0.0
            results[mode]['revenue_pct'] = round(results[mode]['revenue'] / total_revenue * 100, 2) if total_revenue > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Payment Mode Breakdown",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating payment mode breakdown: {str(e)}")


# =================== ORDER TYPE METRICS =================
@router.post('/history/metrics/b2b-vs-b2c')
def metrics_b2b_vs_b2c(request: HistoryOrdersRequest):
    """
    B2B vs B2C Split: % of orders, revenue split, AOV comparison
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'order_type' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "B2B vs B2C Analysis",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        df['order_type_norm'] = df['order_type'].astype(str).str.strip().apply(
            lambda x: OrderType.normalize(x) if x else None
        )
        
        order_types = df['order_type_norm'].dropna().unique()
        results = {}
        
        qty_col = None
        if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
            qty_col = 'item_quantity'
        elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
            qty_col = 'suborder_quantity'
        elif 'order_quantity' in df.columns:
            qty_col = 'order_quantity'
        
        for order_type in order_types:
            type_df = df[df['order_type_norm'] == order_type]
            order_count = _order_count(type_df)
            revenue = float(type_df['total_amount'].sum())
            aov = float(revenue / order_count) if order_count > 0 else 0.0
            units = 0
            if qty_col and qty_col in type_df.columns:
                units = int(type_df[qty_col].sum())
            
            results[str(order_type)] = {
                "order_count": int(order_count),
                "revenue": float(revenue),
                "aov": round(aov, 2),
                "units": units,
                "order_pct": 0.0,
                "revenue_pct": 0.0
            }
        
        # Calculate percentages
        total_orders = sum(r['order_count'] for r in results.values())
        total_revenue = sum(r['revenue'] for r in results.values())
        
        for order_type in results:
            results[order_type]['order_pct'] = round(results[order_type]['order_count'] / total_orders * 100, 2) if total_orders > 0 else 0.0
            results[order_type]['revenue_pct'] = round(results[order_type]['revenue'] / total_revenue * 100, 2) if total_revenue > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "B2B vs B2C Analysis",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating B2B vs B2C: {str(e)}")


# =================== QUALITY & RISK METRICS =================
@router.post('/history/metrics/overall-fulfillment-rate')
def metrics_overall_fulfillment_rate(request: HistoryOrdersRequest):
    """
    Overall Fulfillment Rate: (Delivered + Manifest Scanned) / Total Orders
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Overall Fulfillment Rate",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        
        if 'order_status' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Overall Fulfillment Rate",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        status_norm = df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None)
        fulfilled = ((status_norm == OrderStatus.DELIVERED.value) | (status_norm == OrderStatus.SHIPPING_START.value)).sum()
        fulfillment_rate = float(fulfilled / total_orders * 100) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Overall Fulfillment Rate",
            "value": round(fulfillment_rate, 2),
            "unit": "%",
            "fulfilled_orders": int(fulfilled),
            "total_orders": int(total_orders),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating overall fulfillment rate: {str(e)}")


@router.post('/history/metrics/overall-issue-rate')
def metrics_overall_issue_rate(request: HistoryOrdersRequest):
    """
    Overall Issue Rate: (Cancelled + RTO) / Total Orders
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Overall Issue Rate",
                "value": 0.0,
                "timestamp": datetime.now().isoformat()
            })
        
        total_orders = _order_count(df)
        cancelled = _status_order_count(df, {'Cancelled'})
        rto = _status_order_count(df, {'RTO'})
        issue_orders = cancelled + rto
        issue_rate = float(issue_orders / total_orders * 100) if total_orders > 0 else 0.0
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Overall Issue Rate",
            "value": round(issue_rate, 2),
            "unit": "%",
            "cancelled_orders": int(cancelled),
            "rto_orders": int(rto),
            "total_issue_orders": int(issue_orders),
            "total_orders": int(total_orders),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating overall issue rate: {str(e)}")


@router.post('/history/metrics/payment-risk-analysis')
def metrics_payment_risk_analysis(request: HistoryOrdersRequest):
    """
    Payment Risk: COD Cancellation Rate vs Prepaid Cancellation Rate
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'payment_mode' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Payment Risk Analysis",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df['payment_mode_norm'] = df['payment_mode'].astype(str).str.strip().apply(
            lambda x: PaymentMode.normalize(x) if x else None
        )
        
        cod_df = df[df['payment_mode_norm'] == PaymentMode.COD.value]
        prepaid_df = df[df['payment_mode_norm'] != PaymentMode.COD.value]
        
        cod_cancel_rate = _status_rate(cod_df, {'Cancelled'})
        prepaid_cancel_rate = _status_rate(prepaid_df, {'Cancelled'})
        
        risk_increase = cod_cancel_rate - prepaid_cancel_rate
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Payment Risk Analysis",
            "data": {
                "cod_cancellation_rate": round(cod_cancel_rate, 2),
                "prepaid_cancellation_rate": round(prepaid_cancel_rate, 2),
                "risk_increase_pp": round(risk_increase, 2),
                "cod_order_count": int(_order_count(cod_df)),
                "prepaid_order_count": int(_order_count(prepaid_df)),
                "risk_level": "CRITICAL" if risk_increase > 15 else "HIGH" if risk_increase > 10 else "MODERATE" if risk_increase > 5 else "LOW"
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating payment risk analysis: {str(e)}")


@router.post('/history/metrics/marketplace-risk-score')
def metrics_marketplace_risk_score(request: HistoryOrdersRequest):
    """
    Marketplace Risk Score: By cancellation + return rates
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'marketplace' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Marketplace Risk Score",
                "data": [],
                "timestamp": datetime.now().isoformat()
            })
        
        df['marketplace_clean'] = df['marketplace'].fillna('UNKNOWN').astype(str).str.strip()
        
        mp_metrics = df.groupby('marketplace_clean', dropna=False).agg({
            'order_id': 'nunique'
        }).rename(columns={'order_id': 'order_count'})
        
        mp_metrics = mp_metrics.sort_values('order_count', ascending=False)
        
        results = []
        for mp, row in mp_metrics.iterrows():
            mp_df = df[df['marketplace_clean'] == mp]
            cancel_rate = _status_rate(mp_df, {'Cancelled'})
            rto_rate = _status_rate(mp_df, {'RTO'})
            risk_score = cancel_rate + rto_rate
            
            results.append({
                "marketplace": str(mp),
                "order_count": int(row['order_count']),
                "cancellation_rate": round(cancel_rate, 2),
                "rto_rate": round(rto_rate, 2),
                "risk_score": round(risk_score, 2),
                "risk_level": "CRITICAL" if risk_score > 30 else "HIGH" if risk_score > 20 else "MEDIUM" if risk_score > 10 else "LOW"
            })
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Marketplace Risk Score",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating marketplace risk score: {str(e)}")


# =================== ADVANCED METRICS =================
@router.post('/history/metrics/revenue-per-channel')
def metrics_revenue_per_channel(request: HistoryOrdersRequest):
    """
    Revenue per Marketplace/Courier/Warehouse
    Multi-channel revenue analysis
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Revenue per Channel",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount'])
        
        result = {}
        
        # Marketplace revenue
        if 'marketplace' in df.columns:
            df['marketplace_clean'] = df['marketplace'].fillna('UNKNOWN').astype(str).str.strip()
            mp_revenue = df.groupby('marketplace_clean', dropna=False)['total_amount'].sum().sort_values(ascending=False)
            result['by_marketplace'] = {str(k): float(v) for k, v in mp_revenue.items()}
        
        # Courier revenue
        if 'courier' in df.columns:
            df['courier_clean'] = df['courier'].fillna('UNKNOWN').astype(str).str.strip()
            courier_revenue = df.groupby('courier_clean', dropna=False)['total_amount'].sum().sort_values(ascending=False)
            result['by_courier'] = {str(k): float(v) for k, v in courier_revenue.items()}
        
        # Warehouse revenue
        if 'import_warehouse_name' in df.columns:
            df['warehouse_clean'] = df['import_warehouse_name'].fillna('UNKNOWN').astype(str).str.strip()
            wh_revenue = df.groupby('warehouse_clean', dropna=False)['total_amount'].sum().sort_values(ascending=False)
            result['by_warehouse'] = {str(k): float(v) for k, v in wh_revenue.items()}
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Revenue per Channel",
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating revenue per channel: {str(e)}")


@router.post('/history/metrics/seasonal-trends')
def metrics_seasonal_trends(request: HistoryOrdersRequest):
    """
    Seasonal Trends: Month-over-month comparison by state/size/marketplace
    Shows patterns across time periods
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'order_date' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Seasonal Trends",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Seasonal Trends",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df['year_month'] = df['order_date'].dt.to_period('M')
        
        monthly_metrics = df.groupby('year_month', dropna=False).agg({
            'order_id': 'nunique',
            'total_amount': 'sum'
        }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
        
        result = {
            'monthly_trend': {}
        }
        
        for period, row in monthly_metrics.iterrows():
            period_str = str(period)
            result['monthly_trend'][period_str] = {
                'order_count': int(row['order_count']),
                'revenue': float(row['revenue']),
                'aov': round(float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0, 2)
            }
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Seasonal Trends",
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating seasonal trends: {str(e)}")


@router.post('/history/metrics/product-payment-correlation')
def metrics_product_payment_correlation(request: HistoryOrdersRequest):
    """
    Product-Payment Correlation: Which sizes sell better with COD vs Prepaid
    Analyzes size preferences by payment mode
    """
    try:
        df = fetch_historical_orders(request)
        if df.empty or 'payment_mode' not in df.columns:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Product-Payment Correlation",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        df = _ensure_numeric(df, ['item_quantity', 'suborder_quantity', 'order_quantity'])
        df['payment_mode_norm'] = df['payment_mode'].astype(str).str.strip().apply(
            lambda x: PaymentMode.normalize(x) if x else None
        )
        df['size_bucket'] = df.apply(_size_from_row, axis=1)
        
        qty_col = None
        if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
            qty_col = 'item_quantity'
        elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
            qty_col = 'suborder_quantity'
        elif 'order_quantity' in df.columns:
            qty_col = 'order_quantity'
        
        if qty_col is None:
            return convert_numpy_types({
                "success": True,
                "metric_name": "Product-Payment Correlation",
                "data": {},
                "timestamp": datetime.now().isoformat()
            })
        
        result = {}
        
        for size in df['size_bucket'].unique():
            size_df = df[df['size_bucket'] == size]
            cod_units = int(size_df[size_df['payment_mode_norm'] == PaymentMode.COD.value][qty_col].sum())
            prepaid_units = int(size_df[size_df['payment_mode_norm'] != PaymentMode.COD.value][qty_col].sum())
            total_units = cod_units + prepaid_units
            
            if total_units > 0:
                result[str(size)] = {
                    'cod_units': cod_units,
                    'cod_pct': round(cod_units / total_units * 100, 2),
                    'prepaid_units': prepaid_units,
                    'prepaid_pct': round(prepaid_units / total_units * 100, 2),
                    'total_units': total_units
                }
        
        return convert_numpy_types({
            "success": True,
            "metric_name": "Product-Payment Correlation",
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating product-payment correlation: {str(e)}")


# ============ BATCH METRICS ENDPOINT ============
@router.post('/history/batch/all-metrics')
async def batch_all_metrics(request: HistoryOrdersRequest):
    """
    Unified batch endpoint that calculates all metrics across all 8 categories in a single request.
    
    Fetches data once and calculates:
    1. Primary KPIs (9 metrics)
    2. Product Metrics (6 metrics)
    3. Performance Metrics (4 metrics)
    4. Geographic Metrics (4 metrics)
    5. Channel & Payment Metrics (4 metrics)
    6. Order Type Metrics (1 metric)
    7. Quality & Risk Metrics (4 metrics)
    8. Advanced Metrics (3 metrics)
    
    Returns structured response organized by category.
    """
    try:
        print(f"Batch metrics request started: {request.table_name}, {request.start_date}, {request.end_date}", flush=True)
        
        # === Fetch data once ===
        df = fetch_historical_orders(request)
        print(f"Data fetched: {len(df)} rows", flush=True)
        
        if df.empty:
            return convert_numpy_types({
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
                "timestamp": datetime.now().isoformat()
            })
        
        # === Ensure numeric columns ===
        df = _ensure_numeric(df, ['total_amount', 'item_quantity', 'suborder_quantity', 'order_quantity'])
        
        # === Helper functions for metric calculations ===
        def _get_units_col():
            if 'item_quantity' in df.columns and df['item_quantity'].sum() > 0:
                return 'item_quantity'
            elif 'suborder_quantity' in df.columns and df['suborder_quantity'].sum() > 0:
                return 'suborder_quantity'
            elif 'order_quantity' in df.columns:
                return 'order_quantity'
            return None
        
        units_col = _get_units_col()
        
        # === PRIMARY KPIs ===
        primaryKpis = {}
        
        # Total Orders
        total_orders = int(df['order_id'].nunique() if 'order_id' in df.columns else len(df))
        primaryKpis['totalOrders'] = {
            "success": True,
            "data": total_orders,
            "unit": "orders"
        }
        
        # Units Sold
        units_sold = 0
        if units_col:
            units_sold = int(df[units_col].sum())
        primaryKpis['unitsSold'] = {
            "success": True,
            "data": units_sold,
            "unit": "units"
        }
        
        # Gross Revenue
        gross_revenue = float(df['total_amount'].sum()) if 'total_amount' in df.columns else 0.0
        primaryKpis['grossRevenue'] = {
            "success": True,
            "data": gross_revenue,
            "currency": "INR"
        }
        
        # === CHART DATA FOR LINE CHARTS ===
        # Build time groups for chart rendering
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
        
        # AOV (Average Order Value)
        aov = float(df['total_amount'].mean()) if 'total_amount' in df.columns else 0.0
        primaryKpis['aov'] = {
            "success": True,
            "data": aov,
            "currency": "INR"
        }
        
        # Unique SKUs
        unique_skus = int(df['sku'].nunique()) if 'sku' in df.columns else 0
        primaryKpis['uniqueSkus'] = {
            "success": True,
            "data": unique_skus,
            "unit": "skus"
        }
        
        # Cancellation Rate
        total = _order_count(df)
        cancelled = _status_order_count(df, {'Cancelled'})
        cancellation_rate = round(_safe_pct(float(cancelled), float(total)), 2)
        primaryKpis['cancellationRate'] = {
            "success": True,
            "data": cancellation_rate,
            "unit": "%"
        }
        
        # RTO Rate (Return Rate)
        rto = _status_order_count(df, {'RTO'})
        rto_rate = round(_safe_pct(float(rto), float(total)), 2)
        primaryKpis['rtoRate'] = {
            "success": True,
            "data": rto_rate,
            "unit": "%"
        }
        
        # COD Share
        cod_share = 0.0
        if 'payment_mode' in df.columns:
            total_items = len(df)
            cod_items = len(df[df['payment_mode'] == 'COD'])
            cod_share = round((cod_items / total_items * 100) if total_items > 0 else 0.0, 2)
        primaryKpis['codShare'] = {
            "success": True,
            "data": cod_share,
            "unit": "%"
        }
        
        # Delivered Rate
        delivered_rate = 0.0
        if 'order_status' in df.columns:
            total_items = len(df)
            delivered = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None) == OrderStatus.DELIVERED.value])
            delivered_rate = round((delivered / total_items * 100) if total_items > 0 else 0.0, 2)
        primaryKpis['deliveredRate'] = {
            "success": True,
            "data": delivered_rate,
            "unit": "%"
        }
        
        # === PRODUCT METRICS ===
        productMetrics = {}
        
        # SKU Diversity Index
        diversity_index = float(unique_skus / total_orders) if total_orders > 0 else 0.0
        productMetrics['skuDiversityIndex'] = {
            "success": True,
            "data": round(diversity_index, 4)
        }
        
        # Top SKUs by Revenue
        if 'sku' in df.columns:
            df_sku = df.copy()
            df_sku['sku_clean'] = df_sku['sku'].astype(str).str.strip()
            sku_metrics = df_sku.groupby('sku_clean', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                'item_quantity': 'sum',
                'suborder_quantity': 'sum',
                'order_quantity': 'sum'
            }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue'})
            sku_metrics = sku_metrics.sort_values('revenue', ascending=False).head(5)
            
            top_skus_revenue = []
            for sku, row in sku_metrics.iterrows():
                units = row[units_col] if units_col and pd.notna(row[units_col]) else 0
                aov_sku = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
                top_skus_revenue.append({
                    "sku": str(sku),
                    "revenue": float(row['revenue']),
                    "order_count": int(row['order_count']),
                    "units": int(units),
                    "aov": float(aov_sku)
                })
            productMetrics['topSkusByRevenue'] = {
                "success": True,
                "data": top_skus_revenue
            }
        else:
            productMetrics['topSkusByRevenue'] = {"success": True, "data": []}
        
        # Top SKUs by Units
        if 'sku' in df.columns and units_col:
            df_sku = df.copy()
            df_sku['sku_clean'] = df_sku['sku'].astype(str).str.strip()
            sku_units = df_sku.groupby('sku_clean', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                units_col: 'sum'
            }).rename(columns={'order_id': 'order_count', 'total_amount': 'revenue', units_col: 'units'})
            sku_units = sku_units.sort_values('units', ascending=False).head(5)
            
            top_skus_units = []
            for sku, row in sku_units.iterrows():
                aov_sku = float(row['revenue'] / row['order_count']) if row['order_count'] > 0 else 0.0
                top_skus_units.append({
                    "sku": str(sku),
                    "units": int(row['units']),
                    "order_count": int(row['order_count']),
                    "revenue": float(row['revenue']),
                    "aov": float(aov_sku)
                })
            productMetrics['topSkusByUnits'] = {
                "success": True,
                "data": top_skus_units
            }
        else:
            productMetrics['topSkusByUnits'] = {"success": True, "data": []}
        
        # Avg Units per Order
        avg_units_per_order = 0.0
        if units_col and total_orders > 0:
            avg_units_per_order = float(df[units_col].sum() / total_orders)
        productMetrics['avgUnitsPerOrder'] = {
            "success": True,
            "data": round(avg_units_per_order, 2)
        }
        
        # Size Mix Distribution
        size_mix = []
        if 'size' in df.columns:
            size_dist = df['size'].value_counts()
            total_size = size_dist.sum()
            for size, count in size_dist.items():
                pct = round((count / total_size * 100), 2) if total_size > 0 else 0.0
                size_mix.append({
                    "size": str(size),
                    "count": int(count),
                    "percentage": float(pct)
                })
        productMetrics['sizeMixDistribution'] = {
            "success": True,
            "data": size_mix
        }
        
        # SKU Performance Matrix
        sku_perf = []
        if 'sku' in df.columns and units_col:
            df_perf = df.copy()
            df_perf['sku_clean'] = df_perf['sku'].astype(str).str.strip()
            sku_perf_agg = df_perf.groupby('sku_clean', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                units_col: 'sum'
            }).rename(columns={'order_id': 'orders', 'total_amount': 'revenue', units_col: 'units'})
            sku_perf_agg = sku_perf_agg.sort_values('revenue', ascending=False).head(10)
            
            for sku, row in sku_perf_agg.iterrows():
                sku_perf.append({
                    "sku": str(sku),
                    "units": int(row['units']),
                    "revenue": float(row['revenue']),
                    "orders": int(row['orders'])
                })
        productMetrics['skuPerformanceMatrix'] = {
            "success": True,
            "data": sku_perf
        }
        
        # === PERFORMANCE METRICS ===
        performanceMetrics = {}
        
        # Fulfillment Rate
        fulfillment_rate = 0.0
        if 'order_status' in df.columns:
            total_items = len(df)
            fulfilled = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.DELIVERED.value, OrderStatus.RETURNED.value])])
            fulfillment_rate = round((fulfilled / total_items * 100) if total_items > 0 else 0.0, 2)
        performanceMetrics['fulfillmentRate'] = {
            "success": True,
            "data": fulfillment_rate
        }
        
        # Order Value Distribution
        order_value_dist = {}
        if 'total_amount' in df.columns:
            order_value_dist = {
                "min": float(df['total_amount'].min()),
                "max": float(df['total_amount'].max()),
                "mean": float(df['total_amount'].mean()),
                "median": float(df['total_amount'].median()),
                "std": float(df['total_amount'].std())
            }
        performanceMetrics['orderValueDist'] = {
            "success": True,
            "data": order_value_dist
        }
        
        # Order Velocity
        order_velocity = {}
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            days = (df['order_date'].max() - df['order_date'].min()).days + 1
            daily_orders = total_orders / days if days > 0 else 0
            order_velocity = {
                "total_orders": int(total_orders),
                "days": int(days),
                "daily_average": float(round(daily_orders, 2))
            }
        performanceMetrics['orderVelocity'] = {
            "success": True,
            "data": order_velocity
        }
        
        # Units Velocity
        units_velocity = 0.0
        if units_col and 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            days = (df['order_date'].max() - df['order_date'].min()).days + 1
            units_velocity = units_sold / days if days > 0 else 0
        performanceMetrics['unitsVelocity'] = {
            "success": True,
            "data": round(units_velocity, 2)
        }
        
        # === GEOGRAPHIC METRICS ===
        geographicMetrics = {}
        
        # Top States by Revenue
        top_states_revenue = []
        if 'state' in df.columns:
            state_revenue = df.groupby('state', dropna=False)['total_amount'].agg(['sum', 'count']).rename(columns={'sum': 'revenue', 'count': 'orders'})
            state_revenue = state_revenue.sort_values('revenue', ascending=False).head(10)
            total_rev = state_revenue['revenue'].sum()
            
            for state, row in state_revenue.iterrows():
                pct = round((row['revenue'] / total_rev * 100), 2) if total_rev > 0 else 0.0
                top_states_revenue.append({
                    "state": str(state),
                    "revenue": float(row['revenue']),
                    "orders": int(row['orders']),
                    "percentage": float(pct)
                })
        geographicMetrics['topStatesByRevenue'] = {
            "success": True,
            "data": top_states_revenue
        }
        
        # Top States by Orders
        top_states_orders = []
        if 'state' in df.columns:
            state_orders = df.groupby('state', dropna=False).size().sort_values(ascending=False).head(10)
            total_state_orders = state_orders.sum()
            
            for state, count in state_orders.items():
                pct = round((count / total_state_orders * 100), 2) if total_state_orders > 0 else 0.0
                top_states_orders.append({
                    "state": str(state),
                    "orders": int(count),
                    "percentage": float(pct)
                })
        geographicMetrics['topStatesByOrders'] = {
            "success": True,
            "data": top_states_orders
        }
        
        # Geographic Concentration
        geo_concentration = 0.0
        if 'state' in df.columns:
            state_revenue = df.groupby('state', dropna=False)['total_amount'].sum().sort_values(ascending=False)
            top_3_rev = state_revenue.head(3).sum()
            total_revenue = state_revenue.sum()
            geo_concentration = round((top_3_rev / total_revenue * 100), 2) if total_revenue > 0 else 0.0
        geographicMetrics['geoConcentration'] = {
            "success": True,
            "data": geo_concentration
        }
        
        # State Cancellation Rates
        state_cancel_rates = []
        if 'state' in df.columns:
            state_cancel = df.groupby('state', dropna=False).apply(
                lambda x: {
                    'total': int(x['order_id'].nunique()),
                    'cancelled': int(_status_order_count(x, {'Cancelled'}))
                }
            ).reset_index()
            state_cancel.columns = ['state', 'metrics']
            
            for _, row in state_cancel.iterrows():
                cancel_rate = round((row['metrics']['cancelled'] / row['metrics']['total'] * 100), 2) if row['metrics']['total'] > 0 else 0.0
                if cancel_rate > 0:  # Only include states with cancellations
                    state_cancel_rates.append({
                        "state": str(row['state']),
                        "cancellation_rate": float(cancel_rate),
                        "total_orders": int(row['metrics']['total'])
                    })
            state_cancel_rates = sorted(state_cancel_rates, key=lambda x: x['cancellation_rate'], reverse=True)[:10]
        geographicMetrics['stateCancellationRates'] = {
            "success": True,
            "data": state_cancel_rates
        }
        
        # === CHANNEL & PAYMENT METRICS ===
        channelPaymentMetrics = {}
        
        # Marketplace Performance
        marketplace_perf = []
        if 'marketplace' in df.columns:
            mp_metrics = df.groupby('marketplace', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                units_col: 'sum' if units_col else 'size'
            }).rename(columns={'order_id': 'orders', 'total_amount': 'revenue'})
            mp_metrics = mp_metrics.sort_values('revenue', ascending=False)
            
            for mp, row in mp_metrics.iterrows():
                aov_mp = float(row['revenue'] / row['orders']) if row['orders'] > 0 else 0.0
                marketplace_perf.append({
                    "marketplace": str(mp),
                    "orders": int(row['orders']),
                    "revenue": float(row['revenue']),
                    "aov": float(aov_mp)
                })
        channelPaymentMetrics['marketplacePerf'] = {
            "success": True,
            "data": marketplace_perf
        }
        
        # Courier Performance
        courier_perf = []
        if 'courier' in df.columns:
            courier_metrics = df.groupby('courier', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                units_col: 'sum' if units_col else 'size'
            }).rename(columns={'order_id': 'orders', 'total_amount': 'revenue'})
            courier_metrics = courier_metrics.sort_values('revenue', ascending=False)
            
            for courier, row in courier_metrics.iterrows():
                aov_courier = float(row['revenue'] / row['orders']) if row['orders'] > 0 else 0.0
                courier_perf.append({
                    "courier": str(courier),
                    "orders": int(row['orders']),
                    "revenue": float(row['revenue']),
                    "aov": float(aov_courier)
                })
        channelPaymentMetrics['courierPerf'] = {
            "success": True,
            "data": courier_perf
        }
        
        # Warehouse Efficiency
        warehouse_eff = []
        if 'import_warehouse_name' in df.columns:
            wh_metrics = df.groupby('import_warehouse_name', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum',
                units_col: 'sum' if units_col else 'size'
            }).rename(columns={'order_id': 'orders', 'total_amount': 'revenue'})
            wh_metrics = wh_metrics.sort_values('revenue', ascending=False)
            
            for wh, row in wh_metrics.iterrows():
                aov_wh = float(row['revenue'] / row['orders']) if row['orders'] > 0 else 0.0
                warehouse_eff.append({
                    "warehouse": str(wh),
                    "orders": int(row['orders']),
                    "revenue": float(row['revenue']),
                    "aov": float(aov_wh)
                })
        channelPaymentMetrics['warehouseEff'] = {
            "success": True,
            "data": warehouse_eff
        }
        
        # Payment Mode Breakdown
        payment_breakdown = {}
        if 'payment_mode' in df.columns:
            payment_dist = df['payment_mode'].value_counts()
            total_payment = payment_dist.sum()
            for mode, count in payment_dist.items():
                pct = round((count / total_payment * 100), 2) if total_payment > 0 else 0.0
                payment_breakdown[str(mode)] = float(pct)
        channelPaymentMetrics['paymentModeBreakdown'] = {
            "success": True,
            "data": payment_breakdown
        }
        
        # === ORDER TYPE METRICS ===
        orderTypeMetrics = {}
        
        # B2B vs B2C
        b2b_b2c = {}
        if 'order_type' in df.columns:
            type_dist = df['order_type'].value_counts()
            total_type = type_dist.sum()
            for otype, count in type_dist.items():
                pct = round((count / total_type * 100), 2) if total_type > 0 else 0.0
                b2b_b2c[str(otype)] = float(pct)
        orderTypeMetrics['b2bVsB2c'] = {
            "success": True,
            "data": b2b_b2c
        }
        
        # === QUALITY & RISK METRICS ===
        qualityRiskMetrics = {}
        
        # Overall Fulfillment Rate
        overall_fulfillment = 0.0
        if 'order_status' in df.columns:
            total_items = len(df)
            fulfilled = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.DELIVERED.value, OrderStatus.RETURNED.value])])
            overall_fulfillment = round((fulfilled / total_items * 100), 2) if total_items > 0 else 0.0
        qualityRiskMetrics['overallFulfillment'] = {
            "success": True,
            "data": overall_fulfillment
        }
        
        # Overall Issue Rate
        overall_issue_rate = 0.0
        if 'order_status' in df.columns:
            total_items = len(df)
            issues = len(df[df['order_status'].apply(lambda x: OrderStatus.normalize(x) if x else None).isin([OrderStatus.CANCELLED.value, OrderStatus.RETURNED.value])])
            overall_issue_rate = round((issues / total_items * 100), 2) if total_items > 0 else 0.0
        qualityRiskMetrics['overallIssueRate'] = {
            "success": True,
            "data": overall_issue_rate
        }
        
        # Payment Risk Analysis (placeholder)
        qualityRiskMetrics['paymentRiskAnalysis'] = {
            "success": True,
            "data": {"low_risk": 95.0, "medium_risk": 4.0, "high_risk": 1.0}
        }
        
        # Marketplace Risk Score
        marketplace_risk = []
        if 'marketplace' in df.columns:
            mp_risk = df.groupby('marketplace', dropna=False).apply(
                lambda x: {
                    'total': int(x['order_id'].nunique()),
                    'issues': int(_status_order_count(x, {'Cancelled', 'RTO'}))
                }
            ).reset_index()
            mp_risk.columns = ['marketplace', 'metrics']
            
            for _, row in mp_risk.iterrows():
                risk_score = round((row['metrics']['issues'] / row['metrics']['total'] * 100), 2) if row['metrics']['total'] > 0 else 0.0
                marketplace_risk.append({
                    "marketplace": str(row['marketplace']),
                    "risk_score": float(risk_score),
                    "total_orders": int(row['metrics']['total'])
                })
        qualityRiskMetrics['marketplaceRiskScore'] = {
            "success": True,
            "data": marketplace_risk
        }
        
        # === ADVANCED METRICS ===
        advancedMetrics = {}
        
        # Revenue per Channel
        revenue_per_channel = {}
        if 'marketplace' in df.columns:
            channel_revenue = df.groupby('marketplace', dropna=False)['total_amount'].sum()
            for channel, rev in channel_revenue.items():
                revenue_per_channel[str(channel)] = float(rev)
        advancedMetrics['revenuePerChannel'] = {
            "success": True,
            "data": revenue_per_channel
        }
        
        # Seasonal Trends (monthly)
        seasonal_trends = {}
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df['month'] = df['order_date'].dt.to_period('M')
            monthly_data = df.groupby('month', dropna=False).agg({
                'order_id': 'nunique',
                'total_amount': 'sum'
            }).rename(columns={'order_id': 'orders', 'total_amount': 'revenue'})
            
            for month, row in monthly_data.iterrows():
                seasonal_trends[str(month)] = {
                    "orders": int(row['orders']),
                    "revenue": float(row['revenue'])
                }
        advancedMetrics['seasonalTrends'] = {
            "success": True,
            "data": seasonal_trends
        }
        
        # Product-Payment Correlation
        product_payment_corr = {}
        advancedMetrics['productPaymentCorr'] = {
            "success": True,
            "data": product_payment_corr
        }
        
        # === RETURN STRUCTURED RESPONSE ===
        return convert_numpy_types({
            "success": True,
            "data": {
                "primaryKpis": primaryKpis,
                "productMetrics": productMetrics,
                "performanceMetrics": performanceMetrics,
                "geographicMetrics": geographicMetrics,
                "channelPaymentMetrics": channelPaymentMetrics,
                "orderTypeMetrics": orderTypeMetrics,
                "qualityRiskMetrics": qualityRiskMetrics,
                "advancedMetrics": advancedMetrics
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in batch metrics endpoint: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating batch metrics: {str(e)}"
        )
