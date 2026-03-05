"""
Tool functions for fetching and manipulating data
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import json
import boto3

s3 = boto3.client('s3')

def get_all_orders(start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch orders from EasyEcom API with date windowing support
    
    IMPORTANT: This function ONLY accepts date range parameters.
    Do NOT pass filtering parameters like payment_mode, marketplace, etc.
    Those filters should be applied AFTER fetching using apply_filters().
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_date: End date in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        List of order dictionaries (unfiltered)
    
    Example workflow for filtered data:
        1. orders = get_all_orders(start_date, end_date)  # Fetch all orders
        2. filtered = apply_filters(orders, [{"field": "payment_mode", "operator": "eq", "value": "PrePaid"}])
    """
    api_key = os.getenv("EASYECOM_API_KEY")
    jwt_token = os.getenv("EASYECOM_JWT_TOKEN")
    base_url = os.getenv("EASYECOM_BASE_URL", "https://api.easyecom.io")
    
    if not api_key or not jwt_token:
        raise ValueError("EASYECOM_API_KEY and EASYECOM_JWT_TOKEN must be set in .env")
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    
    # Calculate days difference
    days_diff = (end_dt - start_dt).days
    
    print(f"Date range: {start_date} to {end_date} (Days: {days_diff})")
    
    all_orders = []
    
    # If more than 7 days, implement windowing
    if days_diff > 7:
        print(f"Implementing windowing for {days_diff} days (> 7 days)")
        current_start = start_dt
        window_num = 1
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=7), end_dt)
            
            print(f"Fetching window {window_num}: {current_start} to {current_end}")
            
            # Fetch window with pagination
            window_orders = _fetch_orders_window(
                current_start.strftime("%Y-%m-%d %H:%M:%S"),
                current_end.strftime("%Y-%m-%d %H:%M:%S"),
                api_key,
                jwt_token,
                base_url
            )
            all_orders.extend(window_orders)
            print(f"Window {window_num} completed: {len(window_orders)} orders")
            
            # Move to next window
            current_start = current_end
            window_num += 1
    else:
        # Single fetch for <= 7 days with pagination
        print(f"Fetching all orders for single window ({days_diff} days <= 7)")
        all_orders = _fetch_orders_window(start_date, end_date, api_key, jwt_token, base_url)
    
    print(f"Total orders fetched across all windows/pages: {len(all_orders)}")
    return all_orders

#for any metric, data calculation, first convert to dataframe and then continue.
def convert_to_df(raw: list) -> pd.DataFrame:
    """Convert raw JSON order data to normalized DataFrame"""
    
    #raw = PYTHON LIST

    orders = []
    try:
        # If raw is a JSON string, parse it first
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = raw

        # If parsed is a dict and contains top-level 'data'
        if isinstance(parsed, dict) and 'data' in parsed:
            data_block = parsed.get('data')

            # Common API shape: {'data': {'orders': [...]}}
            if isinstance(data_block, dict) and 'orders' in data_block:
                orders = data_block.get('orders') or []
            # Sometimes 'data' itself is the list of orders
            elif isinstance(data_block, list):
                orders = data_block
            # If 'data' is a single order dict, wrap it
            elif isinstance(data_block, dict):
                orders = [data_block]
            else:
                orders = []

        # If parsed is a dict with 'orders' key at top-level
        elif isinstance(parsed, dict) and 'orders' in parsed:
            orders = parsed.get('orders') or []

        # If parsed is already a list of orders
        elif isinstance(parsed, list):
            orders = parsed

        # If parsed is a single order dict, wrap into list
        elif isinstance(parsed, dict):
            orders = [parsed]

        else:
            raise ValueError(f"Unsupported raw data type: {type(raw)}. Expected JSON string, dict, or list of orders.")

    except Exception as e:
        print(f"Error parsing raw input in convert_to_df: {e}")
        raise

    # Ensure orders is a list
    if orders is None:
        orders = []

    df = pd.json_normalize(orders)

    print("========================")
    print("normalized dataframe: ")
    print(df.head(5), flush=True)
    df.to_csv("normalized_db.csv", index=False)
    print("========================")

    return df


def get_aov(table: pd.DataFrame) -> float:
    """Calculate Average Order Value from orders DataFrame"""

    print("----------------", flush=True)
    print("table in get_aov:", flush=True)
    print(table.head(10), flush=True)
    print("----------------", flush=True)

    try:
        if table.empty:
            return 0.0
        aov = table['total_amount'].astype(float).mean()
        return round(aov, 2) if not pd.isna(aov) else 0.0
    except Exception as e:
        print(f"Error in calculating AOV: {e}")
        return None


def get_total_revenue(table: pd.DataFrame) -> float:
    """Calculate total revenue from orders DataFrame"""
    try:
        if table.empty:
            return 0.0
        revenue = table['total_amount'].astype(float).sum()
        return round(revenue, 2) if not pd.isna(revenue) else 0.0
    except Exception as e:
        print(f"Error in calculating total revenue: {e}")
        return None


def get_order_count(table: pd.DataFrame) -> int:
    """Get total number of orders"""
    try:
        return len(table)
    except Exception as e:
        print(f"Error in calculating order count: {e}")
        return None


def get_order_status_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of order statuses"""
    try:
        if table.empty or 'order_status' not in table.columns:
            return {}
        distribution = table['order_status'].value_counts().to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating order status distribution: {e}")
        return None


def get_payment_mode_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of payment modes"""
    try:
        if table.empty or 'payment_mode' not in table.columns:
            return {}
        distribution = table['payment_mode'].value_counts().to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating payment mode distribution: {e}")
        return None


def get_marketplace_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by marketplace"""
    try:
        if table.empty or 'marketplace' not in table.columns:
            return {}
        distribution = table['marketplace'].value_counts().to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating marketplace distribution: {e}")
        return None


def get_state_wise_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by state"""
    try:
        if table.empty or 'state' not in table.columns:
            return {}
        distribution = table['state'].value_counts().to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating state distribution: {e}")
        return None


def get_city_wise_distribution(table: pd.DataFrame, top_n: int = 10) -> dict:
    """Get distribution of orders by city (top N cities)"""
    try:
        if table.empty or 'city' not in table.columns:
            return {}
        distribution = table['city'].value_counts().head(top_n).to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating city distribution: {e}")
        return None


def get_courier_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by courier service"""
    try:
        if table.empty or 'courier' not in table.columns:
            return {}
        distribution = table['courier'].value_counts().to_dict()
        return distribution
    except Exception as e:
        print(f"Error in calculating courier distribution: {e}")
        return None


def get_average_discount(table: pd.DataFrame) -> float:
    """Calculate average discount amount"""
    try:
        if table.empty:
            return 0.0
        avg_discount = table['total_discount'].astype(float).mean()
        return round(avg_discount, 2) if not pd.isna(avg_discount) else 0.0
    except Exception as e:
        print(f"Error in calculating average discount: {e}")
        return None


def get_average_shipping_charge(table: pd.DataFrame) -> float:
    """Calculate average shipping charge"""
    try:
        if table.empty:
            return 0.0
        avg_shipping = table['total_shipping_charge'].astype(float).mean()
        return round(avg_shipping, 2) if not pd.isna(avg_shipping) else 0.0
    except Exception as e:
        print(f"Error in calculating average shipping charge: {e}")
        return None


def get_average_tax(table: pd.DataFrame) -> float:
    """Calculate average tax amount"""
    try:
        if table.empty:
            return 0.0
        avg_tax = table['total_tax'].astype(float).mean()
        return round(avg_tax, 2) if not pd.isna(avg_tax) else 0.0
    except Exception as e:
        print(f"Error in calculating average tax: {e}")
        return None


# Statistical Tools
def get_statistical_summary(table: pd.DataFrame, field: str) -> dict:
    """Get comprehensive statistical summary for a numeric field"""
    try:
        if table.empty or field not in table.columns:
            return {
                'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0
            }
        
        series = pd.to_numeric(table[field], errors='coerce')
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0
            }
        
        stats = {
            'count': len(clean_series),
            'mean': round(clean_series.mean(), 2),
            'median': round(clean_series.median(), 2),
            'std': round(clean_series.std(), 2),
            'min': round(clean_series.min(), 2),
            'max': round(clean_series.max(), 2),
            'q25': round(clean_series.quantile(0.25), 2),
            'q75': round(clean_series.quantile(0.75), 2)
        }
        return stats
    except Exception as e:
        print(f"Error in calculating statistical summary for {field}: {e}")
        return None


def get_percentile(table: pd.DataFrame, field: str, percentile: float) -> float:
    """Get specific percentile for a numeric field"""
    try:
        if table.empty or field not in table.columns:
            return 0.0
        
        series = pd.to_numeric(table[field], errors='coerce')
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 0.0
        
        result = clean_series.quantile(percentile / 100)
        return round(result, 2) if not pd.isna(result) else 0.0
    except Exception as e:
        print(f"Error in calculating {percentile}th percentile for {field}: {e}")
        return None


def get_top_percentile(table: pd.DataFrame, field: str, percentile: float = 95) -> dict:
    """Get records in top percentile for a field"""
    try:
        if table.empty or field not in table.columns:
            return {
                'threshold': 0.0, 'count': 0, 'percentage': 0.0, 'total_value': 0.0
            }
        
        series = pd.to_numeric(table[field], errors='coerce')
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'threshold': 0.0, 'count': 0, 'percentage': 0.0, 'total_value': 0.0
            }
        
        threshold = clean_series.quantile(percentile / 100)
        # Use boolean indexing safely
        mask = series >= threshold
        top_records = table[mask]
        
        if top_records.empty:
            return {
                'threshold': round(threshold, 2),
                'count': 0,
                'percentage': 0.0,
                'total_value': 0.0
            }
        
        return {
            'threshold': round(threshold, 2),
            'count': len(top_records),
            'percentage': round(len(top_records) / len(table) * 100, 2),
            'total_value': round(pd.to_numeric(top_records[field], errors='coerce').sum(), 2)
        }
    except Exception as e:
        print(f"Error in calculating top {percentile}% for {field}: {e}")
        return None


def get_bottom_percentile(table: pd.DataFrame, field: str, percentile: float = 5) -> dict:
    """Get records in bottom percentile for a field"""
    try:
        if table.empty or field not in table.columns:
            return {
                'threshold': 0.0, 'count': 0, 'percentage': 0.0, 'total_value': 0.0
            }
        
        series = pd.to_numeric(table[field], errors='coerce')
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'threshold': 0.0, 'count': 0, 'percentage': 0.0, 'total_value': 0.0
            }
        
        threshold = clean_series.quantile(percentile / 100)
        # Use boolean indexing safely
        mask = series <= threshold
        bottom_records = table[mask]
        
        if bottom_records.empty:
            return {
                'threshold': round(threshold, 2),
                'count': 0,
                'percentage': 0.0,
                'total_value': 0.0
            }
        
        return {
            'threshold': round(threshold, 2),
            'count': len(bottom_records),
            'percentage': round(len(bottom_records) / len(table) * 100, 2),
            'total_value': round(pd.to_numeric(bottom_records[field], errors='coerce').sum(), 2)
        }
    except Exception as e:
        print(f"Error in calculating bottom {percentile}% for {field}: {e}")
        return None


def get_correlation_matrix(table: pd.DataFrame, fields: list) -> dict:
    """Calculate correlation matrix between numeric fields"""
    try:
        if table.empty or not fields:
            return {}
        
        # Check if all fields exist in the DataFrame
        available_fields = [field for field in fields if field in table.columns]
        
        if len(available_fields) < 2:
            return {"error": "At least 2 valid numeric fields required for correlation"}
        
        numeric_data = table[available_fields].apply(pd.to_numeric, errors='coerce')
        # Remove columns that are all NaN
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {"error": "Insufficient numeric data for correlation calculation"}
        
        corr_matrix = numeric_data.corr().round(3)
        return corr_matrix.to_dict()
    except Exception as e:
        print(f"Error in calculating correlation matrix: {e}")
        return None


def get_conversion_rate(table: pd.DataFrame, success_status: str = 'Delivered') -> float:
    """Calculate order conversion rate based on successful deliveries"""
    try:
        if table.empty:
            return 0.0
        
        total_orders = len(table)
        if total_orders == 0:
            return 0.0
            
        successful_orders = len(table[table['order_status'] == success_status])
        conversion_rate = (successful_orders / total_orders) * 100
        return round(conversion_rate, 2)
    except Exception as e:
        print(f"Error in calculating conversion rate: {e}")
        return None


def get_cod_vs_prepaid_metrics(table: pd.DataFrame) -> dict:
    """Compare COD vs PrePaid payment methods"""
    try:
        if table.empty:
            return {
                'cod': {'count': 0, 'total_revenue': 0.0, 'avg_order_value': 0.0},
                'prepaid': {'count': 0, 'total_revenue': 0.0, 'avg_order_value': 0.0}
            }
        
        cod_orders = table[table['payment_mode'] == 'COD']
        prepaid_orders = table[table['payment_mode'] == 'PrePaid']
        
        # Calculate COD metrics
        if cod_orders.empty:
            cod_revenue = 0.0
            cod_aov = 0.0
        else:
            cod_revenue = cod_orders['total_amount'].astype(float).sum()
            cod_aov = cod_orders['total_amount'].astype(float).mean()
        
        # Calculate PrepPaid metrics
        if prepaid_orders.empty:
            prepaid_revenue = 0.0
            prepaid_aov = 0.0
        else:
            prepaid_revenue = prepaid_orders['total_amount'].astype(float).sum()
            prepaid_aov = prepaid_orders['total_amount'].astype(float).mean()
        
        metrics = {
            'cod': {
                'count': len(cod_orders),
                'total_revenue': round(cod_revenue, 2),
                'avg_order_value': round(cod_aov, 2) if not pd.isna(cod_aov) else 0.0
            },
            'prepaid': {
                'count': len(prepaid_orders),
                'total_revenue': round(prepaid_revenue, 2),
                'avg_order_value': round(prepaid_aov, 2) if not pd.isna(prepaid_aov) else 0.0
            }
        }
        return metrics
    except Exception as e:
        print(f"Error in calculating COD vs PrePaid metrics: {e}")
        return None


def get_common_metrics(data) -> dict:
    """Calculate common business metrics from order data"""
    try:
        # Convert data to DataFrame if needed
        if isinstance(data, list) and len(data) > 0:
            df_data = {"data": data}
            df_json = json.dumps(df_data)
            df = convert_to_df(df_json)
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                return {"error": "No valid data available for metric calculation"}
            df = data
        else:
            return {"error": "No valid data available for metric calculation"}
        
        # Calculate standard metrics
        metrics = {
            "aov": get_aov(df),
            "total_revenue": get_total_revenue(df),
            "order_count": get_order_count(df),
            "order_status_distribution": get_order_status_distribution(df),
            "payment_mode_distribution": get_payment_mode_distribution(df),
            "marketplace_distribution": get_marketplace_distribution(df),
            "total_amount_stats": get_statistical_summary(df, "total_amount")
        }
        return metrics
    except Exception as e:
        print(f"Error in calculating common metrics: {e}")
        return {"error": f"Failed to calculate metrics: {str(e)}"}


def get_geographic_insights(table: pd.DataFrame, top_n: int = 5) -> dict:
    """Get geographic distribution insights"""
    try:
        if table.empty or 'state' not in table.columns or 'city' not in table.columns:
            return {
                'top_states_by_orders': {}, 'top_states_by_revenue': {},
                'top_cities_by_orders': {}, 'top_cities_by_revenue': {},
                'highest_aov_states': {}, 'highest_aov_cities': {}
            }
        
        state_revenue = table.groupby('state')['total_amount'].agg(['count', 'sum', 'mean']).round(2)
        city_revenue = table.groupby('city')['total_amount'].agg(['count', 'sum', 'mean']).round(2)
        
        insights = {
            'top_states_by_orders': state_revenue.nlargest(min(top_n, len(state_revenue)), 'count')['count'].to_dict() if state_revenue.empty == False else {},
            'top_states_by_revenue': state_revenue.nlargest(min(top_n, len(state_revenue)), 'sum')['sum'].to_dict() if state_revenue.empty == False else {},
            'top_cities_by_orders': city_revenue.nlargest(min(top_n, len(city_revenue)), 'count')['count'].to_dict() if city_revenue.empty == False else {},
            'top_cities_by_revenue': city_revenue.nlargest(min(top_n, len(city_revenue)), 'sum')['sum'].to_dict() if city_revenue.empty == False else {},
            'highest_aov_states': state_revenue.nlargest(min(top_n, len(state_revenue)), 'mean')['mean'].to_dict() if state_revenue.empty == False else {},
            'highest_aov_cities': city_revenue.nlargest(min(top_n, len(city_revenue)), 'mean')['mean'].to_dict() if city_revenue.empty == False else {}
        }
        return insights
    except Exception as e:
        print(f"Error in calculating geographic insights: {e}")
        return None

def _fetch_orders_window(start_date: str, end_date: str, api_key: str, jwt_token: str, base_url: str) -> List[Dict]:
    """Fetch all orders for a date window with pagination support"""
    all_orders = []
    url = f"{base_url}/orders/V2/getAllOrders"
    
    params = {
        "limit": 250,
        "start_date": start_date,
        "end_date": end_date
    }
    
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    page = 1
    
    while True:
        try:
            print(f"Fetching page {page} for date range {start_date} to {end_date}")
            response = requests.get(url, params=params, headers=headers)
            
            # Check if we got a 400 Bad Request (end of pagination)
            if response.status_code == 400:
                print(f"Reached end of pagination (400 Bad Request) at page {page}")
                break
                
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") != 200 or "data" not in data:
                print(f"API returned non-200 code or missing data: {data}")
                break
            
            # Extract orders from current page
            page_orders = data["data"].get("orders", [])
            if not page_orders:
                print(f"No orders found on page {page}, ending pagination")
                break
                
            all_orders.extend(page_orders)
            print(f"Fetched {len(page_orders)} orders from page {page}")
            
            # Check for nextUrl to continue pagination
            next_url = data["data"].get("nextUrl")
            if not next_url:
                print(f"No nextUrl found, ending pagination at page {page}")
                break
            
            # Update URL for next request
            # nextUrl is usually a relative path, so prepend base_url
            try:
                if next_url.startswith('/'):
                    url = f"{base_url}{next_url}"
                elif next_url.startswith('http'):
                    url = next_url
                else:
                    url = f"{base_url}/{next_url.lstrip('/')}"
            except Exception as e:
                print(f"Error processing nextUrl '{next_url}': {e}, ending pagination")
                break
            
            # Clear params for subsequent requests since nextUrl contains all needed parameters
            params = {}
            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching orders on page {page}: {e}")
            break
    
    print(f"Total orders fetched for window {start_date} to {end_date}: {len(all_orders)}")
    return all_orders


def apply_filters(data: List[Dict], filters: List[Dict]) -> List[Dict]:
    """
    Apply filters to order data
    
    Args:
        data: List of order records
        filters: List of filter dictionaries with structure:
            [{"field": "payment_mode", "operator": "eq", "value": "PrePaid"}, ...]
    
    Returns:
        Filtered list of orders
    """
    if not filters:
        return data
    
    # Fields that are nested in suborders[] array
    NESTED_FIELDS = {
        "sku", "brand", "category", "size", "productName", "selling_price", 
        "mrp", "model_no", "AccountingSku", "Identifier", "accounting_unit",
        "ean", "marketplace_sku", "item_status", "shipment_type", "sku_type",
        "tax", "tax_rate", "tax_type", "cost", "item_quantity", "description",
        "product_id", "company_product_id", "suborder_id", "suborder_num",
        "suborder_reference_num", "weight", "height", "length", "width"
    }
    
    filtered_data = data
    
    for filter_spec in filters:
        field = filter_spec.get("field")
        operator = filter_spec.get("operator", "eq")
        value = filter_spec.get("value")
        
        # Debug logging
        print(f"[DEBUG FILTER] Field: {field}, Operator: {operator}, Value: {value} (type: {type(value).__name__})", flush=True)
        
        # Special handling for fields nested in suborders array
        if field in NESTED_FIELDS:
            before_count = len(filtered_data)
            if operator == "eq":
                # For numeric comparisons, try to convert both sides to float
                try:
                    numeric_value = float(value)
                    filtered_data = [
                        r for r in filtered_data 
                        if any(
                            float(sub.get(field, 0)) == numeric_value
                            for sub in r.get("suborders", [])
                            if sub.get(field) is not None
                        )
                    ]
                except (ValueError, TypeError):
                    # String comparison (case-insensitive)
                    filtered_data = [
                        r for r in filtered_data 
                        if any(
                            str(sub.get(field, "")).lower() == str(value).lower() 
                            for sub in r.get("suborders", [])
                        )
                    ]
            elif operator == "ne":
                try:
                    numeric_value = float(value)
                    filtered_data = [
                        r for r in filtered_data 
                        if any(
                            float(sub.get(field, 0)) != numeric_value
                            for sub in r.get("suborders", [])
                            if sub.get(field) is not None
                        )
                    ]
                except (ValueError, TypeError):
                    filtered_data = [
                        r for r in filtered_data 
                        if any(
                            str(sub.get(field, "")).lower() != str(value).lower() 
                            for sub in r.get("suborders", [])
                        )
                    ]
            elif operator == "gt":
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        float(sub.get(field, 0)) > float(value)
                        for sub in r.get("suborders", [])
                        if sub.get(field) is not None
                    )
                ]
            elif operator == "lt":
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        float(sub.get(field, 0)) < float(value)
                        for sub in r.get("suborders", [])
                        if sub.get(field) is not None
                    )
                ]
            elif operator == "gte":
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        float(sub.get(field, 0)) >= float(value)
                        for sub in r.get("suborders", [])
                        if sub.get(field) is not None
                    )
                ]
            elif operator == "lte":
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        float(sub.get(field, 0)) <= float(value)
                        for sub in r.get("suborders", [])
                        if sub.get(field) is not None
                    )
                ]
            elif operator == "contains":
                value_str = str(value).lower() if value is not None else ""
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        value_str in str(sub.get(field, "")).lower() 
                        for sub in r.get("suborders", [])
                    )
                ]
            elif operator == "in":
                filtered_data = [
                    r for r in filtered_data 
                    if any(
                        sub.get(field) in value 
                        for sub in r.get("suborders", [])
                    )
                ]
            print(f"[DEBUG FILTER] Nested field '{field}' filter: {before_count} → {len(filtered_data)} records (nested in suborders)", flush=True)
            continue
        
        # Handle top-level fields (payment_mode, marketplace, order_status, state, city, etc.)
        if operator == "eq":
            # Case-insensitive string comparison for string values
            if isinstance(value, str):
                before_count = len(filtered_data)
                filtered_data = [r for r in filtered_data if str(r.get(field, "")).lower() == value.lower()]
                print(f"[DEBUG FILTER] EQ filter: {before_count} → {len(filtered_data)} records (field={field}, value={value})", flush=True)
                
                print(filtered_data)
                
                # Show first few actual values for debugging
                if before_count > 0 and len(filtered_data) == 0:
                    sample_values = [r.get(field) for r in data[:5]]
                    print(f"[DEBUG FILTER] Sample values in data: {sample_values}", flush=True)
            else:
                filtered_data = [r for r in filtered_data if r.get(field) == value]
        elif operator == "ne":
            filtered_data = [r for r in filtered_data if r.get(field) != value]
        elif operator == "gt":
            filtered_data = [r for r in filtered_data if r.get(field, 0) > value]
        elif operator == "lt":
            filtered_data = [r for r in filtered_data if r.get(field, 0) < value]
        elif operator == "gte":
            filtered_data = [r for r in filtered_data if r.get(field, 0) >= value]
        elif operator == "lte":
            filtered_data = [r for r in filtered_data if r.get(field, 0) <= value]
        elif operator == "contains":
            # Safely handle contains operator - convert both to strings
            value_str = str(value).lower() if value is not None else ""
            filtered_data = [r for r in filtered_data if value_str in str(r.get(field, "")).lower()]
        elif operator == "in":
            filtered_data = [r for r in filtered_data if r.get(field) in value]
    
    return filtered_data


def get_schema_info(entity: str = "orders", field: str = None) -> Dict[str, Any]:
    """
    Return schema and metadata information about available data entities
    
    This tool handles schema & data discovery queries like:
    - What fields are available?
    - What are the allowed values for categorical fields?
    - What date ranges are supported?
    - What level of granularity is available?
    
    Args:
        entity: The data entity to get schema for (default: "orders")
        field: Optional - specific field name to get info for (e.g., "payment_mode")
               If provided, returns only that field's schema
    
    Returns:
        Dictionary containing schema information, constraints, and metadata
    """
    if entity == "orders":
        all_fields = {
            "order_id": {
                "type": "string",
                "description": "Unique order identifier",
                "example": "ORD123456",
                "filterable": True,
                "categorical": False
            },
            "marketplace": {
                "type": "string",
                "description": "Sales channel/marketplace name",
                "allowed_values": ["Shopify13", "Flipkart", "Amazon", "Myntra", "Ajio", "Nykaa"],
                "example": "Shopify13",
                "filterable": True,
                "categorical": True
            },
            "payment_mode": {
                "type": "string",
                "description": "Payment method used",
                "allowed_values": ["PrePaid", "COD"],
                "example": "PrePaid",
                "filterable": True,
                "categorical": True
            },
            "order_status": {
                "type": "string",
                "description": "Current order status",
                "allowed_values": ["Open", "Cancelled", "Delivered", "Pending", "Shipped", "RTO"],
                "example": "Delivered",
                "filterable": True,
                "categorical": True
            },
            "total_amount": {
                "type": "float",
                "description": "Total order value",
                "example": 1599.50,
                "filterable": True,
                "categorical": False,
                "unit": "INR"
            },
            "order_date": {
                "type": "datetime",
                "description": "Order creation timestamp",
                "format": "YYYY-MM-DD HH:MM:SS",
                "example": "2026-02-01 14:30:00",
                "filterable": True,
                "categorical": False
            },
            "state": {
                "type": "string",
                "description": "Delivery state/region",
                "allowed_values": [
                                    "Delhi", 
                                    "Andhra Pradesh",
                                    "Arunachal Pradesh",
                                    "Assam",
                                    "Bihar",
                                    "Chhattisgarh",
                                    "Goa",
                                    "Gujarat",
                                    "Haryana",
                                    "Himachal Pradesh",
                                    "Jharkhand",
                                    "Karnataka",
                                    "Kerala",
                                    "Madhya Pradesh",
                                    "Maharashtra",
                                    "Manipur",
                                    "Meghalaya",
                                    "Mizoram",
                                    "Nagaland",
                                    "Odisha",
                                    "Punjab",
                                    "Rajasthan",
                                    "Sikkim",
                                    "Tamil Nadu",
                                    "Telangana",
                                    "Tripura",
                                    "Uttarakhand",
                                    "Uttar Pradesh",
                                    "West Bengal"
                                ],                                       
                "example": "Karnataka",
                "filterable": True,
                "categorical": True,
                "granularity": "state-level"
            },
            "city": {
                "type": "string",
                "description": "Delivery city",
                "example": "Bangalore",
                "filterable": True,
                "categorical": True,
                "granularity": "city-level",
                "note": "200+ cities available"
            },
            "pincode": {
                "type": "string",
                "description": "Delivery postal code",
                "example": "560001",
                "filterable": True,
                "categorical": False
            },
            "customer_name": {
                "type": "string",
                "description": "Customer full name",
                "example": "John Doe",
                "filterable": False,
                "categorical": False
            },
            "customer_email": {
                "type": "string",
                "description": "Customer email address",
                "example": "john@example.com",
                "filterable": False,
                "categorical": False
            },
            "sku": {
                "type": "string",
                "description": "Product SKU",
                "example": "SKU-12345",
                "filterable": True,
                "categorical": False
            },
            "quantity": {
                "type": "integer",
                "description": "Order quantity",
                "example": 2,
                "filterable": True,
                "categorical": False
            }
        }
        
        # If specific field requested, return only that field
        if field:
            if field in all_fields:
                return {
                    "entity": "orders",
                    "field": field,
                    "field_info": all_fields[field]
                }
            else:
                return {
                    "error": f"Field '{field}' not found in entity 'orders'",
                    "available_fields": list(all_fields.keys())
                }
        
        # Return complete schema
        return {
            "entity": "orders",
            "description": "EasyEcom order data from all marketplaces",
            "api_constraints": {
                "max_date_range_days": 7,
                "note": "Queries spanning >7 days will be automatically split into windows"
            },
            "available_fields": all_fields,
            "data_granularity": {
                "temporal": "Order-level (each order is one record)",
                "geographic": "City-level and State-level available",
                "marketplace": "Individual marketplace breakdown available",
                "customer": "Customer identifiable but not segmented by tier"
            },
            "common_queries": [
                "Filter by payment mode (PrePaid/COD)",
                "Filter by marketplace",
                "Filter by state or city",
                "Filter by order status",
                "Filter by date range",
                "Compare marketplaces",
                "Compare payment modes",
                "Compare states/regions"
            ]
        }
    else:
        return {
            "error": f"Unknown entity: {entity}",
            "available_entities": ["orders"]
        }

# Tool registry mapping
TOOL_REGISTRY = {
    "get_all_orders": get_all_orders,
    "get_schema_info": get_schema_info,
    "convert_to_df": convert_to_df,
    "get_aov": get_aov,
    "get_total_revenue": get_total_revenue,
    "get_order_count": get_order_count,
    "get_order_status_distribution": get_order_status_distribution,
    "get_payment_mode_distribution": get_payment_mode_distribution,
    "get_marketplace_distribution": get_marketplace_distribution,
    "get_state_wise_distribution": get_state_wise_distribution,
    "get_city_wise_distribution": get_city_wise_distribution,
    "get_courier_distribution": get_courier_distribution,
    "get_average_discount": get_average_discount,
    "get_average_shipping_charge": get_average_shipping_charge,
    "get_average_tax": get_average_tax,
    "get_statistical_summary": get_statistical_summary,
    "get_percentile": get_percentile,
    "get_top_percentile": get_top_percentile,
    "get_bottom_percentile": get_bottom_percentile,
    "get_correlation_matrix": get_correlation_matrix,
    "get_conversion_rate": get_conversion_rate,
    "get_cod_vs_prepaid_metrics": get_cod_vs_prepaid_metrics,
    "get_geographic_insights": get_geographic_insights,
    "get_common_metrics": get_common_metrics,
}
