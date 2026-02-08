"""
Tool functions for fetching and manipulating data
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any


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
    
    all_orders = []
    
    # If more than 7 days, implement windowing
    if days_diff > 7:
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=7), end_dt)
            
            # Fetch window
            window_orders = _fetch_orders_window(
                current_start.strftime("%Y-%m-%d %H:%M:%S"),
                current_end.strftime("%Y-%m-%d %H:%M:%S"),
                api_key,
                jwt_token,
                base_url
            )
            all_orders.extend(window_orders)
            
            # Move to next window
            current_start = current_end
    else:
        # Single fetch for <= 7 days
        all_orders = _fetch_orders_window(start_date, end_date, api_key, jwt_token, base_url)
    
    return all_orders


def _fetch_orders_window(start_date: str, end_date: str, api_key: str, jwt_token: str, base_url: str) -> List[Dict]:
    """Fetch orders for a single 7-day window"""
    url = f"{base_url}/orders/V2/getAllOrders"
    
    params = {
        "start_date": start_date,
        "end_date": end_date
    }
    
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get("code") == 200 and "data" in data:
            return data["data"].get("orders", [])
        else:
            print(f"API returned non-200 code: {data}")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching orders: {e}")
        return []


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
    
    filtered_data = data
    
    for filter_spec in filters:
        field = filter_spec.get("field")
        operator = filter_spec.get("operator", "eq")
        value = filter_spec.get("value")
        
        if operator == "eq":
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
            filtered_data = [r for r in filtered_data if value.lower() in str(r.get(field, "")).lower()]
        elif operator == "in":
            filtered_data = [r for r in filtered_data if r.get(field) in value]
    
    return filtered_data


# Tool registry mapping
TOOL_REGISTRY = {
    "get_all_orders": get_all_orders,
}
