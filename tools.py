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
                "allowed_values": ["Karnataka", "Maharashtra", "Delhi", "Tamil Nadu", "Gujarat", "Uttar Pradesh", "West Bengal", "Rajasthan"],
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
}
