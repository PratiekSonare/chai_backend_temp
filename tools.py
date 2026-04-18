"""
Tool functions for fetching and manipulating data
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import ast
import pandas as pd
import numpy as np
import json
import boto3
import re
import signal
import sys
import math
from utils.type_converters import convert_numpy_types

s3 = boto3.client('s3')

MAX_CUSTOM_CALC_CODE_LEN = 8000
MAX_CUSTOM_RESULT_SIZE = 10_000_000
CUSTOM_CALC_TIMEOUT_SECONDS = 5

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
def _is_valid_metric_name(metric_name: str) -> bool:
    if not isinstance(metric_name, str) or not metric_name:
        return False
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', metric_name) is not None


def _validate_custom_calculation_code(calculation_code: str) -> tuple[bool, str]:
    """Validate custom calculation code before execution."""
    if not isinstance(calculation_code, str) or not calculation_code.strip():
        return False, "calculation_code must be a non-empty string"

    if len(calculation_code) > MAX_CUSTOM_CALC_CODE_LEN:
        return False, f"calculation_code exceeds {MAX_CUSTOM_CALC_CODE_LEN} characters"

    forbidden_names = {
        "__import__", "open", "eval", "exec", "compile", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr"
    }
    forbidden_roots = {
        "os", "sys", "subprocess", "socket", "requests", "boto3", "importlib", "pathlib"
    }

    try:
        tree = ast.parse(calculation_code)
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"

    assigns_result = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Import statements are not allowed"

        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return False, f"Forbidden identifier used: {node.id}"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith('__'):
                return False, "Dunder attribute access is not allowed"
            if isinstance(node.value, ast.Name) and node.value.id in forbidden_roots:
                return False, f"Forbidden module access: {node.value.id}"

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            return False, f"Forbidden function call: {node.func.id}"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'result':
                    assigns_result = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == 'result':
                assigns_result = True

    if not assigns_result:
        return False, "calculation_code must assign output to 'result'"

    return True, "ok"


def _execute_custom_code_with_timeout(calculation_code: str, local_vars: dict, safe_builtins: dict) -> None:
    """Execute custom code with timeout where possible."""
    timeout_supported = hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm')
    old_handler = None

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Calculation exceeded {CUSTOM_CALC_TIMEOUT_SECONDS} second limit")

    try:
        if timeout_supported:
            try:
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(CUSTOM_CALC_TIMEOUT_SECONDS)
            except (ValueError, OSError):
                # signal.alarm is only available in the main thread on some runtimes.
                timeout_supported = False

        # Use one shared execution namespace so comprehensions/lambdas can
        # resolve variables like `df` consistently.
        exec_scope = {"__builtins__": safe_builtins, **local_vars}
        exec(calculation_code, exec_scope, exec_scope)

        # Persist any values (including `result`) back to local_vars for callers.
        local_vars.update(exec_scope)
    finally:
        if timeout_supported:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# ===================================================================
# ORDER TOOLS
# ===================================================================
def get_all_orders(start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch and aggregate daily orders from S3 for a date range.

    S3 layout:
        s3://chupps-data-portal/orders/YYYY-MM/YYYY-MM-DD.json

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
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    if start_dt > end_dt:
        raise ValueError("start_date must be less than or equal to end_date")

    bucket_name = "chupps-data-portal"
    root_prefix = "orders"

    print(f"Fetching orders from S3: {start_date} to {end_date}")

    all_orders = []
    skipped_files = []

    current_day = start_dt.date()
    end_day = end_dt.date()

    while current_day <= end_day:
        month_folder = current_day.strftime("%Y-%m")
        file_name = current_day.strftime("%Y-%m-%d.json")
        object_key = f"{root_prefix}/{month_folder}/{file_name}"

        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            body_bytes = response["Body"].read()
            payload = None
            
            # Try UTF-8 first with error handling
            try:
                decoded_text = body_bytes.decode("utf-8")
                payload = json.loads(decoded_text)
            except UnicodeDecodeError as e:
                print(f"WARNING: UTF-8 decode error at position {e.start} in {object_key}")
                # Fallback 1: UTF-8 with error replacement
                try:
                    decoded_text = body_bytes.decode("utf-8", errors="replace")
                    payload = json.loads(decoded_text)
                    print(f"  Recovered using UTF-8 with error replacement")
                except json.JSONDecodeError:
                    print(f"  JSON parse failed after UTF-8 recovery, trying repair...")
                    # Fallback 2: Try to repair truncated JSON
                    try:
                        payload = _repair_json(decoded_text)
                        if payload:
                            print(f"  Recovered by repairing truncated JSON")
                        else:
                            raise ValueError("JSON repair returned None")
                    except Exception as repair_e:
                        print(f"  JSON repair failed: {repair_e}")
                        # Fallback 3: Try latin-1
                        try:
                            decoded_text = body_bytes.decode("latin-1")
                            payload = json.loads(decoded_text)
                            print(f"  Recovered using latin-1 encoding")
                        except json.JSONDecodeError:
                            print(f"  JSON parse failed with latin-1")
                            # Last resort: Try to repair latin-1 text
                            try:
                                payload = _repair_json(decoded_text)
                                if payload:
                                    print(f"  Recovered by repairing latin-1 JSON")
                                else:
                                    raise ValueError("Unable to repair JSON from any encoding")
                            except Exception as final_repair:
                                print(f"ERROR: Could not recover {object_key}: {final_repair}")
                                skipped_files.append(object_key)
                                payload = None
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON parse error in {object_key} at line {e.lineno} col {e.colno}: {e.msg}")
                # Try repair for initial UTF-8 decode
                try:
                    payload = _repair_json(decoded_text if 'decoded_text' in locals() else body_bytes.decode("utf-8", errors="replace"))
                    if payload:
                        print(f"  Recovered using JSON repair")
                    else:
                        raise ValueError("JSON repair failed")
                except Exception as repair_e:
                    print(f"  Could not repair JSON: {repair_e}")
                    skipped_files.append(object_key)
                    payload = None

            if payload is None:
                # Skip this file and continue to next
                current_day += timedelta(days=1)
                continue

            if isinstance(payload, list):
                day_orders = payload
            elif isinstance(payload, dict):
                if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("orders"), list):
                    day_orders = payload["data"]["orders"]
                elif isinstance(payload.get("orders"), list):
                    day_orders = payload["orders"]
                else:
                    day_orders = [payload] if payload else []
            else:
                day_orders = []

            all_orders.extend(day_orders)
            print(f"Fetched {len(day_orders)} orders from s3://{bucket_name}/{object_key}")
            
        except Exception as e:
            error_code = None
            if hasattr(e, "response") and isinstance(getattr(e, "response"), dict):
                error_code = e.response.get("Error", {}).get("Code")

            if error_code in {"NoSuchKey", "404", "NotFound"}:
                print(f"No file found for {current_day}: s3://{bucket_name}/{object_key}")
            else:
                print(f"Error fetching s3://{bucket_name}/{object_key}: {e}")
                skipped_files.append(object_key)

        current_day += timedelta(days=1)

    if skipped_files:
        print(f"\n⚠️  WARNING: Skipped {len(skipped_files)} corrupted files:")
        for f in skipped_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files) - 5} more")
        print(f"These files should be investigated and regenerated from source.\n")

    print(f"Total orders fetched across {(end_day - start_dt.date()).days + 1 - len(skipped_files)} valid S3 daily files: {len(all_orders)}")
    return all_orders


def _repair_json(text: str) -> Optional[Dict | List]:
    """
    Attempt to repair truncated or malformed JSON by finding last valid closing bracket.
    Returns parsed JSON if successful, None otherwise.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Try original first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find last valid JSON object/array by finding closing ] or }
    # Strategy: Find last ] or }, truncate there, and try parsing
    last_bracket_idx = max(
        text.rfind(']'),
        text.rfind('}')
    )
    
    if last_bracket_idx <= 0:
        return None
    
    try:
        truncated = text[:last_bracket_idx + 1]
        return json.loads(truncated)
    except json.JSONDecodeError:
        # If truncating to last bracket didn't work, try to find valid JSON array of objects
        # Look for patterns like [{"key": value}, ...
        try:
            # Try to extract array if wrapped in array
            if text.strip().startswith('['):
                # Find all complete objects within array
                import re
                object_pattern = r'\{[^{}]*\}'
                matches = re.findall(object_pattern, text)
                if matches:
                    valid_objects = []
                    for match in matches:
                        try:
                            valid_objects.append(json.loads(match))
                        except:
                            pass
                    if valid_objects:
                        return valid_objects
        except Exception:
            pass
        
        return None


#for any metric, data calculation, first convert to dataframe and then continue.
def convert_to_df(raw: list) -> pd.DataFrame:
    """Convert raw JSON order data to normalized DataFrame"""
    
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
            if isinstance(data_block, dict) and 'orders' in data_block:
                orders = data_block.get('orders') or []
            elif isinstance(data_block, list):
                orders = data_block
            elif isinstance(data_block, dict):
                orders = [data_block]
            else:
                orders = []
        elif isinstance(parsed, dict) and 'orders' in parsed:
            orders = parsed.get('orders') or []
        elif isinstance(parsed, list):
            orders = parsed
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

    # Explode suborders: each suborder becomes a row, with suborder_ prefix for its fields
    exploded_rows = []
    for order in orders:
        suborders = order.get('suborders')
        if isinstance(suborders, list) and len(suborders) > 0:
            for sub in suborders:
                # Copy parent order fields
                row = {k: v for k, v in order.items() if k != 'suborders'}
                # Add suborder fields with prefix
                for subk, subv in sub.items():
                    row[f'suborder_{subk}'] = subv
                exploded_rows.append(row)
        else:
            # No suborders, just add the order as is
            row = {k: v for k, v in order.items() if k != 'suborders'}
            exploded_rows.append(row)

    df = pd.json_normalize(exploded_rows)

    print("========================")
    print("columns:", list(df.columns), flush=True)
    print("========================")
    print("normalized dataframe: ")
    print(df.head(5), flush=True)
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
        result = round(aov, 2) if not pd.isna(aov) else 0.0
        return convert_numpy_types(result)
    except Exception as e:
        print(f"Error in calculating AOV: {e}")
        return None


def get_total_revenue(table: pd.DataFrame) -> float:
    """Calculate total revenue from orders DataFrame"""
    try:
        if table.empty:
            return 0.0
        revenue = table['total_amount'].astype(float).sum()
        result = round(revenue, 2) if not pd.isna(revenue) else 0.0
        return convert_numpy_types(result)
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
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating order status distribution: {e}")
        return None


def get_payment_mode_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of payment modes"""
    try:
        if table.empty or 'payment_mode' not in table.columns:
            return {}
        distribution = table['payment_mode'].value_counts().to_dict()
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating payment mode distribution: {e}")
        return None


def get_marketplace_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by marketplace"""
    try:
        if table.empty or 'marketplace' not in table.columns:
            return {}
        distribution = table['marketplace'].value_counts().to_dict()
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating marketplace distribution: {e}")
        return None


def get_state_wise_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by state"""
    try:
        if table.empty or 'state' not in table.columns:
            return {}
        distribution = table['state'].value_counts().to_dict()
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating state distribution: {e}")
        return None


def get_city_wise_distribution(table: pd.DataFrame, top_n: int = 10) -> dict:
    """Get distribution of orders by city (top N cities)"""
    try:
        if table.empty or 'city' not in table.columns:
            return {}
        distribution = table['city'].value_counts().head(top_n).to_dict()
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating city distribution: {e}")
        return None


def get_courier_distribution(table: pd.DataFrame) -> dict:
    """Get distribution of orders by courier service"""
    try:
        if table.empty or 'courier' not in table.columns:
            return {}
        distribution = table['courier'].value_counts().to_dict()
        return convert_numpy_types(distribution)
    except Exception as e:
        print(f"Error in calculating courier distribution: {e}")
        return None


def get_average_discount(table: pd.DataFrame) -> float:
    """Calculate average discount amount"""
    try:
        if table.empty:
            return 0.0
        avg_discount = table['total_discount'].astype(float).mean()
        result = round(avg_discount, 2) if not pd.isna(avg_discount) else 0.0
        return convert_numpy_types(result)
    except Exception as e:
        print(f"Error in calculating average discount: {e}")
        return None


def get_average_shipping_charge(table: pd.DataFrame) -> float:
    """Calculate average shipping charge"""
    try:
        if table.empty:
            return 0.0
        avg_shipping = table['total_shipping_charge'].astype(float).mean()
        result = round(avg_shipping, 2) if not pd.isna(avg_shipping) else 0.0
        return convert_numpy_types(result)
    except Exception as e:
        print(f"Error in calculating average shipping charge: {e}")
        return None


def get_average_tax(table: pd.DataFrame) -> float:
    """Calculate average tax amount"""
    try:
        if table.empty:
            return 0.0
        avg_tax = table['total_tax'].astype(float).mean()
        result = round(avg_tax, 2) if not pd.isna(avg_tax) else 0.0
        return convert_numpy_types(result)
    except Exception as e:
        print(f"Error in calculating average tax: {e}")
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
        return convert_numpy_types(round(conversion_rate, 2))
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
        return convert_numpy_types(metrics)
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
        return convert_numpy_types(metrics)
    except Exception as e:
        print(f"Error in calculating common metrics: {e}")
        return {"error": f"Failed to calculate metrics: {str(e)}"}


def _error_response(metric_name: str, error: str, code: str) -> dict:
    """Helper for consistent error format"""
    return {
        "success": False,
        "metric_name": metric_name,
        "error": error,
        "calculation_code": code[:600] if isinstance(code, str) else "",
        "result": None
    }


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
        return convert_numpy_types(insights)
    except Exception as e:
        print(f"Error in calculating geographic insights: {e}")
        return None


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

def get_cancelled_count(table: pd.DataFrame) -> dict:
    """Return count cancelled orders"""
    try:
        if table.empty:
            return {
                "cancel_count": 0
            }

        cancel_count = (table['order_status'] == "Cancelled").sum()
        return {
            "cancel_count": int(cancel_count) 
        }    
    except Exception as e:
        print(f"Error in calculating geographic insights: {e}")
        return None

# ===================================================================
# PROFIT TOOLS
# ===================================================================
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

def get_vendor_cost_sheet(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY) -> pd.DataFrame:
    """
    Fetch vendor_cost_sheet from Supabase & return as List[Dict].
    (Directly returns the raw data — not the same as orders table)
    """
    if not supabase_url or not supabase_key:
        raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")

    # Fetch the entire table
    response = (
        supabase.table("vendor_cost_sheet")        # ← Change if your actual table name is different
        .select("*")
        .execute()
    )

    # response.data is already a List[Dict]
    data: List[Dict] = response.data

    if not data:
        print("Warning: vendor_cost_sheet returned no data.")

    return pd.DataFrame(data) #return dataframe


def _ensure_dataframe(data: Union[pd.DataFrame, list]) -> pd.DataFrame:
    """Helper to force input into a pandas DataFrame.
    
    IMPORTANT: This function expects either:
    - A pandas DataFrame
    - A list of dictionaries
    
    DO NOT pass individual dicts! Always wrap in a list: [record]
    
    Examples:
    ✅ _ensure_dataframe([{"MRP": 100, "Final price": 80}])  # Good
    ❌ _ensure_dataframe({"MRP": 100, "Final price": 80})     # Bad - will fail
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Defensive handling: if someone passes a dict, wrap it in a list first
        return pd.DataFrame([data])
    return data  # Assume it's already a DataFrame

def _get_clean_series(table: Union[pd.DataFrame, list], col_name: str) -> pd.Series:
    """Helper to safely extract a numeric series from a column."""
    df = _ensure_dataframe(table)
    if col_name not in df.columns or df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col_name], errors='coerce').fillna(0)

def get_margin(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """Calculates row-wise Margin % and returns the sum."""
    df = _ensure_dataframe(table)
    mrp = _get_clean_series(df, "MRP")
    cost = _get_clean_series(df, "Final price")
    
    if mrp.empty and cost.empty:
        return None

    gp = mrp - cost
    # Handle division by zero: if MRP is 0, margin is 0
    margins = (gp / mrp).where(mrp != 0, 0) * 100
    return round(float(margins.sum()), 2)

def get_gross_profit(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """Returns the total sum of Gross Profit across all items."""
    df = _ensure_dataframe(table)
    mrp = _get_clean_series(df, "MRP")
    cost = _get_clean_series(df, "Final price")
    
    if mrp.empty and cost.empty:
        return None
        
    return round(float((mrp - cost).sum()), 2)

def get_markup(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """Calculates row-wise Markup and returns the total sum."""
    df = _ensure_dataframe(table)
    mrp = _get_clean_series(df, "MRP")
    cost = _get_clean_series(df, "Final price")
    
    if mrp.empty and cost.empty:
        return None

    gp = mrp - cost
    # Handle division by zero: if Cost is 0, markup is 0
    markups = (gp / cost).where(cost != 0, 0)
    return round(float(markups.sum()), 4)

def get_selling_price(table: Union[pd.DataFrame, list]) -> float:
    """Returns total sum of MRP."""
    return round(float(_get_clean_series(table, "MRP").sum()), 2)

def get_cost_price(table: Union[pd.DataFrame, list]) -> float:
    """Returns total sum of Final price."""
    return round(float(_get_clean_series(table, "Final price").sum()), 2)

def get_cost_to_price_ratio(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """Cost-to-Price Ratio % = (Total Cost / Total Selling Price) * 100"""
    cp = get_cost_price(table)
    sp = get_selling_price(table)
    if sp == 0:
        return None
    return round((cp / sp) * 100, 2)
    
   
# ===================================================================
# COMMON TOOLS
# =================================================================== 
def execute_custom_calculation(
    table: pd.DataFrame, 
    calculation_code: str, 
    metric_name: str = "custom_metric"
) -> dict:
    """
    Safe execution of LLM-generated Python code for custom metrics on a DataFrame.
    Designed for REPL-like usage where the LLM can generate one-off calculations.
    """
    if not isinstance(calculation_code, str) or not calculation_code.strip():
        return {
            "success": False,
            "error": "Calculation code cannot be empty",
            "metric_name": metric_name,
            "calculation_code": ""
        }

    try:
        # Basic validations
        if not _is_valid_metric_name(metric_name):
            return _error_response(metric_name, "Invalid metric name. Use only letters, numbers, and underscores.", calculation_code)

        if table.empty:
            return _error_response(metric_name, "DataFrame is empty", calculation_code)

        # Strict code validation (this is the most important layer)
        is_valid, validation_message = _validate_custom_calculation_code(calculation_code)
        if not is_valid:
            return _error_response(metric_name, f"Code validation failed: {validation_message}", calculation_code)

        # Prepare safe environment
        local_vars = {
            'df': table.copy(),           # always give a copy
            'pd': pd,
            'np': np,
            'math': math,
            'datetime': datetime,
            'result': None,
        }

        safe_builtins = {
            # Core functions
            'len': len, 'max': max, 'min': min, 'sum': sum, 'abs': abs,
            'round': round, 'sorted': sorted, 'enumerate': enumerate,
            'range': range, 'zip': zip, 'any': any, 'all': all,
            'map': map, 'filter': filter, 'reversed': reversed,
            'next': next, 'slice': slice,
            # Types
            'bool': bool, 'int': int, 'float': float, 'str': str,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'type': type, 'object': object, 'complex': complex,
            'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
            'frozenset': frozenset, 'property': property,
            # Exception classes
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'KeyError': KeyError, 'IndexError': IndexError, 'ZeroDivisionError': ZeroDivisionError,
            'AttributeError': AttributeError, 'StopIteration': StopIteration, 'AssertionError': AssertionError,
            'NotImplementedError': NotImplementedError, 'RuntimeError': RuntimeError, 'OSError': OSError,
            'ImportError': ImportError, 'NameError': NameError, 'UnboundLocalError': UnboundLocalError,
            'ArithmeticError': ArithmeticError, 'OverflowError': OverflowError, 'EOFError': EOFError,
            'IOError': IOError, 'LookupError': LookupError, 'MemoryError': MemoryError,
            'ReferenceError': ReferenceError, 'TabError': TabError, 'SystemExit': SystemExit,
            # Utility functions
            'isinstance': isinstance, 'issubclass': issubclass, 'id': id, 'hash': hash,
            'all': all, 'any': any, 'callable': callable, 'chr': chr, 'ord': ord,
            'divmod': divmod, 'pow': pow, 'sum': sum, 'min': min, 'max': max,
            'abs': abs, 'repr': repr, 'str': str, 'format': format,
            'bin': bin, 'oct': oct, 'hex': hex,
            # Safe constants
            'True': True, 'False': False, 'None': None,
        }

        # Execute with timeout protection
        _execute_custom_code_with_timeout(calculation_code, local_vars, safe_builtins)

        result = local_vars.get('result')
        if result is None:
            return _error_response(
                metric_name, 
                "Your code must assign the final value to a variable named `result`", 
                calculation_code
            )

        # Size guard
        if sys.getsizeof(result) > MAX_CUSTOM_RESULT_SIZE:
            return _error_response(
                metric_name, 
                f"Result too large ({sys.getsizeof(result)} bytes). Max: {MAX_CUSTOM_RESULT_SIZE} bytes", 
                calculation_code
            )

        result = convert_numpy_types(result)

        return {
            "success": True,
            "metric_name": metric_name,
            "result": result,
            "calculation_code": calculation_code,
            "row_count": len(table)
        }

    except TimeoutError:
        return _error_response(metric_name, f"Calculation timed out after {CUSTOM_CALC_TIMEOUT_SECONDS} seconds", calculation_code)
    except Exception as e:
        return _error_response(metric_name, f"Execution error: {type(e).__name__}: {str(e)}", calculation_code)


def apply_filters(table: List[Dict], filters: List[Dict]) -> List[Dict]:
    """
    Apply filters to order table (ALSO WORKS FOR PROFIT TABLE, SINCE NO COMMON PARAMS)
    
    Args:
        data: List of order records
        filters: List of filter dictionaries with structure:
            [{"field": "payment_mode", "operator": "eq", "value": "PrePaid"}, ...]
    
    Returns:
        Filtered list of orders
    """
    # Accept either a list of dicts or a pandas DataFrame
    if isinstance(table, pd.DataFrame):
        try:
            table = table.to_dict(orient='records')
        except Exception:
            # If conversion fails, fall back to empty list
            table = []

    if not filters:
        return table
    
    # Fields that are nested in suborders[] array
    NESTED_FIELDS = {
        "sku", "brand", "category", "size", "productName", "selling_price", 
        "mrp", "model_no", "AccountingSku", "Identifier", "accounting_unit",
        "ean", "marketplace_sku", "item_status", "shipment_type", "sku_type",
        "tax", "tax_rate", "tax_type", "cost", "item_quantity", "description",
        "product_id", "company_product_id", "suborder_id", "suborder_num",
        "suborder_reference_num", "weight", "height", "length", "width"
    }
    
    filtered_data = table
    
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
                    sample_values = [r.get(field) for r in table[:5]]
                    print(f"[DEBUG FILTER] Sample values in table: {sample_values}", flush=True)
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
        return convert_numpy_types(stats)
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
        final_result = round(result, 2) if not pd.isna(result) else 0.0
        return convert_numpy_types(final_result)
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
        
        result = {
            'threshold': round(threshold, 2),
            'count': len(top_records),
            'percentage': round(len(top_records) / len(table) * 100, 2),
            'total_value': round(pd.to_numeric(top_records[field], errors='coerce').sum(), 2)
        }
        return convert_numpy_types(result)
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
        
        result = {
            'threshold': round(threshold, 2),
            'count': len(bottom_records),
            'percentage': round(len(bottom_records) / len(table) * 100, 2),
            'total_value': round(pd.to_numeric(bottom_records[field], errors='coerce').sum(), 2)
        }
        return convert_numpy_types(result)
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
        return convert_numpy_types(corr_matrix.to_dict())
    except Exception as e:
        print(f"Error in calculating correlation matrix: {e}")
        return None

# ===================================================================
# PAYMENT CYCLE TOOLS
# ===================================================================
def get_payment_cycle_data(distributor_name: str = None) -> List[Dict]:
    """
    Fetch payment cycle and cash discount data from Supabase.
    
    Args:
        distributor_name: Optional filter for specific distributor
    
    Returns:
        List of distributor payment cycle records with fields:
        - PARTY NAME (distributor name)
        - MARGIN (margin %)
        - CD (cash discount %)
        - PAYMENT CYCLE (days)
        - ASM NAME
        - REMARK
    """
    if not supabase:
        raise ValueError("Supabase client not initialized. Check SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        query = supabase.table("payment_cycle_and_cash_discount").select("*")
        
        # Apply optional filter
        if distributor_name:
            query = query.eq("PARTY NAME", distributor_name)
        
        response = query.execute()
        data: List[Dict] = response.data
        
        if not data:
            print("Warning: payment_cycle_and_cash_discount returned no data.")
        
        return data
    
    except Exception as e:
        print(f"Error fetching payment cycle data: {e}")
        raise

def get_avg_margin(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """
    Calculate average margin across all distributors.
    
    Returns the arithmetic mean of MARGIN field.
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns:
        return None
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce').dropna()
    if len(margins) == 0:
        return None
    
    return round(float(margins.mean()), 2)

def get_weighted_avg_margin(table: Union[pd.DataFrame, list], volume_column: str = None) -> Optional[float]:
    """
    Calculate weighted average margin using sales volume if available.
    
    Formula: SUM(margin × volume) / SUM(volume)
    Falls back to simple average if volume_column not available or not provided.
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns:
        return None
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce')
    
    # If volume column provided and exists, use weighted average
    if volume_column and volume_column in df.columns:
        try:
            volumes = pd.to_numeric(df[volume_column], errors='coerce')
            weighted_sum = (margins * volumes).sum()
            total_volume = volumes.sum()
            
            if total_volume == 0:
                return None
            
            return round(float(weighted_sum / total_volume), 2)
        except Exception:
            # Fall back to simple average if weighting fails
            pass
    
    # Simple average fallback
    clean_margins = margins.dropna()
    if len(clean_margins) == 0:
        return None
    
    return round(float(clean_margins.mean()), 2)

def get_margin_per_payment_day(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """
    Calculate efficiency score: Average Margin per Payment Day.
    
    Formula: AVG(margin / payment_cycle_days)
    Higher score = better efficiency (higher margin, shorter cycle)
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns or "PAYMENT CYCLE" not in df.columns:
        return None
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce')
    cycles = pd.to_numeric(df["PAYMENT CYCLE"], errors='coerce')
    
    # Avoid division by zero
    mask = cycles > 0
    if not mask.any():
        return None
    
    efficiency = (margins[mask] / cycles[mask]).mean()
    return round(float(efficiency), 4) if not pd.isna(efficiency) else None

def get_total_margin_exposure(table: Union[pd.DataFrame, list], avg_monthly_sales_column: str = None) -> Optional[float]:
    """
    Calculate total margin exposure: SUM(margin × estimated_monthly_sales).
    
    This represents credit risk from high-margin distributors.
    If sales_column not provided, estimates using order count as proxy.
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns:
        return None
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce')
    
    # Use provided sales column or estimate
    if avg_monthly_sales_column and avg_monthly_sales_column in df.columns:
        sales = pd.to_numeric(df[avg_monthly_sales_column], errors='coerce')
    else:
        # Estimate: uniform distribution
        sales = pd.Series([1.0] * len(df))
    
    exposure = (margins * sales).sum()
    return round(float(exposure), 2) if not pd.isna(exposure) else None

def get_high_risk_distributors(table: Union[pd.DataFrame, list], margin_threshold: float = 30, cycle_threshold: int = 45) -> List[Dict]:
    """
    Identify high-risk distributors: margin > threshold AND payment_cycle > threshold.
    
    Args:
        table: Distributor records
        margin_threshold: Minimum margin % to flag (default: 30%)
        cycle_threshold: Maximum payment days before flagged as risky (default: 45)
    
    Returns:
        List of high-risk distributor records with fields:
        - PARTY NAME
        - MARGIN
        - PAYMENT CYCLE
        - CD (cash discount)
        - Risk score (calculated)
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns or "PAYMENT CYCLE" not in df.columns:
        return []
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce')
    cycles = pd.to_numeric(df["PAYMENT CYCLE"], errors='coerce')
    
    # Identify high-risk: high margin + long payment cycle
    high_risk_mask = (margins > margin_threshold) & (cycles > cycle_threshold)
    
    if not high_risk_mask.any():
        return []
    
    high_risk_df = df[high_risk_mask].copy()
    
    # Calculate risk score (higher = riskier)
    high_risk_df["risk_score"] = (
        (pd.to_numeric(high_risk_df["MARGIN"], errors='coerce') / 100) * 
        (pd.to_numeric(high_risk_df["PAYMENT CYCLE"], errors='coerce') / 45)
    ).round(3)
    
    # Sort by risk score descending
    high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)
    
    return high_risk_df[["PARTY NAME", "MARGIN", "PAYMENT CYCLE", "CD", "risk_score"]].to_dict('records')

def get_cycle_efficiency_score(table: Union[pd.DataFrame, list]) -> Optional[float]:
    """
    Calculate cycle efficiency score for portfolio.
    
    Formula: AVG((100 - margin) / payment_cycle_days)
    Lower margin + shorter cycle = lower score = better efficiency for you
    This inverts traditional logic: you want low scores (low margin exposure, fast turnaround)
    """
    df = _ensure_dataframe(table)
    if df.empty or "MARGIN" not in df.columns or "PAYMENT CYCLE" not in df.columns:
        return None
    
    margins = pd.to_numeric(df["MARGIN"], errors='coerce')
    cycles = pd.to_numeric(df["PAYMENT CYCLE"], errors='coerce')
    
    # Avoid division by zero
    mask = cycles > 0
    if not mask.any():
        return None
    
    # (100 - margin) / cycle: lower is better
    efficiency = ((100 - margins[mask]) / cycles[mask]).mean()
    return round(float(efficiency), 4) if not pd.isna(efficiency) else None

def get_payment_cycle_distribution(table: Union[pd.DataFrame, list]) -> dict:
    """
    Get distribution of payment cycles (grouped by day ranges).
    
    Returns counts by cycle ranges: <15 days, 15-30 days, 30-45 days, >45 days
    """
    df = _ensure_dataframe(table)
    if df.empty or "PAYMENT CYCLE" not in df.columns:
        return {}
    
    cycles = pd.to_numeric(df["PAYMENT CYCLE"], errors='coerce')
    
    distribution = {
        "very_short_cycle_0_15": int(((cycles >= 0) & (cycles < 15)).sum()),
        "short_cycle_15_30": int(((cycles >= 15) & (cycles < 30)).sum()),
        "medium_cycle_30_45": int(((cycles >= 30) & (cycles < 45)).sum()),
        "long_cycle_45plus": int((cycles >= 45).sum())
    }
    
    return distribution

def get_cash_discount_stats(table: Union[pd.DataFrame, list]) -> dict:
    """
    Calculate statistics for cash discount (CD) field.
    
    Returns: sum, mean, max, min, count of non-null CD values
    """
    df = _ensure_dataframe(table)
    if df.empty or "CD" not in df.columns:
        return {"sum": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0, "count": 0}
    
    cds = pd.to_numeric(df["CD"], errors='coerce').dropna()
    
    if len(cds) == 0:
        return {"sum": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0, "count": 0}
    
    return {
        "sum": round(float(cds.sum()), 2),
        "mean": round(float(cds.mean()), 2),
        "max": round(float(cds.max()), 2),
        "min": round(float(cds.min()), 2),
        "count": len(cds)
    }

#queries regarding listing highest / lowest, margin above 30% etc. to  be handled by CustomMetricCalculator()
PROFIT_TOOL_REGISTRY = {
    "get_vendor_cost_sheet": get_vendor_cost_sheet,
    "apply_filters": apply_filters,
    "get_cost_price": get_cost_price,
    "get_selling_price": get_selling_price, #current selling price, if price_i is not null
    "get_gross_profit": get_gross_profit, #selling-cost #for single sku => will get filtered data
    "get_margin": get_margin, #gp/selling*100 #for single sku => will get filtered data
    "get_markup": get_markup, #gp/cost
    "get_cost_to_price_ratio": get_cost_to_price_ratio,
    "execute_custom_calculation": execute_custom_calculation,  # Dynamic code generation for custom metrics
    "get_statistical_summary": get_statistical_summary,
    "get_percentile": get_percentile,
    "get_top_percentile": get_top_percentile,
    "get_bottom_percentile": get_bottom_percentile,
    "get_correlation_matrix": get_correlation_matrix
}

PAYMENT_CYCLE_TOOL_REGISTRY = {
    "get_payment_cycle_data": get_payment_cycle_data,
    "apply_filters": apply_filters,
    "get_avg_margin": get_avg_margin,
    "get_weighted_avg_margin": get_weighted_avg_margin,
    "get_margin_per_payment_day": get_margin_per_payment_day,
    "get_total_margin_exposure": get_total_margin_exposure,
    "get_high_risk_distributors": get_high_risk_distributors,
    "get_cycle_efficiency_score": get_cycle_efficiency_score,
    "get_payment_cycle_distribution": get_payment_cycle_distribution,
    "get_cash_discount_stats": get_cash_discount_stats,
    "execute_custom_calculation": execute_custom_calculation,
    "get_schema_info": get_schema_info,
    "get_statistical_summary": get_statistical_summary,
    "get_percentile": get_percentile,
}

# Tool registry mapping
ORDERS_TOOL_REGISTRY = {
    "get_all_orders": get_all_orders,
    "apply_filters": apply_filters,  # Enable early filtering optimization
    "get_schema_info": get_schema_info,
    "convert_to_df": convert_to_df,
    "execute_custom_calculation": execute_custom_calculation,  # Dynamic code generation for custom metrics
    "get_aov": get_aov,
    "get_total_revenue": get_total_revenue,
    "get_order_count": get_order_count,
    "get_cancelled_count": get_cancelled_count,
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


# ------------------ DEPRECATED -------------------
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
