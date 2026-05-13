import os
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest, DateRangeOrdersRequest
from utils.type_converters import convert_numpy_types
from tools import get_all_orders

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/cancellation/chart/bar')
def cancellation_bar_chart(request: OrdersMetricsRequest):
    """
        Return time range-wise cancelled and returned orders count.
        Depending on date-range, return time-based cancellation/return counts.
        If daily, use monday, tuesday, wednesday...
        If weekly, use date-ranges.
        If monthly, mention months.
        Returns both cancelled and returned order counts for time-based bar charts.
    """
    try:
        orders_data = request.orders
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        # Convert order_date to datetime
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.dropna(subset=['order_date'])
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {
                    "cancelled": [],
                    "returned": []
                },
                "totals": {
                    "cancelled": 0,
                    "returned": 0
                }
            })
        
        # Expand suborders to get item-level data for cancelled/returned quantities
        expanded_data = []
        for _, order in df.iterrows():
            if 'suborders' in order and isinstance(order['suborders'], list):
                for suborder in order['suborders']:
                    if isinstance(suborder, dict) and 'items' in suborder:
                        for item in suborder['items']:
                            if isinstance(item, dict):
                                expanded_data.append({
                                    'order_date': order['order_date'],
                                    'cancelled_quantity': item.get('cancelled_quantity', 0),
                                    'returned_quantity': item.get('returned_quantity', 0),
                                    'order_status': order.get('order_status', ''),
                                    'shipping_status': order.get('shipping_status', '')
                                })
        
        if not expanded_data:
            # Fallback: use order-level status for cancellation detection
            for _, order in df.iterrows():
                is_cancelled = 1 if 'cancel' in str(order.get('order_status', '')).lower() or 'cancel' in str(order.get('shipping_status', '')).lower() else 0
                is_returned = 1 if 'return' in str(order.get('order_status', '')).lower() or 'return' in str(order.get('shipping_status', '')).lower() else 0
                
                expanded_data.append({
                    'order_date': order['order_date'],
                    'cancelled_quantity': is_cancelled,
                    'returned_quantity': is_returned,
                    'order_status': order.get('order_status', ''),
                    'shipping_status': order.get('shipping_status', '')
                })
        
        items_df = pd.DataFrame(expanded_data)
        
        # Get date range
        min_date = items_df['order_date'].min()
        max_date = items_df['order_date'].max()
        date_range_days = (max_date - min_date).days
        
        # Determine chart type based on date range
        if date_range_days <= 7:
            # Daily view
            chart_type = "daily"
            items_df['date_group'] = items_df['order_date'].dt.date
            grouped = items_df.groupby('date_group').agg({
                'cancelled_quantity': 'sum',
                'returned_quantity': 'sum'
            }).reset_index()
            labels = [d.strftime('%a') for d in grouped['date_group']]
                
        elif date_range_days <= 90:
            # Weekly view
            chart_type = "weekly"
            items_df['week_start'] = items_df['order_date'].dt.to_period('W').dt.start_time
            items_df['date_group'] = items_df['week_start'].dt.date
            grouped = items_df.groupby('date_group').agg({
                'cancelled_quantity': 'sum',
                'returned_quantity': 'sum'
            }).reset_index()
            labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped['date_group']]
                
        else:
            # Monthly view
            chart_type = "monthly"
            items_df['date_group'] = items_df['order_date'].dt.to_period('M').dt.start_time
            grouped = items_df.groupby('date_group').agg({
                'cancelled_quantity': 'sum',
                'returned_quantity': 'sum'
            }).reset_index()
            labels = [d.strftime('%b %Y') for d in grouped['date_group']]
        
        # Prepare data for bar chart
        cancelled_data = grouped['cancelled_quantity'].tolist()
        returned_data = grouped['returned_quantity'].tolist()
        
        # Calculate overall totals
        total_cancelled = int(items_df['cancelled_quantity'].sum())
        total_returned = int(items_df['returned_quantity'].sum())
        
        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {
                "cancelled": cancelled_data,
                "returned": returned_data
            },
            "totals": {
                "cancelled": total_cancelled,
                "returned": total_returned
            },
            "date_range_days": int(date_range_days),
            "total_orders_analyzed": len(df)
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating cancellation chart data: {str(e)}"
            }
        )


@router.post('/cancellation/rto')
def calculate_section_metrics(request: DateRangeOrdersRequest) -> Dict:
    """
    Calculate comprehensive metrics for a section (cancelled, returned, pending_returns).
    Fetches orders from the date range and returns metrics.
    """
    # Fetch orders based on date range
    start_datetime = f"{request.start_date} 00:00:00"
    end_datetime = f"{request.end_date} 23:59:59"
    orders_list = get_all_orders(start_datetime, end_datetime)
    df = pd.DataFrame(orders_list)
    section_type = request.order_type or "cancelled"
    return _calculate_metrics_from_df(df, section_type)


def _calculate_metrics_from_df(df: pd.DataFrame, section_type: str = "cancelled") -> Dict:
    """Calculate metrics from a DataFrame"""
    if df.empty:
        return {
            'count': 0,
            'top_states': [],
            'top_pincodes': [],
            'payment_mode_breakdown': [],
            'marketplace_breakdown': [],
            'fulfillment_stage_breakdown': {},
            'courier_breakdown': [],
            'top_skus': [],
            'revenue_impact': 0,
            'avg_days_to_cancellation': 0,
            'repeat_offenders': [],
            'warehouse_breakdown': [],
            'orders': []
        }
    
    # Determine pincode column
    pincode_col = None
    for candidate in ['pin_code', 'pincode', 'pin', 'pinCode', 'postal_code']:
        if candidate in df.columns:
            pincode_col = candidate
            break
    
    metrics = {
        'count': len(df),
        'top_states': top_counts(df['state']) if 'state' in df.columns else [],
        'top_pincodes': top_counts(df[pincode_col], key_name='pincode') if pincode_col and pincode_col in df.columns else [],
    }
    
    # Payment mode breakdown
    if 'payment_mode' in df.columns:
        payment_counts = df['payment_mode'].fillna('Unknown').value_counts()
        metrics['payment_mode_breakdown'] = [
            {'mode': str(k), 'count': int(v), 'percentage': round(v/len(df)*100, 2)} 
            for k, v in payment_counts.items()
        ]
    else:
        metrics['payment_mode_breakdown'] = []
    
    # Marketplace breakdown
    if 'marketplace' in df.columns:
        marketplace_counts = df['marketplace'].fillna('Unknown').value_counts()
        metrics['marketplace_breakdown'] = [
            {'marketplace': str(k), 'count': int(v), 'percentage': round(v/len(df)*100, 2)} 
            for k, v in marketplace_counts.items()
        ]
    else:
        metrics['marketplace_breakdown'] = []
    
    # Fulfillment stage: pre-manifest vs post-manifest
    fulfillment_breakdown = {'pre_manifest': 0, 'post_manifest': 0}
    if 'manifest_date' in df.columns:
        fulfillment_breakdown['pre_manifest'] = int((df['manifest_date'].isna()).sum())
        fulfillment_breakdown['post_manifest'] = int((df['manifest_date'].notna()).sum())
    metrics['fulfillment_stage_breakdown'] = fulfillment_breakdown
    
    # Courier breakdown
    if 'courier' in df.columns:
        courier_counts = df['courier'].fillna('Unknown').value_counts().head(10)
        metrics['courier_breakdown'] = [
            {'courier': str(k), 'count': int(v), 'percentage': round(v/len(df)*100, 2)} 
            for k, v in courier_counts.items()
        ]
    else:
        metrics['courier_breakdown'] = []
    
    # Top SKUs by cancellation (from suborders/items)
    top_skus_list = []
    for _, order in df.iterrows():
        if 'suborders' in order and isinstance(order['suborders'], list):
            for suborder in order['suborders']:
                if isinstance(suborder, dict) and 'items' in suborder:
                    for item in suborder['items']:
                        if isinstance(item, dict) and 'sku_code' in item:
                            top_skus_list.append(item['sku_code'])
    
    if top_skus_list:
        sku_series = pd.Series(top_skus_list)
        sku_counts = sku_series.value_counts().head(10)
        metrics['top_skus'] = [
            {'sku': str(k), 'count': int(v)} 
            for k, v in sku_counts.items()
        ]
    else:
        metrics['top_skus'] = []
    
    # Revenue impact
    if 'total_amount' in df.columns:
        try:
            revenue_lost = float(pd.to_numeric(df['total_amount'], errors='coerce').fillna(0).sum())
            metrics['revenue_impact'] = round(revenue_lost, 2)
        except Exception as e:
            logger.warning(f"Error calculating revenue impact: {e}")
            metrics['revenue_impact'] = 0.0
    else:
        metrics['revenue_impact'] = 0.0
    
    # Average days to cancellation
    avg_days = 0.0
    if 'order_date' in df.columns and 'last_update_date' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], errors='coerce')
            df_copy['last_update_date'] = pd.to_datetime(df_copy['last_update_date'], errors='coerce')
            valid_dates = df_copy[(df_copy['order_date'].notna()) & (df_copy['last_update_date'].notna())]
            if not valid_dates.empty:
                avg_days = float((valid_dates['last_update_date'] - valid_dates['order_date']).dt.total_seconds().mean() / (24 * 3600))
                avg_days = round(avg_days, 2)
        except Exception as e:
            logger.warning(f"Error calculating average days to cancellation: {e}")
    metrics['avg_days_to_cancellation'] = avg_days
    
    # Repeat offenders (order_id appearing multiple times)
    if 'order_id' in df.columns:
        order_counts = df['order_id'].value_counts()
        repeat_offenders = order_counts[order_counts > 1].head(5)
        metrics['repeat_offenders'] = [
            {'order_id': int(k), 'count': int(v)} 
            for k, v in repeat_offenders.items()
        ]
    else:
        metrics['repeat_offenders'] = []
    
    # Warehouse breakdown
    if 'import_warehouse_id' in df.columns:
        warehouse_counts = df['import_warehouse_id'].fillna('Unknown').value_counts().head(10)
        metrics['warehouse_breakdown'] = [
            {'warehouse_id': str(k), 'count': int(v), 'percentage': round(v/len(df)*100, 2)} 
            for k, v in warehouse_counts.items()
        ]
    else:
        metrics['warehouse_breakdown'] = []
    
    return metrics


@router.post('/cancellation/rto/dashboard')
def rto_dashboard(request: DateRangeOrdersRequest):
    """
    RTO dashboard endpoint.
    Returns separated payloads for:
    - cancelled orders (from S3 via order_status filtering)
    - returned orders (from EasyEcom getAllReturns API)
    - pending returns (from EasyEcom getPendingReturns API)
    Plus comprehensive metrics: payment mode, marketplace, fulfillment stage, courier, SKUs, revenue impact, etc.
    """
    try:
        start_datetime = f"{request.start_date} 00:00:00"
        end_datetime = f"{request.end_date} 23:59:59"

        # Fetch cancelled orders from S3
        orders = get_all_orders(start_datetime, end_datetime)

        api_key = os.getenv("EASYECOM_API_KEY")
        jwt_token = os.getenv("EASYECOM_JWT_TOKEN")
        
        # Fetch returned and pending returns from EasyEcom APIs
        completed_returns = []
        pending_returns = []
        
        if api_key and jwt_token:
            try:
                completed_returns = fetch_returns_for_window(start_datetime, end_datetime, api_key, jwt_token)
                pending_returns = fetch_pending_returns_for_window(start_datetime, end_datetime, api_key, jwt_token)
            except Exception as e:
                logger.warning(f"Failed to fetch returns from EasyEcom API: {e}. Continuing with empty returns.")
        else:
            logger.warning("EASYECOM_API_KEY or EASYECOM_JWT_TOKEN not configured. Skipping returns API fetch.")

        if not orders and not completed_returns and not pending_returns:
            return convert_numpy_types({
                "success": True,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "totals": {"orders": 0, "cancelled": 0, "returned": 0, "pending_returns": 0},
                "cancelled": {
                    "count": 0, "top_states": [], "top_pincodes": [], 
                    "payment_mode_breakdown": [], "marketplace_breakdown": [],
                    "fulfillment_stage_breakdown": {}, "courier_breakdown": [], "top_skus": [],
                    "revenue_impact": 0, "avg_days_to_cancellation": 0, "repeat_offenders": [],
                    "warehouse_breakdown": [], "orders": []
                },
                "returned": {
                    "count": 0, "top_states": [], "top_pincodes": [],
                    "payment_mode_breakdown": [], "marketplace_breakdown": [],
                    "fulfillment_stage_breakdown": {}, "courier_breakdown": [], "top_skus": [],
                    "revenue_impact": 0, "avg_days_to_cancellation": 0, "repeat_offenders": [],
                    "warehouse_breakdown": [], "orders": []
                },
                "pending_returns": {
                    "count": 0, "top_states": [], "top_pincodes": [],
                    "payment_mode_breakdown": [], "marketplace_breakdown": [],
                    "fulfillment_stage_breakdown": {}, "courier_breakdown": [], "top_skus": [],
                    "revenue_impact": 0, "avg_days_to_cancellation": 0, "repeat_offenders": [],
                    "warehouse_breakdown": [], "orders": []
                }
            })

        # Process cancelled orders from S3
        cancelled_df = pd.DataFrame()
        cancelled_orders = []
        cancelled_metrics = {}
        
        if orders:
            df = pd.DataFrame(orders)
            # Normalize status for cancelled detection
            status_series = df['order_status'].astype(str).str.strip().str.lower() if 'order_status' in df.columns else pd.Series([''] * len(df))
            cancelled_mask = status_series.str.contains('cancel', na=False)
            cancelled_df = df[cancelled_mask]
            
            if not cancelled_df.empty:
                cancelled_metrics = _calculate_metrics_from_df(cancelled_df, "cancelled")
                max_rows = 1000
                cancelled_orders = cancelled_df.to_dict(orient='records')[:max_rows]
                cancelled_metrics['orders'] = cancelled_orders
            else:
                cancelled_metrics = _calculate_metrics_from_df(pd.DataFrame(), "cancelled")
        
        # Process completed returns from API
        returned_metrics = {}
        if completed_returns:
            returned_df = pd.DataFrame(completed_returns)
            returned_metrics = _calculate_metrics_from_df(returned_df, "returned")
            max_rows = 1000
            returned_metrics['orders'] = completed_returns[:max_rows]
        else:
            returned_metrics = _calculate_metrics_from_df(pd.DataFrame(), "returned")
        
        # Process pending returns from API
        pending_metrics = {}
        if pending_returns:
            pending_df = pd.DataFrame(pending_returns)
            pending_metrics = _calculate_metrics_from_df(pending_df, "pending_returns")
        else:
            pending_metrics = _calculate_metrics_from_df(pd.DataFrame(), "pending_returns")
        
        totals = {
            'orders': int(len(orders)) if orders else 0,
            'cancelled': cancelled_metrics.get('count', 0),
            'returned': returned_metrics.get('count', 0),
            'pending_returns': pending_metrics.get('count', 0)
        }

        return convert_numpy_types({
            'success': True,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'totals': totals,
            'cancelled': cancelled_metrics if cancelled_metrics else _calculate_metrics_from_df(pd.DataFrame(), "cancelled"),
            'returned': returned_metrics if returned_metrics else _calculate_metrics_from_df(pd.DataFrame(), "returned"),
            'pending_returns': pending_metrics if pending_metrics else _calculate_metrics_from_df(pd.DataFrame(), "pending_returns")
        })

    except Exception as e:
        logger.error(f"Error generating RTO dashboard: {e}")
        raise HTTPException(status_code=500, detail={
            'success': False,
            'error': f'Error generating RTO dashboard: {str(e)}'
        })


def top_counts(series: pd.Series, key_name: str = 'key', top_n: int = 10) -> List[Dict]:
    """
    Helper function to extract top N counts from a pandas Series.
    Returns list of dicts with key_name and 'count' fields.
    """
    if series is None or series.empty:
        return []
    vc = series.fillna('UNKNOWN').astype(str).str.strip().value_counts().head(top_n)
    return [{key_name: k, 'count': int(v)} for k, v in vc.items()]


def fetch_returns_for_window(
    start_date: str,
    end_date: str,
    api_key: str,
    jwt_token: str,
    base_url: str = "https://api.easyecom.io",
) -> List[Dict]:
    """
    Fetch all completed returns (credit notes) from EasyEcom /orders/getAllReturns endpoint
    with pagination support, filtering by created_after date.
    
    Args:
        start_date: Date string in format YYYY-MM-DD HH:MM:SS to start fetching from (used as created_after)
        end_date: End date (stored but not directly used in API call; filtering may be needed post-API)
        api_key: EasyEcom API key
        jwt_token: EasyEcom JWT token
        base_url: Base URL for EasyEcom API
        
    Returns:
        List of credit_note objects from the API response
    """
    all_returns = []
    url = f"{base_url}/orders/getAllReturns"
    
    # Extract date-only portion from start_date (format: YYYY-MM-DD HH:MM:SS -> YYYY-MM-DD)
    created_after_date = start_date.split(' ')[0] if ' ' in start_date else start_date
    
    params = {
        "limit": 250,
        "created_after": created_after_date,
    }
    
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }
    
    page = 1
    
    while True:
        try:
            logger.info(f"Fetching page {page} of completed returns from {created_after_date}")
            response = requests.get(url, params=params, headers=headers, timeout=60)
            
            # API returns 400 when pagination ends
            if response.status_code == 400:
                logger.info(f"Pagination ended with 400 at page {page}")
                break
            
            response.raise_for_status()
            payload = response.json()
            
            if payload.get("code") != 200 or "data" not in payload:
                logger.warning(f"API returned unexpected payload at page {page}: {payload}")
                break
            
            page_returns = payload["data"].get("credit_notes", [])
            if not page_returns:
                logger.info(f"No returns found on page {page}, stopping")
                break
            
            all_returns.extend(page_returns)
            logger.info(f"Fetched {len(page_returns)} completed returns on page {page}")
            
            next_url = payload["data"].get("nextUrl")
            if not next_url:
                logger.info(f"No nextUrl found at page {page}, stopping")
                break
            
            # Handle nextUrl: can be relative, absolute, or partial path
            if next_url.startswith("/"):
                url = f"{base_url}{next_url}"
            elif next_url.startswith("http"):
                url = next_url
            else:
                url = f"{base_url}/{next_url.lstrip('/')}"
            
            # nextUrl already includes query params, clear params for subsequent requests
            params = {}
            page += 1
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching completed returns on page {page}: {e}")
            break
    
    logger.info(f"Total completed returns fetched: {len(all_returns)}")
    return all_returns


def fetch_pending_returns_for_window(
    start_date: str,
    end_date: str,
    api_key: str,
    jwt_token: str,
    base_url: str = "https://api.easyecom.io",
) -> List[Dict]:
    """
    Fetch all pending returns from EasyEcom /getPendingReturns endpoint
    with pagination support, filtering by created_after date.
    
    Args:
        start_date: Date string in format YYYY-MM-DD HH:MM:SS to start fetching from (used as created_after)
        end_date: End date (stored but not directly used in API call; filtering may be needed post-API)
        api_key: EasyEcom API key
        jwt_token: EasyEcom JWT token
        base_url: Base URL for EasyEcom API
        
    Returns:
        List of pending_return objects from the API response
    """
    all_returns = []
    url = f"{base_url}/getPendingReturns"
    
    # Extract date-only portion from start_date
    created_after_date = start_date.split(' ')[0] if ' ' in start_date else start_date
    
    params = {
        "limit": 250,
        "created_after": created_after_date,
    }
    
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }
    
    page = 1
    
    while True:
        try:
            logger.info(f"Fetching page {page} of pending returns from {created_after_date}")
            response = requests.get(url, params=params, headers=headers, timeout=60)
            
            # API returns 400 when pagination ends
            if response.status_code == 400:
                logger.info(f"Pagination ended with 400 at page {page}")
                break
            
            response.raise_for_status()
            payload = response.json()
            
            if payload.get("code") != 200 or "data" not in payload:
                logger.warning(f"API returned unexpected payload at page {page}: {payload}")
                break
            
            page_returns = payload["data"].get("pending_returns", [])
            if not page_returns:
                logger.info(f"No pending returns found on page {page}, stopping")
                break
            
            all_returns.extend(page_returns)
            logger.info(f"Fetched {len(page_returns)} pending returns on page {page}")
            
            next_url = payload["data"].get("nextUrl")
            if not next_url:
                logger.info(f"No nextUrl found at page {page}, stopping")
                break
            
            # Handle nextUrl: can be relative, absolute, or partial path
            if next_url.startswith("/"):
                url = f"{base_url}{next_url}"
            elif next_url.startswith("http"):
                url = next_url
            else:
                url = f"{base_url}/{next_url.lstrip('/')}"
            
            # nextUrl already includes query params, clear params for subsequent requests
            params = {}
            page += 1
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching pending returns on page {page}: {e}")
            break
    
    logger.info(f"Total pending returns fetched: {len(all_returns)}")
    return all_returns


@router.post('/cancellation/returns/direct')
def fetch_returns_by_date_range(request: DateRangeOrdersRequest):
    """
    Fetch completed and pending returns directly from EasyEcom APIs.
    Returns separated payloads for completed returns and pending returns,
    plus per-section top states and pincodes and totals.
    """
    try:
        api_key = os.getenv("EASYECOM_API_KEY")
        jwt_token = os.getenv("EASYECOM_JWT_TOKEN")
        
        if not api_key or not jwt_token:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": "EASYECOM_API_KEY and EASYECOM_JWT_TOKEN not configured"
                }
            )
        
        start_datetime = f"{request.start_date} 00:00:00"
        end_datetime = f"{request.end_date} 23:59:59"
        
        # Fetch both completed and pending returns
        completed_returns = fetch_returns_for_window(start_datetime, end_datetime, api_key, jwt_token)
        pending_returns = fetch_pending_returns_for_window(start_datetime, end_datetime, api_key, jwt_token)
        
        if not completed_returns and not pending_returns:
            return convert_numpy_types({
                "success": True,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "totals": {"completed_returns": 0, "pending_returns": 0},
                "completed_returns": {"count": 0, "top_states": [], "top_pincodes": [], "orders": []},
                "pending_returns": {"count": 0, "top_states": [], "top_pincodes": [], "orders": []}
            })
        
        # Process completed returns
        max_rows = 1000
        completed_orders = completed_returns[:max_rows]
        
        # Extract states and pincodes from completed returns
        completed_states = [order.get("forward_shipment_customer_state") for order in completed_returns if order.get("forward_shipment_customer_state")]
        completed_pincodes = [order.get("forward_shipment_customer_pin_code") for order in completed_returns if order.get("forward_shipment_customer_pin_code")]
        
        completed_top_states = top_counts(pd.Series(completed_states), key_name='key') if completed_states else []
        completed_top_pincodes = top_counts(pd.Series(completed_pincodes), key_name='pincode') if completed_pincodes else []
        
        # Process pending returns
        pending_orders = pending_returns[:max_rows]
        
        # Extract states and pincodes from pending returns
        pending_states = [order.get("forward_shipment_customer_state") for order in pending_returns if order.get("forward_shipment_customer_state")]
        pending_pincodes = [order.get("forward_shipment_customer_pin_code") for order in pending_returns if order.get("forward_shipment_customer_pin_code")]
        
        pending_top_states = top_counts(pd.Series(pending_states), key_name='key') if pending_states else []
        pending_top_pincodes = top_counts(pd.Series(pending_pincodes), key_name='pincode') if pending_pincodes else []
        
        totals = {
            'completed_returns': int(len(completed_returns)),
            'pending_returns': int(len(pending_returns))
        }
        
        return convert_numpy_types({
            'success': True,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'totals': totals,
            'completed_returns': {
                'count': totals['completed_returns'],
                'top_states': completed_top_states,
                'top_pincodes': completed_top_pincodes,
                'orders': completed_orders
            },
            'pending_returns': {
                'count': totals['pending_returns'],
                'top_states': pending_top_states,
                'top_pincodes': pending_top_pincodes,
                'orders': pending_orders
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching returns: {e}")
        raise HTTPException(status_code=500, detail={
            'success': False,
            'error': f'Error fetching returns: {str(e)}'
        })