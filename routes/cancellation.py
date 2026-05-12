import pandas as pd
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest, DateRangeOrdersRequest
from utils.type_converters import convert_numpy_types
from tools import get_all_orders

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
def rto_dashboard(request: DateRangeOrdersRequest):
    """
    RTO dashboard endpoint.
    Returns separated payloads for cancelled and returned orders within date range,
    plus per-section top states and pincodes and totals.
    """
    try:
        start_datetime = f"{request.start_date} 00:00:00"
        end_datetime = f"{request.end_date} 23:59:59"

        orders = get_all_orders(start_datetime, end_datetime)

        if not orders:
            return convert_numpy_types({
                "success": True,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "totals": {"orders": 0, "cancelled": 0, "returned": 0},
                "cancelled": {"count": 0, "top_states": [], "top_pincodes": [], "orders": []},
                "returned": {"count": 0, "top_states": [], "top_pincodes": [], "orders": []}
            })

        import pandas as pd

        df = pd.DataFrame(orders)

        # Normalize status
        status_series = df['order_status'].astype(str).str.strip().str.lower() if 'order_status' in df.columns else pd.Series([''] * len(df))

        cancelled_mask = status_series.str.contains('cancel', na=False)
        returned_mask = status_series.str.contains('return|rto', na=False)

        cancelled_df = df[cancelled_mask]
        returned_df = df[returned_mask]

        def top_counts(series, key_name='key', top_n=10):
            if series is None or series.empty:
                return []
            vc = series.fillna('UNKNOWN').astype(str).str.strip().value_counts().head(top_n)
            return [{key_name: k, 'count': int(v)} for k, v in vc.items()]

        # Determine pincode column name variants
        pincode_col = None
        for candidate in ['pin_code', 'pincode', 'pin', 'pinCode', 'postal_code']:
            if candidate in df.columns:
                pincode_col = candidate
                break

        cancelled_top_states = top_counts(cancelled_df['state']) if 'state' in cancelled_df.columns else []
        returned_top_states = top_counts(returned_df['state']) if 'state' in returned_df.columns else []

        cancelled_top_pincodes = top_counts(cancelled_df[pincode_col], key_name='pincode') if pincode_col and pincode_col in cancelled_df.columns else []
        returned_top_pincodes = top_counts(returned_df[pincode_col], key_name='pincode') if pincode_col and pincode_col in returned_df.columns else []

        # Limit orders list to reasonable size for response
        max_rows = 1000
        cancelled_orders = cancelled_df.to_dict(orient='records')[:max_rows]
        returned_orders = returned_df.to_dict(orient='records')[:max_rows]

        totals = {
            'orders': int(len(df)),
            'cancelled': int(len(cancelled_df)),
            'returned': int(len(returned_df))
        }

        return convert_numpy_types({
            'success': True,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'totals': totals,
            'cancelled': {
                'count': totals['cancelled'],
                'top_states': cancelled_top_states,
                'top_pincodes': cancelled_top_pincodes,
                'orders': cancelled_orders
            },
            'returned': {
                'count': totals['returned'],
                'top_states': returned_top_states,
                'top_pincodes': returned_top_pincodes,
                'orders': returned_orders
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail={
            'success': False,
            'error': f'Error generating RTO dashboard: {str(e)}'
        })