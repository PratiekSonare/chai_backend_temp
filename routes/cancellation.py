import pandas as pd
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest
from utils.type_converters import convert_numpy_types

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