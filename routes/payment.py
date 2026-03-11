import pandas as pd
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest
from utils.type_converters import convert_numpy_types

router = APIRouter()

@router.post('/payment/chart/radial')
def payment_radial(request: OrdersMetricsRequest):
    """
        Return time range-wise cod orders and prepaid orders for stacked bar charts.
        Depending on date-range, return time-based COD and prepaid counts.
        If daily, use monday, tuesday, wednesday...
        If weekly, use date-ranges.
        If monthly, mention months.
        Returns both individual counts and time-based breakdown for stacked charts.
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
        
        # Ensure payment_mode column exists
        if 'payment_mode' not in df.columns:
            df['payment_mode'] = 'Unknown'
        
        if df.empty:
            return convert_numpy_types({
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {
                    "cod": [],
                    "prepaid": []
                },
                "totals": {
                    "cod": 0,
                    "prepaid": 0
                }
            })
        
        # Get date range
        min_date = df['order_date'].min()
        max_date = df['order_date'].max()
        date_range_days = (max_date - min_date).days
        
        # Determine chart type based on date range
        if date_range_days <= 7:
            # Daily view
            chart_type = "daily"
            df['date_group'] = df['order_date'].dt.date
            grouped = df.groupby(['date_group', 'payment_mode']).size().unstack(fill_value=0)
            labels = [d.strftime('%a') for d in grouped.index]  # Mon, Tue, Wed, etc.
                
        elif date_range_days <= 90:
            # Weekly view
            chart_type = "weekly"
            df['week_start'] = df['order_date'].dt.to_period('W').dt.start_time
            df['date_group'] = df['week_start'].dt.date
            grouped = df.groupby(['date_group', 'payment_mode']).size().unstack(fill_value=0)
            labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped.index]
                
        else:
            # Monthly view
            chart_type = "monthly"
            df['date_group'] = df['order_date'].dt.to_period('M').dt.start_time
            grouped = df.groupby(['date_group', 'payment_mode']).size().unstack(fill_value=0)
            labels = [d.strftime('%b %Y') for d in grouped.index]
        
        # Prepare data for stacked bar chart
        cod_data = []
        prepaid_data = []
        
        for _, row in grouped.iterrows():
            cod_count = int(row.get('COD', 0))
            # Sum all non-COD payment modes as prepaid
            prepaid_count = int(sum(v for k, v in row.items() if k != 'COD'))
            
            cod_data.append(cod_count)
            prepaid_data.append(prepaid_count)
        
        # Calculate overall totals
        total_payment_breakdown = df['payment_mode'].value_counts().to_dict()
        total_cod = int(total_payment_breakdown.get('COD', 0))
        total_prepaid = int(sum(count for mode, count in total_payment_breakdown.items() if mode != 'COD'))
        
        return convert_numpy_types({
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "datasets": {
                "cod": cod_data,
                "prepaid": prepaid_data
            },
            "totals": {
                "cod": total_cod,
                "prepaid": total_prepaid
            },
            "date_range_days": int(date_range_days)
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating payment chart data: {str(e)}"
            }
        )