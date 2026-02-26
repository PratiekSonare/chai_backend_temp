import pandas as pd
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest

router = APIRouter()

@router.post('/revenue/chart/line')
def revenue_line_chart(request: OrdersMetricsRequest):
    """
        Depending on date-range, return time-based total revenue and AOV.
        If daily, use monday, tuesday, wednesday...
        If weekly, use date-ranges.
        If monthly, mention months.
        Returns both revenue and AOV for charting.
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
        
        # Convert total_amount to numeric
        if 'total_amount' in df.columns:
            df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(0)
        
        if df.empty:
            return {
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {
                    "revenue": [],
                    "aov": []
                }
            }
        
        # Get date range
        min_date = df['order_date'].min()
        max_date = df['order_date'].max()
        date_range_days = (max_date - min_date).days
        
        # Determine chart type based on date range
        if date_range_days <= 7:
            # Daily view
            chart_type = "daily"
            df['date_group'] = df['order_date'].dt.date
            grouped = df.groupby('date_group').agg({
                'total_amount': 'sum',
                'order_date': 'size'  # count of orders
            })
            labels = [d.strftime('%A') for d in grouped.index]  # Monday, Tuesday, etc.
                
        elif date_range_days <= 90:
            # Weekly view
            chart_type = "weekly"
            df['week_start'] = df['order_date'].dt.to_period('W').dt.start_time
            df['date_group'] = df['week_start'].dt.date
            grouped = df.groupby('date_group').agg({
                'total_amount': 'sum',
                'order_date': 'size'  # count of orders
            })
            labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped.index]
                
        else:
            # Monthly view
            chart_type = "monthly"
            df['date_group'] = df['order_date'].dt.to_period('M').dt.start_time
            grouped = df.groupby('date_group').agg({
                'total_amount': 'sum',
                'order_date': 'size'  # count of orders
            })
            labels = [d.strftime('%B %Y') for d in grouped.index]
        
        # Prepare data for chart.js
        revenue_data = []
        aov_data = []
        
        for _, row in grouped.iterrows():
            revenue = float(row['total_amount'])
            order_count = int(row['order_date'])
            aov = revenue / order_count if order_count > 0 else 0
            revenue_data.append(revenue)
            aov_data.append(aov)
        
        return {
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "titles": ["Total Revenue", "Average Order Value"],
            "datasets": {
                "revenue": revenue_data,
                "aov": aov_data
            },
            "total_revenue": float(df['total_amount'].sum()),
            "overall_aov": float(df['total_amount'].sum() / len(df)) if len(df) > 0 else 0,
            "date_range_days": int(date_range_days)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating revenue chart data: {str(e)}"
            }
        )
