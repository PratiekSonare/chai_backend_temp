import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from datetime import datetime
from models import OrdersMetricsRequest

router = APIRouter()

@router.post('/orders/metrics')
def calculate_orders_metrics(request: OrdersMetricsRequest):
    """
    Calculate comprehensive order metrics from order data
    
    Request body:
    {
        "orders": [...]  // Array of order objects
    }
    
    Response: Comprehensive metrics including volume, revenue, payment, cancellation, time-based, geographic, and product metrics
    """
    try:
        orders_data = request.orders
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        # Flatten suborders data
        suborders_list = []
        for _, order in df.iterrows():
            if 'suborders' in order and order['suborders']:
                for suborder in order['suborders']:
                    suborder_row = {
                        'sku': suborder.get('sku'),
                        'productName': suborder.get('productName'),
                        'selling_price': suborder.get('selling_price', 0),
                        'cost': suborder.get('cost', 0),
                        'mrp': suborder.get('mrp', 0),
                        'item_quantity': suborder.get('item_quantity', 0),
                        'cancelled_quantity': suborder.get('cancelled_quantity', 0),
                        'shipped_quantity': suborder.get('shipped_quantity', 0),
                        'suborder_history': suborder.get('suborder_history', {}),
                        'order_date': order.get('order_date')
                    }
                    suborders_list.append(suborder_row)
        
        suborders_df = pd.DataFrame(suborders_list) if suborders_list else pd.DataFrame()
        
        # Convert date columns to datetime
        date_columns = ['order_date', 'invoice_date', 'import_date', 'last_update_date', 'available_after']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['total_amount', 'total_tax', 'total_shipping_charge', 'order_quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if not suborders_df.empty:
            suborder_numeric_cols = ['selling_price', 'cost', 'mrp', 'item_quantity', 'cancelled_quantity', 'shipped_quantity']
            for col in suborder_numeric_cols:
                if col in suborders_df.columns:
                    suborders_df[col] = pd.to_numeric(suborders_df[col], errors='coerce').fillna(0)
        
        # ============ VOLUME METRICS ============
        total_orders = len(df)
        
        if not suborders_df.empty:
            total_skus_sold = suborders_df['shipped_quantity'].sum()
            net_skus_sold = suborders_df['item_quantity'].sum() - suborders_df['cancelled_quantity'].sum()
            unique_skus_ordered = suborders_df['sku'].nunique() if 'sku' in suborders_df.columns else 0
        else:
            total_skus_sold = 0
            net_skus_sold = 0
            unique_skus_ordered = 0
        
        # State-wise order count
        state_wise_orders = df['state'].value_counts().to_dict() if 'state' in df.columns else {}
        
        volume_metrics = {
            "total_orders": int(total_orders),
            "total_skus_sold": int(total_skus_sold),
            "net_skus_sold": int(net_skus_sold),
            "unique_skus_ordered": int(unique_skus_ordered),
            "state_wise_order_count": state_wise_orders
        }
        
        # ============ REVENUE METRICS ============
        gross_revenue = df['total_amount'].sum()
        
        if not suborders_df.empty:
            gross_cost = suborders_df['cost'].sum() * suborders_df['item_quantity'].sum() / suborders_df['item_quantity'].sum() if suborders_df['item_quantity'].sum() > 0 else 0
            gross_margin = gross_revenue - gross_cost if gross_cost > 0 else 0
            avg_selling_price = suborders_df['selling_price'].mean() if len(suborders_df) > 0 else 0
        else:
            gross_margin = 0
            avg_selling_price = 0
        
        aov = gross_revenue / total_orders if total_orders > 0 else 0
        
        revenue_metrics = {
            "gross_revenue": float(gross_revenue),
            "gross_margin": float(gross_margin),
            "aov": float(aov),
            "asp": float(avg_selling_price)
        }
        
        # ============ PAYMENT METRICS ============
        payment_breakdown = df['payment_mode'].value_counts().to_dict() if 'payment_mode' in df.columns else {}
        cod_orders = payment_breakdown.get('COD', 0)
        prepaid_orders = sum(count for mode, count in payment_breakdown.items() if mode != 'COD')
        
        payment_metrics = {
            "cod_vs_prepaid": {
                "cod_orders": int(cod_orders),
                "prepaid_orders": int(prepaid_orders),
                "cod_percentage": float(cod_orders / total_orders * 100) if total_orders > 0 else 0,
                "prepaid_percentage": float(prepaid_orders / total_orders * 100) if total_orders > 0 else 0
            },
            "payment_mode_breakdown": payment_breakdown
        }
        
        # ============ CANCELLATION METRICS ============
        cancelled_orders = len(df[df['order_status'] == 'Cancelled']) if 'order_status' in df.columns else 0
        cancellation_rate = (cancelled_orders / total_orders * 100) if total_orders > 0 else 0
        
        # RTO Rate calculation (assuming RTO orders have specific status)
        rto_orders = len(df[df['order_status'].str.contains('RTO|Return', case=False, na=False)]) if 'order_status' in df.columns else 0
        rto_rate = (rto_orders / total_orders * 100) if total_orders > 0 else 0
        
        # Highest state RTO
        if 'state' in df.columns and 'order_status' in df.columns:
            state_rto = df[df['order_status'].str.contains('RTO|Return', case=False, na=False)]['state'].value_counts()
            highest_state_rto = state_rto.index[0] if len(state_rto) > 0 else None
        else:
            highest_state_rto = None
        
        cancellation_metrics = {
            "rto_rate": float(rto_rate),
            "cancellation_rate": float(cancellation_rate),
            "cancelled_orders": int(cancelled_orders),
            "rto_orders": int(rto_orders),
            "highest_state_rto": highest_state_rto
        }
        
        # ============ TIME-BASED METRICS ============
        if 'order_date' in df.columns:
            df['order_day'] = df['order_date'].dt.date
            day_wise_frequency = df['order_day'].value_counts().sort_index().to_dict()
            
            # 7-day moving average
            daily_counts = df['order_day'].value_counts().sort_index()
            moving_avg_7d = daily_counts.rolling(window=7, min_periods=1).mean().to_dict()
        else:
            day_wise_frequency = {}
            moving_avg_7d = {}
        
        # Processing times calculation
        processing_times = {}
        if not suborders_df.empty:
            for _, suborder in suborders_df.iterrows():
                if 'suborder_history' in suborder and suborder['suborder_history']:
                    history = suborder['suborder_history']
                    
                    # Inventory assignment time
                    if history.get('inventory_assigned_datetime') and 'order_date' in suborder:
                        inv_time = (pd.to_datetime(history['inventory_assigned_datetime']) - pd.to_datetime(suborder['order_date'])).total_seconds() / 3600
                        processing_times.setdefault('inventory_assignment_times', []).append(inv_time)
                    
                    # QC time
                    if history.get('qc_pass_datetime') and history.get('inventory_assigned_datetime'):
                        qc_time = (pd.to_datetime(history['qc_pass_datetime']) - pd.to_datetime(history['inventory_assigned_datetime'])).total_seconds() / 3600
                        processing_times.setdefault('qc_times', []).append(qc_time)
        
        avg_inventory_time = np.mean(processing_times.get('inventory_assignment_times', [0])) if processing_times.get('inventory_assignment_times') else 0
        avg_qc_time = np.mean(processing_times.get('qc_times', [0])) if processing_times.get('qc_times') else 0
        avg_processing_time = avg_inventory_time + avg_qc_time
        
        time_based_metrics = {
            "day_wise_frequency": {str(k): int(v) for k, v in day_wise_frequency.items()},
            "moving_average_7d": {str(k): float(v) for k, v in moving_avg_7d.items()},
            "avg_inventory_assignment_time_hours": float(avg_inventory_time),
            "avg_qc_time_hours": float(avg_qc_time),
            "avg_order_processing_time_hours": float(avg_processing_time)
        }
        
        # ============ GEOGRAPHIC METRICS ============
        revenue_by_state = df.groupby('state')['total_amount'].sum().to_dict() if 'state' in df.columns else {}
        
        cancellation_by_state = {}
        if 'state' in df.columns and 'order_status' in df.columns:
            cancelled_by_state = df[df['order_status'] == 'Cancelled'].groupby('state').size()
            total_by_state = df.groupby('state').size()
            cancellation_by_state = ((cancelled_by_state / total_by_state) * 100).fillna(0).to_dict()
        
        geographic_metrics = {
            "state_wise_order_count": state_wise_orders,
            "revenue_by_state": {k: float(v) for k, v in revenue_by_state.items()},
            "cancellation_by_state": {k: float(v) for k, v in cancellation_by_state.items()}
        }
        
        # ============ PRODUCT METRICS ============
        if not suborders_df.empty:
            # Top SKUs by quantity
            top_skus_qty = suborders_df.groupby('sku')['item_quantity'].sum().sort_values(ascending=False).head(10).to_dict()
            
            # Top SKUs by revenue
            suborders_df['revenue'] = suborders_df['selling_price'] * suborders_df['item_quantity']
            top_skus_revenue = suborders_df.groupby('sku')['revenue'].sum().sort_values(ascending=False).head(10).to_dict()
            
            # Top products by name
            top_products = suborders_df.groupby('productName')['item_quantity'].sum().sort_values(ascending=False).head(10).to_dict() if 'productName' in suborders_df.columns else {}
        else:
            top_skus_qty = {}
            top_skus_revenue = {}
            top_products = {}
        
        product_metrics = {
            "top_skus_by_quantity": {k: int(v) for k, v in top_skus_qty.items()},
            "top_skus_by_revenue": {k: float(v) for k, v in top_skus_revenue.items()},
            "top_products_by_quantity": {k: int(v) for k, v in top_products.items()}
        }
        
        # ============ SUMMARY RESPONSE ============
        return {
            "success": True,
            "total_orders_analyzed": int(total_orders),
            "analysis_timestamp": datetime.now().isoformat(),
            "metrics": {
                "volume_metrics": volume_metrics,
                "revenue_metrics": revenue_metrics,
                "payment_metrics": payment_metrics,
                "cancellation_metrics": cancellation_metrics,
                "time_based_metrics": time_based_metrics,
                "geographic_metrics": geographic_metrics,
                "product_metrics": product_metrics
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error calculating metrics: {str(e)}"
            }
        )

@router.post('/orders/chart/count')
def volume_count(request: OrdersMetricsRequest):
    """
        Depending on date-range, return monthly / weekly / daily count of orders and unique SKUs.
        If daily, use monday, tuesday, wednesday...
        If weekly, use date-ranges.
        If monthly, mention months.
        Returns both order count and unique SKU count for charting.
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
            return {
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "datasets": {
                    "order_count": [],
                    "unique_sku_count": []
                }
            }
        
        # Flatten suborders data for SKU analysis
        suborders_list = []
        for _, order in df.iterrows():
            if 'suborders' in order and order['suborders']:
                for suborder in order['suborders']:
                    suborder_row = {
                        'order_date': order['order_date'],
                        'sku': suborder.get('sku'),
                        'item_quantity': suborder.get('item_quantity', 0)
                    }
                    if suborder_row['sku']:  # Only include if SKU exists
                        suborders_list.append(suborder_row)
        
        suborders_df = pd.DataFrame(suborders_list) if suborders_list else pd.DataFrame()
        
        # Get date range
        min_date = df['order_date'].min()
        max_date = df['order_date'].max()
        date_range_days = (max_date - min_date).days
        
        # Determine chart type based on date range
        if date_range_days <= 7:
            # Daily view
            chart_type = "daily"
            df['date_group'] = df['order_date'].dt.date
            grouped_orders = df.groupby('date_group').size()
            labels = [d.strftime('%a') for d in grouped_orders.index]  # Mon, Tue, Wed, etc.
            
            # Group suborders by same date grouping for SKU count
            if not suborders_df.empty:
                suborders_df['date_group'] = pd.to_datetime(suborders_df['order_date']).dt.date
                grouped_skus = suborders_df.groupby('date_group')['sku'].nunique()
            else:
                grouped_skus = pd.Series([], dtype='int64')
                
        elif date_range_days <= 90:
            # Weekly view
            chart_type = "weekly"
            df['week_start'] = df['order_date'].dt.to_period('W').dt.start_time
            df['date_group'] = df['week_start'].dt.date
            grouped_orders = df.groupby('date_group').size()
            labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped_orders.index]
            
            # Group suborders by same weekly grouping for SKU count
            if not suborders_df.empty:
                suborders_df['week_start'] = pd.to_datetime(suborders_df['order_date']).dt.to_period('W').dt.start_time
                suborders_df['date_group'] = suborders_df['week_start'].dt.date
                grouped_skus = suborders_df.groupby('date_group')['sku'].nunique()
            else:
                grouped_skus = pd.Series([], dtype='int64')
                
        else:
            # Monthly view
            chart_type = "monthly"
            df['date_group'] = df['order_date'].dt.to_period('M').dt.start_time
            grouped_orders = df.groupby('date_group').size()
            labels = [d.strftime('%b %Y') for d in grouped_orders.index]
            
            # Group suborders by same monthly grouping for SKU count
            if not suborders_df.empty:
                suborders_df['date_group'] = pd.to_datetime(suborders_df['order_date']).dt.to_period('M').dt.start_time
                grouped_skus = suborders_df.groupby('date_group')['sku'].nunique()
            else:
                grouped_skus = pd.Series([], dtype='int64')
        
        # Prepare data for chart.js - align both datasets to same date groups
        order_counts = []
        sku_counts = []
        
        for date_group in grouped_orders.index:
            order_counts.append(int(grouped_orders[date_group]))
            sku_counts.append(int(grouped_skus.get(date_group, 0)))
        
        return {
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "titles": ["Total Orders", "Unique SKUs"],
            "datasets": {
                "order_count": order_counts,
                "unique_sku_count": sku_counts
            },
            "total_orders": int(len(df)),
            "total_unique_skus": int(suborders_df['sku'].nunique() if not suborders_df.empty else 0),
            "date_range_days": int(date_range_days)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating chart data: {str(e)}"
            }
        )
