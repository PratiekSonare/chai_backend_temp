import pandas as pd
from fastapi import APIRouter, HTTPException
from models import OrdersMetricsRequest

router = APIRouter()

@router.post('/payment/chart/radial')
def payment_radial(request: OrdersMetricsRequest):
    """
        Return cod orders, prepaid orders.
    """
    try:
        orders_data = request.orders
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        payment_breakdown = df['payment_mode'].value_counts().to_dict() if 'payment_mode' in df.columns else {}
        cod_orders = payment_breakdown.get('COD', 0)
        prepaid_orders = sum(count for mode, count in payment_breakdown.items() if mode != 'COD')
        
        if df.empty:
            return {
                "success": True,
                "labels": ["COD, PrePaid"],
                "datasets": {
                    "cod": 0,
                    "aov": 0
                }
            }
        
        return {
            "success": True,
            "labels": ["COD", "PrePaid"],
            "data": {
                "cod": int(cod_orders),
                "prepaid": int(prepaid_orders)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating revenue chart data: {str(e)}"
            }
        )