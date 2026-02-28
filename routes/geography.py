import pandas as pd
from fastapi import APIRouter, HTTPException
from models import GeographyRequest

router = APIRouter()

@router.post('/geography/chart/pincode')
def pincode_list(request: GeographyRequest):
    """
       take state input and return the count(pincodes) within that state.
    """
    try:
        orders_data = request.orders
        target_state = request.state
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        if not target_state:
            raise HTTPException(status_code=400, detail="State parameter is required")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        # Check if required columns exist
        if 'state' not in df.columns or 'pin_code' not in df.columns:
            raise HTTPException(status_code=400, detail="Required geographic data (state, pin_code) not found in orders")
        
        # Remove rows with null state or pincode values
        df_clean = df.dropna(subset=['state', 'pin_code'])
        
        # Filter by target state (case-insensitive)
        state_filtered = df_clean[df_clean['state'].str.lower() == target_state.lower()]
        
        if state_filtered.empty:
            return {
                "success": True,
                "message": f"No orders found for state: {target_state}",
                "state": target_state,
                "pincode_count": 0,
                "pincodes": [],
                "pincode_details": {}
            }
        
        # Get unique pincodes for the target state
        unique_pincodes = state_filtered['pin_code'].unique().tolist()
        pincode_count = len(unique_pincodes)
        
        # Get order count per pincode for additional insights
        pincode_order_counts = state_filtered['pin_code'].value_counts().to_dict()
        
        # Sort pincodes by order count in decreasing order
        sorted_pincode_counts = dict(sorted(pincode_order_counts.items(), key=lambda item: item[1], reverse=True))
        unique_pincodes = list(sorted_pincode_counts.keys())
        
        return {
            "success": True,
            "state": target_state,
            "pincode_count": pincode_count,
            "pincodes": unique_pincodes,
            "pincode_details": {
                pincode: {
                    "order_count": sorted_pincode_counts[pincode]
                }
                for pincode in unique_pincodes
            },
            "chart_data": {
                "labels": list(sorted_pincode_counts.keys()),
                "values": list(sorted_pincode_counts.values())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error processing pincode data: {str(e)}"
            }
        )
