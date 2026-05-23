import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import logging
import re
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Model paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "lightgbm_footwear_prediction_model_jan.joblib")

# Global model - loaded on first request
_model = None


def load_model():
    """Load the LightGBM model"""
    global _model
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info(f"✓ LightGBM Model loaded successfully from {MODEL_PATH}")
        logger.info(f"Model type: {type(_model).__name__}")
    
    return _model


def get_model_feature_names(model):
    """Extract feature names from LightGBM model (handles both Booster and LGBMClassifier)"""
    # If it's an LGBMClassifier (sklearn wrapper), get the booster
    if hasattr(model, 'booster_'):
        booster = model.booster_
    else:
        booster = model
    
    # Try to get feature names from booster
    try:
        # For Booster, feature_name is a method
        if hasattr(booster, 'feature_name'):
            feature_names = booster.feature_name()
            if feature_names and isinstance(feature_names, (list, tuple)):
                return feature_names
    except (TypeError, AttributeError):
        pass
    
    # Try feature_names_ property
    if hasattr(booster, 'feature_names_'):
        return booster.feature_names_
    
    # Fallback
    return None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    # Order data fields that the model expects
    order_data: Dict[str, Any] = None
    
    # Alternative: accept as list for batch processing
    orders: Optional[List[Dict[str, Any]]] = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    prediction: Optional[float] = None
    prediction_probability: Optional[float] = None
    class_0_prob: Optional[float] = None
    class_1_prob: Optional[float] = None
    message: str


def transform_order_to_features(order: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw order data into model features"""
    
    def clean_size(size_str):
        """Extract numeric size from string"""
        if pd.isna(size_str) or size_str is None:
            return np.nan
        s = str(size_str).replace(',', '').strip()
        s = re.sub(r'[^\d.]', '', s)
        return pd.to_numeric(s, errors='coerce')
    
    def parse_sku_and_extract_size(sku_str):
        """Parse SKU and extract size from end"""
        sku_str = str(sku_str).strip().upper()
        if '-' in sku_str:
            parts = sku_str.rsplit('-', 1)
            if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
                return parts[0], pd.to_numeric(parts[1], errors='coerce')
        return sku_str, np.nan
    
    # Parse date
    order_date_raw = order.get('order_date', '')
    order_date_str = str(order_date_raw).strip()
    date_to_parse = order_date_str.split()[0] if ' ' in order_date_str else order_date_str
    order_date = pd.to_datetime(date_to_parse, errors='coerce')
    if pd.isna(order_date):
        order_date = datetime.now()
    order_date_hour = order_date.hour if pd.notna(order_date) else 0
    
    # Parse SKUs
    raw_sku = order.get('sku', 'UNKNOWN')
    clean_sku_val, extracted_sku_size = parse_sku_and_extract_size(raw_sku)
    
    raw_suborder_sku = order.get('suborder_sku', 'UNKNOWN')
    clean_suborder_sku_val, extracted_suborder_sku_size = parse_sku_and_extract_size(raw_suborder_sku)
    
    # Get sizes
    order_size_from_data = clean_size(order.get('size', 0))
    final_size = extracted_sku_size if pd.notna(extracted_sku_size) else order_size_from_data
    
    suborder_size_from_data = clean_size(order.get('suborder_size', 0))
    final_suborder_size = extracted_suborder_sku_size if pd.notna(extracted_suborder_sku_size) else suborder_size_from_data
    
    # Get selling price
    suborder_selling_price = float(order.get('suborder_selling_price', 0))
    if suborder_selling_price > 0:
        suborder_selling_price = (int(suborder_selling_price // 100) + 1) * 99
    
    return {
        'order_id': order.get('order_id'),
        'invoice_id': str(order.get('invoice_id', '')),
        'item_quantity': int(order.get('item_quantity', 0)),
        'suborder_quantity': int(order.get('suborder_quantity', 0)),
        'clean_sku': clean_sku_val,
        'clean_suborder_sku': clean_suborder_sku_val,
        'marketplace': str(order.get('marketplace', 'UNKNOWN')).upper(),
        'payment_mode': str(order.get('payment_mode', 'UNKNOWN')).upper(),
        'state': str(order.get('state', 'UNKNOWN')).upper(),
        'billing_state': str(order.get('billing_state', 'UNKNOWN')).upper(),
        'size': final_size if pd.notna(final_size) else 0,
        'suborder_size': final_suborder_size if pd.notna(final_suborder_size) else 0,
        'suborder_selling_price': suborder_selling_price,
        'order_date_hour': order_date_hour
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Run a single data point (or batch) through the LightGBM model
    
    Process:
    1. Accept raw order data
    2. Transform into suitable features
    3. Run through LightGBM model
    4. Return prediction probability
    
    Example request body:
    {
        "order_data": {
            "marketplace": "MYNTRA PPMP",
            "order_status": "Delivered",
            "payment_mode": "PrePaid",
            "pincode_bin": 110001,
            ...
        }
    }
    
    Or for batch:
    {
        "orders": [
            {"marketplace": "MYNTRA PPMP", ...},
            {"marketplace": "FLIPKART", ...}
        ]
    }
    """
    try:
        model = load_model()
        
        # Determine if single or batch prediction
        if request.order_data:
            orders = [request.order_data]
            is_single = True
        elif request.orders:
            orders = request.orders
            is_single = False
        else:
            raise ValueError("Either 'order_data' or 'orders' must be provided")
        
        # Transform orders to features
        transformed_records = []
        for order in orders:
            try:
                features = transform_order_to_features(order)
                transformed_records.append(features)
            except Exception as e:
                logger.warning(f"Failed to transform order {order.get('order_id', 'unknown')}: {e}")
                continue
        
        if not transformed_records:
            raise ValueError("No valid orders could be transformed")
        
        # Create DataFrame
        transformed_df = pd.DataFrame(transformed_records)
        
        # Fill NaN values
        transformed_df = transformed_df.fillna(0)
        
        # Get model's expected features
        model_features = get_model_feature_names(model)
        
        if not model_features:
            # If no feature names stored, use all columns from transformed_df
            model_features = list(transformed_df.columns)
            logger.warning("Model has no stored feature names, using all available features")
        
        # Ensure all model features exist in transformed_df
        for feature in model_features:
            if feature not in transformed_df.columns:
                transformed_df[feature] = 0
        
        # Select only model features in correct order
        X = transformed_df[model_features].astype(np.float32)
        
        # Make predictions
        predictions_proba = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        
        if is_single:
            # Single prediction
            pred_proba = float(predictions_proba[0])
            pred_class = int(predictions[0])
            
            return PredictionResponse(
                success=True,
                prediction=float(pred_class),
                prediction_probability=pred_proba,
                class_0_prob=float(1 - pred_proba),
                class_1_prob=pred_proba,
                message=f"✓ Prediction complete. Class: {pred_class}, Probability: {pred_proba:.4f}"
            )
        else:
            # Batch prediction
            proba_list = predictions_proba.tolist()
            
            return PredictionResponse(
                success=True,
                message=f"✓ Batch predictions complete for {len(orders)} orders. Probabilities: {proba_list}"
            )
    
    except FileNotFoundError as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model files not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/predict/info")
def get_prediction_info():
    """Get information about the prediction model"""
    try:
        model = load_model()
        feature_names = get_model_feature_names(model)
        
        return {
            "status": "ready",
            "model_type": str(type(model).__name__),
            "feature_count": len(feature_names) if feature_names else "unknown",
            "feature_names": feature_names,
            "model_path": MODEL_PATH
        }
    except Exception as e:
        logger.error(f"Error fetching model info: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/predict/example")
def get_prediction_example():
    """Get an example training instance for prediction"""
    # DynamoDB converted to clean JSON for prediction endpoint
    example_instance = {
        "invoice_id": "619506623_33",
        "billing_state": "Maharashtra",
        "canonical_sku": "11200-900-7",
        "city": "Bhiwandi",
        "courier": "SelfShip",
        "import_warehouse_name": "Moder Godam",
        "item_quantity": 30,
        "marketplace": "B2B",
        "marketplace_sku": "11200-900-7",
        "order_date": "2026-05-06 19:41:42",
        "order_id": 525263621,
        "order_quantity": 1103,
        "order_status": "Open",
        "order_type": "STN",
        "payment_mode": "COD",
        "pin_code": "421302",
        "size": "7",
        "sku": "11200-900-7",
        "source_file": "s3://chupps-data-portal/orders/2026-05",
        "source_month": "2026-05",
        "state": "Maharashtra",
        "suborder_cost": 699,
        "suborder_marketplace_sku": "11200-900-7",
        "suborder_model_no": "Chupster",
        "suborder_mrp": 1299,
        "suborder_productName": "11200-900_chupster_Pista Green_Men",
        "suborder_quantity": 30,
        "suborder_selling_price": 0.3,
        "suborder_size": "7",
        "suborder_sku": "11200-900-7",
        "total_amount": 11.0300
    }
    
    return {
        "request_body": {
            "order_data": example_instance
        },
        "description": "Send this example as POST body to /predict endpoint",
        "curl_example": f"""curl -X POST http://localhost:5002/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"order_data": {example_instance}}}'"""
    }
