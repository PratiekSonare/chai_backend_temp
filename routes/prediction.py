import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Tuple
import logging
import re
import numpy as np
from datetime import datetime
import pickle
import sys

logger = logging.getLogger(__name__)

router = APIRouter()


class FeatureEngineer:
    """Feature engineering for footwear orders."""

    def __init__(self):
        self.category_encodings = {}

    def fit_target_encoding(self, df: pd.DataFrame, labels: np.ndarray,
                           categorical_cols: List[str], smoothing: float = 1.0) -> None:
        """Fit target encoding for categorical features using Bayesian smoothing."""
        global_rate = labels.mean()

        for col in categorical_cols:
            category_stats = {}
            if col not in df.columns or df[col].isnull().all():
                continue

            for category in df[col].unique():
                if pd.isna(category):
                    continue
                mask = df[col] == category
                category_labels = labels[mask]

                n = len(category_labels)
                category_mean = category_labels.mean() if n > 0 else global_rate
                smoothed = (n * category_mean + smoothing * global_rate) / (n + smoothing)
                category_stats[category] = smoothed

            self.category_encodings[col] = category_stats

    def engineer_features(self, orders: List[Dict], fit_encodings: bool = False,
                         labels: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Transform orders into feature matrix using actual schema (31 fields)."""
        records = []
        extracted_labels = []

        for order in orders:
            status_val = str(order.get('order_status', '')).strip().upper()
            order_type_val = str(order.get('order_type', '')).strip().upper()

            if status_val == 'RETURNED':
                continue
            if order_type_val == 'STN':
                continue

            try:
                order_date_raw = order.get('order_date', '')
                order_date_str = str(order_date_raw).strip()

                date_to_parse = None
                if order_date_str:
                    if ' ' in order_date_str:
                        date_to_parse = order_date_str.split()[0]
                    else:
                        date_to_parse = order_date_str

                order_date = pd.to_datetime(date_to_parse, errors='coerce')
                if pd.isna(order_date):
                    order_date = datetime.now()

                order_date_hour = order_date.hour if pd.notna(order_date) else 0

                def clean_size(size_str):
                    if pd.isna(size_str): return np.nan
                    s = str(size_str).replace(',', '').strip()
                    s = re.sub(r'[^\d.]', '', s)
                    return pd.to_numeric(s, errors='coerce')

                def parse_sku_and_extract_size(sku_str):
                    sku_str = str(sku_str).strip().upper()
                    if '-' in sku_str:
                        parts = sku_str.rsplit('-', 1)
                        if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
                            return parts[0], pd.to_numeric(parts[1], errors='coerce')
                    return sku_str, np.nan

                raw_sku = order.get('sku', 'UNKNOWN')
                clean_sku_val, extracted_sku_size = parse_sku_and_extract_size(raw_sku)

                raw_suborder_sku = order.get('suborder_sku', 'UNKNOWN')
                clean_suborder_sku_val, extracted_suborder_sku_size = parse_sku_and_extract_size(raw_suborder_sku)

                order_size_from_data = clean_size(order.get('size', 0))
                final_size = extracted_sku_size if pd.notna(extracted_sku_size) else order_size_from_data

                order_suborder_size_from_data = clean_size(order.get('suborder_size', 0))
                final_suborder_size = extracted_suborder_sku_size if pd.notna(extracted_suborder_sku_size) else order_suborder_size_from_data

                suborder_selling_price = float(order.get('suborder_selling_price', 0))
                if suborder_selling_price > 0:
                    suborder_selling_price = (int(suborder_selling_price // 100) + 1) * 99

                record = {
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
                    'size': final_size,
                    'suborder_size': final_suborder_size,
                    'suborder_selling_price': suborder_selling_price,
                    'order_date_hour': order_date_hour
                }

                if status_val == 'CANCELLED':
                    extracted_labels.append(1)
                else:
                    extracted_labels.append(0)

                records.append(record)
            except (ValueError, TypeError, KeyError):
                continue

        df = pd.DataFrame(records)

        if len(df) == 0:
            return df, np.array([])

        for col in ['size', 'suborder_size']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median() if not df[col].empty else 0, inplace=True)

        cols_to_drop_raw = [
            'canonical_sku', 'marketplace_sku', 'order_type', 'courier', 'import_warehouse_name'
        ]
        cols_to_drop_raw = [col for col in cols_to_drop_raw if col in df.columns]
        if cols_to_drop_raw:
            df = df.drop(columns=cols_to_drop_raw)

        categorical_cols = ['marketplace', 'payment_mode', 'state', 'clean_sku', 'clean_suborder_sku', 'billing_state']
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        current_labels = np.array(extracted_labels) if extracted_labels else np.array([])

        if fit_encodings and current_labels.size > 0:
            self.fit_target_encoding(df, current_labels, categorical_cols)

        for col in categorical_cols:
            if col in self.category_encodings:
                df[f'{col}_encoded'] = df[col].map(self.category_encodings[col])
                df[f'{col}_encoded'].fillna(current_labels.mean() if current_labels.size > 0 else 0.5, inplace=True)

        cols_to_drop_after_encoding = [col for col in categorical_cols if f'{col}_encoded' in df.columns and col in df.columns]
        if cols_to_drop_after_encoding:
            df = df.drop(columns=cols_to_drop_after_encoding)

        return df, current_labels

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using learned encodings (inference mode)."""
        df = df.copy()
        
        categorical_cols = ['marketplace', 'payment_mode', 'state', 'clean_sku', 'clean_suborder_sku', 'billing_state']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if col in self.category_encodings:
                df[f'{col}_encoded'] = df[col].map(self.category_encodings[col])
                df[f'{col}_encoded'].fillna(0.5, inplace=True)
        
        cols_to_drop = [col for col in categorical_cols if f'{col}_encoded' in df.columns and col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        return df

# Model paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "lightgbm_footwear_prediction_model_jan.joblib")
FEATURE_ENGINEER_PATH = os.path.join(BASE_PATH, "feature_engineer_jan.joblib")

# Global model and feature engineer - loaded on first request
_model = None
_feature_engineer = None


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


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to remap __mp_main__.FeatureEngineer to current module's FeatureEngineer"""
    def find_class(self, module, name):
        logger.debug(f"Unpickler looking for: {module}.{name}")
        # Remap __mp_main__.FeatureEngineer to current module's FeatureEngineer
        if module == '__mp_main__' and name == 'FeatureEngineer':
            logger.info(f"Found __mp_main__.FeatureEngineer, remapping to current module")
            # Register the FeatureEngineer class in sys.modules so pickle can find it
            import types
            if '__mp_main__' not in sys.modules:
                sys.modules['__mp_main__'] = types.ModuleType('__mp_main__')
            
            # Get the FeatureEngineer class from the current module
            fe_class = FeatureEngineer
            sys.modules['__mp_main__'].FeatureEngineer = fe_class
            logger.info(f"Successfully registered FeatureEngineer in sys.modules['__mp_main__']")
            return fe_class
        return super().find_class(module, name)


def load_feature_engineer():
    """Load the feature engineer for categorical encoding"""
    global _feature_engineer
    
    if _feature_engineer is None:
        if not os.path.exists(FEATURE_ENGINEER_PATH):
            logger.warning(f"Feature engineer file not found at {FEATURE_ENGINEER_PATH}. Using default instance.")
            _feature_engineer = FeatureEngineer()
            return _feature_engineer
        
        # Pre-register FeatureEngineer in __mp_main__ to help with unpickling
        import types
        if '__mp_main__' not in sys.modules:
            sys.modules['__mp_main__'] = types.ModuleType('__mp_main__')
        sys.modules['__mp_main__'].FeatureEngineer = FeatureEngineer
        
        loaded_successfully = False
        
        try:
            # Try standard joblib load first (may fail with __mp_main__ reference)
            logger.info(f"Attempting to load feature engineer from {FEATURE_ENGINEER_PATH}")
            _feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
            logger.info(f"✓ Feature Engineer loaded via joblib from {FEATURE_ENGINEER_PATH}")
            loaded_successfully = True
        except (AttributeError, ModuleNotFoundError, TypeError) as e:
            logger.warning(f"joblib.load failed: {e}. Trying custom unpickler...")
            try:
                # Use custom unpickler as fallback
                with open(FEATURE_ENGINEER_PATH, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    _feature_engineer = unpickler.load()
                logger.info(f"✓ Feature Engineer loaded via CustomUnpickler from {FEATURE_ENGINEER_PATH}")
                loaded_successfully = True
            except Exception as e2:
                logger.warning(f"Custom unpickler failed: {e2}")
                # Try dill as last resort if available
                try:
                    import dill
                    logger.info("Trying dill for unpickling...")
                    with open(FEATURE_ENGINEER_PATH, 'rb') as f:
                        _feature_engineer = dill.load(f)
                    logger.info(f"✓ Feature Engineer loaded via dill from {FEATURE_ENGINEER_PATH}")
                    loaded_successfully = True
                except ImportError:
                    logger.warning("dill not available, skipping dill attempt")
                except Exception as e3:
                    logger.warning(f"dill also failed: {e3}")
        
        # If all loading attempts failed, use a default instance
        if not loaded_successfully:
            logger.warning(f"Could not load feature engineer from file. Using default FeatureEngineer instance (no pre-trained encodings).")
            _feature_engineer = FeatureEngineer()
        
        logger.info(f"Feature Engineer type: {type(_feature_engineer).__name__}")
        logger.info(f"Feature Engineer category_encodings count: {len(_feature_engineer.category_encodings)}")
    
    return _feature_engineer


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
    
    # Extract first suborder if nested structure
    suborder = None
    if isinstance(order.get('suborders'), list) and len(order.get('suborders', [])) > 0:
        suborder = order['suborders'][0]
    
    # Parse SKUs - prioritize suborder sku if available
    if suborder:
        raw_sku = suborder.get('sku', order.get('sku', 'UNKNOWN'))
        raw_suborder_sku = suborder.get('sku', 'UNKNOWN')
    else:
        raw_sku = order.get('sku', 'UNKNOWN')
        raw_suborder_sku = order.get('suborder_sku', 'UNKNOWN')
    
    clean_sku_val, extracted_sku_size = parse_sku_and_extract_size(raw_sku)
    clean_suborder_sku_val, extracted_suborder_sku_size = parse_sku_and_extract_size(raw_suborder_sku)
    
    # Get sizes
    if suborder:
        order_size_from_data = clean_size(suborder.get('size', order.get('size', 0)))
        suborder_size_from_data = clean_size(suborder.get('size', order.get('suborder_size', 0)))
    else:
        order_size_from_data = clean_size(order.get('size', 0))
        suborder_size_from_data = clean_size(order.get('suborder_size', 0))
    
    final_size = extracted_sku_size if pd.notna(extracted_sku_size) else order_size_from_data
    final_suborder_size = extracted_suborder_sku_size if pd.notna(extracted_suborder_sku_size) else suborder_size_from_data
    
    # Get quantities
    if suborder:
        item_qty = int(suborder.get('item_quantity', order.get('item_quantity', 0)))
        suborder_qty = int(suborder.get('suborder_quantity', order.get('suborder_quantity', 0)))
    else:
        item_qty = int(order.get('item_quantity', 0))
        suborder_qty = int(order.get('suborder_quantity', 0))
    
    # Get selling price from suborder
    if suborder:
        suborder_selling_price = float(suborder.get('selling_price', order.get('suborder_selling_price', 0)))
    else:
        suborder_selling_price = float(order.get('suborder_selling_price', 0))
    
    if suborder_selling_price > 0:
        suborder_selling_price = (int(suborder_selling_price // 100) + 1) * 99
    
    return {
        'order_id': order.get('order_id'),
        'invoice_id': str(order.get('invoice_id', '')),
        'item_quantity': item_qty,
        'suborder_quantity': suborder_qty,
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
        feature_engineer = load_feature_engineer()
        
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
        
        print("Raw features after transformation:", transformed_records, flush=True)

        # Create DataFrame
        transformed_df = pd.DataFrame(transformed_records)
        
        # Fill NaN values
        transformed_df = transformed_df.fillna(0)
        
        # Apply feature engineer transformation (categorical encoding)
        logger.info("Applying feature engineer transformations...")
        try:
            if hasattr(feature_engineer, 'transform'):
                # Apply categorical encoding via feature engineer
                encoded_df = feature_engineer.transform(transformed_df)
                logger.info(f"✓ Feature engineer applied. Shape before: {transformed_df.shape}, after: {encoded_df.shape}")
                transformed_df = encoded_df
            else:
                logger.warning(f"Feature engineer has no transform method. Using raw features.")
        except Exception as e:
            logger.warning(f"Feature engineer transformation failed: {e}. Proceeding with raw features.")
        
        print("Features after feature engineering:", transformed_df.head(), flush=True)
        
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
