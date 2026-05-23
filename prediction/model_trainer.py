"""
Daily Warm-Start GBDT Model Trainer for Footwear Order Status Prediction
Integrates with existing S3 order pipeline
"""

import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import boto3
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFAULT_AWS_REGION = "ap-south-1"
DEFAULT_BUCKET = "chupps-data-portal"
ORDERS_PREFIX = "orders"
PREDICTIONS_PREFIX = "predictions"
MODELS_PREFIX = "models"
LABEL_WINDOW_DAYS = 30  # Train on last 30 days (closing return window)
PREDICTION_WINDOW_DAYS = 7  # Validate on last 7 days
RETURN_THRESHOLD_DAYS = 30  # Orders older than this are considered "finalized"

# Model hyperparameters
MODEL_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_threads': 8,
    'verbose': -1,
}


class S3DataHandler:
    """Fetch orders from S3 and manage model artifacts."""
    
    def __init__(self, region: str = DEFAULT_AWS_REGION):
        self.s3_client = boto3.client('s3', region_name=region)
        self.bucket = DEFAULT_BUCKET
    
    def fetch_orders_from_s3(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Fetch all orders from S3 for date range.
        S3 structure: orders/YYYY-MM/YYYY-MM-DD.json
        """
        all_orders = []
        current_date = start_date
        
        while current_date <= end_date:
            month_folder = current_date.strftime("%Y-%m")
            day_file = current_date.strftime("%Y-%m-%d")
            key = f"{ORDERS_PREFIX}/{month_folder}/{day_file}.json"
            
            try:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                data = json.loads(response['Body'].read().decode('utf-8'))
                all_orders.extend(data)
                print(f"✓ Loaded {len(data)} orders from {key}")
            except self.s3_client.exceptions.NoSuchKey:
                print(f"⚠ No data at {key}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON decode error in {key}: {e}")
            
            current_date += timedelta(days=1)
        
        return all_orders
    
    def upload_predictions(self, predictions: List[Dict], predict_date: date) -> str:
        """Upload predictions to S3."""
        month_folder = predict_date.strftime("%Y-%m")
        day_file = predict_date.strftime("%Y-%m-%d")
        key = f"{PREDICTIONS_PREFIX}/{month_folder}/{day_file}-predictions.json"
        
        body = json.dumps(predictions, ensure_ascii=True)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        print(f"✓ Uploaded predictions to s3://{self.bucket}/{key}")
        return key
    
    def upload_model(self, model: lgb.Booster, model_date: date) -> str:
        """Upload trained model to S3."""
        month_folder = model_date.strftime("%Y-%m")
        day_file = model_date.strftime("%Y-%m-%d")
        key = f"{MODELS_PREFIX}/{month_folder}/{day_file}-model.pkl"
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model.save_model(tmp.name)
            with open(tmp.name, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=f.read(),
                )
            os.unlink(tmp.name)
        
        print(f"✓ Uploaded model to s3://{self.bucket}/{key}")
        return key
    
    def download_model(self, model_date: date) -> Optional[lgb.Booster]:
        """Download model from S3 for warm-start."""
        month_folder = model_date.strftime("%Y-%m")
        day_file = model_date.strftime("%Y-%m-%d")
        key = f"{MODELS_PREFIX}/{month_folder}/{day_file}-model.pkl"
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                tmp.write(response['Body'].read())
                tmp.flush()
                model = lgb.Booster(model_file=tmp.name)
                os.unlink(tmp.name)
            print(f"✓ Loaded previous model from {key}")
            return model
        except self.s3_client.exceptions.NoSuchKey:
            print(f"⚠ No previous model found at {key} (first run?)")
            return None


class LabelAssigner:
    """Assign labels based on sliding window to handle return delays."""
    
    @staticmethod
    def assign_labels(orders: List[Dict], current_date: date) -> Tuple[List[Dict], np.ndarray]:
        """
        Assign binary labels (1 = Returned/Cancelled, 0 = Other).
        Only label orders that are "old enough" (> RETURN_THRESHOLD_DAYS ago)
        to ensure their status is finalized.
        """
        labels = []
        valid_orders = []
        
        cutoff_date = current_date - timedelta(days=RETURN_THRESHOLD_DAYS)
        
        for order in orders:
            # Parse order_date
            try:
                order_date_str = order.get('order_date', '')
                if isinstance(order_date_str, str):
                    order_date = datetime.strptime(order_date_str.split()[0], "%Y-%m-%d").date()
                else:
                    continue
            except (ValueError, AttributeError):
                continue
            
            # Only label if order is old enough to have finalized status
            if order_date < cutoff_date:
                status = order.get('order_status', '').strip()
                label = 1 if status in ['Returned', 'Cancelled'] else 0
                labels.append(label)
                valid_orders.append(order)
        
        print(f"Assigned {len(valid_orders)} labeled orders "
              f"(cutoff: before {cutoff_date}, positive rate: {sum(labels)/len(labels) if labels else 0:.2%})")
        
        return valid_orders, np.array(labels)


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
        import re
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
            except (ValueError, TypeError, KeyError) as e:
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
            else:
                pass

        cols_to_drop_after_encoding = [col for col in categorical_cols if f'{col}_encoded' in df.columns and col in df.columns]
        if cols_to_drop_after_encoding:
            df = df.drop(columns=cols_to_drop_after_encoding)

        return df, current_labels


class ModelTrainer:
    """Train and manage warm-start GBDT models."""
    
    def __init__(self, s3_handler: S3DataHandler):
        self.s3_handler = s3_handler
        self.model = None
    
    def train(self, train_df: pd.DataFrame, train_labels: np.ndarray,
              val_df: pd.DataFrame, val_labels: np.ndarray,
              prev_model_date: Optional[date] = None) -> Tuple[lgb.Booster, Dict]:
        """
        Train or warm-start model.
        If prev_model_date is provided, uses that as initialization (warm-start).
        """
        # Select feature columns (exclude order_id, order_date, order_status)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['order_id', 'order_date', 'order_status']]
        
        X_train = train_df[feature_cols].astype(np.float32)
        X_val = val_df[feature_cols].astype(np.float32)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=train_labels)
        val_data = lgb.Dataset(X_val, label=val_labels, reference=train_data)
        
        # Load previous model for warm-start
        init_model = None
        if prev_model_date:
            init_model = self.s3_handler.download_model(prev_model_date)
        
        # Train with warm-start
        print(f"Training with warm-start: {'Yes' if init_model else 'No (first run)'}")
        
        self.model = lgb.train(
            MODEL_PARAMS,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(5),
                lgb.log_evaluation(period=10)
            ],
            init_model=init_model
        )
        
        # Evaluate
        val_preds = self.model.predict(X_val)
        auc = roc_auc_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, (val_preds > 0.5).astype(int), average='binary'
        )
        
        metrics = {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_features': len(feature_cols),
            'feature_names': feature_cols
        }
        
        print(f"Validation AUC: {auc:.4f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return self.model, metrics


class BatchPredictor:
    """Apply trained model to new orders."""
    
    def __init__(self, model: lgb.Booster, feature_engineer: FeatureEngineer):
        self.model = model
        self.feature_engineer = feature_engineer
    
    def predict(self, orders: List[Dict], model_date: date) -> List[Dict]:
        """
        Predict on incoming orders (no labels available yet).
        """
        df = self.feature_engineer.engineer_features(orders, fit_encodings=False)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['order_id', 'order_date', 'order_status']]
        
        X = df[feature_cols].astype(np.float32)
        predictions = self.model.predict(X)
        
        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            pred_prob = predictions[idx]
            results.append({
                'order_id': row['order_id'],
                'order_date': row['order_date'].isoformat(),
                'prediction': {
                    'p_returned': float(pred_prob),
                    'p_cancelled': float(1 - pred_prob),
                    'predicted_class': 'Returned' if pred_prob > 0.5 else 'Cancelled',
                    'model_date': model_date.isoformat()
                }
            })
        
        print(f"Generated {len(results)} predictions")
        return results


class MetricsAggregator:
    """Aggregate SKU-level and global metrics from predictions."""
    
    @staticmethod
    def aggregate_metrics(predictions: List[Dict]) -> Dict:
        """
        Calculate SKU-level and global return rate metrics.
        """
        df = pd.DataFrame(predictions)
        df['p_returned'] = df['prediction'].apply(lambda x: x['p_returned'])
        df['predicted_class'] = df['prediction'].apply(lambda x: x['predicted_class'])
        
        metrics = {
            'global': {
                'mean_return_prob': float(df['p_returned'].mean()),
                'std_return_prob': float(df['p_returned'].std()),
                'median_return_prob': float(df['p_returned'].median()),
                'high_risk_count': int((df['p_returned'] > 0.7).sum()),
                'low_risk_count': int((df['p_returned'] < 0.3).sum()),
            },
            'by_predicted_class': {
                'returned': int((df['predicted_class'] == 'Returned').sum()),
                'cancelled': int((df['predicted_class'] == 'Cancelled').sum()),
            }
        }
        
        print(f"Global Return Probability: {metrics['global']['mean_return_prob']:.2%}")
        print(f"High Risk Orders (>70%): {metrics['global']['high_risk_count']}")
        
        return metrics


def run_daily_training_pipeline(train_date: date) -> None:
    """
    Full daily training pipeline:
    1. Fetch labeled data (sliding window: train_date - 30 to train_date - 1)
    2. Feature engineering + fitting target encodings
    3. Train model with warm-start (using previous day's model)
    4. Evaluate and upload model
    5. Predict on new data (train_date)
    6. Aggregate metrics
    """
    print(f"\n{'='*60}")
    print(f"DAILY TRAINING PIPELINE: {train_date}")
    print(f"{'='*60}\n")
    
    s3_handler = S3DataHandler()
    label_assigner = LabelAssigner()
    feature_engineer = FeatureEngineer()
    metrics_agg = MetricsAggregator()
    
    # --- PHASE 1: Data Preparation ---
    print("PHASE 1: Data Preparation")
    print("-" * 40)
    
    train_start = train_date - timedelta(days=LABEL_WINDOW_DAYS)
    train_end = train_date - timedelta(days=1)
    
    print(f"Fetching labeled data from {train_start} to {train_end}")
    labeled_orders = s3_handler.fetch_orders_from_s3(train_start, train_end)
    
    if not labeled_orders:
        print("✗ No labeled data available")
        return
    
    labeled_orders, labels = label_assigner.assign_labels(labeled_orders, train_date)
    
    # --- PHASE 2: Feature Engineering ---
    print("\nPHASE 2: Feature Engineering")
    print("-" * 40)
    
    train_df = feature_engineer.engineer_features(labeled_orders, 
                                                  fit_encodings=True, 
                                                  labels=labels)
    
    # Split into train/validation (80/20 split on time)
    split_idx = int(len(train_df) * 0.8)
    train_df_split = train_df.iloc[:split_idx]
    val_df_split = train_df.iloc[split_idx:]
    
    train_labels_split = labels[:split_idx]
    val_labels_split = labels[split_idx:]
    
    print(f"Train set: {len(train_df_split)} | Validation set: {len(val_df_split)}")
    
    # --- PHASE 3: Model Training (Warm-Start) ---
    print("\nPHASE 3: Model Training (Warm-Start GBDT)")
    print("-" * 40)
    
    trainer = ModelTrainer(s3_handler)
    prev_model_date = train_date - timedelta(days=1)
    
    model, train_metrics = trainer.train(
        train_df_split, train_labels_split,
        val_df_split, val_labels_split,
        prev_model_date=prev_model_date
    )
    
    # Upload model
    model_key = s3_handler.upload_model(model, train_date)
    
    # --- PHASE 4: Batch Prediction ---
    print("\nPHASE 4: Batch Prediction on New Orders")
    print("-" * 40)
    
    predict_orders = s3_handler.fetch_orders_from_s3(train_date, train_date)
    if not predict_orders:
        print("⚠ No new orders to predict")
        return
    
    predictor = BatchPredictor(model, feature_engineer)
    predictions = predictor.predict(predict_orders, train_date)
    
    # Upload predictions
    s3_handler.upload_predictions(predictions, train_date)
    
    # --- PHASE 5: Metrics Aggregation ---
    print("\nPHASE 5: Metrics Aggregation")
    print("-" * 40)
    
    metrics = metrics_agg.aggregate_metrics(predictions)
    
    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE: {train_date}")
    print(f"Model uploaded: {model_key}")
    print(f"Predictions: {len(predictions)} orders classified")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example: Train on 2026-05-09 using data from 2026-04-09 to 2026-05-08
    today = date.today()
    run_daily_training_pipeline(today)
