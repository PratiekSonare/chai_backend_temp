"""
Footwear Order Status Prediction Module

Provides daily warm-start GBDT model training and batch inference
for predicting order returns/cancellations based on 26-column tabular data.

Architecture:
  1. Label Assignment: Sliding window (T-30 to T-1) for finalized status
  2. Feature Engineering: Target encoding for high-cardinality features + temporal features
  3. Model Training: Warm-start LightGBM using previous day's model
  4. Batch Prediction: Apply to incoming orders (T+0 batch of 500)
  5. Metrics Aggregation: SKU-level and global return rate metrics

Usage:
  from backend.prediction.model_trainer import run_daily_training_pipeline
  from datetime import date
  
  run_daily_training_pipeline(date.today())

Config:
  - LABEL_WINDOW_DAYS: 30 (days to train on, labeled data only)
  - RETURN_THRESHOLD_DAYS: 30 (age cutoff for finalized labels)
  - PREDICTION_WINDOW_DAYS: 7 (validation window)
  - MODEL_PARAMS: LightGBM hyperparameters
"""

__version__ = "1.0.0"
__author__ = "Chupps Data Team"
