"""
Quick test script for order status prediction pipeline.

Tests each component independently to validate:
  1. S3 connectivity
  2. Data fetching and labeling
  3. Feature engineering
  4. Model training
  5. Batch prediction
  6. Metrics aggregation

Run: python -m prediction.test_pipeline
"""

import os
import sys
from datetime import date, timedelta
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.model_trainer import (
    S3DataHandler,
    LabelAssigner,
    FeatureEngineer,
    ModelTrainer,
    BatchPredictor,
    MetricsAggregator,
)


def test_s3_connectivity():
    """Test 1: Verify S3 access."""
    print("\n" + "="*60)
    print("TEST 1: S3 Connectivity")
    print("="*60)
    
    try:
        s3 = S3DataHandler()
        
        # Try to list bucket contents
        response = s3.s3_client.list_objects_v2(
            Bucket=s3.bucket,
            Prefix="orders/",
            MaxKeys=5
        )
        
        if 'Contents' in response:
            print(f"✓ S3 bucket '{s3.bucket}' accessible")
            print(f"  Found {len(response['Contents'])} objects in orders/ prefix")
            for obj in response['Contents'][:3]:
                print(f"    - {obj['Key']}")
        else:
            print(f"⚠ S3 bucket '{s3.bucket}' is empty or no 'orders/' prefix")
        
        return True
    
    except Exception as e:
        print(f"✗ S3 connectivity failed: {e}")
        print("  Check AWS credentials in ~/.aws/credentials")
        return False


def test_data_fetching():
    """Test 2: Fetch sample orders from S3."""
    print("\n" + "="*60)
    print("TEST 2: Data Fetching")
    print("="*60)
    
    try:
        s3 = S3DataHandler()
        
        # Try to fetch last 5 days of orders
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        
        print(f"Fetching orders from {start_date} to {end_date}...")
        orders = s3.fetch_orders_from_s3(start_date, end_date)
        
        if orders:
            print(f"✓ Fetched {len(orders)} orders")
            
            # Show sample order structure
            sample = orders[0]
            print(f"\n  Sample order structure:")
            for key in ['order_id', 'sku', 'marketplace', 'order_status', 'total_amount']:
                value = sample.get(key, 'N/A')
                print(f"    {key}: {value}")
            
            return True, orders
        else:
            print("⚠ No orders found (data may not exist for this date range)")
            return False, []
    
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        return False, []


def test_labeling(orders):
    """Test 3: Label assignment."""
    print("\n" + "="*60)
    print("TEST 3: Label Assignment")
    print("="*60)
    
    try:
        assigner = LabelAssigner()
        current_date = date.today()
        
        labeled_orders, labels = assigner.assign_labels(orders, current_date)
        
        if len(labeled_orders) > 0:
            pos_rate = labels.mean()
            print(f"✓ Assigned {len(labeled_orders)} labels")
            print(f"  Positive rate (Returned/Cancelled): {pos_rate:.2%}")
            print(f"  Distribution: {(labels == 0).sum()} negative, {(labels == 1).sum()} positive")
            
            return True, labeled_orders, labels
        else:
            print("⚠ No labeled orders (all orders may be too recent)")
            return False, [], None
    
    except Exception as e:
        print(f"✗ Labeling failed: {e}")
        return False, [], None


def test_feature_engineering(labeled_orders, labels):
    """Test 4: Feature engineering."""
    print("\n" + "="*60)
    print("TEST 4: Feature Engineering")
    print("="*60)
    
    try:
        engineer = FeatureEngineer()
        
        df = engineer.engineer_features(
            labeled_orders,
            fit_encodings=True,
            labels=labels
        )
        
        if len(df) > 0:
            print(f"✓ Engineered {len(df)} records")
            print(f"  Features: {len(df.columns)}")
            print(f"\n  Feature columns:")
            for col in df.columns[:10]:
                print(f"    - {col}")
            if len(df.columns) > 10:
                print(f"    ... and {len(df.columns) - 10} more")
            
            print(f"\n  Data types:")
            print(f"    Numeric: {(df.dtypes == 'float32').sum() + (df.dtypes == 'float64').sum()}")
            print(f"    Object: {(df.dtypes == 'object').sum()}")
            
            return True, df, engineer
        else:
            print("✗ No records engineered")
            return False, None, None
    
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_model_training(train_df, labels):
    """Test 5: Model training."""
    print("\n" + "="*60)
    print("TEST 5: Model Training")
    print("="*60)
    
    try:
        s3 = S3DataHandler()
        trainer = ModelTrainer(s3)
        
        # Split data
        split_idx = int(len(train_df) * 0.7)
        train_df_split = train_df.iloc[:split_idx]
        val_df_split = train_df.iloc[split_idx:]
        
        train_labels_split = labels[:split_idx]
        val_labels_split = labels[split_idx:]
        
        print(f"Training set: {len(train_df_split)} | Validation set: {len(val_df_split)}")
        print("Training model (this may take 1-2 minutes)...")
        
        model, metrics = trainer.train(
            train_df_split, train_labels_split,
            val_df_split, val_labels_split,
            prev_model_date=None  # No warm-start for test
        )
        
        print(f"✓ Model trained successfully")
        print(f"  Validation AUC: {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        
        if metrics['auc'] < 0.6:
            print(f"\n  ⚠ WARNING: AUC is below 0.6 (poor model)")
            print(f"    This may indicate:")
            print(f"    - Insufficient data for this window")
            print(f"    - Features need refinement")
            print(f"    - Classes are imbalanced")
        
        return True, model
    
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batch_prediction(model, feature_engineer, orders):
    """Test 6: Batch prediction."""
    print("\n" + "="*60)
    print("TEST 6: Batch Prediction")
    print("="*60)
    
    try:
        if not model or not orders:
            print("⚠ Model or orders not available for prediction")
            return False, []
        
        predictor = BatchPredictor(model, feature_engineer)
        predictions = predictor.predict(orders[:min(100, len(orders))], date.today())
        
        if predictions:
            print(f"✓ Generated {len(predictions)} predictions")
            
            # Show statistics
            p_returned = [p['prediction']['p_returned'] for p in predictions]
            import statistics
            print(f"\n  Prediction statistics:")
            print(f"    Mean return probability: {statistics.mean(p_returned):.2%}")
            print(f"    Median: {statistics.median(p_returned):.2%}")
            print(f"    Std Dev: {statistics.stdev(p_returned) if len(p_returned) > 1 else 0:.2%}")
            
            high_risk = sum(1 for p in p_returned if p > 0.7)
            low_risk = sum(1 for p in p_returned if p < 0.3)
            print(f"    High risk (>70%): {high_risk} orders")
            print(f"    Low risk (<30%): {low_risk} orders")
            
            print(f"\n  Sample prediction:")
            sample = predictions[0]
            print(f"    Order ID: {sample['order_id']}")
            print(f"    P(Returned): {sample['prediction']['p_returned']:.2%}")
            print(f"    Predicted: {sample['prediction']['predicted_class']}")
            
            return True, predictions
        else:
            print("✗ No predictions generated")
            return False, []
    
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_metrics_aggregation(predictions):
    """Test 7: Metrics aggregation."""
    print("\n" + "="*60)
    print("TEST 7: Metrics Aggregation")
    print("="*60)
    
    try:
        if not predictions:
            print("⚠ No predictions to aggregate")
            return False
        
        metrics = MetricsAggregator.aggregate_metrics(predictions)
        
        print(f"✓ Aggregated metrics")
        print(f"\n  Global Metrics:")
        for key, value in metrics['global'].items():
            print(f"    {key}: {value}")
        
        print(f"\n  By Class:")
        for key, value in metrics['by_predicted_class'].items():
            print(f"    {key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"✗ Metrics aggregation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ORDER STATUS PREDICTION PIPELINE TEST SUITE")
    print("#"*60)
    
    results = {}
    
    # Test 1: S3 Connectivity
    results['s3_connectivity'] = test_s3_connectivity()
    if not results['s3_connectivity']:
        print("\n✗ Cannot proceed without S3 access")
        return
    
    # Test 2: Data Fetching
    success, orders = test_data_fetching()
    results['data_fetching'] = success
    if not success or not orders:
        print("\n✗ Cannot proceed without data")
        return
    
    # Test 3: Labeling
    success, labeled_orders, labels = test_labeling(orders)
    results['labeling'] = success
    if not success or not labeled_orders:
        print("\n✗ Cannot proceed without labeled data")
        return
    
    # Test 4: Feature Engineering
    success, df, engineer = test_feature_engineering(labeled_orders, labels)
    results['feature_engineering'] = success
    if not success or df is None:
        print("\n✗ Cannot proceed without engineered features")
        return
    
    # Test 5: Model Training
    success, model = test_model_training(df, labels)
    results['model_training'] = success
    if not success or model is None:
        print("\n⚠ Skipping prediction tests (no model)")
    else:
        # Test 6: Batch Prediction
        success, predictions = test_batch_prediction(model, engineer, orders)
        results['batch_prediction'] = success
        
        # Test 7: Metrics Aggregation
        if success and predictions:
            results['metrics_aggregation'] = test_metrics_aggregation(predictions)
        else:
            results['metrics_aggregation'] = False
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready for deployment.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")


if __name__ == "__main__":
    main()
