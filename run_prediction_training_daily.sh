#!/bin/bash
# Daily Order Status Prediction Model Training Script
# Runs at 01:00 AM daily to update warm-start GBDT model

# Load environment
source /home/pratiek/.bashrc
cd /home/pratiek/Downloads/chupps/backend

# Activate Python environment (adjust to your environment)
python -m venv /tmp/pred-env --clear 2>/dev/null || true
source /tmp/pred-env/bin/activate

# Install dependencies if needed
pip install -q lightgbm scikit-learn pandas boto3 python-dotenv 2>/dev/null || true

# Run training pipeline
echo "[$(date)] Starting daily order status prediction training..."
python -m prediction.model_trainer >> /var/log/chupps/daily_prediction_train.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ Training completed successfully"
else
    echo "[$(date)] ✗ Training failed with exit code $EXIT_CODE"
    # Send alert to monitoring
    # curl -X POST https://alerts.example.com/failed -d "task=prediction_training"
fi

exit $EXIT_CODE
