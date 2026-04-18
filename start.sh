source venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 5000 2>&1 | tee -a uvicorn.log

# 2>&1 | tee -a uvicorn.log

# gunicorn app:app \
#   --worker-class uvicorn.workers.UvicornWorker \
#   --workers 4 \
#   --bind 0.0.0.0:5000 \
#   --log-level info \
#   --access-logfile - \
#   --error-logfile - \
#   --timeout 120 \
#   --graceful-timeout 120