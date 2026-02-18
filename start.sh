source venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 5000 2>&1 | tee -a server.log
