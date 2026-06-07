#!/bin/bash
# Collect weekly inventory snapshot (Wednesdays)
set -e

cd /home/ubuntu/chupps/backend

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting weekly inventory snapshot collection..."
python snapshot_collector.py
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Weekly snapshot collection complete."
