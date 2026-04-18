#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="$SCRIPT_DIR/venv/bin"

if [ -f "$SCRIPT_DIR/.env" ]; then
	set -a
	source "$SCRIPT_DIR/.env"
	set +a
fi

UVICORN_CMD="$VENV_BIN/uvicorn app:app --reload --host 0.0.0.0 --port 5000"
CELERY_CMD="$VENV_BIN/celery -A celery_app.celery_app worker --loglevel=info"

open_terminal() {
	local title="$1"
	local cmd="$2"

	if command -v gnome-terminal >/dev/null 2>&1; then
		gnome-terminal --title="$title" -- bash -lc "cd '$SCRIPT_DIR' && $cmd; exec bash"
		return 0
	fi

	if command -v x-terminal-emulator >/dev/null 2>&1; then
		x-terminal-emulator -e bash -lc "cd '$SCRIPT_DIR' && $cmd; exec bash"
		return 0
	fi

	if command -v konsole >/dev/null 2>&1; then
		konsole --new-tab -p tabtitle="$title" -e bash -lc "cd '$SCRIPT_DIR' && $cmd; exec bash"
		return 0
	fi

	if command -v xfce4-terminal >/dev/null 2>&1; then
		xfce4-terminal --title="$title" --hold -e "bash -lc \"cd '$SCRIPT_DIR' && $cmd\""
		return 0
	fi

	return 1
}

run_in_background() {
	local title="$1"
	local cmd="$2"
	local log_file="$SCRIPT_DIR/${3:-${title,,}.log}"

	if command -v nohup >/dev/null 2>&1; then
		nohup bash -lc "cd '$SCRIPT_DIR' && $cmd" >"$log_file" 2>&1 &
		printf '%s started in background. Log: %s\n' "$title" "$log_file"
		return 0
	fi

	return 1
}

if ! open_terminal "Uvicorn API" "$UVICORN_CMD"; then
	if ! run_in_background "Uvicorn API" "$UVICORN_CMD" "uvicorn.log"; then
		echo "Could not find a supported terminal emulator to open Uvicorn automatically."
		echo "Run manually: cd '$SCRIPT_DIR' && $UVICORN_CMD"
	fi
fi

if ! open_terminal "Celery Worker" "$CELERY_CMD"; then
	if ! run_in_background "Celery Worker" "$CELERY_CMD" "celery.log"; then
		echo "Could not find a supported terminal emulator to open Celery automatically."
		echo "Run manually: cd '$SCRIPT_DIR' && $CELERY_CMD"
	fi
fi

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