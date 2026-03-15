import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from redis import Redis  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency fallback
    Redis = None

REQUEST_LOG_STORAGE = os.getenv("REQUEST_LOG_STORAGE", "auto").strip().lower()
REQUEST_LOG_TTL_SECONDS = int(os.getenv("REQUEST_LOG_TTL_SECONDS", "3600"))

DEFAULT_LOG_DIR = Path(os.getenv("TMPDIR", "/tmp")) / "chupps" / "request_logs"
LOG_DIR = Path(os.getenv("REQUEST_LOG_DIR", DEFAULT_LOG_DIR))
LOG_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "").strip()

_lock = threading.Lock()
_seq_by_request: dict[str, int] = {}
_redis_client: Any | None = None
_redis_init_failed = False


def _redis_keys(request_id: str) -> tuple[str, str]:
    safe_request_id = "".join(ch for ch in request_id if ch.isalnum() or ch in ("-", "_"))
    if not safe_request_id:
        safe_request_id = "unknown"
    return (f"request_logs:{safe_request_id}:events", f"request_logs:{safe_request_id}:seq")


def _storage_mode() -> str:
    if REQUEST_LOG_STORAGE in ("redis", "file", "auto"):
        return REQUEST_LOG_STORAGE
    return "auto"


def _should_use_redis() -> bool:
    mode = _storage_mode()
    if mode == "file":
        return False
    if mode == "redis":
        return True
    return bool(REDIS_URL)


def _get_redis_client() -> Any | None:
    global _redis_client, _redis_init_failed
    if not _should_use_redis() or Redis is None:
        return None
    if _redis_client is not None:
        return _redis_client
    if _redis_init_failed:
        return None

    try:
        _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_init_failed = True
        _redis_client = None
        return None


def _append_request_log_redis(
    request_id: str,
    step_key: str,
    summary: str,
    details: str | None = None,
    status: str = "INFO",
    wait_ms: int | None = None,
) -> dict[str, Any]:
    client = _get_redis_client()
    if client is None:
        raise RuntimeError("Redis client unavailable")

    logs_key, seq_key = _redis_keys(request_id)
    current = int(client.incr(seq_key))
    record: dict[str, Any] = {
        "request_id": request_id,
        "sequence": current,
        "step_key": step_key,
        "summary": summary,
        "details": details,
        "status": status,
        "wait_ms": wait_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if step_key == "TOOL_EXECUTION_COMPLETE":
        existing_raw = client.lrange(logs_key, 0, -1)
        existing_records: list[dict[str, Any]] = []
        for raw in existing_raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    existing_records.append(parsed)
            except json.JSONDecodeError:
                continue

        removed = _remove_latest_tool_execution_start(existing_records)
        if removed:
            pipe = client.pipeline()
            pipe.delete(logs_key)
            for existing in existing_records:
                pipe.rpush(logs_key, json.dumps(existing, ensure_ascii=False))
            pipe.rpush(logs_key, json.dumps(record, ensure_ascii=False))
            pipe.expire(logs_key, REQUEST_LOG_TTL_SECONDS)
            pipe.expire(seq_key, REQUEST_LOG_TTL_SECONDS)
            pipe.execute()
        else:
            pipe = client.pipeline()
            pipe.rpush(logs_key, json.dumps(record, ensure_ascii=False))
            pipe.expire(logs_key, REQUEST_LOG_TTL_SECONDS)
            pipe.expire(seq_key, REQUEST_LOG_TTL_SECONDS)
            pipe.execute()
    else:
        pipe = client.pipeline()
        pipe.rpush(logs_key, json.dumps(record, ensure_ascii=False))
        pipe.expire(logs_key, REQUEST_LOG_TTL_SECONDS)
        pipe.expire(seq_key, REQUEST_LOG_TTL_SECONDS)
        pipe.execute()

    return record


def _read_request_logs_redis(request_id: str, since_sequence: int = 0) -> list[dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        raise RuntimeError("Redis client unavailable")

    logs_key, _ = _redis_keys(request_id)
    raw_logs = client.lrange(logs_key, 0, -1)
    logs: list[dict[str, Any]] = []
    for raw in raw_logs:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if int(record.get("sequence", 0)) > since_sequence:
            logs.append(record)
    return logs


def _get_latest_sequence_redis(request_id: str) -> int:
    client = _get_redis_client()
    if client is None:
        raise RuntimeError("Redis client unavailable")

    _, seq_key = _redis_keys(request_id)
    seq = client.get(seq_key)
    if seq is None:
        return 0
    try:
        return int(seq)
    except ValueError:
        return 0


def _request_log_path(request_id: str) -> Path:
    safe_request_id = "".join(ch for ch in request_id if ch.isalnum() or ch in ("-", "_"))
    if not safe_request_id:
        safe_request_id = "unknown"
    return LOG_DIR / f"{safe_request_id}.jsonl"


def _read_latest_sequence_from_file(path: Path) -> int:
    if not path.exists():
        return 0

    latest = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                latest = max(latest, int(record.get("sequence", 0)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return latest


def _read_all_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)

    return records


def _remove_latest_tool_execution_start(records: list[dict[str, Any]]) -> bool:
    """Remove the newest TOOL_EXECUTION_START entry if present."""
    for idx in range(len(records) - 1, -1, -1):
        if records[idx].get("step_key") == "TOOL_EXECUTION_START":
            records.pop(idx)
            return True
    return False


def append_request_log(
    request_id: str,
    step_key: str,
    summary: str,
    details: str | None = None,
    status: str = "INFO",
    wait_ms: int | None = None,
) -> dict[str, Any]:
    """Append a log event for a request and return the persisted record."""
    if _should_use_redis():
        try:
            return _append_request_log_redis(
                request_id=request_id,
                step_key=step_key,
                summary=summary,
                details=details,
                status=status,
                wait_ms=wait_ms,
            )
        except Exception:
            # Fall back to local file storage when Redis is not reachable.
            pass

    with _lock:
        path = _request_log_path(request_id)
        if request_id not in _seq_by_request:
            _seq_by_request[request_id] = _read_latest_sequence_from_file(path)

        current = _seq_by_request.get(request_id, 0) + 1
        _seq_by_request[request_id] = current

        record: dict[str, Any] = {
            "request_id": request_id,
            "sequence": current,
            "step_key": step_key,
            "summary": summary,
            "details": details,
            "status": status,
            "wait_ms": wait_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if step_key == "TOOL_EXECUTION_COMPLETE":
            existing_records = _read_all_records(path)
            removed = _remove_latest_tool_execution_start(existing_records)

            if removed:
                with path.open("w", encoding="utf-8") as handle:
                    for existing in existing_records:
                        handle.write(json.dumps(existing, ensure_ascii=False) + "\n")
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record


def read_request_logs(request_id: str, since_sequence: int = 0) -> list[dict[str, Any]]:
    """Read log records for a request after the given sequence number."""
    if _should_use_redis():
        try:
            return _read_request_logs_redis(request_id=request_id, since_sequence=since_sequence)
        except Exception:
            # Fall back to local file storage when Redis is not reachable.
            pass

    path = _request_log_path(request_id)
    if not path.exists():
        return []

    logs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("sequence", 0)) > since_sequence:
                logs.append(record)

    return logs


def get_latest_sequence(request_id: str) -> int:
    """Return latest known sequence number for a request."""
    if _should_use_redis():
        try:
            return _get_latest_sequence_redis(request_id)
        except Exception:
            # Fall back to local file storage when Redis is not reachable.
            pass

    if request_id in _seq_by_request:
        return _seq_by_request[request_id]

    return _read_latest_sequence_from_file(_request_log_path(request_id))
