import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import redis
except Exception:
    redis = None

PRESET_7D = "7d"
PRESET_30D = "30d"
PRESET_ALL = "all"
SUPPORTED_PRESETS = (PRESET_7D, PRESET_30D, PRESET_ALL)

DEFAULT_ALL_TIME_START = os.getenv("HISTORY_CACHE_ALL_TIME_START", "2025-09-01 00:00:00")
CACHE_TTL_SECONDS = int(os.getenv("HISTORY_KPI_CACHE_TTL_SECONDS", "1200"))
REDIS_URL = os.getenv("REDIS_URL", "").strip()

_redis_client: Any | None = None
_redis_init_failed = False


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    return datetime.fromisoformat(value_str.replace("Z", "+00:00"))


def get_redis_client() -> Any | None:
    global _redis_client, _redis_init_failed
    if redis is None or not REDIS_URL:
        return None
    if _redis_init_failed:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_init_failed = True
        _redis_client = None
        return None


def resolve_preset(start_date: Optional[str], end_date: Optional[str], filters: Optional[dict]) -> Optional[str]:
    if isinstance(filters, dict) and filters:
        return None

    try:
        start_dt = _parse_dt(start_date)
        end_dt = _parse_dt(end_date)
    except Exception:
        return None

    if not start_dt or not end_dt or start_dt > end_dt:
        return None

    days = (end_dt.date() - start_dt.date()).days + 1
    if days == 7:
        return PRESET_7D
    if days == 30:
        return PRESET_30D

    try:
        all_start = _parse_dt(DEFAULT_ALL_TIME_START)
    except Exception:
        all_start = None

    if all_start and start_dt.date() <= all_start.date() and abs((datetime.now().date() - end_dt.date()).days) <= 2:
        return PRESET_ALL

    return None


def build_preset_window(preset: str) -> tuple[str, str]:
    now = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    if preset == PRESET_7D:
        start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif preset == PRESET_30D:
        start = (now - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif preset == PRESET_ALL:
        parsed = _parse_dt(DEFAULT_ALL_TIME_START)
        if parsed is None:
            raise ValueError("Invalid HISTORY_CACHE_ALL_TIME_START")
        start = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    return (
        start.isoformat(sep=" ", timespec="seconds"),
        now.isoformat(sep=" ", timespec="seconds"),
    )


def _version_key(table_name: str) -> str:
    return f"history:kpi:version:{table_name}"


def current_cache_version(table_name: str) -> int:
    client = get_redis_client()
    if not client:
        return 1
    try:
        raw = client.get(_version_key(table_name))
        if raw is None:
            return 1
        return int(raw)
    except Exception:
        return 1


def bump_cache_version(table_name: str) -> int:
    client = get_redis_client()
    if not client:
        return 1
    try:
        return int(client.incr(_version_key(table_name)))
    except Exception:
        return current_cache_version(table_name)


def preset_cache_key(table_name: str, preset: str, version: int | None = None) -> str:
    cache_version = version if version is not None else current_cache_version(table_name)
    return f"history:kpi:all:{table_name}:{preset}:v{cache_version}"


def batch_all_metrics_cache_key(table_name: str, preset: str, version: int | None = None) -> str:
    cache_version = version if version is not None else current_cache_version(table_name)
    return f"history:kpi:batch-all-metrics:{table_name}:{preset}:v{cache_version}"


def get_cached_preset_payload(table_name: str, preset: str) -> dict | None:
    client = get_redis_client()
    if not client:
        return None
    key = preset_cache_key(table_name, preset)
    try:
        payload = client.get(key)
        if not payload:
            return None
        data = json.loads(payload)
        if isinstance(data, dict):
            data["source"] = "preset-cache"
        return data
    except Exception:
        return None


def set_cached_preset_payload(table_name: str, preset: str, payload: dict, version: int | None = None) -> None:
    client = get_redis_client()
    if not client:
        return
    key = preset_cache_key(table_name, preset, version=version)
    try:
        client.set(key, json.dumps(payload, default=str), ex=CACHE_TTL_SECONDS)
    except Exception:
        return


def get_cached_batch_all_metrics_payload(table_name: str, preset: str) -> dict | None:
    client = get_redis_client()
    if not client:
        return None
    key = batch_all_metrics_cache_key(table_name, preset)
    try:
        payload = client.get(key)
        if not payload:
            return None
        data = json.loads(payload)
        if isinstance(data, dict):
            data["source"] = "preset-cache"
        return data
    except Exception:
        return None


def set_cached_batch_all_metrics_payload(table_name: str, preset: str, payload: dict, version: int | None = None) -> None:
    client = get_redis_client()
    if not client:
        return
    key = batch_all_metrics_cache_key(table_name, preset, version=version)
    try:
        client.set(key, json.dumps(payload, default=str), ex=CACHE_TTL_SECONDS)
    except Exception:
        return


def enqueue_preset_refresh(table_name: str, preset: str, version: int | None = None) -> None:
    try:
        from celery_app import celery_app

        celery_app.send_task(
            "history_kpi.precompute_preset",
            kwargs={
                "table_name": table_name,
                "preset": preset,
                "cache_version": version,
            },
        )
    except Exception:
        return


def enqueue_all_presets_refresh(table_name: str, version: int | None = None) -> None:
    for preset in SUPPORTED_PRESETS:
        enqueue_preset_refresh(table_name=table_name, preset=preset, version=version)


def enqueue_batch_all_metrics_refresh(table_name: str, preset: str, version: int | None = None) -> None:
    try:
        from celery_app import celery_app

        celery_app.send_task(
            "history_kpi.precompute_batch_all_metrics_preset",
            kwargs={
                "table_name": table_name,
                "preset": preset,
                "cache_version": version,
            },
        )
    except Exception:
        return


def enqueue_all_batch_all_metrics_refresh(table_name: str, version: int | None = None) -> None:
    for preset in SUPPORTED_PRESETS:
        enqueue_batch_all_metrics_refresh(table_name=table_name, preset=preset, version=version)
