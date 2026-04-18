from datetime import datetime
import asyncio

from celery_app import celery_app
from models import HistoryOrdersRequest
from services.history_kpi_cache import (
    SUPPORTED_PRESETS,
    build_preset_window,
    current_cache_version,
    set_cached_preset_payload,
    set_cached_batch_all_metrics_payload,
)


@celery_app.task(name="history_kpi.precompute_preset")
def precompute_preset(table_name: str, preset: str, cache_version: int | None = None) -> dict:
    if preset not in SUPPORTED_PRESETS:
        return {"success": False, "error": f"Unsupported preset: {preset}"}

    if cache_version is None:
        cache_version = current_cache_version(table_name)

    start_date, end_date = build_preset_window(preset)
    request = HistoryOrdersRequest(
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
        filters=None,
    )

    # Local import avoids circular import during module initialization.
    from routes.historyOrders import build_kpi_response

    payload = build_kpi_response(request=request, source="celery")
    set_cached_preset_payload(
        table_name=table_name,
        preset=preset,
        payload=payload,
        version=cache_version,
    )

    return {
        "success": True,
        "preset": preset,
        "table_name": table_name,
        "cache_version": cache_version,
        "computed_at": datetime.now().isoformat(),
    }


@celery_app.task(name="history_kpi.refresh_all_presets")
def refresh_all_presets(table_name: str, cache_version: int | None = None) -> dict:
    for preset in SUPPORTED_PRESETS:
        precompute_preset.delay(
            table_name=table_name,
            preset=preset,
            cache_version=cache_version,
        )
    return {
        "success": True,
        "table_name": table_name,
        "presets": list(SUPPORTED_PRESETS),
    }


@celery_app.task(name="history_kpi.precompute_batch_all_metrics_preset")
def precompute_batch_all_metrics_preset(table_name: str, preset: str, cache_version: int | None = None) -> dict:
    if preset not in SUPPORTED_PRESETS:
        return {"success": False, "error": f"Unsupported preset: {preset}"}

    if cache_version is None:
        cache_version = current_cache_version(table_name)

    start_date, end_date = build_preset_window(preset)
    request = HistoryOrdersRequest(
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
        filters=None,
    )

    # Local import avoids circular import during module initialization.
    from routes.historyOrders import batch_all_metrics

    payload = asyncio.run(batch_all_metrics(request=request))
    set_cached_batch_all_metrics_payload(
        table_name=table_name,
        preset=preset,
        payload=payload,
        version=cache_version,
    )

    return {
        "success": True,
        "preset": preset,
        "table_name": table_name,
        "cache_version": cache_version,
        "computed_at": datetime.now().isoformat(),
    }
