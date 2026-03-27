from datetime import datetime
import ast
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


def _default_end_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _default_start_date() -> str:
    return "2025-09-01 00:00:00"

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str


class ExecuteRequest(BaseModel):
    query: str
    plan: Dict[str, Any]
    summarized_query: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str

class OrdersMetricsRequest(BaseModel):
    orders: List[Dict[str, Any]]

class GeographyRequest(BaseModel):
    orders: List[Dict[str, Any]]
    state: str

class HistoryOrdersRequest(BaseModel):
    """Request model for historical orders queries from DyanmoDB"""
    table_name: str = "history-orders"
    start_date: str = Field(default_factory=_default_start_date)
    end_date: str = Field(default_factory=_default_end_date)
    filters: Optional[Dict[str, Any]] = None

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def normalize_datetime_bounds(cls, value: Any, info):
        if value is None:
            return _default_start_date() if info.field_name == "start_date" else _default_end_date()

        value_str = str(value).strip()
        if not value_str:
            return _default_start_date() if info.field_name == "start_date" else _default_end_date()

        try:
            parsed = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("start_date/end_date must be valid date or datetime values") from exc

        is_date_only = len(value_str) == 10 and value_str[4] == "-" and value_str[7] == "-"
        if is_date_only:
            if info.field_name == "start_date":
                parsed = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=0)

        return parsed.isoformat(sep=" ", timespec="seconds")