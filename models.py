from pydantic import BaseModel
from typing import List, Dict, Any

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    service: str

class LogsRequest(BaseModel):
    request_id: str

class OrdersMetricsRequest(BaseModel):
    orders: List[Dict[str, Any]]
