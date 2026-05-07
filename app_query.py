"""
Query Service - Dedicated FastAPI instance for data search workflow agent
Handles: /plan, /query, /execute, /query/logs, /query/*/cancel endpoints
Runs on: QUERY_PORT (default 5001)

This service contains the LangGraph-based workflow for query processing,
planning, and execution. It maintains request logging, cancellation registry,
and SSE streaming for real-time log updates.
"""

import os
import uuid
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import json
from decimal import Decimal

# Load environment variables from .env file
load_dotenv()

# Custom JSON encoder to handle numpy and pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        return super().default(obj)


# Override FastAPI's JSON response encoder (fallback)
def custom_json_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return jsonable_encoder(obj)


# Import route modules
from routes.health import router as health_router
from routes.query import router as query_router


app = FastAPI(
    title="Query Service - Data Search Workflow Agent",
    description="FastAPI server for data search planning, querying, and execution via LangGraph workflow",
    version="1.0.0"
)

async def startup_event():
    """Print startup banner once when the application starts."""
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('QUERY_PORT', 5001))

    print(f"\n{'='*70}", flush=True)
    print(f"🔍 QUERY SERVICE - Data Search Workflow Agent", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"📡 Service: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"🧠 Plan: POST http://{host}:{port}/plan", flush=True)
    print(f"🔍 Query: POST http://{host}:{port}/query", flush=True)
    print(f"📖 Docs: http://{host}:{port}/docs", flush=True)
    print(f"📋 ReDoc: http://{host}:{port}/redoc", flush=True)
    print(f"{'='*70}\n", flush=True)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a stable request id for correlation across query processing and log polling."""
    incoming_request_id = request.headers.get("X-Request-ID")
    request_id = incoming_request_id or str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Custom JSON Response class
from fastapi.responses import JSONResponse
import typing

class CustomJSONResponse(JSONResponse):
    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            cls=NumpyEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# Set default response class
app.default_response_class = CustomJSONResponse


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000",
        "https://www.engineermonke.space",
        "https://engineermonke.space",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-Request-ID",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"]
)


# Include routers
app.include_router(health_router)
app.include_router(query_router)


@app.on_event("startup")
async def on_startup():
    """Run startup event"""
    await startup_event()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('QUERY_PORT', 5001))
    uvicorn.run(app, host=host, port=port)
