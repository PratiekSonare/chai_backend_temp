"""
Metrics Service - Dedicated FastAPI instance for metric and chart calculations
Handles: /orders/*, /revenue/*, /payment/*, /cancellation/*, /geography/*, /history/* endpoints
Runs on: METRICS_PORT (default 5002)

This service is optimized for data aggregation, metric calculations,
and chart generation without the complexity of the query workflow engine.
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
from routes.orders import router as orders_router
from routes.revenue import router as revenue_router
from routes.payment import router as payment_router
from routes.cancellation import router as cancellation_router
from routes.geography import router as geography_router
from routes.historyOrders import router as history_orders_router
from routes.prediction import router as prediction_router
from routes.forecast import router as forecast_router
from routes.inventory import router as inventory_router


app = FastAPI(
    title="Metrics Service - Metric & Chart Calculations",
    description="FastAPI server for orders metrics, revenue analysis, and chart generation",
    version="1.0.0"
)

async def startup_event():
    """Print startup banner once when the application starts."""
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('METRICS_PORT', 5002))

    print(f"\n{'='*70}", flush=True)
    print(f"📊 METRICS SERVICE - Metric & Chart Calculations", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"📡 Service: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"📦 Orders: POST http://{host}:{port}/orders/metrics", flush=True)
    print(f"💰 Revenue: POST http://{host}:{port}/revenue/chart/line", flush=True)
    print(f"💳 Payment: POST http://{host}:{port}/payment/chart/radial", flush=True)
    print(f"❌ Cancellation: POST http://{host}:{port}/cancellation/chart/bar", flush=True)
    print(f"🗺️  Geography: POST http://{host}:{port}/geography/chart/pincode", flush=True)
    print(f"📦 Inventory: GET http://{host}:{port}/inventory/snapshot", flush=True)
    print(f"📖 Docs: http://{host}:{port}/docs", flush=True)
    print(f"📋 ReDoc: http://{host}:{port}/redoc", flush=True)
    print(f"{'='*70}\n", flush=True)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a stable request id for correlation and logging."""
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
app.include_router(orders_router)
app.include_router(revenue_router)
app.include_router(payment_router)
app.include_router(cancellation_router)
app.include_router(geography_router)
app.include_router(history_orders_router)
app.include_router(prediction_router)
app.include_router(forecast_router)
app.include_router(inventory_router)


@app.on_event("startup")
async def on_startup():
    """Run startup event"""
    await startup_event()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('METRICS_PORT', 5002))
    uvicorn.run(app, host=host, port=port)
