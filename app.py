import os
import uuid
import uvicorn
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

# Override FastAPI's JSON response encoder
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
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    return jsonable_encoder(obj)

# Import route modules
from routes.health import router as health_router
from routes.query import router as query_router
from routes.orders import router as orders_router
from routes.revenue import router as revenue_router
from routes.payment import router as payment_router
from routes.cancellation import router as cancellation_router
from routes.geography import router as geography_router
from routes.reasoning import router as reasoning_router

app = FastAPI(
    title="Order Analysis Workflow API",
    description="FastAPI server for processing order analysis queries",
    version="1.0.0"
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a stable request id for correlation across query processing and log polling."""
    incoming_request_id = request.headers.get("X-Request-ID")
    request_id = incoming_request_id or str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Configure FastAPI to use custom JSON encoder for numpy/pandas types
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

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000"
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
app.include_router(orders_router)
app.include_router(revenue_router)
app.include_router(payment_router)
app.include_router(cancellation_router)
app.include_router(geography_router)
app.include_router(reasoning_router)

if __name__ == '__main__':
    # Load environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    print("api-key: ", os.getenv('OPENROUTER_API_KEY'))

    print(f"\n{'='*60}", flush=True)
    print(f"🚀 Order Analysis Workflow Server (FastAPI)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"📡 Server: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"🧠 Plan: POST http://{host}:{port}/plan", flush=True)
    print(f"🔍 Query: POST http://{host}:{port}/query", flush=True)
    print(f"🧠 Reasoning: http://{host}:{port}/reasoning/", flush=True)
    print(f"📖 Docs: http://{host}:{port}/docs", flush=True)
    print(f"📋 ReDoc: http://{host}:{port}/redoc", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )