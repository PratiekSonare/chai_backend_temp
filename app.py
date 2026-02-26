import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import route modules
from routes.health import router as health_router
from routes.websocket import router as websocket_router  
from routes.query import router as query_router
from routes.orders import router as orders_router
from routes.revenue import router as revenue_router

app = FastAPI(
    title="Order Analysis Workflow API",
    description="FastAPI server for processing order analysis queries",
    version="1.0.0"
)

# Enable CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(websocket_router)
app.include_router(query_router)
app.include_router(orders_router)
app.include_router(revenue_router)

if __name__ == '__main__':
    # Load environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 Order Analysis Workflow Server (FastAPI)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"📡 Server: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"🧠 Plan: POST http://{host}:{port}/plan", flush=True)
    print(f"🔍 Query: POST http://{host}:{port}/query", flush=True)
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