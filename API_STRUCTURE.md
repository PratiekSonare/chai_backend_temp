# API Structure Documentation

## New Modular Structure

The FastAPI application has been reorganized into a cleaner, more maintainable structure:

```
backend/
├── app.py                 # Main FastAPI application entry point
├── models.py              # Pydantic models for requests/responses
├── routes/                # API route modules
│   ├── __init__.py
│   ├── health.py          # Health checks and examples
│   ├── websocket.py       # SSE endpoints for logging
│   ├── query.py           # Query processing endpoints (/plan, /query)
│   ├── orders.py          # Order-related endpoints (/orders/*)
│   └── revenue.py         # Revenue-related endpoints (/revenue/*)
├── utils/                 # Utility modules
│   ├── __init__.py
│   └── connection_manager.py  # SSE connection management
├── workflow.py            # Existing workflow logic
├── tools.py              # Existing tools
└── llm_providers.py       # Existing LLM providers
```

## API Endpoints

### Health & Utilities
- `GET /health` - Health check
- `GET /examples` - API usage examples

### Query Processing
- `POST /plan` - Generate execution plan without running query
- `POST /query` - Execute full query workflow

### Orders
- `POST /orders/metrics` - Calculate comprehensive order metrics
- `POST /orders/chart/count` - Get order count chart data

### Revenue
- `POST /revenue/chart/line` - Get revenue and AOV chart data

### SSE
- `GET /sse/logs` - Real-time log streaming
- `GET /logs/{request_id}` - Get logs for specific request

## Benefits of New Structure

1. **Separation of Concerns**: Each route module handles specific functionality
2. **Maintainability**: Easier to find and modify specific endpoints
3. **Scalability**: Easy to add new route modules
4. **Code Reusability**: Shared models and utilities
5. **Testing**: Each module can be tested independently

## Usage

The main `app.py` simply imports and registers all the route modules:

```python
from routes.health import router as health_router
# ... other imports

app.include_router(health_router)
# ... other routers
```

## Adding New Endpoints

1. Create a new file in the `routes/` directory
2. Define your routes using FastAPI's `APIRouter`
3. Import and include the router in `app.py`

Example:
```python
# routes/new_feature.py
from fastapi import APIRouter
router = APIRouter()

@router.get('/new-endpoint')
def new_endpoint():
    return {"message": "Hello from new feature"}

# app.py
from routes.new_feature import router as new_feature_router
app.include_router(new_feature_router)
```

## Files to Keep

- `app_backup.py` - Original monolithic app.py (backup)
- `app_old.py` - Previous version before modularization
- `app.py` - New modular version (active)