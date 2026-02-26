from fastapi import APIRouter
from models import HealthResponse

router = APIRouter()

@router.get('/health', response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "service": "Order Analysis Workflow API"
    }

@router.get('/examples')
def get_examples():
    """Get example queries and API usage"""
    return {
        "endpoints": {
            "/plan": "POST - Generate execution plan without running the query",
            "/query": "POST - Execute the full query workflow and return results",
            "/ws/logs": "WebSocket - Real-time log streaming for all requests",
            "/logs/{request_id}": "GET - Retrieve logs for a specific request ID"
        },
        "schema_discovery_queries": [
            "What fields are available in the orders data?",
            "What are the allowed values for payment_mode?",
            "Which date ranges does this API support?",
            "What marketplaces do we have data for?",
            "Show me the complete data schema",
            "What enum values are available for order_status?"
        ],
        "standard_queries": [
            "Show me orders from last 5 days with payment mode prepaid",
            "Get all open orders from last week",
            "Orders from Karnataka in last 10 days",
            "Show COD orders from last 3 days"
        ],
        "metric_analysis_queries": [
            "Calculate AOV from the past 2 days",
            "What is the average order value and revenue from last 7 days?",
            "Calculate total revenue and order count for last month",
            "Show me key metrics for orders from last week"
        ],
        "comparison_queries": [
            "Compare orders between Shopify13 and Flipkart from the last 10 days",
            "Compare prepaid vs COD orders from last week",
            "Compare Karnataka vs Maharashtra order volumes in last 15 days",
            "Compare Shopify13, Flipkart, and Amazon sales from last month"
        ]
    }
