"""
FastAPI Server for Order Analysis Workflow
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import workflow components
from workflow import app as workflow_app, AgentState

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

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    service: str


@app.get('/health', response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "order-analysis-workflow"
    }

@app.post('/plan')
def generate_plan(request: QueryRequest):
    """
    Generate execution plan for a natural language query
    
    Request body:
    {
        "query": "Show me orders from last 5 days with payment mode prepaid"
    }
    
    Response:
    {
        "success": true,
        "query": "...",
        "plan": {
            "query_type": "standard",
            "steps": [...],
            "manipulation": {...}
        }
    }
    """
    try:
        user_query = request.query
        
        # Initialize state for planning only
        initial_state = AgentState(
            user_query=user_query,
            plan=None,
            tool_result_refs={},
            tool_result_schemas={},
            current_step_index=0,
            filters=None,
            final_result_ref=None,
            error=None,
            retry_count=0,
            comparison_mode=False,
            comparison_groups=None,
            group_results=None,
            group_schemas=None,
            current_group_index=0,
            aggregated_metrics=None,
            comparison_results=None,
            insights=None,
            metric_results=None,
            metric_analysis=None
        )
        
        # Run only the planning node
        from workflow import planning_node
        result = planning_node(initial_state)
        
        # Check for planning errors
        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": result["error"]
                }
            )
        
        # Return the generated plan
        if result.get("plan"):
            return {
                "success": True,
                "query": user_query,
                "plan": result["plan"]
            }
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No plan generated"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@app.post('/query')
def process_query(request: QueryRequest):
    """
    Process a natural language query
    
    Request body:
    {
        "query": "Show me orders from last 5 days with payment mode prepaid"
    }
    
    Response:
    {
        "success": true,
        "data": [...],
        "insights": "...",
        "metadata": {...}
    }
    """
    try:
        user_query = request.query
        
        # Determine query type
        is_comparison = any(word in user_query.lower() for word in ["compare", "vs", "versus", "between"])
        is_metric = any(word in user_query.lower() for word in ["aov", "average order", "revenue", "metrics", "calculate", "total"])
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            plan=None,
            tool_result_refs={},
            tool_result_schemas={},
            current_step_index=0,
            filters=None,
            final_result_ref=None,
            error=None,
            retry_count=0,
            comparison_mode=is_comparison,
            comparison_groups=None,
            group_results=None,
            group_schemas=None,
            current_group_index=0,
            aggregated_metrics=None,
            comparison_results=None,
            insights=None,
            metric_results=None,
            metric_analysis=None
        )
        
        # Run workflow
        result = workflow_app.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": result["error"]
                }
            )
        
        # Get final result
        if result.get("final_result_ref"):
            from workflow import get_cached_result
            final_data = get_cached_result(result["final_result_ref"])
            
            # For comparison queries
            if is_comparison and isinstance(final_data, dict) and "insights" in final_data:
                return {
                    "success": True,
                    "query_type": "comparison",
                    "insights": final_data["insights"],
                    "comparison_data": final_data.get("comparison_data"),
                    "detailed_metrics": final_data.get("detailed_metrics")
                }
            
            # For metric analysis queries
            elif isinstance(final_data, dict) and "metrics" in final_data and "analysis" in final_data:
                return {
                    "success": True,
                    "query_type": "metric_analysis",
                    "query": final_data["query"],
                    "metrics": final_data["metrics"],
                    "analysis": final_data["analysis"],
                    "metrics_calculated": final_data.get("metrics_calculated", [])
                }
            
            # For standard queries
            else:
                return {
                    "success": True,
                    "query_type": "standard",
                    "count": len(final_data) if isinstance(final_data, list) else 1,
                    "data": final_data[:100] if isinstance(final_data, list) else final_data,  # Limit to 100 records
                    "total_records": len(final_data) if isinstance(final_data, list) else 1
                }
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No result generated"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@app.get('/examples')
def get_examples():
    """Get example queries and API usage"""
    return {
        "endpoints": {
            "/plan": "POST - Generate execution plan without running the query",
            "/query": "POST - Execute the full query workflow and return results"
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

if __name__ == '__main__':
    import uvicorn
    
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
