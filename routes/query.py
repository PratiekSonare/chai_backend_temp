import uuid
from fastapi import APIRouter, HTTPException
from models import QueryRequest
from workflow import app as workflow_app, AgentState, is_comparison_query
from utils.connection_manager import manager

router = APIRouter()

@router.post('/plan')
async def generate_plan(request: QueryRequest):
    try:
        user_query = request.query.strip()
        
        # Create initial state for planning
        initial_state = AgentState(
            user_query=user_query,
            summarized_query=None,
            plan=None,
            # ... other fields with default values
        )
        
        # Use planner node only for plan generation
        from workflow import planner
        result = planner(initial_state)
        
        return {
            "success": True,
            "query": user_query,
            "plan": result.get("plan", []),
            "is_comparison": result.get("comparison_mode", False)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@router.post('/query')
async def process_query(request: QueryRequest):
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
    # Generate unique request ID
    user_query = request.query.strip()
    is_comparison = is_comparison_query(user_query)
    request_id = str(uuid.uuid4())[:8]
    
    try:
        await manager.log_request_start(request_id, user_query)
        await manager.log_request_step(request_id, "initialize", "Initializing workflow")
        
        initial_state = AgentState(
            user_query=user_query,
            summarized_query=None,
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
        await manager.log_request_step(request_id, "execute_workflow", "Running order analysis workflow")
        result = workflow_app.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            await manager.log_request_end(request_id, True, error=result["error"])
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": result["error"],
                    "request_id": request_id
                }
            )
        
        # Get final result
        if result.get("final_result_ref"):
            from workflow import get_cached_result
            final_data = get_cached_result(result["final_result_ref"])
            
            response_data = {
                "success": True,
                "request_id": request_id,
                **final_data
            }
            
            # Add insights if available
            if result.get("insights"):
                response_data["insights"] = result["insights"]
            
            # Add metric analysis if available
            if result.get("metric_analysis"):
                response_data["metric_analysis"] = result["metric_analysis"]
            
            # Add comparison results if available
            if result.get("comparison_results"):
                response_data["comparison_results"] = result["comparison_results"]
                response_data["comparison_groups"] = result.get("comparison_groups", [])
            
            await manager.log_request_end(request_id, False)
            return response_data
        
        await manager.log_request_end(request_id, False, error="No result generated")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No result generated",
                "request_id": request_id
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
