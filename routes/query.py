import uuid
import numpy as np
import pandas as pd
from decimal import Decimal
from fastapi import APIRouter, HTTPException, Header, Request
from models import QueryRequest
from workflow import app as workflow_app, AgentState
from utils.request_log_store import append_request_log, read_request_logs, get_latest_sequence

def convert_numpy_types(obj):
    """Recursively convert numpy types and pandas objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        # Convert DataFrame/Series to dict but handle NaN values
        result = obj.where(pd.notnull(obj), None).to_dict()
        return convert_numpy_types(result)  # Recursively clean the dict
    elif isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):  # Handle NaN and infinity
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Handle NaN values in arrays
        result = obj.tolist()
        return convert_numpy_types(result)  # Recursively clean the list
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, 'item'):  # numpy scalars
        item = obj.item()
        return convert_numpy_types(item)  # Recursively check the item
    elif hasattr(obj, 'tolist'):  # numpy arrays
        result = obj.tolist()
        return convert_numpy_types(result)  # Recursively clean the list
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif pd.isna(obj):  # Handle individual scalar NaN values (after DataFrame/Series check)
        return None
    else:
        return obj

router = APIRouter()


@router.get('/query/logs/{request_id}')
async def get_query_logs(request_id: str, since: int = 0):
    logs = read_request_logs(request_id, since_sequence=since)
    return {
        "success": True,
        "request_id": request_id,
        "logs": logs,
        "next_sequence": get_latest_sequence(request_id)
    }

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
        
        # Convert any numpy types in the result to JSON-serializable types
        result = convert_numpy_types(result)
        
        response_data = {
            "success": True,
            "query": user_query,
            "plan": result.get("plan", []),
            "is_comparison": result.get("comparison_mode", False)
        }
        
        # Convert the entire response as well
        response_data = convert_numpy_types(response_data)
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@router.post('/query')
async def process_query(
    request: QueryRequest,
    raw_request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID")
):
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

    print("running query...", flush=True)

    # Use request id from middleware/header if available.
    user_query = request.query.strip()
    request_id = getattr(raw_request.state, "request_id", None) or x_request_id or str(uuid.uuid4())[:8]

    append_request_log(
        request_id=request_id,
        step_key="REQUEST_START",
        summary="Query request accepted",
        details=user_query[:200],
        status="START"
    )
    
    try:
        async def workflow_logger(
            req_id: str,
            step_key: str,
            summary: str,
            payload: dict | None = None,
        ):
            payload = payload or {}
            append_request_log(
                request_id=req_id,
                step_key=step_key,
                summary=summary,
                details=payload.get("details"),
                status=payload.get("status", "INFO"),
                wait_ms=payload.get("wait_ms"),
            )

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
            logger=workflow_logger,
            request_id=request_id,
            comparison_mode=False,  # Will be set by planning LLM
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
        print("running state...", flush=True)
        result = await workflow_app.ainvoke(initial_state)

        append_request_log(
            request_id=request_id,
            step_key="WORKFLOW_COMPLETE",
            summary="Workflow execution finished",
            status="COMPLETE"
        )

        # Check for errors
        if result.get("error"):
            append_request_log(
                request_id=request_id,
                step_key="WORKFLOW_ERROR",
                summary="Workflow returned error",
                details=result["error"],
                status="ERROR"
            )
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
            print(f"retrieving cached result for: {result['final_result_ref']}", flush=True)
            from workflow import get_cached_result
            
            try:
                final_data = get_cached_result(result["final_result_ref"])
                print(f"cached result retrieved, type: {type(final_data)}, length: {len(final_data) if isinstance(final_data, (list, dict)) else 'N/A'}", flush=True)
                
                # Convert numpy types to JSON-serializable types
                final_data = convert_numpy_types(final_data)
                
                # Get query type from the plan
                query_type = result.get("plan", {}).get("query_type", "standard")
                
                # Handle different response formats based on query type
                if query_type == "comparison" and isinstance(final_data, dict) and "insights" in final_data:
                    # Comparison query response
                    response_data = {
                        "success": True,
                        "query_type": "comparison",
                        "request_id": request_id,
                        "logs": read_request_logs(request_id),
                        "summarized_query": result.get("summarized_query", ""),
                        "insights": final_data["insights"],
                        "comparison_data": final_data.get("comparison_data"),
                        "detailed_metrics": final_data.get("detailed_metrics")
                    }
                elif query_type == "metric_analysis" and isinstance(final_data, dict) and "metrics" in final_data and ("analysis" in final_data or "insights" in final_data):
                    # Metric analysis query response
                    response_data = {
                        "success": True,
                        "query_type": "metric_analysis",
                        "request_id": request_id,
                        "logs": read_request_logs(request_id),
                        "summarized_query": result.get("summarized_query", ""),
                        "query": final_data["query"],
                        "metrics": final_data["metrics"],
                        "analysis": final_data.get("analysis", final_data.get("insights", "")),
                        "metrics_calculated": final_data.get("metrics_calculated", []),
                        "plan": final_data["plan"]
                    }
                elif query_type == "schema_discovery" and isinstance(final_data, dict):
                    # Schema discovery response
                    response_data = {
                        "success": True,
                        "query_type": "schema_discovery", 
                        "request_id": request_id,
                        "logs": read_request_logs(request_id),
                        "summarized_query": result.get("summarized_query", ""),
                        **final_data  # Include all schema data
                    }
                else:
                    # Standard query response (data list or other)
                    # Check if final_data is already a complete response structure
                    if isinstance(final_data, dict) and "success" in final_data and "data" in final_data:
                        # final_data is already a complete response, return it directly
                        response_data = final_data
                        # Update with current request info if needed
                        response_data["request_id"] = request_id
                        response_data["logs"] = read_request_logs(request_id)
                        if result.get("summarized_query"):
                            response_data["summarized_query"] = result.get("summarized_query")
                    else:
                        # final_data is raw data, wrap it in response structure
                        record_count = len(final_data) if isinstance(final_data, list) else 1
                        response_data = {
                            "success": True,
                            "query_type": "standard",
                            "request_id": request_id,
                            "logs": read_request_logs(request_id),
                            "summarized_query": result.get("summarized_query", ""),
                            "count": record_count,
                            "data": final_data,
                            "total_records": record_count
                        }
                
                print("response constructed successfully", flush=True)
                
            except Exception as cache_error:
                print(f"error retrieving cached result: {cache_error}", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "success": False,
                        "error": f"Cache retrieval error: {str(cache_error)}",
                        "request_id": request_id
                    }
                )

            # Add insights if available
            try:
                if result.get("insights"):
                    print(f"adding insights: {type(result['insights'])}", flush=True)
                    response_data["insights"] = result["insights"]
                
                # Add metric analysis if available
                if result.get("metric_analysis"):
                    print(f"adding metric analysis: {type(result['metric_analysis'])}", flush=True)
                    response_data["metric_analysis"] = result["metric_analysis"]
                
                # Add comparison results if available
                if result.get("comparison_results"):
                    print(f"adding comparison results: {type(result['comparison_results'])}", flush=True)
                    response_data["comparison_results"] = result["comparison_results"]
                    response_data["comparison_groups"] = result.get("comparison_groups", [])
                
                print("optional fields added successfully", flush=True)
                
            except Exception as field_error:
                print(f"error adding optional fields: {field_error}", flush=True)
                # Continue without the optional fields rather than failing
            
            try:
                print(f"returning response_data keys: {list(response_data.keys())}", flush=True)
                # Convert all numpy types in response data to JSON-serializable types
                response_data = convert_numpy_types(response_data)
                return response_data
            except Exception as return_error:
                print(f"error in return statement: {return_error}", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "success": False,
                        "error": f"Response return error: {str(return_error)}",
                        "request_id": request_id
                    }
                )
        
        # Fallback: If final_result_ref is missing but we have tool_result_refs, try to use the last result
        elif result.get("tool_result_refs") and result.get("plan") and result["plan"].get("steps"):
            print("final_result_ref missing, trying fallback...", flush=True)
            from workflow import get_cached_result
            
            try:
                # Get the last step result
                last_step = result["plan"]["steps"][-1]
                if "save_as" in last_step and last_step["save_as"] in result["tool_result_refs"]:
                    last_result_ref = result["tool_result_refs"][last_step["save_as"]]
                    final_data = get_cached_result(last_result_ref)
                    
                    # Convert numpy types to JSON-serializable types
                    final_data = convert_numpy_types(final_data)
                    
                    response_data = {
                        "success": True,
                        "request_id": request_id,
                        "logs": read_request_logs(request_id),
                        "data": final_data,
                        "metadata": {"source": "fallback_last_step"}
                    }
                    
                    # Convert the entire response data as well
                    response_data = convert_numpy_types(response_data)
                    
                    print("fallback successful", flush=True)
                    return response_data
                else:
                    print("fallback failed: save_as missing or not in tool_result_refs", flush=True)
            except Exception as fallback_error:
                print(f"fallback error: {fallback_error}", flush=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No result generated",
                "request_id": request_id
            }
        )
        
    except HTTPException as http_error:
        append_request_log(
            request_id=request_id,
            step_key="REQUEST_ERROR",
            summary="HTTP exception in /query",
            details=str(http_error.detail),
            status="ERROR"
        )
        raise http_error

    except Exception as e:
        append_request_log(
            request_id=request_id,
            step_key="REQUEST_ERROR",
            summary="Unhandled exception in /query",
            details=str(e),
            status="ERROR"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )
