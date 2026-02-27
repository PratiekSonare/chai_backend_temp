import uuid
from fastapi import APIRouter, HTTPException
from models import QueryRequest
from workflow import app as workflow_app, AgentState
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

    print("running query...", flush=True)

    # Generate unique request ID
    user_query = request.query.strip()
    request_id = str(uuid.uuid4())[:8]
    
    try:
        await manager.log_request_start(request_id, user_query)
        await manager.log_request_step(request_id, "initialize", "Initializing workflow")
        
        # Analyze query type (will be determined by planning LLM)
        await manager.log_request_step(request_id, "analyze_query", "Planning LLM will determine query type")
        
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
        await manager.log_request_step(request_id, "execute_workflow", "Running order analysis workflow")

        print("running state...", flush=True)
        result = workflow_app.invoke(initial_state)
        
        # Debug: Print the result structure
        print("workflow result keys:", list(result.keys()) if isinstance(result, dict) else type(result), flush=True)
        print("final_result_ref:", result.get("final_result_ref"), flush=True)
        print("tool_result_refs:", result.get("tool_result_refs"), flush=True)
        print("error:", result.get("error"), flush=True)
        
        # Debug: Show what the planning LLM determined
        if result.get("plan"):
            plan = result["plan"]
            print(f"planning LLM determined query_type: {plan.get('query_type')}", flush=True)
            print(f"planning LLM set comparison_mode: {result.get('comparison_mode')}", flush=True)
            # Log the workflow type that was executed
            query_type = plan.get('query_type', 'standard')
            await manager.log_request_step(request_id, "workflow_executed", f"Executed {query_type} workflow")
        
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
            await manager.log_request_step(request_id, "process_results", "Processing final results")
            print(f"retrieving cached result for: {result['final_result_ref']}", flush=True)
            from workflow import get_cached_result
            
            try:
                final_data = get_cached_result(result["final_result_ref"])
                print(f"cached result retrieved, type: {type(final_data)}, length: {len(final_data) if isinstance(final_data, (list, dict)) else 'N/A'}", flush=True)
                
                # Get query type from the plan
                query_type = result.get("plan", {}).get("query_type", "standard")
                print(f"query type: {query_type}", flush=True)
                
                # Handle different response formats based on query type
                if query_type == "comparison" and isinstance(final_data, dict) and "insights" in final_data:
                    # Comparison query response
                    response_data = {
                        "success": True,
                        "query_type": "comparison",
                        "request_id": request_id,
                        "summarized_query": result.get("summarized_query", ""),
                        "insights": final_data["insights"],
                        "comparison_data": final_data.get("comparison_data"),
                        "detailed_metrics": final_data.get("detailed_metrics")
                    }
                elif query_type == "metric_analysis" and isinstance(final_data, dict) and "metrics" in final_data and "analysis" in final_data:
                    # Metric analysis query response
                    response_data = {
                        "success": True,
                        "query_type": "metric_analysis",
                        "request_id": request_id,
                        "summarized_query": result.get("summarized_query", ""),
                        "query": final_data["query"],
                        "metrics": final_data["metrics"],
                        "analysis": final_data["analysis"],
                        "metrics_calculated": final_data.get("metrics_calculated", [])
                    }
                elif query_type == "schema_discovery" and isinstance(final_data, dict):
                    # Schema discovery response
                    response_data = {
                        "success": True,
                        "query_type": "schema_discovery", 
                        "request_id": request_id,
                        "summarized_query": result.get("summarized_query", ""),
                        **final_data  # Include all schema data
                    }
                else:
                    # Standard query response (data list or other)
                    record_count = len(final_data) if isinstance(final_data, list) else 1
                    response_data = {
                        "success": True,
                        "query_type": "standard",
                        "request_id": request_id,
                        "summarized_query": result.get("summarized_query", ""),
                        "count": record_count,
                        "data": final_data,
                        "total_records": record_count
                    }
                
                print("response constructed successfully", flush=True)
                
            except Exception as cache_error:
                print(f"error retrieving cached result: {cache_error}", flush=True)
                await manager.log_request_end(request_id, True, error=str(cache_error))
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
                print("logging request end...", flush=True)
                
                # Enhanced logging based on query type
                if response_data["query_type"] == "comparison":
                    await manager.log_request_end(request_id, False, "Comparison analysis completed successfully")
                elif response_data["query_type"] == "metric_analysis":
                    metrics_count = len(response_data.get("metrics_calculated", []))
                    await manager.log_request_end(request_id, False, f"Metric analysis completed - calculated {metrics_count} metrics")
                elif response_data["query_type"] == "schema_discovery":
                    await manager.log_request_end(request_id, False, "Schema discovery completed successfully")
                else:
                    record_count = response_data.get("count", 0)
                    await manager.log_request_end(request_id, False, f"Query completed successfully - returned {record_count} records")
                
                print("request logged successfully", flush=True)
            except Exception as log_error:
                print(f"logging error (non-fatal): {log_error}", flush=True)
                # Continue even if logging fails
            
            try:
                print(f"returning response_data keys: {list(response_data.keys())}", flush=True)
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
                    
                    response_data = {
                        "success": True,
                        "request_id": request_id,
                        "data": final_data,
                        "metadata": {"source": "fallback_last_step"}
                    }
                    
                    print("fallback successful", flush=True)
                    await manager.log_request_end(request_id, False)
                    return response_data
                else:
                    print("fallback failed: save_as missing or not in tool_result_refs", flush=True)
            except Exception as fallback_error:
                print(f"fallback error: {fallback_error}", flush=True)
        
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
