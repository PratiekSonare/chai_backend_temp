from typing import TypedDict, Annotated, Literal, Callable, Any
from langgraph.graph import StateGraph, END
import operator
import pandas as pd
import json
import asyncio

# Import tools and LLM providers
from tools import TOOL_REGISTRY, apply_filters
from llm_providers import planning_llm, filtering_llm, grouping_llm, insight_llm

# State schema
class AgentState(TypedDict):
    user_query: str
    summarized_query: str | None
    plan: dict | None
    tool_result_refs: dict[str, str]  # {save_as: cache_key} - references only
    tool_result_schemas: dict[str, dict]  # {save_as: schema_with_enums} - lightweight
    current_step_index: int
    filters: list[dict] | None
    final_result_ref: str | None  # Reference to final result
    error: str | None
    retry_count: int
    logger: Any  # Logger function for SSE
    request_id: str | None  # Request ID for logging
    # Comparison-specific fields
    comparison_mode: bool
    comparison_groups: list[dict] | None  # [{group_id, filters}, ...]
    comparison_param: str | None  # The field being compared (e.g., "state", "marketplace", "payment_mode")
    group_results: dict[str, str] | None  # {group_id: cache_ref}
    group_schemas: dict[str, dict] | None  # {group_id: schema}
    current_group_index: int
    aggregated_metrics: dict[str, dict] | None  # {group_id: metrics}
    comparison_results: dict | None
    insights: str | None
    # Metric analysis fields
    metric_results: dict | None  # Results from metric calculations
    metric_analysis: str | None  # LLM analysis of metrics
    metrics_calculated: list[str] | None

# Simple in-memory cache (use Redis for production)
RESULT_CACHE = {}


async def emit_step_event(
    state: AgentState,
    step_key: str,
    summary: str,
    status: str = "INFO",
    details: str | None = None,
    wait_ms: int | None = None,
) -> None:
    """Emit workflow event through request-scoped logger if available."""
    if not state.get("logger"):
        return

    payload = {
        "status": status,
        "details": details,
    }
    if wait_ms is not None:
        payload["wait_ms"] = wait_ms

    await state["logger"](
        state.get("request_id", "unknown"),
        step_key,
        summary,
        payload,
    )


async def gate_next_step(state: AgentState, next_step: str, wait_ms: int = 500) -> None:
    """Emit pending event before enforced debounce so frontend can show upcoming step."""
    await emit_step_event(
        state,
        "NEXT_STEP_PENDING",
        f"{next_step}",
        status="PENDING",
        wait_ms=wait_ms,
    )
    await asyncio.sleep(wait_ms / 1000)

def cache_result(data: dict | list, key: str) -> str:
    """Cache result and return reference key"""
    RESULT_CACHE[key] = data
    return key

def get_cached_result(key: str) -> dict | list:
    """Retrieve result from cache"""
    return RESULT_CACHE.get(key)

def extract_schema_with_enums(data: dict | list, sample_size: int = 100) -> dict:
    """Extract schema with categorical value examples for LLM understanding"""

    # Handle pandas DataFrame
    if hasattr(data, 'empty'):  # Check if it's a DataFrame-like object
        if data.empty:
            return {}
        
        # Convert DataFrame to schema format
        sample = data.head(sample_size) if len(data) > sample_size else data
        schema = {}
        
        for column in data.columns:
            if column != 'pickup_address':
                # Get unique values for categorical detection
                unique_values = set()
                try:
                    unique_vals = sample[column].dropna().unique()
                    for val in unique_vals:
                        if val is not None:
                            # Handle complex data types (lists, dicts) that can't be hashed
                            try:
                                # Try to convert to string for simple types
                                if isinstance(val, (list, dict, tuple)):
                                    # For complex types, use JSON representation
                                    unique_values.add(json.dumps(val, sort_keys=True, default=str))
                                else:
                                    unique_values.add(str(val))
                            except (TypeError, ValueError):
                                # Fallback for any other unhashable types
                                unique_values.add(str(type(val).__name__))
                except Exception as e:
                    # If unique() fails, just skip this column's enum detection
                    print(f"Warning: Could not extract unique values for column '{column}': {e}")
                    unique_values = set()
            
            # Sample value for example
            sample_value = None
            if not sample[column].empty:
                non_null_values = sample[column].dropna()
                if len(non_null_values) > 0:
                    sample_value = non_null_values.iloc[0]
                    # Convert complex types to JSON for display
                    if isinstance(sample_value, (list, dict, tuple)):
                        try:
                            sample_value = json.dumps(sample_value, default=str)
                        except:
                            sample_value = str(sample_value)
            
            # If a certain column field is [object Object]
            value_type = type(sample_value).__name__ if sample_value is not None else "object"

            # If few unique values (< 20), treat as categorical
            is_categorical = len(unique_values) < 20 and len(unique_values) > 0

            schema[column] = {
                "type": value_type,
                "example": sample_value,
                "is_categorical": is_categorical,
                "enum": sorted(list(unique_values)) if is_categorical else None
            }
        
        return schema
    
    # Handle empty data
    if not data:
        return {}
    
    # Handle scalar values (int, float, numpy scalars)
    if isinstance(data, (int, float)) or hasattr(data, 'dtype'):
        # Single scalar value - create a simple schema
        return {
            "value": {
                "type": type(data).__name__,
                "example": float(data) if hasattr(data, 'dtype') else data,
                "is_categorical": False,
                "enum": None
            }
        }
    
    # Handle list of records
    if isinstance(data, list):
        if len(data) == 0:
            return {}
        
        # Sample records to find all unique categorical values
        sample = data[:sample_size] if len(data) > sample_size else data
        schema = {}
        
        # Build schema from first record
        first_record = data[0]
        for key, value in first_record.items():
            value_type = type(value).__name__
            
            # Collect unique values for categorical detection
            unique_values = set()
            for record in sample:
                if key in record and record[key] is not None:
                    unique_values.add(str(record[key]))
            
            # If few unique values (< 20), treat as categorical
            is_categorical = len(unique_values) < 20 and len(unique_values) > 0
            
            schema[key] = {
                "type": value_type,
                "example": value,
                "is_categorical": is_categorical,
                "enum": sorted(list(unique_values)) if is_categorical else None
            }
        
        return schema
    
    # Handle single dict
    else:
        schema = {}
        for key, value in data.items():
            schema[key] = {
                "type": type(value).__name__,
                "example": value,
                "is_categorical": False,
                "enum": None
            }
        return schema

# Node functions
async def planning_node(state: AgentState) -> AgentState:
    """Planning LLM generates execution plan"""
    await emit_step_event(
        state,
        "PLANNING_START",
        "Planning execution...",
        status="START",
        details=f"Query: {state['user_query'][:120]}",
    )
    
    print(f"🧠 [PLANNING] Query: '{state['user_query'][:60]}...' | Error: {state.get('error', 'None')}", flush=True)
    try:
        await gate_next_step(state, "Calling planning model...", wait_ms=500)
        # Call your planning LLM here
        plan_response = planning_llm.invoke(state["user_query"])
        
        if not plan_response.get("success"):
            return {
                **state,
                "error": "Planning failed: " + plan_response.get("error", "Unknown error")
            }
        
        await emit_step_event(
            state,
            "PLANNING_COMPLETE",
            f"Plan created: {plan_response['plan'].get('query_type', 'unknown')} query",
            status="COMPLETE",
        )
        
        print(f"✅ [PLANNING] Plan created: {plan_response['plan'].get('query_type', 'unknown')} query", flush=True)
        
        return {
            **state,
            "summarized_query": plan_response["summarized_query"],
            "plan": plan_response["plan"],
            "current_step_index": 0,
            "error": None
        }
    
    except Exception as e:
        await emit_step_event(
            state,
            "PLANNING_ERROR",
            "Planning failed",
            status="ERROR",
            details=str(e),
        )
        
        print(f"❌ [PLANNING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Planning error: {str(e)}"}

async def execute_tool_node(state: AgentState) -> AgentState:
    """Execute the current step's tool
    
    TODO: PARALLEL EXECUTION ENHANCEMENT
    Current implementation executes steps sequentially. For better performance,
    steps with identical depends_on values could be executed in parallel.
    
    Example optimization opportunity:
    - step3a: get_total_revenue (depends_on: ["step2"])
    - step3b: get_aov (depends_on: ["step2"])  
    - step3c: get_order_count (depends_on: ["step2"])
    
    These three could run simultaneously since they all depend only on step2.
    """
    step_idx = state["current_step_index"]
    total_steps = len(state["plan"]["steps"]) if state["plan"] else 0
    
    await emit_step_event(
        state,
        "EXECUTE_STEP_START",
        f"Step {step_idx + 1}/{total_steps}",
        status="START",
        details=f"Retry: {state.get('retry_count', 0)}",
    )
    
    print(f"🔧 [EXECUTE_TOOL] Step {step_idx + 1}/{total_steps} | Retry: {state.get('retry_count', 0)}", flush=True)

    #print(state, flush=True)

    try:
        plan = state["plan"]
        step = plan["steps"][state["current_step_index"]]

        # Resolve dependencies from cache
        resolved_params = step["params"].copy()
        for dep_id in step["depends_on"]:
            dep_step = next(s for s in plan["steps"] if s["id"] == dep_id)
            ref_key = state["tool_result_refs"].get(dep_step["save_as"])
            if not ref_key:
                return {
                    **state,
                    "error": f"Dependency {dep_step['save_as']} not found"
                }
            # Retrieve actual data from cache for execution
            dep_data = get_cached_result(ref_key)
            
            # Replace placeholder values in params with actual dependency data
            for param_key, param_value in resolved_params.items():
                if isinstance(param_value, str) and param_value == f"{{{{{dep_step['save_as']}}}}}":
                    resolved_params[param_key] = dep_data
        
        # Emit next-step event, wait 500ms, then execute tool.
        await gate_next_step(state, f"Executing {step['tool']}...", wait_ms=500)
        await emit_step_event(
            state,
            "TOOL_EXECUTION_START",
            f"Executing {step['tool']}",
            status="START",
        )

        # Execute tool (map tool name to actual function)
        tool_function = TOOL_REGISTRY.get(step["tool"])
        if not tool_function:
            return {**state, "error": f"Tool {step['tool']} not found"}

        result = tool_function(**resolved_params)
        
        # Log successful execution
        if isinstance(result, list):
            await emit_step_event(
                state,
                "TOOL_EXECUTION_COMPLETE",
                f"Executed {step['tool']}",
                status="COMPLETE",
                details=f"Records: {len(result)}",
            )
            print(f"✅ [EXECUTE_TOOL] Tool executed: {step['tool']} | Records: {len(result)}", flush=True)
        elif isinstance(result, dict) and "available_fields" in result:
            await emit_step_event(
                state,
                "TOOL_EXECUTION_COMPLETE",
                f"Executed {step['tool']}",
                status="COMPLETE",
                details="Schema info retrieved",
            )
            print(f"✅ [EXECUTE_TOOL] Tool executed: {step['tool']} | Schema info retrieved", flush=True)
        else:
            await emit_step_event(
                state,
                "TOOL_EXECUTION_COMPLETE",
                f"Executed {step['tool']}",
                status="COMPLETE",
                details=f"Result type: {type(result).__name__}",
            )
            print(f"✅ [EXECUTE_TOOL] Tool executed: {step['tool']} | Result type: {type(result).__name__}", flush=True)
        
        # Cache the actual result data (not in state)
        result_key = cache_result(result, key=step["save_as"])
        
        # Extract schema with categorical values (goes in state - lightweight)
        schema = extract_schema_with_enums(result)
        
        # Update state with references and schemas only
        tool_result_refs = state["tool_result_refs"].copy()
        tool_result_refs[step["save_as"]] = result_key
        
        tool_result_schemas = state["tool_result_schemas"].copy()
        tool_result_schemas[step["save_as"]] = schema
        
        return {
            **state,
            "tool_result_refs": tool_result_refs,
            "tool_result_schemas": tool_result_schemas,
            "current_step_index": state["current_step_index"] + 1,
            "error": None,
            "retry_count": 0
        }
        
    except Exception as e:
        # Retry logic
        await emit_step_event(
            state,
            "EXECUTE_STEP_ERROR",
            "Tool execution error",
            status="ERROR",
            details=str(e),
        )
        if state["retry_count"] < 2:
            return {
                **state,
                "retry_count": state["retry_count"] + 1,
                "error": f"Tool execution retry {state['retry_count'] + 1}: {str(e)}"
            }
        
        print(f"❌ [EXECUTE_TOOL] Error: {str(e)}", flush=True)
        return {**state, "error": f"Tool execution failed: {str(e)}"}
    
async def filtering_node(state: AgentState) -> AgentState:
    """Filtering LLM generates filter parameters using schema with categorical values"""
    print(f"🔍 [FILTERING] Extracting filters from query", flush=True)
    try:
        await emit_step_event(
            state,
            "FILTERING_START",
            "Preparing filters...",
            status="START",
        )

        plan = state["plan"]
        # Get schema (not data!) from the last step
        last_step = plan["steps"][-1]
        schema = state["tool_result_schemas"][last_step["save_as"]]

        await gate_next_step(state, "Calling filtering model...", wait_ms=500)
        
        # Call filtering LLM with schema including enum values
        # LLM can now learn: "prepaid" -> "PrePaid" from enum list
        filter_response = filtering_llm.invoke({
            "query": state["user_query"],
            "schema": schema,  # Includes enum values for categorical fields
            "manipulation_type": plan["manipulation"]["type"]
        })

        print(f"✅ [FILTERING] Filters extracted: {len(filter_response.get('filters', []))} filter(s)", flush=True)
        await emit_step_event(
            state,
            "FILTERING_COMPLETE",
            f"Extracted {len(filter_response.get('filters', []))} filter(s)",
            status="COMPLETE",
        )
        

        # print(state, flush=True)

        return {
            **state,
            "filters": filter_response.get("filters", []),
            "error": None
        }
        
    except Exception as e:
        await emit_step_event(
            state,
            "FILTERING_ERROR",
            "Filtering failed",
            status="ERROR",
            details=str(e),
        )
        print(f"❌ [FILTERING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Filtering error: {str(e)}"}

def apply_filters_node(state: AgentState) -> AgentState:
    """Apply filters to the result"""
    print(f"📊 [APPLY_FILTERS] Applying {len(state.get('filters', []))} filter(s)", flush=True)
    
    try:
        plan = state["plan"]
        last_step = plan["steps"][-1]
        ref_key = state["tool_result_refs"][last_step["save_as"]]
        
        # Retrieve actual data from cache
        result_data = get_cached_result(ref_key)
        
        filtered_data = apply_filters(result_data, state["filters"])
        
        # Cache filtered result and store reference
        final_result = {"data": filtered_data, "plan": state["plan"]}
        final_ref = cache_result(final_result, key="final_result")
        
        print(f"✅ [APPLY_FILTERS] Filtered: {len(result_data) if isinstance(result_data, list) else 1} → {len(filtered_data) if isinstance(filtered_data, list) else 1} records", flush=True)
        
        return {
            **state,
            "final_result_ref": final_ref,
            "error": None
        }
    except Exception as e:
        print(f"❌ [APPLY_FILTERS] Error: {str(e)}", flush=True)
        return {**state, "error": f"Filter application error: {str(e)}"}

def no_filter_node(state: AgentState) -> AgentState:
    """Direct output when no filtering needed"""
    print(f"➡️  [NO_FILTER] Skipping filters, returning raw results", flush=True)
    
    plan = state["plan"]
    last_step = plan["steps"][-1]
    ref_key = state["tool_result_refs"][last_step["save_as"]]
    
    # Retrieve data and wrap with plan
    data = get_cached_result(ref_key)
    final_result = {"data": data, "plan": state["plan"]}
    final_ref = cache_result(final_result, key="final_result")
    
    return {
        **state,
        "final_result_ref": final_ref,
        "error": None
    }

def metric_processing_node(state: AgentState) -> AgentState:
    """Execute metric calculations based on the plan"""
    print(f"📊 [METRIC_PROCESSING] Executing metric calculations", flush=True)
    
    try:
        # Get the final data after all tool executions
        plan = state["plan"]
        final_step = plan["steps"][-1]
        
        if state.get("final_result_ref"):
            # Data already processed (filtered)
            data_ref = state["final_result_ref"]
        else:
            # Use last step result
            data_ref = state["tool_result_refs"][final_step["save_as"]]
        
        # Get the actual data
        result_data = get_cached_result(data_ref)
        
        # Execute all metric calculations mentioned in the plan
        metric_results = {}
        
        # Extract which metrics were requested based on the plan steps
        metric_tools = [
            step for step in plan["steps"]
            if (
                (step["tool"].startswith("get_") and step["tool"] not in ["get_all_orders", "get_schema_info"])
                or step["tool"] == "execute_custom_calculation"
            )
        ]
        
        # print("state[tool_result_refs]: ", state["tool_result_refs"])
        # print("metric_tools: ", metric_tools)

        for metric_step in metric_tools:
            tool_name = metric_step["tool"]
            save_as = metric_step["save_as"]
            # print("tool_name: ", tool_name)
            # print("save_as: ", save_as)
            if save_as in state["tool_result_refs"]:
                # Result already calculated
                cached_metric = get_cached_result(state["tool_result_refs"][save_as])

                if tool_name == "execute_custom_calculation" and isinstance(cached_metric, dict):
                    custom_metric_name = metric_step.get("params", {}).get("metric_name") or save_as
                    if cached_metric.get("success") and custom_metric_name in cached_metric:
                        metric_results[custom_metric_name] = cached_metric.get(custom_metric_name)
                    else:
                        metric_results[custom_metric_name] = cached_metric
                    print(f"Found cached custom metric result for {custom_metric_name}: {metric_results[custom_metric_name]}")
                else:
                    metric_results[tool_name] = cached_metric
                    print(f"Found cached metric result for {tool_name}: {metric_results[tool_name]}")
        
        # print("==========>>>>> plan", flush=True)
        # print(state['plan'], flush=True)

        # If no specific metrics in plan, calculate common metrics
        if not metric_results:
            # Use the common metrics tool
            metric_results = TOOL_REGISTRY["get_common_metrics"](result_data)
        
        # Cache metric results
        if metric_results:
            metric_ref = cache_result(metric_results, key="metric_results")
            normalized_metrics = {"overall": metric_results}
            metrics_calculated = list(metric_results.keys()) if isinstance(metric_results, dict) else []
            print(f"✅ [METRIC_PROCESSING] Calculated {len(metric_results)} metrics", flush=True)
            
            # print("metric_results: ", metric_results)
            # print("metric_ref: ", metric_ref)

            return {
                **state,
                "metric_results": metric_results,
                "aggregated_metrics": normalized_metrics,
                "metrics_calculated": metrics_calculated,
                "final_result_ref": metric_ref,
                "error": None
            }
        else:
            return {
                **state,
                "error": "No metrics could be calculated from the data"
            }
        
    except Exception as e:
        print(f"❌ [METRIC_PROCESSING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Metric processing error: {str(e)}"}

async def grouping_node(state: AgentState) -> AgentState:
    """Identify comparison groups and create parallel execution branches"""
    print(f"🔀 [GROUPING] Identifying comparison groups", flush=True)
    
    try:
        await emit_step_event(
            state,
            "GROUPING_START",
            "Preparing comparison groups...",
            status="START",
        )

        await gate_next_step(state, "Calling grouping model...", wait_ms=500)
        # Call grouping LLM to extract comparison dimensions
        grouping_response = grouping_llm.invoke({
            "query": state["user_query"],
            "plan": state["plan"]
        })
        
        groups = grouping_response.get("groups", [])
        
        # Extract comparison parameter (the field being compared)
        comparison_param = None
        if groups and len(groups) > 0:
            # Get the field being compared from the first group's filters
            first_group_filters = groups[0].get("filters", {})
            if first_group_filters:
                # Take the first (and typically only) filter field as the comparison dimension
                comparison_param = list(first_group_filters.keys())[0]
        
        print(f"✅ [GROUPING] Found {len(groups)} groups comparing by '{comparison_param}': {[g['group_id'] for g in groups]}", flush=True)
        await emit_step_event(
            state,
            "GROUPING_COMPLETE",
            f"Found {len(groups)} group(s)",
            status="COMPLETE",
            details=f"comparison_param={comparison_param}",
        )
        
        return {
            **state,
            "comparison_groups": grouping_response["groups"],
            "comparison_param": comparison_param,
            "group_results": {},
            "group_schemas": {},
            "current_group_index": 0,
            "error": None
        }
    except Exception as e:
        await emit_step_event(
            state,
            "GROUPING_ERROR",
            "Grouping failed",
            status="ERROR",
            details=str(e),
        )
        print(f"❌ [GROUPING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Grouping error: {str(e)}"}

async def parallel_fetch_node(state: AgentState) -> AgentState:
    """Execute data fetches for current comparison group"""
    current_idx = state["current_group_index"]
    total_groups = len(state["comparison_groups"]) if state["comparison_groups"] else 0
    group_id = state["comparison_groups"][current_idx]["group_id"] if state["comparison_groups"] else "unknown"
    print(f"📥 [PARALLEL_FETCH] Fetching group {current_idx + 1}/{total_groups}: {group_id}", flush=True)
    
    try:
        await emit_step_event(
            state,
            "PARALLEL_FETCH_START",
            f"Fetching {group_id}",
            status="START",
            details=f"Group {current_idx + 1}/{total_groups}",
        )

        plan = state["plan"]
        current_group = state["comparison_groups"][state["current_group_index"]]
        
        # IMPORTANT: Only pass parameters that the tool function accepts
        # For get_all_orders(), only start_date and end_date are valid parameters
        # Group-specific filters (e.g., payment_mode, marketplace) are applied AFTER fetching
        tool_function = TOOL_REGISTRY.get(plan["tool"])
        if not tool_function:
            return {**state, "error": f"Tool {plan['tool']} not found"}

        await gate_next_step(state, f"Executing {plan['tool']} for {group_id}...", wait_ms=500)
        
        # Fetch data using only base parameters (date range)
        result = tool_function(**plan["base_params"])
        
        # Apply group-specific filters to the fetched data
        # This is where payment_mode, marketplace, etc. filtering happens
        if current_group.get("filters"):
            from tools import apply_filters
            # Convert dict filters to list format expected by apply_filters
            filter_list = [
                {"field": field, "operator": "eq", "value": value}
                for field, value in current_group["filters"].items()
            ]
            result = apply_filters(result, filter_list)
        
        # Cache result
        result_key = cache_result(result, key=f"group_{current_group['group_id']}")
        schema = extract_schema_with_enums(result)
        
        print(f"✅ [PARALLEL_FETCH] Fetched {len(result) if isinstance(result, list) else 1} records for {current_group['group_id']}", flush=True)
        await emit_step_event(
            state,
            "PARALLEL_FETCH_COMPLETE",
            f"Fetched data for {current_group['group_id']}",
            status="COMPLETE",
            details=f"Records: {len(result) if isinstance(result, list) else 1}",
        )
        
        group_results = state["group_results"].copy()
        group_results[current_group["group_id"]] = result_key
        
        group_schemas = state["group_schemas"].copy()
        group_schemas[current_group["group_id"]] = schema
        
        return {
            **state,
            "group_results": group_results,
            "group_schemas": group_schemas,
            "current_group_index": state["current_group_index"] + 1,
            "error": None
        }
        
    except Exception as e:
        await emit_step_event(
            state,
            "PARALLEL_FETCH_ERROR",
            "Parallel fetch failed",
            status="ERROR",
            details=str(e),
        )
        print(f"❌ [PARALLEL_FETCH] Error: {str(e)}", flush=True)
        return {**state, "error": f"Parallel fetch error: {str(e)}"}

def aggregation_node(state: AgentState) -> AgentState:
    """Compute metrics for each comparison group"""
    num_groups = len(state.get("group_results", {}))
    print(f"📈 [AGGREGATION] Computing metrics for {num_groups} group(s)", flush=True)
    
    try:
        aggregated_metrics = {}
        
        for group_id, result_ref in state["group_results"].items():
            data = get_cached_result(result_ref)
            
            # Helper function to safely convert to float
            def safe_float(value, default=0.0):
                """Convert value to float, handling strings and None"""
                try:
                    if value is None or value == "":
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Calculate metrics with type safety
            total_revenue = 0.0
            if isinstance(data, list):
                for record in data:
                    total_revenue += safe_float(record.get("total_amount", 0))
            else:
                total_revenue = safe_float(data.get("total_amount", 0))
            
            metrics = {
                "count": len(data) if isinstance(data, list) else 1,
                "total_revenue": total_revenue,
                "avg_order_value": 0,
                "payment_mode_distribution": {},
                "order_status_distribution": {},
                "top_cities": {},
                "top_states": {}
            }
            
            if metrics["count"] > 0:
                metrics["avg_order_value"] = metrics["total_revenue"] / metrics["count"]
            
            # Distribution calculations
            if isinstance(data, list):
                for record in data:
                    # Payment mode
                    pm = record.get("payment_mode", "Unknown")
                    metrics["payment_mode_distribution"][pm] = metrics["payment_mode_distribution"].get(pm, 0) + 1
                    
                    # Order status
                    status = record.get("order_status", "Unknown")
                    metrics["order_status_distribution"][status] = metrics["order_status_distribution"].get(status, 0) + 1
                    
                    # Cities
                    city = record.get("city", "Unknown")
                    metrics["top_cities"][city] = metrics["top_cities"].get(city, 0) + 1
                    
                    # States
                    billing_state = record.get("billing_state", "Unknown")
                    metrics["top_states"][billing_state] = metrics["top_states"].get(billing_state, 0) + 1
            
            aggregated_metrics[group_id] = metrics
        
        print(f"✅ [AGGREGATION] Metrics computed for: {list(aggregated_metrics.keys())}", flush=True)
        
        return {
            **state,
            "aggregated_metrics": aggregated_metrics,
            "error": None
        }
        
    except Exception as e:
        print(f"❌ [AGGREGATION] Error: {str(e)}", flush=True)
        return {**state, "error": f"Aggregation error: {str(e)}"}

def comparison_node(state: AgentState) -> AgentState:
    """Perform comparison logic between groups"""
    print(f"⚖️  [COMPARISON] Comparing groups", flush=True)
    
    try:
        metrics = state.get("aggregated_metrics")
        if not metrics:
            return {**state, "error": "No aggregated metrics available for comparison"}
        
        group_ids = list(metrics.keys())
        num_groups = len(group_ids)
        
        if num_groups < 2:
            return {**state, "error": "Need at least 2 groups for comparison"}
        
        # Two-group pairwise comparison
        if num_groups == 2:
            group_a, group_b = group_ids[0], group_ids[1]
            metrics_a = metrics[group_a]
            metrics_b = metrics[group_b]
            
            comparison_results = {
                "comparison_type": "pairwise",
                "comparison_mode": True,
                "comparison_param": state.get("comparison_param"),
                "groups": {"a": group_a, "b": group_b},
                "order_count": {
                    "a": metrics_a["count"],
                    "b": metrics_b["count"],
                    "diff": metrics_b["count"] - metrics_a["count"],
                    "diff_pct": ((metrics_b["count"] - metrics_a["count"]) / metrics_a["count"] * 100) if metrics_a["count"] > 0 else 0
                },
                "total_revenue": {
                    "a": metrics_a["total_revenue"],
                    "b": metrics_b["total_revenue"],
                    "diff": metrics_b["total_revenue"] - metrics_a["total_revenue"],
                    "diff_pct": ((metrics_b["total_revenue"] - metrics_a["total_revenue"]) / metrics_a["total_revenue"] * 100) if metrics_a["total_revenue"] > 0 else 0
                },
                "avg_order_value": {
                    "a": metrics_a["avg_order_value"],
                    "b": metrics_b["avg_order_value"],
                    "diff": metrics_b["avg_order_value"] - metrics_a["avg_order_value"],
                    "diff_pct": ((metrics_b["avg_order_value"] - metrics_a["avg_order_value"]) / metrics_a["avg_order_value"] * 100) if metrics_a["avg_order_value"] > 0 else 0
                },
                "winner_by_volume": group_a if metrics_a["count"] > metrics_b["count"] else group_b,
                "winner_by_revenue": group_a if metrics_a["total_revenue"] > metrics_b["total_revenue"] else group_b,
                "winner_by_avg_value": group_a if metrics_a["avg_order_value"] > metrics_b["avg_order_value"] else group_b
            }
            
            print(f"✅ [COMPARISON] {group_a} vs {group_b} | Winner by volume: {comparison_results['winner_by_volume']}", flush=True)
        
        # Multi-group comparison (N > 2)
        else:
            # Use first group as baseline for comparison
            baseline_id = group_ids[0]
            baseline_metrics = metrics[baseline_id]
            
            # Build comparison summary for all groups
            group_summaries = {}
            for group_id in group_ids:
                group_metrics = metrics[group_id]
                group_summaries[group_id] = {
                    "order_count": group_metrics["count"],
                    "total_revenue": group_metrics["total_revenue"],
                    "avg_order_value": group_metrics["avg_order_value"],
                    "payment_mode_distribution": group_metrics["payment_mode_distribution"],
                    "order_status_distribution": group_metrics["order_status_distribution"],
                    "top_cities": group_metrics["top_cities"],
                    "top_states": group_metrics["top_states"]
                }
            
            # Compare each group to baseline
            comparisons_to_baseline = {}
            for group_id in group_ids[1:]:  # Skip baseline itself
                group_metrics = metrics[group_id]
                comparisons_to_baseline[group_id] = {
                    "order_count_diff": group_metrics["count"] - baseline_metrics["count"],
                    "order_count_diff_pct": ((group_metrics["count"] - baseline_metrics["count"]) / baseline_metrics["count"] * 100) if baseline_metrics["count"] > 0 else 0,
                    "revenue_diff": group_metrics["total_revenue"] - baseline_metrics["total_revenue"],
                    "revenue_diff_pct": ((group_metrics["total_revenue"] - baseline_metrics["total_revenue"]) / baseline_metrics["total_revenue"] * 100) if baseline_metrics["total_revenue"] > 0 else 0,
                    "avg_order_value_diff": group_metrics["avg_order_value"] - baseline_metrics["avg_order_value"],
                    "avg_order_value_diff_pct": ((group_metrics["avg_order_value"] - baseline_metrics["avg_order_value"]) / baseline_metrics["avg_order_value"] * 100) if baseline_metrics["avg_order_value"] > 0 else 0
                }
            
            # Identify overall winners across all groups
            winner_by_volume = max(group_ids, key=lambda gid: metrics[gid]["count"])
            winner_by_revenue = max(group_ids, key=lambda gid: metrics[gid]["total_revenue"])
            winner_by_avg_value = max(group_ids, key=lambda gid: metrics[gid]["avg_order_value"])
            
            comparison_results = {
                "comparison_type": "multi_group",
                "comparison_mode": True,
                "comparison_param": state.get("comparison_param"),
                "num_groups": num_groups,
                "groups": group_ids,
                "baseline": baseline_id,
                "group_summaries": group_summaries,
                "comparisons_to_baseline": comparisons_to_baseline,
                "overall_winners": {
                    "by_volume": winner_by_volume,
                    "by_revenue": winner_by_revenue,
                    "by_avg_value": winner_by_avg_value
                }
            }
            
            print(f"✅ [COMPARISON] {num_groups} groups compared | Winner by volume: {winner_by_volume}, by revenue: {winner_by_revenue}", flush=True)
        
        return {
            **state,
            "comparison_results": comparison_results,
            "error": None
        }
        
    except Exception as e:
        print(f"❌ [COMPARISON] Error: {str(e)}", flush=True)
        return {**state, "error": f"Comparison error: {str(e)}"}

async def insight_generation_node(state: AgentState) -> AgentState:
    """Generate natural language insights for comparison and metric analysis."""
    print(f"💡 [INSIGHTS] Generating natural language summary", flush=True)
    
    try:
        await emit_step_event(
            state,
            "INSIGHT_START",
            "Generating insights...",
            status="START",
        )

        # Call insight LLM to generate natural language summary
        # Extract date range from plan base_params or first step params
        date_range = {}
        if state["plan"].get("base_params"):
            base_params = state["plan"]["base_params"]
            if "start_date" in base_params and "end_date" in base_params:
                date_range = {
                    "start_date": base_params["start_date"],
                    "end_date": base_params["end_date"]
                }
        
        # Fallback: extract from first step if base_params not available
        if not date_range and state["plan"].get("steps"):
            first_step = state["plan"]["steps"][0]
            step_params = first_step.get("params", {})
            if "start_date" in step_params and "end_date" in step_params:
                date_range = {
                    "start_date": step_params["start_date"],
                    "end_date": step_params["end_date"]
                }
        
        query_type = state["plan"].get("query_type", "standard")
        is_metric_query = query_type in ["metric_analysis", "custom_metric_generation"]

        normalized_metrics = state.get("aggregated_metrics")
        if not normalized_metrics and is_metric_query and state.get("metric_results"):
            normalized_metrics = {"overall": state["metric_results"]}

        await gate_next_step(state, "Calling insight model...", wait_ms=500)
        insight_response = insight_llm.invoke({
            "query": state["user_query"],
            "metrics": normalized_metrics or {},
            "comparison": state["comparison_results"] if query_type == "comparison" else None,
            "date_range": date_range,
            "analysis_mode": query_type,
            "raw_metrics": state.get("metric_results", {})
        })
        
        print(f"✅ [INSIGHTS] Generated {len(insight_response['insights'])} chars of insights", flush=True)
        await emit_step_event(
            state,
            "INSIGHT_COMPLETE",
            "Insights generated",
            status="COMPLETE",
            details=f"Chars: {len(insight_response.get('insights', ''))}",
        )

        analysis_text = insight_response.get("analysis", insight_response.get("insights", ""))
        metrics_calculated = state.get("metrics_calculated")
        if not metrics_calculated and isinstance(state.get("metric_results"), dict):
            metrics_calculated = list(state["metric_results"].keys())

        # Unified internal payload used by route adapters.
        final_payload = {
            "query": state["user_query"],
            "analysis_mode": query_type,
            "insights": insight_response.get("insights", ""),
            "analysis": analysis_text,
            "metrics": state.get("metric_results"),
            "metrics_calculated": metrics_calculated or [],
            "comparison_data": state.get("comparison_results") if query_type == "comparison" else None,
            "detailed_metrics": normalized_metrics,
            "plan": state["plan"]
        }

        final_ref = cache_result(final_payload, key="final_insight_result")
        
        return {
            **state,
            "insights": insight_response.get("insights", ""),
            "metric_analysis": analysis_text if is_metric_query else state.get("metric_analysis"),
            "final_result_ref": final_ref,
            "error": None
        }
        
    except Exception as e:
        await emit_step_event(
            state,
            "INSIGHT_ERROR",
            "Insight generation failed",
            status="ERROR",
            details=str(e),
        )
        print(f"❌ [INSIGHTS] Error: {str(e)}", flush=True)
        return {**state, "error": f"Insight generation error: {str(e)}"}

# Conditional edges
def should_continue_execution(state: AgentState) -> Literal["execute_tool", "check_manipulation", "error"]:
    """Check if more tools need execution"""
    if state["error"]:
        return "error"
    
    if state["current_step_index"] < len(state["plan"]["steps"]):
        return "execute_tool"
    
    # Schema discovery queries go directly to output (no filtering needed)
    if state["plan"].get("query_type") == "schema_discovery":
        return "check_manipulation"  # Will route to no_filter
    
    return "check_manipulation"

def needs_manipulation(state: AgentState) -> Literal["filtering", "no_filter", "metric_processing", "error"]:
    """Check if filtering/manipulation is required"""
    if state["error"]:
        return "error"
    
    # Schema discovery queries never need filtering
    if state["plan"].get("query_type") == "schema_discovery":
        return "no_filter"
    
    # Metric analysis and custom metric generation queries go directly to metric processing
    if state["plan"].get("query_type") in ["metric_analysis", "custom_metric_generation"]:
        return "metric_processing"
    
    if state["plan"]["manipulation"]["required"]:
        return "filtering"
    
    return "no_filter"

def check_error(state: AgentState) -> Literal["replan", "end"]:
    """Decide whether to replan or abort"""
    # If planning failed or too many retries, abort
    if "Planning" in state["error"] or state["retry_count"] >= 2:
        return "end"
    
    # Otherwise, attempt to replan
    return "replan"

def is_comparison_query(state: AgentState) -> Literal["grouping", "execute_tool", "error"]:
    """Check if query requires comparison or is schema discovery"""
    if state["error"]:
        return "error"
    
    query_type = state["plan"].get("query_type")
    
    # Schema discovery and metric queries go through standard execute_tool flow
    # They just return schema info or processed metrics instead of data
    if query_type in ["schema_discovery", "metric_analysis", "custom_metric_generation"]:
        return "execute_tool"
    
    if query_type == "comparison":
        return "grouping"
    
    return "execute_tool"

def all_groups_fetched(state: AgentState) -> Literal["parallel_fetch", "aggregation", "error"]:
    """Check if all comparison groups have data"""
    if state["error"]:
        return "error"
    
    expected = len(state["comparison_groups"])
    actual = state["current_group_index"]
    
    if actual < expected:
        return "parallel_fetch"  # Continue fetching
    
    return "aggregation"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes - Standard flow
workflow.add_node("planning", planning_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("filtering", filtering_node)
workflow.add_node("apply_filters", apply_filters_node)
workflow.add_node("no_filter", no_filter_node)

# Add nodes - Metric analysis flow
workflow.add_node("metric_processing", metric_processing_node)

# Add nodes - Comparison flow
workflow.add_node("grouping", grouping_node)
workflow.add_node("parallel_fetch", parallel_fetch_node)
workflow.add_node("aggregation", aggregation_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("insight_generation", insight_generation_node)

# Set entry point
workflow.set_entry_point("planning")

# Add edges - Planning branches to comparison or standard flow
workflow.add_conditional_edges(
    "planning",
    is_comparison_query,
    {
        "grouping": "grouping",
        "execute_tool": "execute_tool",
        "error": "error_handler"
    }
)

# Standard flow
workflow.add_conditional_edges(
    "execute_tool",
    should_continue_execution,
    {
        "execute_tool": "execute_tool",  # Loop for multiple steps
        "check_manipulation": "check_manipulation",
        "error": "error_handler"
    }
)

# Add a routing node for manipulation check
workflow.add_node("check_manipulation", lambda s: s)
workflow.add_conditional_edges(
    "check_manipulation",
    needs_manipulation,
    {
        "filtering": "filtering",
        "no_filter": "no_filter", 
        "metric_processing": "metric_processing",
        "error": "error_handler"
    }
)

workflow.add_edge("filtering", "apply_filters")
workflow.add_edge("apply_filters", END)
workflow.add_edge("no_filter", END)

# Metric analysis flow
workflow.add_edge("metric_processing", "insight_generation")

# Comparison flow
workflow.add_edge("grouping", "parallel_fetch")
workflow.add_conditional_edges(
    "parallel_fetch",
    all_groups_fetched,
    {
        "parallel_fetch": "parallel_fetch",  # Loop until all groups done
        "aggregation": "aggregation",
        "error": "error_handler"
    }
)
workflow.add_edge("aggregation", "comparison")
workflow.add_edge("comparison", "insight_generation")
workflow.add_edge("insight_generation", END)

# Error handling
workflow.add_node("error_handler", lambda s: s)
workflow.add_conditional_edges(
    "error_handler",
    check_error,
    {
        "replan": "planning",
        "end": END
    }
)

# Compile
app = workflow.compile()

# Usage Examples - Only run when executed directly
if __name__ == '__main__':
    # Example 1: Standard query
    initial_state_standard = AgentState(
        user_query="Show me orders from last 5 days with payment mode prepaid",
        summarized_query=None,
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
        comparison_param=None,
        group_results=None,
        group_schemas=None,
        current_group_index=0,
        aggregated_metrics=None,
        comparison_results=None,
        insights=None,
        metric_results=None,
        metric_analysis=None,
        metrics_calculated=None
    )

    # Example 2: Comparison query
    initial_state_comparison = AgentState(
        user_query="Compare orders between Shopify13 & Flipkart from the last 10 days",
        summarized_query=None,
        plan=None,
        tool_result_refs={},
        tool_result_schemas={},
        current_step_index=0,
        filters=None,
        final_result_ref=None,
        error=None,
        retry_count=0,
        comparison_mode=True,
        comparison_groups=None,
        comparison_param=None,
        group_results=None,
        group_schemas=None,
        current_group_index=0,
        aggregated_metrics=None,
        comparison_results=None,
        insights=None,
        metric_results=None,
        metric_analysis=None,
        metrics_calculated=None
    )

    # Example 3: Metric analysis query
    initial_state_metric = AgentState(
        user_query="What is the average order value and revenue from last 7 days?",
        summarized_query=None,
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
        comparison_param=None,
        group_results=None,
        group_schemas=None,
        current_group_index=0,
        aggregated_metrics=None,
        comparison_results=None,
        insights=None,
        metric_results=None,
        metric_analysis=None,
        metrics_calculated=None
    )

    # Run standard query
    result_standard = app.invoke(initial_state_standard)

    if result_standard["final_result_ref"]:
        final_data = get_cached_result(result_standard["final_result_ref"])
        print(f"Plan: {final_data['plan']}")
        data = final_data['data']
        if isinstance(data, list):
            print(f"Found {len(data)} records")
        else:
            print(f"Result: {data}")
    else:
        print(f"Error: {result_standard['error']}")

    # Run comparison query
    result_comparison = app.invoke(initial_state_comparison)

    if result_comparison["final_result_ref"]:
        final_data = get_cached_result(result_comparison["final_result_ref"])
        print(f"Plan: {final_data['plan']}")
        print(f"\nComparison Insights:\n{final_data['insights']}")
        print(f"\nComparison Data: {final_data['comparison_data']}")
    else:
        print(f"Error: {result_comparison['error']}")

    # Run metric analysis query
    print(f"\n{'='*50}")
    print("METRIC ANALYSIS EXAMPLE")
    print(f"{'='*50}")
    
    result_metric = app.invoke(initial_state_metric)

    if result_metric["final_result_ref"]:
        final_data = get_cached_result(result_metric["final_result_ref"])
        print(f"Plan: {final_data['plan']}")
        print(f"\nQuery: {final_data['query']}")
        print(f"\nMetrics: {final_data.get('metrics')}")
        print(f"\nAnalysis:\n{final_data.get('analysis', final_data.get('insights', ''))}")
    else:
        print(f"Error: {result_metric['error']}")