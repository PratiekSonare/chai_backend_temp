from typing import TypedDict, Annotated, Literal, Callable, Any, Optional, Union
from langgraph.graph import StateGraph, END
import operator
import pandas as pd
import json
import asyncio
import uuid

# Import tools and LLM providers
from tools import apply_filters, get_gross_profit, get_margin, get_markup, get_cost_price, get_selling_price, get_cost_to_price_ratio
from tools import ORDERS_TOOL_REGISTRY, PROFIT_TOOL_REGISTRY, PAYMENT_CYCLE_TOOL_REGISTRY
from llm_providers import query_categorization_llm, planning_llm, filtering_llm, grouping_llm, insight_llm, custom_calculation_llm

# State schema
class AgentState(TypedDict):
    user_query: str
    data_source: str | None
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
    cancel_checker: Any  # Callable(request_id) -> bool
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
RESULT_CACHE: dict[str, Any] = {}

# Default tool registry (orders by default). Will be switched after categorization.
TOOL_REGISTRY = ORDERS_TOOL_REGISTRY

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


async def gate_next_step(state: AgentState, next_step: str, wait_ms: int = 0) -> None:
    """Emit pending event before enforced debounce so frontend can show upcoming step."""
    await emit_step_event(
        state,
        "NEXT_STEP_PENDING",
        f"{next_step}",
        status="PENDING",
        wait_ms=wait_ms,
    )
    await asyncio.sleep(wait_ms / 1000)


REQUEST_CANCELLED_ERROR = "Request cancelled by client"


def _is_request_cancelled(state: AgentState) -> bool:
    checker = state.get("cancel_checker")
    if not checker:
        return False
    try:
        return bool(checker(state.get("request_id")))
    except Exception:
        return False


async def _cancelled_state(state: AgentState, step_key: str, summary: str) -> AgentState:
    await emit_step_event(
        state,
        step_key,
        summary,
        status="CANCELLED",
    )
    return {
        **state,
        "error": REQUEST_CANCELLED_ERROR,
    }


def cache_result(data: dict | list, key: str) -> str:
    """Cache result and return reference key"""
    cache_key = f"{key}_{uuid.uuid4().hex[:8]}"
    RESULT_CACHE[cache_key] = data
    return cache_key


def get_cached_result(key: str) -> dict | list:
    """Retrieve result from cache"""
    return RESULT_CACHE.get(key)

def extract_schema_with_enums(
    data: Union[pd.DataFrame, dict, list], 
    sample_size: int = 100,
    exclude_columns: list = None
) -> dict:
    """
    Extract schema with categorical enums for LLM understanding.
    
    Supports:
    - pandas DataFrame
    - dict (single record or metadata)
    - list (list of dicts or simple list)
    """
    if exclude_columns is None:
        exclude_columns = ['pickup_address']
    
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return _extract_dataframe_schema(data, sample_size, exclude_columns)
    
    # Handle plain dict
    elif isinstance(data, dict):
        return _extract_dict_schema(data)
    
    # Handle list
    elif isinstance(data, list):
        return _extract_list_schema(data, sample_size)
    
    else:
        return {"type": type(data).__name__, "note": "Unsupported input type"}


def _extract_dataframe_schema(
    df: pd.DataFrame, 
    sample_size: int = 100,
    exclude_columns: list = None
) -> dict:
    """Extract detailed schema from a pandas DataFrame"""
    if df.empty:
        return {"note": "DataFrame is empty"}
    
    sample = df.head(sample_size) if len(df) > sample_size else df
    schema = {}
    
    for column in df.columns:
        if column in exclude_columns:
            continue
            
        # --- Unique values for categorical detection ---
        unique_values: set = set()
        try:
            # Use value_counts for efficiency and accuracy
            value_counts = df[column].value_counts(dropna=True)
            # Take top unique values (limit for performance + prompt size)
            for val in value_counts.index[:50]:
                if pd.isna(val):
                    continue
                try:
                    if isinstance(val, (list, dict, tuple)):
                        unique_values.add(json.dumps(val, sort_keys=True, default=str))
                    else:
                        unique_values.add(str(val))
                except (TypeError, ValueError):
                    unique_values.add(f"<unhashable_{type(val).__name__}>")
        except Exception as e:
            print(f"Warning: Could not extract uniques for '{column}': {e}")
        
        # --- Sample value ---
        sample_value = None
        non_null = df[column].dropna()
        if len(non_null) > 0:
            raw_value = non_null.iloc[0]
            if isinstance(raw_value, (list, dict, tuple)):
                try:
                    sample_value = json.dumps(raw_value, default=str)
                except:
                    sample_value = str(raw_value)
            else:
                sample_value = raw_value
        
        # Determine basic type
        value_type = type(sample_value).__name__ if sample_value is not None else "object"
        
        # Categorical detection: few unique values
        is_categorical = 0 < len(unique_values) <= 20
        
        schema[column] = {
            "type": value_type,
            "example": sample_value,
            "is_categorical": is_categorical,
            "enum": sorted(list(unique_values)) if is_categorical else None,
            "unique_count": len(unique_values) if is_categorical else None
        }
    
    return schema


def _extract_dict_schema(data: dict) -> dict:
    """Handle single dictionary (one record)"""
    schema = {}
    for key, value in data.items():
        if isinstance(value, (list, dict, tuple)):
            try:
                example = json.dumps(value, default=str)
            except:
                example = str(value)
        else:
            example = value
            
        schema[key] = {
            "type": type(value).__name__,
            "example": example,
            "is_categorical": False,
            "enum": None
        }
    return schema


def _extract_list_schema(data: list, sample_size: int = 100) -> dict:
    """Handle list input"""
    if not data:
        return {"type": "list", "length": 0, "note": "Empty list"}
    
    # If list of dictionaries → convert to DataFrame for consistent schema
    if isinstance(data[0], dict):
        try:
            df = pd.DataFrame(data[:sample_size])
            return _extract_dataframe_schema(df, sample_size=sample_size)
        except Exception as e:
            return {
                "type": "list[dict]",
                "length": len(data),
                "error": str(e),
                "sample": data[:3]
            }
    
    # Simple list (strings, numbers, etc.)
    else:
        unique_values = set()
        try:
            for item in data[:sample_size]:
                unique_values.add(str(item))
        except:
            pass
            
        is_categorical = len(unique_values) <= 20
        
        return {
            "type": "list",
            "length": len(data),
            "is_categorical": is_categorical,
            "enum": sorted(list(unique_values)) if is_categorical else None,
            "sample": data[:5]
        }


# Node functions
async def query_categorization_node(state: AgentState) -> AgentState:

    """QueryCategorizationLLM() categorizes user_query into data_sources"""
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "CATEGORIZING_CANCELLED", "Categorization skipped due to cancellation")

    await emit_step_event(
        state,
        "CATEGORIZING_START",
        "Choosing data source...",
        status="START",
        details=f"Query: {state['user_query'][:120]}",
    )

    print(f"🧠 [CATEGORIZING] Query: '{state['user_query'][:60]}...' | Error: {state.get('error', 'None')}", flush=True)

    try:
        await gate_next_step(state, "Choosing a data source...", wait_ms=500)

        query_categorization_response = query_categorization_llm.invoke(state["user_query"])

        print("llm response: ", query_categorization_response, flush=True)
        data_source = query_categorization_response.get('data_source')

        if not query_categorization_response.get("success"):
            return {
                **state,
                "error": "Categorization failed: " + query_categorization_response.get("error", "Unknown error")
            }

        # Set global TOOL_REGISTRY based on chosen data source
        try:
            global TOOL_REGISTRY
            if data_source == "order":
                TOOL_REGISTRY = ORDERS_TOOL_REGISTRY
            elif data_source == "profit":
                TOOL_REGISTRY = PROFIT_TOOL_REGISTRY
            elif data_source == "payment_cycle":
                TOOL_REGISTRY = PAYMENT_CYCLE_TOOL_REGISTRY
        except Exception:
            # Fallback to orders registry in case of any issue
            TOOL_REGISTRY = ORDERS_TOOL_REGISTRY

        await emit_step_event(
            state,
            "CATEGORIZING_COMPLETE",
            f"Data source chosen: {data_source} query",
            status="COMPLETE",
        )

        print(f"✅ [CATEGORIZING] Data source: {data_source}", flush=True)

        return {
            **state,
            "data_source": data_source,
            "current_step_index": 0,
            "error": None
        }
    
    except Exception as e:
        await emit_step_event(
            state,
            "CATEGORIZING_ERROR",
            "Categorizing failed",
            status="ERROR",
            details=str(e),
        )
        
        print(f"❌ [CATEGORIZING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Categorizing error: {str(e)}"}


async def planning_node(state: AgentState) -> AgentState:
    """Planning LLM generates execution plan"""
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "PLANNING_CANCELLED", "Planning skipped due to cancellation")

    # Early exit if previous step had error
    if state.get("error"):
        # print("state: ", state, flush=True)
        print(f"⏭️ [PLANNING] Skipped due to previous error", flush=True)
        return state

    await emit_step_event(
        state,
        "PLANNING_START",
        "Planning execution...",
        status="START",
        details=f"Query: {state['user_query'][:120]}",
    )
    
    print(f"🧠 [PLANNING] Query: '{state['user_query'][:60]}...' | Error: {state.get('error', 'None')}", flush=True)

    try:
        await gate_next_step(state, "Generating a plan...", wait_ms=500)

        data_source = state.get("data_source")
        
        print(f"data_source in workflow planning_node: ", data_source, flush=True)

        # Now call invoke with clean string
        plan_response = planning_llm.invoke(
            query=state["user_query"], 
            data_source=state["data_source"]          # ← Now correctly passing string "order" or "profit"
        )
        
        if not plan_response.get("success"):
            return {
                **state,
                "error": "Planning failed: " + plan_response.get("error", "Unknown error")
            }
        
        await emit_step_event(
            state,
            "PLANNING_COMPLETE",
            f"Plan created: {plan_response.get('plan', {}).get('query_type', 'unknown')} query",
            status="COMPLETE",
        )
        
        print(f"✅ [PLANNING] Plan created for {data_source} query", flush=True)
        
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

    if _is_request_cancelled(state):
        return await _cancelled_state(state, "EXECUTE_CANCELLED", "Step execution skipped due to cancellation")
    
    await emit_step_event(
        state,
        "EXECUTE_STEP_START",
        f"Step {step_idx + 1}/{total_steps}",
        status="START",
        details=f"Retry: {state.get('retry_count', 0)}",
    )
    
    print(f"🔧 [EXECUTE_TOOL] Step {step_idx + 1}/{total_steps} | Retry: {state.get('retry_count', 0)}", flush=True)

   

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
        if _is_request_cancelled(state):
            return await _cancelled_state(state, "EXECUTE_CANCELLED", "Step execution cancelled before tool run")

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

        if _is_request_cancelled(state):
            return await _cancelled_state(state, "EXECUTE_CANCELLED", "Step execution cancelled after tool run")
        
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
        if state["retry_count"] < 1:
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

        await gate_next_step(state, "Filtering data...", wait_ms=500)
        
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
    if _is_request_cancelled(state):
        return {**state, "error": REQUEST_CANCELLED_ERROR}

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
    if _is_request_cancelled(state):
        return {**state, "error": REQUEST_CANCELLED_ERROR}

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
                (step["tool"].startswith("get_") and step["tool"] not in ["get_all_orders", "get_schema_info", "get_vendor_cost_sheet"])
                or step["tool"] == "execute_custom_calculation"
            )
        ]

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
        if not metric_results and (state.get("data_source") == "order"):
            # Use the common metrics tool
            metric_results = TOOL_REGISTRY["get_common_metrics"](result_data)
        
        # Cache metric results
        if metric_results:
            metric_ref = cache_result(metric_results, key="metric_results")
            normalized_metrics = {"overall": metric_results}
            metrics_calculated = list(metric_results.keys()) if isinstance(metric_results, dict) else []
            print(f"✅ [METRIC_PROCESSING] Calculated {len(metric_results)} metrics", flush=True)
            
            print("metric_results: ", metric_results, flush=True)
            print("metric_ref: ", metric_ref, flush=True)

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


async def custom_calculation_node(state: AgentState) -> AgentState:
    """Execute custom metric calculation using ReAct pattern with CustomCalculationLLM"""
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "CUSTOM_CALC_CANCELLED", "Custom calculation skipped due to cancellation")

    print(f"🧠 [CUSTOM_CALCULATION] Executing custom metric generation with ReAct pattern", flush=True)
    
    try:
        await emit_step_event(
            state,
            "CUSTOM_CALC_START",
            "Initiating custom metric calculation...",
            status="START",
        )
        
        plan = state["plan"]

        # Always source DataFrame input from the latest convert_to_df output.
        # final_result_ref can point to metric caches by this stage.
        df_step = None
        for step in reversed(plan.get("steps", [])):
            if step.get("tool") == "convert_to_df":
                df_step = step
                break

        if not df_step:
            return {**state, "error": "No convert_to_df step found for custom calculation"}

        df_ref = state["tool_result_refs"].get(df_step["save_as"])
        if not df_ref:
            return {**state, "error": "No DataFrame reference available for custom calculation"}

        result_data = get_cached_result(df_ref)
        if not isinstance(result_data, pd.DataFrame):
            return {**state, "error": "Custom calculation requires a DataFrame input"}

        # Get schema info aligned with the DataFrame source step
        schema_info = state.get("tool_result_schemas", {}).get(df_step["save_as"], {})
        
        # Extract intent from the plan (summarized_query or explicit intent)
        intent = plan.get("summarized_query", state["summarized_query"] or "Calculate custom metric")
        
        # Bind the current DataFrame to the calculator executor expected by ReAct loop.
        def _bound_custom_executor(calculation_code: str, metric_name: str = "custom_metric") -> dict:
            return TOOL_REGISTRY["execute_custom_calculation"](
                table=result_data,
                calculation_code=calculation_code,
                metric_name=metric_name,
            )

        # Invoke CustomCalculationLLM with ReAct pattern
        calc_result = custom_calculation_llm.invoke({
            "query": state["user_query"],
            "intent": intent,
            "data": result_data,
            "schema": schema_info,
            "date_range": plan.get("base_params", {}),
            "executor": _bound_custom_executor,
        })
        
        if not calc_result.get("success"):
            error_msg = calc_result.get("error", "Custom calculation failed")
            print(f"❌ [CUSTOM_CALCULATION] {error_msg}", flush=True)
            return {
                **state,
                "error": f"Custom calculation error: {error_msg}",
                "metric_results": calc_result.get("last_observation")
            }
        
        # Extract result and metadata
        final_result = calc_result.get("final_result")
        metric_results = {
            "custom_metric": final_result,
            "calculation_code": calc_result.get("calculation_code"),
            "iterations": calc_result.get("iterations"),
            "metadata": calc_result.get("metadata", {})
        }
        
        # Cache the results
        metric_ref = cache_result(metric_results, key="custom_metric_results")
        normalized_metrics = {"overall": metric_results}
        
        print(f"✅ [CUSTOM_CALCULATION] Custom metric generated successfully in {calc_result.get('iterations', 1)} iteration(s)", flush=True)
        
        await emit_step_event(
            state,
            "CUSTOM_CALC_COMPLETE",
            f"Custom metric calculated: {final_result}",
            status="SUCCESS",
        )
        
        return {
            **state,
            "metric_results": metric_results,
            "aggregated_metrics": normalized_metrics,
            "metrics_calculated": ["custom_metric"],
            "final_result_ref": metric_ref,
            "error": None
        }
        
    except Exception as e:
        print(f"❌ [CUSTOM_CALCULATION] Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {**state, "error": f"Custom calculation error: {str(e)}"}


async def grouping_node(state: AgentState) -> AgentState:
    """Identify comparison groups and create parallel execution branches"""
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "GROUPING_CANCELLED", "Grouping skipped due to cancellation")

    print(f"🔀 [GROUPING] Identifying comparison groups", flush=True)
    
    try:
        await emit_step_event(
            state,
            "GROUPING_START",
            "Preparing comparison groups...",
            status="START",
        )

        await gate_next_step(state, "Grouping data...", wait_ms=500)
        
        # IMPORTANT: Pass data_source to the grouping LLM so it uses the correct schema!
        # Without this, grouping_llm defaults to "order" mode and ignores profit schema
        # This causes it to group by non-existent fields like "product_name" for profit queries
        grouping_response = grouping_llm.invoke({
            "query": state["user_query"],
            "plan": state["plan"],
            "data_source": state.get("data_source", "order")  # ← CRITICAL: Pass the data source (profit or order)
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
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "PARALLEL_FETCH_CANCELLED", "Group fetch skipped due to cancellation")

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
        if _is_request_cancelled(state):
            return await _cancelled_state(state, "PARALLEL_FETCH_CANCELLED", "Group fetch cancelled before tool run")
        
        # Fetch data using only base parameters (date range)
        result = tool_function(**plan["base_params"])

        if _is_request_cancelled(state):
            return await _cancelled_state(state, "PARALLEL_FETCH_CANCELLED", "Group fetch cancelled after tool run")
        
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

#COMPARISON - adding conditional if data_source === "order" OR "profit"
def aggregation_node(state: AgentState) -> AgentState:
    """Compute metrics for each comparison group"""
    if _is_request_cancelled(state):
        return {**state, "error": REQUEST_CANCELLED_ERROR}

    data_source = state.get("data_source")
    num_groups = len(state.get("group_results", {}))
    print(f"📈 [AGGREGATION] Computing metrics for {num_groups} group(s)", flush=True)

    if data_source == "order":
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
    
    elif data_source == "profit":
        try:
            aggregated_metrics = {}
            
            for group_id, result_ref in state["group_results"].items():
                data = get_cached_result(result_ref)
                
                # Ensure data is always a list for uniform processing
                if not isinstance(data, list):
                    data = [data] if data else []
                
                # Helper to safely get numeric value from record
                def safe_float(value, default=0.0):
                    try:
                        if value is None or value == "" or value == "None":
                            return default
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                group_metrics = []
                
                for record in data:
                    # Convert record to dict if it's a tuple/row from DB
                    if not isinstance(record, dict):
                        try:
                            record = dict(record)  # for psycopg2 RealDictRow or similar
                        except (TypeError, ValueError):
                            record = {}
                    
                    # Calculate all profit metrics using your imported tools
                    # IMPORTANT: Wrap record in a list when passing to tools!
                    # Tools like get_cost_price() expect Union[pd.DataFrame, list], not a single dict
                    # Passing a dict causes: "'dict' object has no attribute 'columns'"
                    metrics = {
                        "style_name": record.get("Style Name") or record.get("style_name"),
                        "collection_name": record.get("Collection Name") or record.get("collection_name"),
                        "brand": record.get("BRAND") or record.get("brand"),
                        "gender": record.get("Gender") or record.get("gender"),
                        "factory": record.get("FACTORY") or record.get("factory"),
                        "cost_price": get_cost_price([record]),  # ← Wrap in list!
                        "selling_price": get_selling_price([record]),  # ← Wrap in list!
                        "gross_profit": get_gross_profit([record]),  # ← Wrap in list!
                        "gross_margin": get_margin([record]),           # in % | Wrap in list!
                        "markup": get_markup([record]),                 # as ratio | Wrap in list!
                        "markup_percent": round(get_markup([record]) * 100, 2) if get_markup([record]) is not None else None,  # Wrap in list!
                        "cost_to_price_ratio": get_cost_to_price_ratio([record]),  # in % | Wrap in list!
                    }
                    
                    # Add raw columns for transparency
                    metrics["final_price_raw"] = safe_float(record.get("Final price"))
                    metrics["price_1_raw"] = safe_float(record.get("PRICE_1"))
                    metrics["mrp_raw"] = safe_float(record.get("MRP"))
                    
                    group_metrics.append(metrics)
                
                aggregated_metrics[group_id] = {
                    "items": group_metrics,
                    "count": len(group_metrics),
                    "summary": {
                        "avg_gross_margin": round(sum(m["gross_margin"] for m in group_metrics if m["gross_margin"] is not None) / len([m for m in group_metrics if m["gross_margin"] is not None]) or 0, 2),
                        "avg_markup_percent": round(sum(m["markup_percent"] for m in group_metrics if m["markup_percent"] is not None) / len([m for m in group_metrics if m["markup_percent"] is not None]) or 0, 2),
                        "total_gross_profit": round(sum(m["gross_profit"] for m in group_metrics if m["gross_profit"] is not None), 2),
                    }
                }
            
            print(f"✅ [AGGREGATION] Profit metrics computed for: {list(aggregated_metrics.keys())}", flush=True)
            
            return {
                **state,
                "aggregated_metrics": aggregated_metrics,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ [AGGREGATION] Profit Error: {str(e)}", flush=True)
            return {**state, "error": f"Profit aggregation error: {str(e)}"}
    
    elif data_source == "payment_cycle":
        try:
            aggregated_metrics = {}
            
            for group_id, result_ref in state["group_results"].items():
                data = get_cached_result(result_ref)
                
                # Ensure data is always a list for uniform processing
                if not isinstance(data, list):
                    data = [data] if data else []
                
                # Helper to safely get numeric value from record
                def safe_float(value, default=0.0):
                    try:
                        if value is None or value == "" or value == "None":
                            return default
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                # Import payment cycle tools
                from tools import (
                    get_avg_margin, get_weighted_avg_margin, get_margin_per_payment_day,
                    get_total_margin_exposure, get_high_risk_distributors, get_cycle_efficiency_score,
                    get_payment_cycle_distribution, get_cash_discount_stats
                )
                
                df_data = pd.DataFrame(data) if data else pd.DataFrame()
                
                # Calculate payment cycle metrics
                metrics = {
                    "distributor_count": len(data),
                    "avg_margin": get_avg_margin(data),
                    "margin_per_payment_day": get_margin_per_payment_day(data),
                    "cycle_efficiency_score": get_cycle_efficiency_score(data),
                    "total_margin_exposure": get_total_margin_exposure(data),
                    "payment_cycle_distribution": get_payment_cycle_distribution(data),
                    "cash_discount_stats": get_cash_discount_stats(data),
                    "high_risk_distributors": get_high_risk_distributors(data),
                }
                
                aggregated_metrics[group_id] = metrics
            
            print(f"✅ [AGGREGATION] Payment cycle metrics computed for: {list(aggregated_metrics.keys())}", flush=True)
            
            return {
                **state,
                "aggregated_metrics": aggregated_metrics,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ [AGGREGATION] Payment Cycle Error: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return {**state, "error": f"Payment cycle aggregation error: {str(e)}"}
    
    else:
        # Fallback for unknown data_source
        return {**state, "error": f"Unsupported data_source: {data_source}"}

def comparison_node(state: AgentState) -> AgentState:
    """Perform comparison logic between groups - supports both order and profit data"""
    if _is_request_cancelled(state):
        return {**state, "error": REQUEST_CANCELLED_ERROR}

    print(f"⚖️  [COMPARISON] Comparing groups", flush=True)
    
    try:
        metrics = state.get("aggregated_metrics")
        if not metrics:
            return {**state, "error": "No aggregated metrics available for comparison"}
        
        group_ids = list(metrics.keys())
        num_groups = len(group_ids)
        
        if num_groups < 2:
            return {**state, "error": "Need at least 2 groups for comparison"}
        
        data_source = state.get("data_source")
        
        # ====================== PROFIT COMPARISON ======================
        if data_source == "profit":
            comparison_results = {
                "comparison_type": "pairwise" if num_groups == 2 else "multi_group",
                "comparison_mode": True,
                "comparison_param": state.get("comparison_param"),
                "data_source": "profit",
                "num_groups": num_groups,
                "groups": group_ids,
            }
            
            if num_groups == 2:
                # Pairwise comparison (most common for styles)
                group_a, group_b = group_ids[0], group_ids[1]
                metrics_a = metrics[group_a]
                metrics_b = metrics[group_b]
                
                # Extract summary for easier access
                summary_a = metrics_a.get("summary", {})
                summary_b = metrics_b.get("summary", {})
                
                comparison_results.update({
                    "groups": {"a": group_a, "b": group_b},
                    "item_count": {
                        "a": metrics_a.get("count", 0),
                        "b": metrics_b.get("count", 0),
                        "diff": metrics_b.get("count", 0) - metrics_a.get("count", 0),
                        "diff_pct": round(((metrics_b.get("count", 0) - metrics_a.get("count", 0)) / metrics_a.get("count", 1) * 100), 2) 
                                   if metrics_a.get("count", 0) > 0 else 0
                    },
                    "avg_gross_margin": {
                        "a": summary_a.get("avg_gross_margin", 0),
                        "b": summary_b.get("avg_gross_margin", 0),
                        "diff": round(summary_b.get("avg_gross_margin", 0) - summary_a.get("avg_gross_margin", 0), 2),
                        "diff_pct": round(((summary_b.get("avg_gross_margin", 0) - summary_a.get("avg_gross_margin", 0)) / 
                                         summary_a.get("avg_gross_margin", 1) * 100), 2) if summary_a.get("avg_gross_margin", 0) > 0 else 0
                    },
                    "avg_markup_percent": {
                        "a": summary_a.get("avg_markup_percent", 0),
                        "b": summary_b.get("avg_markup_percent", 0),
                        "diff": round(summary_b.get("avg_markup_percent", 0) - summary_a.get("avg_markup_percent", 0), 2),
                        "diff_pct": round(((summary_b.get("avg_markup_percent", 0) - summary_a.get("avg_markup_percent", 0)) / 
                                         summary_a.get("avg_markup_percent", 1) * 100), 2) if summary_a.get("avg_markup_percent", 0) > 0 else 0
                    },
                    "total_gross_profit": {
                        "a": summary_a.get("total_gross_profit", 0),
                        "b": summary_b.get("total_gross_profit", 0),
                        "diff": round(summary_b.get("total_gross_profit", 0) - summary_a.get("total_gross_profit", 0), 2),
                        "diff_pct": round(((summary_b.get("total_gross_profit", 0) - summary_a.get("total_gross_profit", 0)) / 
                                         summary_a.get("total_gross_profit", 1) * 100), 2) if summary_a.get("total_gross_profit", 0) > 0 else 0
                    },
                    "winner_by_margin": group_a if summary_a.get("avg_gross_margin", 0) > summary_b.get("avg_gross_margin", 0) else group_b,
                    "winner_by_total_profit": group_a if summary_a.get("total_gross_profit", 0) > summary_b.get("total_gross_profit", 0) else group_b,
                    "winner_by_markup": group_a if summary_a.get("avg_markup_percent", 0) > summary_b.get("avg_markup_percent", 0) else group_b,
                })
                
                print(f"✅ [COMPARISON] Profit: {group_a} vs {group_b} | "
                      f"Margin Winner: {comparison_results['winner_by_margin']}", flush=True)
            else:
                # Multi-group comparison for profit
                baseline_id = group_ids[0]
                baseline_summary = metrics[baseline_id].get("summary", {})
                
                group_summaries = {}
                comparisons_to_baseline = {}
                
                for group_id in group_ids:
                    group_summary = metrics[group_id].get("summary", {})
                    group_summaries[group_id] = {
                        "count": metrics[group_id].get("count", 0),
                        "avg_gross_margin": group_summary.get("avg_gross_margin", 0),
                        "avg_markup_percent": group_summary.get("avg_markup_percent", 0),
                        "total_gross_profit": group_summary.get("total_gross_profit", 0),
                    }
                
                for group_id in group_ids[1:]:
                    group_summary = metrics[group_id].get("summary", {})
                    comparisons_to_baseline[group_id] = {
                        "margin_diff": round(group_summary.get("avg_gross_margin", 0) - baseline_summary.get("avg_gross_margin", 0), 2),
                        "margin_diff_pct": round(((group_summary.get("avg_gross_margin", 0) - baseline_summary.get("avg_gross_margin", 0)) / 
                                                baseline_summary.get("avg_gross_margin", 1) * 100), 2) 
                                           if baseline_summary.get("avg_gross_margin", 0) > 0 else 0,
                        "total_profit_diff": round(group_summary.get("total_gross_profit", 0) - baseline_summary.get("total_gross_profit", 0), 2),
                        "total_profit_diff_pct": round(((group_summary.get("total_gross_profit", 0) - baseline_summary.get("total_gross_profit", 0)) / 
                                                      baseline_summary.get("total_gross_profit", 1) * 100), 2) 
                                                if baseline_summary.get("total_gross_profit", 0) > 0 else 0,
                    }
                
                # Overall winners
                winner_by_margin = max(group_ids, key=lambda gid: metrics[gid].get("summary", {}).get("avg_gross_margin", 0))
                winner_by_profit = max(group_ids, key=lambda gid: metrics[gid].get("summary", {}).get("total_gross_profit", 0))
                winner_by_markup = max(group_ids, key=lambda gid: metrics[gid].get("summary", {}).get("avg_markup_percent", 0))
                
                comparison_results.update({
                    "baseline": baseline_id,
                    "group_summaries": group_summaries,
                    "comparisons_to_baseline": comparisons_to_baseline,
                    "overall_winners": {
                        "by_margin": winner_by_margin,
                        "by_total_profit": winner_by_profit,
                        "by_markup": winner_by_markup
                    }
                })
                
                print(f"✅ [COMPARISON] Profit: {num_groups} groups compared | "
                      f"Best Margin: {winner_by_margin}", flush=True)
            
            # Return profit comparison results
            return {
                **state,
                "comparison_results": comparison_results,
                "error": None
            }
        
        # ====================== ORDER COMPARISON ======================
        elif data_source == "order":
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
        
        # ====================== PAYMENT CYCLE COMPARISON ======================
        elif data_source == "payment_cycle":
            comparison_results = {
                "comparison_type": "pairwise" if num_groups == 2 else "multi_group",
                "comparison_mode": True,
                "comparison_param": state.get("comparison_param"),
                "data_source": "payment_cycle",
                "num_groups": num_groups,
                "groups": group_ids,
            }
            
            if num_groups == 2:
                # Pairwise comparison
                group_a, group_b = group_ids[0], group_ids[1]
                metrics_a = metrics[group_a]
                metrics_b = metrics[group_b]
                
                comparison_results.update({
                    "groups": {"a": group_a, "b": group_b},
                    "distributor_count": {
                        "a": metrics_a.get("distributor_count", 0),
                        "b": metrics_b.get("distributor_count", 0),
                        "diff": metrics_b.get("distributor_count", 0) - metrics_a.get("distributor_count", 0),
                    },
                    "avg_margin": {
                        "a": metrics_a.get("avg_margin", 0),
                        "b": metrics_b.get("avg_margin", 0),
                        "diff": round(metrics_b.get("avg_margin", 0) - metrics_a.get("avg_margin", 0), 2),
                    },
                    "margin_per_payment_day": {
                        "a": metrics_a.get("margin_per_payment_day", 0),
                        "b": metrics_b.get("margin_per_payment_day", 0),
                        "diff": round(metrics_b.get("margin_per_payment_day", 0) - metrics_a.get("margin_per_payment_day", 0), 4),
                    },
                    "cycle_efficiency_score": {
                        "a": metrics_a.get("cycle_efficiency_score", 0),
                        "b": metrics_b.get("cycle_efficiency_score", 0),
                        "diff": round(metrics_b.get("cycle_efficiency_score", 0) - metrics_a.get("cycle_efficiency_score", 0), 4),
                    },
                    "total_margin_exposure": {
                        "a": metrics_a.get("total_margin_exposure", 0),
                        "b": metrics_b.get("total_margin_exposure", 0),
                        "diff": round(metrics_b.get("total_margin_exposure", 0) - metrics_a.get("total_margin_exposure", 0), 2),
                    },
                    "high_risk_count": {
                        "a": len(metrics_a.get("high_risk_distributors", [])),
                        "b": len(metrics_b.get("high_risk_distributors", [])),
                        "diff": len(metrics_b.get("high_risk_distributors", [])) - len(metrics_a.get("high_risk_distributors", [])),
                    },
                    "winner_by_margin": group_a if metrics_a.get("avg_margin", 0) > metrics_b.get("avg_margin", 0) else group_b,
                    "winner_by_efficiency": group_a if metrics_a.get("margin_per_payment_day", 0) > metrics_b.get("margin_per_payment_day", 0) else group_b,
                    "lower_risk_group": group_a if len(metrics_a.get("high_risk_distributors", [])) < len(metrics_b.get("high_risk_distributors", [])) else group_b,
                })
                
                print(f"✅ [COMPARISON] Payment Cycle: {group_a} vs {group_b} | "
                      f"Margin Winner: {comparison_results['winner_by_margin']}", flush=True)
            else:
                # Multi-group comparison
                group_summaries = {}
                for group_id in group_ids:
                    group_summaries[group_id] = {
                        "distributor_count": metrics[group_id].get("distributor_count", 0),
                        "avg_margin": metrics[group_id].get("avg_margin", 0),
                        "margin_per_payment_day": metrics[group_id].get("margin_per_payment_day", 0),
                        "cycle_efficiency_score": metrics[group_id].get("cycle_efficiency_score", 0),
                        "total_margin_exposure": metrics[group_id].get("total_margin_exposure", 0),
                        "high_risk_count": len(metrics[group_id].get("high_risk_distributors", [])),
                    }
                
                # Find winners
                winner_by_margin = max(group_ids, key=lambda gid: metrics[gid].get("avg_margin", 0))
                winner_by_efficiency = max(group_ids, key=lambda gid: metrics[gid].get("margin_per_payment_day", 0))
                lowest_risk_group = min(group_ids, key=lambda gid: len(metrics[gid].get("high_risk_distributors", [])))
                
                comparison_results.update({
                    "group_summaries": group_summaries,
                    "overall_winners": {
                        "by_margin": winner_by_margin,
                        "by_efficiency": winner_by_efficiency,
                        "lowest_risk": lowest_risk_group
                    }
                })
                
                print(f"✅ [COMPARISON] Payment Cycle: {num_groups} groups compared | "
                      f"Best Margin: {winner_by_margin}, Most Efficient: {winner_by_efficiency}", flush=True)
            
            return {
                **state,
                "comparison_results": comparison_results,
                "error": None
            }
        
        else:
            # Unsupported data source
            return {**state, "error": f"Comparison not supported for data_source: {data_source}"}
            
    except Exception as e:
        print(f"❌ [COMPARISON] Error: {str(e)}", flush=True)
        return {**state, "error": f"Comparison error: {str(e)}"}

async def insight_generation_node(state: AgentState) -> AgentState:
    """Generate natural language insights for comparison and metric analysis."""
    if _is_request_cancelled(state):
        return await _cancelled_state(state, "INSIGHT_CANCELLED", "Insight generation skipped due to cancellation")

    print(f"💡 [INSIGHTS] Generating natural language summary", flush=True)
    
    # print("STATE: ", state, flush=True)
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
        query_type = state["plan"].get("query_type", "standard")
        is_metric_query = query_type in ["metric_analysis", "custom_metric_generation"]

        normalized_metrics = state.get("aggregated_metrics")
        if not normalized_metrics and is_metric_query and state.get("metric_results"):
            normalized_metrics = {"overall": state["metric_results"]}

        await gate_next_step(state, "Generating intelligent insights...", wait_ms=500)
        if _is_request_cancelled(state):
            return await _cancelled_state(state, "INSIGHT_CANCELLED", "Insight generation cancelled before LLM call")

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
    if not state.get("error"):
        return "end"

    # If planning failed or too many retries, abort
    if "Planning" in state["error"] or "Invalid precomputed plan" in state["error"] or state["retry_count"] >= 2:
        return "end"
    
    # Otherwise, attempt to replan
    return "replan"

def route_metric_processing(state: AgentState) -> Literal["custom_calculation", "insight_generation", "error"]:
    """Route after metric processing to custom calculation or insight generation"""
    if state.get("error"):
        return "error"
    
    query_type = state["plan"].get("query_type", "standard")
    
    # Custom metric generation goes to custom calculation node
    if query_type == "custom_metric_generation":
        return "custom_calculation"
    
    # All other metric types go to insight generation
    return "insight_generation"

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


def prepare_start_node(state: AgentState) -> AgentState:
    """Validate precomputed plan before routing into the graph."""
    plan = state.get("plan")
    if not plan:
        return state

    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return {
            **state,
            "error": "Invalid precomputed plan: expected object with steps list"
        }

    return state


def route_start(state: AgentState) -> Literal["query_categorization", "planning", "grouping", "execute_tool", "error"]:
    """Main router at the beginning of the workflow"""
    if state.get("error"):
        return "error"

    plan = state.get("plan")
    
    # If a valid precomputed plan already exists, skip categorization + planning
    if plan and isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        if plan.get("query_type") == "comparison":
            return "grouping"
        return "execute_tool"

    # Normal flow: start with categorization
    return "query_categorization"


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes - Standard flow
workflow.add_node("query_categorization", query_categorization_node)
workflow.add_node("route_start", prepare_start_node)
workflow.add_node("planning", planning_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("filtering", filtering_node)
workflow.add_node("apply_filters", apply_filters_node)
workflow.add_node("no_filter", no_filter_node)

# Add nodes - Metric analysis flow
workflow.add_node("metric_processing", metric_processing_node)
workflow.add_node("custom_calculation", custom_calculation_node)

# Add nodes - Comparison flow
workflow.add_node("grouping", grouping_node)
workflow.add_node("parallel_fetch", parallel_fetch_node)
workflow.add_node("aggregation", aggregation_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("insight_generation", insight_generation_node)


workflow.set_entry_point("route_start")

# Start routing - skip planning when plan is already present
workflow.add_conditional_edges(
    "route_start",
    route_start,
    {
        "query_categorization": "query_categorization",
        "planning": "planning",
        "grouping": "grouping",
        "execute_tool": "execute_tool",
        "error": "error_handler",
    }
)

# After choosing data source, head to creating a plan
workflow.add_edge("query_categorization", "planning")

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

# Metric analysis flow - route metric processing to custom calculation or insight generation
workflow.add_conditional_edges(
    "metric_processing",
    route_metric_processing,
    {
        "custom_calculation": "custom_calculation",
        "insight_generation": "insight_generation",
        "error": "error_handler"
    }
)

# Custom calculation routes to insight generation
workflow.add_edge("custom_calculation", "insight_generation")

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