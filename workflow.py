from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

# Import tools and LLM providers
from tools import TOOL_REGISTRY, apply_filters
from llm_providers import planning_llm, filtering_llm, grouping_llm, insight_llm

# State schema
class AgentState(TypedDict):
    user_query: str
    plan: dict | None
    tool_result_refs: dict[str, str]  # {save_as: cache_key} - references only
    tool_result_schemas: dict[str, dict]  # {save_as: schema_with_enums} - lightweight
    current_step_index: int
    filters: list[dict] | None
    final_result_ref: str | None  # Reference to final result
    error: str | None
    retry_count: int
    # Comparison-specific fields
    comparison_mode: bool
    comparison_groups: list[dict] | None  # [{group_id, filters}, ...]
    group_results: dict[str, str] | None  # {group_id: cache_ref}
    group_schemas: dict[str, dict] | None  # {group_id: schema}
    current_group_index: int
    aggregated_metrics: dict[str, dict] | None  # {group_id: metrics}
    comparison_results: dict | None
    insights: str | None

# Simple in-memory cache (use Redis for production)
RESULT_CACHE = {}

def cache_result(data: dict | list, key: str) -> str:
    """Cache result and return reference key"""
    RESULT_CACHE[key] = data
    return key

def get_cached_result(key: str) -> dict | list:
    """Retrieve result from cache"""
    return RESULT_CACHE.get(key)

def extract_schema_with_enums(data: dict | list, sample_size: int = 100) -> dict:
    """Extract schema with categorical value examples for LLM understanding"""
    if not data:
        return {}
    
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
def planning_node(state: AgentState) -> AgentState:
    """Planning LLM generates execution plan"""
    print(f"🧠 [PLANNING] Query: '{state['user_query'][:60]}...' | Error: {state.get('error', 'None')}", flush=True)
    try:
        # Call your planning LLM here
        plan_response = planning_llm.invoke(state["user_query"])
        
        if not plan_response.get("success"):
            return {
                **state,
                "error": "Planning failed: " + plan_response.get("error", "Unknown error")
            }
        
        print(f"✅ [PLANNING] Plan created: {plan_response['plan'].get('query_type', 'unknown')} query", flush=True)
        return {
            **state,
            "plan": plan_response["plan"],
            "current_step_index": 0,
            "error": None
        }
    except Exception as e:
        print(f"❌ [PLANNING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Planning error: {str(e)}"}

def execute_tool_node(state: AgentState) -> AgentState:
    """Execute the current step's tool"""
    step_idx = state["current_step_index"]
    total_steps = len(state["plan"]["steps"]) if state["plan"] else 0
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
            resolved_params[dep_step["save_as"]] = get_cached_result(ref_key)
        
        # Execute tool (map tool name to actual function)
        print(f"✅ [EXECUTE_TOOL] Tool executed: {step['tool']} | Records: {len(result) if isinstance(result, list) else 1}", flush=True)
        
        tool_function = TOOL_REGISTRY.get(step["tool"])
        if not tool_function:
            return {**state, "error": f"Tool {step['tool']} not found"}
        
        result = tool_function(**resolved_params)
        
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
        print(f"❌ [EXECUTE_TOOL] Error: {str(e)}", flush=True)
        
    except Exception as e:
        # Retry logic
        if state["retry_count"] < 2:
            return {
                **state,
                "retry_count": state["retry_count"] + 1,
                "error": f"Tool execution retry {state['retry_count'] + 1}: {str(e)}"
            }
        return {**state, "error": f"Tool execution failed: {str(e)}"}
print(f"🔍 [FILTERING] Extracting filters from query", flush=True)
    
def filtering_node(state: AgentState) -> AgentState:
    """Filtering LLM generates filter parameters using schema with categorical values"""
    try:
        plan = state["plan"]
        # Get schema (not data!) from the last step
        last_step = plan["steps"][-1]
        schema = state["tool_result_schemas"][last_step["save_as"]]
        
        # Call filtering LLM with schema including enum values
        # LLM can now learn: "prepaid" -> "PrePaid" from enum list
        filter_response = filtering_llm.invoke({
            "query": state["user_query"],
            "schema": schema,  # Includes enum values for categorical fields
            "manipulation_type": plan["manipulation"]["type"]
        })

        print(f"✅ [FILTERING] Filters extracted: {len(filter_response.get('filters', []))} filter(s)", flush=True)
        
        return {
            **state,
            "filters": filter_response.get("filters", []),
            "error": None
        }
        
    except Exception as e:
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
        final_ref = cache_result(filtered_data, key="final_result")
        
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
    
    # Reference the cached result as final result
    return {
        **state,
        "final_result_ref": ref_key,
        "error": None
    }

def grouping_node(state: AgentState) -> AgentState:
    """Identify comparison groups and create parallel execution branches"""
    print(f"🔀 [GROUPING] Identifying comparison groups", flush=True)
    
    try:
        # Call grouping LLM to extract comparison dimensions
        grouping_response = grouping_llm.invoke({
            "query": state["user_query"],
            "plan": state["plan"]
        })
        
        groups = grouping_response.get("groups", [])
        print(f"✅ [GROUPING] Found {len(groups)} groups: {[g['group_id'] for g in groups]}", flush=True)
        
        return {
            **state,
            "comparison_groups": grouping_response["groups"],
            "group_results": {},
            "group_schemas": {},
            "current_group_index": 0,
            "error": None
        }
    except Exception as e:
        print(f"❌ [GROUPING] Error: {str(e)}", flush=True)
        return {**state, "error": f"Grouping error: {str(e)}"}

def parallel_fetch_node(state: AgentState) -> AgentState:
    """Execute data fetches for current comparison group"""
    current_idx = state["current_group_index"]
    total_groups = len(state["comparison_groups"]) if state["comparison_groups"] else 0
    group_id = state["comparison_groups"][current_idx]["group_id"] if state["comparison_groups"] else "unknown"
    print(f"📥 [PARALLEL_FETCH] Fetching group {current_idx + 1}/{total_groups}: {group_id}", flush=True)
    
    try:
        plan = state["plan"]
        current_group = state["comparison_groups"][state["current_group_index"]]
        
        # IMPORTANT: Only pass parameters that the tool function accepts
        # For get_all_orders(), only start_date and end_date are valid parameters
        # Group-specific filters (e.g., payment_mode, marketplace) are applied AFTER fetching
        tool_function = TOOL_REGISTRY.get(plan["tool"])
        if not tool_function:
            return {**state, "error": f"Tool {plan['tool']} not found"}
        
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
                "top_cities": {}
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
                    "top_cities": group_metrics["top_cities"]
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

def insight_generation_node(state: AgentState) -> AgentState:
    """Generate natural language insights from comparison"""
    print(f"💡 [INSIGHTS] Generating natural language summary", flush=True)
    
    try:
        # Call insight LLM to generate natural language summary
        insight_response = insight_llm.invoke({
            "query": state["user_query"],
            "metrics": state["aggregated_metrics"],
            "comparison": state["comparison_results"]
        })
        
        print(f"✅ [INSIGHTS] Generated {len(insight_response['insights'])} chars of insights", flush=True)
        
        # Cache insights as final result
        final_ref = cache_result({
            "insights": insight_response["insights"],
            "comparison_data": state["comparison_results"],
            "detailed_metrics": state["aggregated_metrics"]
        }, key="comparison_final")
        
        return {
            **state,
            "insights": insight_response["insights"],
            "final_result_ref": final_ref,
            "error": None
        }
        
    except Exception as e:
        print(f"❌ [INSIGHTS] Error: {str(e)}", flush=True)
        return {**state, "error": f"Insight generation error: {str(e)}"}


# Conditional edges
def should_continue_execution(state: AgentState) -> Literal["execute_tool", "check_manipulation", "error"]:
    """Check if more tools need execution"""
    if state["error"]:
        return "error"
    
    if state["current_step_index"] < len(state["plan"]["steps"]):
        return "execute_tool"
    
    return "check_manipulation"

def needs_manipulation(state: AgentState) -> Literal["filtering", "no_filter", "error"]:
    """Check if filtering/manipulation is required"""
    if state["error"]:
        return "error"
    
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
    """Check if query requires comparison"""
    if state["error"]:
        return "error"
    
    if state["plan"].get("query_type") == "comparison":
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
        "error": "error_handler"
    }
)

workflow.add_edge("filtering", "apply_filters")
workflow.add_edge("apply_filters", END)
workflow.add_edge("no_filter", END)

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
        insights=None
    )

    # Example 2: Comparison query
    initial_state_comparison = AgentState(
        user_query="Compare orders between Shopify13 & Flipkart from the last 10 days",
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
        group_results=None,
        group_schemas=None,
        current_group_index=0,
        aggregated_metrics=None,
        comparison_results=None,
        insights=None
    )

    # Run standard query
    result_standard = app.invoke(initial_state_standard)

    if result_standard["final_result_ref"]:
        final_data = get_cached_result(result_standard["final_result_ref"])
        if isinstance(final_data, list):
            print(f"Found {len(final_data)} records")
        else:
            print(f"Result: {final_data}")
    else:
        print(f"Error: {result_standard['error']}")

    # Run comparison query
    result_comparison = app.invoke(initial_state_comparison)

    if result_comparison["final_result_ref"]:
        final_data = get_cached_result(result_comparison["final_result_ref"])
        print(f"\nComparison Insights:\n{final_data['insights']}")
        print(f"\nComparison Data: {final_data['comparison_data']}")
    else:
        print(f"Error: {result_comparison['error']}")