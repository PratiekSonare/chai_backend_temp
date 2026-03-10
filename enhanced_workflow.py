"""
Enhanced workflow with integrated reasoning capabilities
"""
from typing import TypedDict, Annotated, Literal, Callable, Any
from langgraph.graph import StateGraph, END
import operator
import pandas as pd
import json

# Import existing components
from tools import TOOL_REGISTRY, apply_filters
from llm_providers import planning_llm, filtering_llm, grouping_llm, insight_llm, metric_llm

# Import new reasoning agents
from reasoning_agents import create_reasoning_agent

# Enhanced state schema with reasoning capabilities
class EnhancedAgentState(TypedDict):
    # Existing fields
    user_query: str
    summarized_query: str | None
    plan: dict | None
    tool_result_refs: dict[str, str]
    tool_result_schemas: dict[str, dict]
    current_step_index: int
    filters: list[dict] | None
    final_result_ref: str | None
    error: str | None
    retry_count: int
    logger: Any
    request_id: str | None
    comparison_mode: bool
    comparison_groups: list[dict] | None
    comparison_param: str | None
    group_results: dict[str, str] | None
    group_schemas: dict[str, dict] | None
    current_group_index: int
    aggregated_metrics: dict[str, dict] | None
    comparison_results: dict | None
    insights: str | None
    metric_results: dict | None
    metric_analysis: str | None
    
    # New reasoning fields
    reasoning_mode: Literal["standard", "react", "cot", "meta"] | None
    reasoning_trace: list[dict] | None
    reasoning_context: dict | None
    intermediate_reasoning_results: dict | None
    confidence_score: float | None
    reasoning_justification: str | None
    multi_step_plan: list[dict] | None
    verification_results: dict | None


# Simple in-memory cache (use Redis for production)
ENHANCED_RESULT_CACHE = {}

def cache_enhanced_result(data: dict | list, key: str) -> str:
    """Cache result and return reference key"""
    ENHANCED_RESULT_CACHE[key] = data
    return key

def get_cached_enhanced_result(key: str) -> dict | list:
    """Retrieve result from cache"""
    return ENHANCED_RESULT_CACHE.get(key)


async def meta_reasoning_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """
    Meta-reasoning node that decides which reasoning approach to use
    """
    if state.get("logger"):
        await state["logger"](state.get("request_id", "unknown"), "META_REASONING", "Analyzing query complexity and choosing reasoning approach")
    
    try:
        meta_agent = create_reasoning_agent("meta")
        
        # Analyze query to determine reasoning approach
        meta_result = meta_agent.invoke({
            "query": state["user_query"],
            "context": {
                "has_comparison": "compare" in state["user_query"].lower(),
                "has_calculation": any(word in state["user_query"].lower() 
                                    for word in ["calculate", "average", "total", "count"]),
                "query_length": len(state["user_query"].split()),
                "existing_plan": state.get("plan")
            },
            "available_reasoning_types": ["react", "cot", "standard"]
        })
        
        reasoning_mode = meta_result.get("recommended_approach", "standard")
        reasoning_params = meta_result.get("reasoning_parameters", {})
        
        if state.get("logger"):
            await state["logger"](
                state.get("request_id", "unknown"), 
                "META_REASONING", 
                f"Selected reasoning approach: {reasoning_mode} - {meta_result.get('justification', '')}"
            )
        
        return {
            **state,
            "reasoning_mode": reasoning_mode,
            "reasoning_context": reasoning_params,
            "reasoning_justification": meta_result.get("justification", "")
        }
        
    except Exception as e:
        if state.get("logger"):
            await state["logger"](state.get("request_id", "unknown"), "META_REASONING", f"Error: {str(e)}, falling back to standard")
        
        return {
            **state,
            "reasoning_mode": "standard",
            "reasoning_justification": f"Error in meta-reasoning: {str(e)}, using standard approach"
        }


async def react_reasoning_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """
    ReACT reasoning node for complex multi-step queries
    """
    if state.get("logger"):
        await state["logger"](state.get("request_id", "unknown"), "REACT_REASONING", "Starting ReACT reasoning process")
    
    try:
        react_agent = create_reasoning_agent("react")
        
        # Prepare available tools for the ReACT agent
        available_tools = [
            {
                "name": "get_orders", 
                "description": "Fetch order data for date range",
                "parameters": {"start_date": "string", "end_date": "string"}
            },
            {
                "name": "apply_filters",
                "description": "Filter data by conditions",
                "parameters": {"filters": "array"}
            },
            {
                "name": "calculate_metrics",
                "description": "Calculate business metrics (revenue, AOV, count)",
                "parameters": {"data": "object", "metrics": "array"}
            },
            {
                "name": "analyze_distribution", 
                "description": "Analyze data distribution across dimensions",
                "parameters": {"data": "object", "dimension": "string"}
            }
        ]
        
        reasoning_context = state.get("reasoning_context", {})
        
        react_result = react_agent.invoke({
            "query": state["user_query"],
            "available_tools": available_tools,
            "context": {
                "existing_data": state.get("tool_result_refs", {}),
                "schemas": state.get("tool_result_schemas", {}),
                **reasoning_context
            },
            "goal": reasoning_context.get("goal", f"Comprehensively answer: {state['user_query']}")
        })
        
        if state.get("logger"):
            await state["logger"](
                state.get("request_id", "unknown"), 
                "REACT_REASONING", 
                f"ReACT reasoning completed with {len(react_result.get('reasoning_trace', []))} steps"
            )
        
        return {
            **state,
            "reasoning_trace": react_result.get("reasoning_trace", []),
            "intermediate_reasoning_results": react_result,
            "insights": react_result.get("answer", "")
        }
        
    except Exception as e:
        if state.get("logger"):
            await state["logger"](state.get("request_id", "unknown"), "REACT_REASONING", f"Error: {str(e)}")
        
        return {
            **state,
            "error": f"ReACT reasoning error: {str(e)}"
        }


async def cot_reasoning_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """
    Chain of Thought reasoning node for analytical queries
    """
    if state.get("logger"):
        await state["logger"](state.get("request_id", "unknown"), "COT_REASONING", "Starting Chain of Thought reasoning")
    
    try:
        cot_agent = create_reasoning_agent("cot")
        
        # Prepare data for CoT analysis
        analysis_data = {}
        for ref_key in state.get("tool_result_refs", {}).values():
            cached_data = get_cached_enhanced_result(ref_key)
            if cached_data:
                analysis_data.update(cached_data)
        
        reasoning_context = state.get("reasoning_context", {})
        analysis_type = reasoning_context.get("analysis_type", "general")
        
        cot_result = cot_agent.invoke({
            "query": state["user_query"],
            "data": analysis_data,
            "analysis_type": analysis_type,
            "steps": reasoning_context.get("custom_steps")
        })
        
        if state.get("logger"):
            await state["logger"](
                state.get("request_id", "unknown"), 
                "COT_REASONING", 
                f"CoT reasoning completed with {len(cot_result.get('reasoning_steps', []))} steps"
            )
        
        return {
            **state,
            "reasoning_trace": cot_result.get("reasoning_steps", []),
            "intermediate_reasoning_results": cot_result,
            "insights": cot_result.get("final_answer", ""),
            "multi_step_plan": cot_result.get("chain_of_thought_trace", [])
        }
        
    except Exception as e:
        if state.get("logger"):
            await state["logger"](state.get("request_id", "unknown"), "COT_REASONING", f"Error: {str(e)}")
        
        return {
            **state,
            "error": f"Chain of Thought reasoning error: {str(e)}"
        }


async def verification_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """
    Verification node to validate reasoning results and calculate confidence
    """
    if state.get("logger"):
        await state["logger"](state.get("request_id", "unknown"), "VERIFICATION", "Verifying reasoning results")
    
    try:
        reasoning_results = state.get("intermediate_reasoning_results", {})
        reasoning_trace = state.get("reasoning_trace", [])
        
        # Calculate confidence score based on reasoning quality
        confidence_score = await _calculate_confidence_score(
            state["user_query"], 
            reasoning_results, 
            reasoning_trace
        )
        
        # Verify logical consistency
        verification_result = await _verify_logical_consistency(
            state["user_query"],
            reasoning_results,
            state.get("tool_result_refs", {})
        )
        
        if state.get("logger"):
            await state["logger"](
                state.get("request_id", "unknown"), 
                "VERIFICATION", 
                f"Verification complete - Confidence: {confidence_score:.2f}"
            )
        
        return {
            **state,
            "confidence_score": confidence_score,
            "verification_results": verification_result
        }
        
    except Exception as e:
        if state.get("logger"):
            await state["logger"](state.get("request_id", "unknown"), "VERIFICATION", f"Verification error: {str(e)}")
        
        return {
            **state,
            "confidence_score": 0.5,  # Default moderate confidence
            "verification_results": {"error": str(e)}
        }


async def _calculate_confidence_score(query: str, reasoning_results: dict, reasoning_trace: list) -> float:
    """Calculate confidence score for reasoning results"""
    score = 0.5  # Base score
    
    # Factor 1: Length and depth of reasoning
    if reasoning_trace:
        trace_length = len(reasoning_trace)
        if trace_length >= 3:
            score += 0.2
        elif trace_length >= 2:
            score += 0.1
    
    # Factor 2: Success of reasoning process
    if reasoning_results.get("success", False):
        score += 0.2
    
    # Factor 3: Presence of supporting evidence
    if reasoning_results.get("answer") or reasoning_results.get("final_answer"):
        score += 0.1
    
    # Factor 4: No errors in reasoning
    if not reasoning_results.get("error"):
        score += 0.1
    
    # Factor 5: Reasoning completeness
    if "insights" in reasoning_results or "reasoning_summary" in reasoning_results:
        score += 0.1
    
    return min(1.0, score)


async def _verify_logical_consistency(query: str, reasoning_results: dict, data_refs: dict) -> dict:
    """Verify logical consistency of reasoning results"""
    verification = {
        "consistent": True,
        "issues": [],
        "recommendations": []
    }
    
    # Check if answers exist
    answer = reasoning_results.get("answer") or reasoning_results.get("final_answer")
    if not answer:
        verification["consistent"] = False
        verification["issues"].append("No final answer provided")
    
    # Check if reasoning trace exists for complex queries
    reasoning_trace = reasoning_results.get("reasoning_trace", [])
    if len(query.split()) > 10 and len(reasoning_trace) < 2:
        verification["issues"].append("Complex query but insufficient reasoning depth")
        verification["recommendations"].append("Consider using more detailed reasoning approach")
    
    return verification


def should_use_reasoning(state: EnhancedAgentState) -> Literal["meta_reasoning", "standard_planning"]:
    """Conditional function to decide if advanced reasoning is needed"""
    query = state["user_query"].lower()
    
    # Use reasoning for complex queries
    complex_indicators = [
        len(query.split()) > 15,  # Long queries
        "analyze" in query,
        "explain" in query,
        "why" in query,
        "how does" in query,
        "what factors" in query,
        query.count("and") > 2,  # Multiple conditions
        query.count("or") > 1
    ]
    
    if any(complex_indicators):
        return "meta_reasoning"
    
    return "standard_planning"


def route_reasoning_type(state: EnhancedAgentState) -> Literal["react_reasoning", "cot_reasoning", "standard_execution"]:
    """Route to appropriate reasoning type based on meta-reasoning decision"""
    reasoning_mode = state.get("reasoning_mode", "standard")
    
    if reasoning_mode == "react":
        return "react_reasoning"
    elif reasoning_mode == "cot":
        return "cot_reasoning"
    else:
        return "standard_execution"


def should_verify(state: EnhancedAgentState) -> Literal["verification", "finalize"]:
    """Decide if verification is needed"""
    reasoning_mode = state.get("reasoning_mode")
    
    if reasoning_mode in ["react", "cot"]:
        return "verification"
    
    return "finalize"


# Build enhanced workflow
def build_enhanced_workflow():
    """Build the enhanced workflow with reasoning capabilities"""
    workflow = StateGraph(EnhancedAgentState)
    
    # Add reasoning nodes
    workflow.add_node("meta_reasoning", meta_reasoning_node)
    workflow.add_node("react_reasoning", react_reasoning_node)
    workflow.add_node("cot_reasoning", cot_reasoning_node)
    workflow.add_node("verification", verification_node)
    
    # Import existing nodes (would need to adapt them to EnhancedAgentState)
    # workflow.add_node("standard_planning", planning_llm)
    # workflow.add_node("standard_execution", execute_tool_node)
    # ... other existing nodes
    
    # Set entry point with reasoning decision
    workflow.set_entry_point("meta_reasoning")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "meta_reasoning",
        route_reasoning_type,
        {
            "react_reasoning": "react_reasoning",
            "cot_reasoning": "cot_reasoning", 
            "standard_execution": "standard_execution"
        }
    )
    
    # Add verification routing
    workflow.add_conditional_edges(
        "react_reasoning",
        should_verify,
        {
            "verification": "verification",
            "finalize": END
        }
    )
    
    workflow.add_conditional_edges(
        "cot_reasoning", 
        should_verify,
        {
            "verification": "verification",
            "finalize": END
        }
    )
    
    # Connect verification to end
    workflow.add_edge("verification", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    enhanced_app = build_enhanced_workflow()
    
    # Test with complex reasoning query
    test_state = EnhancedAgentState(
        user_query="Analyze why our revenue dropped last month compared to the previous month, considering marketplace performance, payment modes, and regional trends. What factors might have contributed to this decline?",
        summarized_query=None,
        plan=None,
        tool_result_refs={},
        tool_result_schemas={},
        current_step_index=0,
        filters=None,
        final_result_ref=None,
        error=None,
        retry_count=0,
        logger=None,
        request_id="test_001",
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
        reasoning_mode=None,
        reasoning_trace=None,
        reasoning_context=None,
        intermediate_reasoning_results=None,
        confidence_score=None,
        reasoning_justification=None,
        multi_step_plan=None,
        verification_results=None
    )
    
    print("Testing enhanced reasoning workflow...")
    # result = enhanced_app.invoke(test_state)
    # print(f"Reasoning mode: {result.get('reasoning_mode')}")
    # print(f"Confidence: {result.get('confidence_score')}")
    # print(f"Insights: {result.get('insights', '')[:200]}...")