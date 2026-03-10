"""
Reasoning API routes for enhanced query processing
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, Literal
from pydantic import BaseModel
import uuid
import json

# Import reasoning components
from reasoning_agents import create_reasoning_agent
from enhanced_planning_llm import reasoning_planning_llm
from utils.connection_manager import manager

router = APIRouter(prefix="/reasoning", tags=["reasoning"])


class ReasoningQueryRequest(BaseModel):
    query: str
    reasoning_mode: Optional[Literal["auto", "react", "cot", "standard"]] = "auto"
    max_reasoning_steps: Optional[int] = 10
    include_trace: Optional[bool] = True
    context: Optional[dict] = None


class ReasoningPlanRequest(BaseModel):
    query: str
    reasoning_mode: Optional[Literal["auto", "react", "cot", "standard"]] = "auto"
    include_complexity_analysis: Optional[bool] = True


@router.post('/analyze-complexity')
async def analyze_query_complexity(request: ReasoningQueryRequest):
    """
    Analyze the complexity of a query to determine optimal reasoning approach
    
    This endpoint helps determine:
    - Whether advanced reasoning is needed
    - Which reasoning approach would be best
    - Expected complexity and challenges
    """
    try:
        # Use meta-reasoning agent to analyze complexity
        meta_agent = create_reasoning_agent("meta")
        
        analysis = meta_agent.invoke({
            "query": request.query,
            "context": request.context or {},
            "available_reasoning_types": ["react", "cot", "standard"]
        })
        
        return {
            "success": True,
            "query": request.query,
            "complexity_analysis": analysis,
            "recommendations": {
                "reasoning_approach": analysis.get("recommended_approach"),
                "justification": analysis.get("justification"),
                "expected_steps": request.max_reasoning_steps if analysis.get("recommended_approach") != "standard" else 1
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )


@router.post('/plan-with-reasoning')
async def create_reasoning_plan(request: ReasoningPlanRequest):
    """
    Generate an execution plan with reasoning capabilities
    
    Creates a detailed plan that incorporates:
    - Step-by-step reasoning approach
    - Verification checkpoints
    - Intermediate insight generation
    - Confidence tracking
    """
    try:
        # Determine reasoning mode
        reasoning_mode = request.reasoning_mode
        if reasoning_mode == "auto":
            # Use meta-reasoning to decide
            meta_agent = create_reasoning_agent("meta")
            meta_result = meta_agent.invoke({
                "query": request.query,
                "context": {},
                "available_reasoning_types": ["react", "cot", "standard"]
            })
            reasoning_mode = meta_result.get("recommended_approach", "standard")
        
        # Generate reasoning-enhanced plan
        plan_result = reasoning_planning_llm.invoke({
            "query": request.query,
            "reasoning_mode": reasoning_mode,
            "schema": {}  # Would typically include actual schema
        })
        
        response_data = {
            "success": True,
            "query": request.query,
            "reasoning_mode": reasoning_mode,
            "plan": plan_result.get("plan", {}),
            "reasoning_metadata": plan_result.get("reasoning_metadata", {})
        }
        
        if request.include_complexity_analysis:
            response_data["complexity_analysis"] = plan_result.get("reasoning_metadata", {}).get("complexity_analysis", {})
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )


@router.post('/react-reasoning')
async def process_with_react(request: ReasoningQueryRequest):
    """
    Process a query using ReACT (Reasoning + Acting) approach
    
    Best for:
    - Complex queries requiring multiple steps
    - Queries that need tool usage and observation
    - Exploratory analysis where the path isn't predetermined
    """
    try:
        request_id = str(uuid.uuid4())[:8]
        
        # Create logger for reasoning trace
        async def reasoning_logger(req_id: str, step: str, message: str):
            await manager.log_request_step(req_id, step, message)
        
        await manager.log_request_start(request_id, f"ReACT: {request.query}")
        
        # Create ReACT agent
        react_agent = create_reasoning_agent("react")
        
        # Available tools for ReACT agent (simplified for demo)
        available_tools = [
            {
                "name": "get_orders",
                "description": "Fetch order data for date range",
                "parameters": {"start_date": "string", "end_date": "string"}
            },
            {
                "name": "calculate_metrics", 
                "description": "Calculate business metrics",
                "parameters": {"data": "object", "metrics": "array"}
            },
            {
                "name": "analyze_trends",
                "description": "Analyze temporal trends",
                "parameters": {"data": "object", "time_dimension": "string"}
            }
        ]
        
        # Execute ReACT reasoning
        result = react_agent.invoke({
            "query": request.query,
            "available_tools": available_tools,
            "context": request.context or {},
            "goal": f"Comprehensively answer: {request.query}"
        })
        
        await manager.log_request_step(
            request_id, 
            "REACT_COMPLETE", 
            f"Completed ReACT reasoning with {len(result.get('reasoning_trace', []))} steps"
        )
        
        response_data = {
            "success": True,
            "request_id": request_id,
            "query": request.query,
            "reasoning_mode": "react",
            "answer": result.get("answer", ""),
            "confidence_estimation": "ReACT confidence calculation would go here"
        }
        
        if request.include_trace:
            response_data["reasoning_trace"] = result.get("reasoning_trace", [])
            response_data["reasoning_summary"] = result.get("reasoning_summary", "")
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )


@router.post('/cot-reasoning') 
async def process_with_cot(request: ReasoningQueryRequest):
    """
    Process a query using Chain of Thought reasoning
    
    Best for:
    - Analytical queries requiring step-by-step logic
    - Queries asking "why" or "how"
    - Comparative analysis
    - Trend analysis and pattern recognition
    """
    try:
        request_id = str(uuid.uuid4())[:8]
        
        await manager.log_request_start(request_id, f"CoT: {request.query}")
        
        # Create CoT agent
        cot_agent = create_reasoning_agent("cot")
        
        # Determine analysis type
        analysis_type = _determine_cot_analysis_type(request.query)
        
        # Execute Chain of Thought reasoning
        result = cot_agent.invoke({
            "query": request.query,
            "data": request.context or {},
            "analysis_type": analysis_type
        })
        
        await manager.log_request_step(
            request_id,
            "COT_COMPLETE",
            f"Completed CoT reasoning with {len(result.get('reasoning_steps', []))} steps"
        )
        
        response_data = {
            "success": True,
            "request_id": request_id, 
            "query": request.query,
            "reasoning_mode": "cot",
            "analysis_type": analysis_type,
            "answer": result.get("final_answer", ""),
            "key_insights": _extract_key_insights(result)
        }
        
        if request.include_trace:
            response_data["reasoning_steps"] = result.get("reasoning_steps", [])
            response_data["chain_of_thought_trace"] = result.get("chain_of_thought_trace", [])
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )


@router.post('/compare-reasoning')
async def compare_reasoning_approaches(request: ReasoningQueryRequest):
    """
    Compare different reasoning approaches for the same query
    
    Useful for:
    - Understanding which approach works best for different query types
    - Research and optimization
    - Demonstrating reasoning capabilities
    """
    try:
        request_id = str(uuid.uuid4())[:8]
        await manager.log_request_start(request_id, f"Compare: {request.query}")
        
        results = {}
        
        # Test with standard approach (if applicable)
        if len(request.query.split()) <= 15:  # Simple enough for standard
            try:
                # This would use your existing planning LLM
                results["standard"] = {
                    "approach": "standard",
                    "result": "Standard planning approach result",
                    "execution_time": "~1-2 seconds",
                    "best_for": "Simple, direct queries"
                }
            except Exception as e:
                results["standard"] = {"error": str(e)}
        
        # Test with ReACT
        try:
            react_agent = create_reasoning_agent("react")
            react_result = react_agent.invoke({
                "query": request.query,
                "available_tools": [],  # Simplified for comparison
                "context": request.context or {},
                "goal": f"Answer: {request.query}"
            })
            
            results["react"] = {
                "approach": "react",
                "result": react_result.get("answer", ""),
                "steps": len(react_result.get("reasoning_trace", [])),
                "best_for": "Complex queries requiring multiple tools",
                "reasoning_trace_length": len(react_result.get("reasoning_trace", []))
            }
        except Exception as e:
            results["react"] = {"error": str(e)}
        
        # Test with CoT
        try:
            cot_agent = create_reasoning_agent("cot")
            cot_result = cot_agent.invoke({
                "query": request.query,
                "data": request.context or {},
                "analysis_type": _determine_cot_analysis_type(request.query)
            })
            
            results["cot"] = {
                "approach": "cot",
                "result": cot_result.get("final_answer", ""),
                "steps": len(cot_result.get("reasoning_steps", [])),
                "best_for": "Analytical queries requiring logical reasoning",
                "reasoning_steps_count": len(cot_result.get("reasoning_steps", []))
            }
        except Exception as e:
            results["cot"] = {"error": str(e)}
        
        # Recommend best approach
        recommendation = _recommend_best_approach(request.query, results)
        
        return {
            "success": True,
            "request_id": request_id,
            "query": request.query,
            "comparison_results": results,
            "recommendation": recommendation,
            "summary": {
                "approaches_tested": len([k for k, v in results.items() if "error" not in v]),
                "approaches_failed": len([k for k, v in results.items() if "error" in v])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )


@router.get('/examples')
async def get_reasoning_examples():
    """
    Get examples of queries that work well with different reasoning approaches
    """
    return {
        "react_examples": [
            {
                "query": "Analyze why our Flipkart sales dropped last month and suggest 3 action items",
                "reasoning": "Requires multiple tools: get orders, calculate metrics, analyze trends, external research"
            },
            {
                "query": "Find the top 3 underperforming states and create a intervention plan for each",
                "reasoning": "Complex multi-step process requiring data gathering, analysis, and strategic thinking"
            }
        ],
        "cot_examples": [
            {
                "query": "Why might prepaid orders have higher AOV than COD orders?",
                "reasoning": "Requires step-by-step logical analysis of customer behavior and payment psychology"
            },
            {
                "query": "Compare the revenue impact of festival seasons vs regular periods",
                "reasoning": "Systematic comparative analysis with multiple factors to consider"
            }
        ],
        "standard_examples": [
            {
                "query": "Show me orders from last 7 days with COD payment",
                "reasoning": "Simple data retrieval, no complex reasoning needed"
            },
            {
                "query": "Calculate total revenue for Karnataka in October",
                "reasoning": "Direct calculation, straightforward query"
            }
        ]
    }


def _determine_cot_analysis_type(query: str) -> str:
    """Determine the appropriate CoT analysis type"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    elif any(word in query_lower for word in ["trend", "over time", "growth", "decline"]):
        return "trend"
    elif any(word in query_lower for word in ["distribution", "breakdown", "split", "across"]):
        return "distribution"
    elif any(word in query_lower for word in ["why", "cause", "reason", "factors"]):
        return "causal"
    else:
        return "general"


def _extract_key_insights(cot_result: dict) -> list:
    """Extract key insights from CoT reasoning result"""
    insights = []
    
    reasoning_steps = cot_result.get("reasoning_steps", [])
    for step in reasoning_steps:
        step_insights = step.get("insights", {})
        for key, value in step_insights.items():
            insights.append({"step": step.get("step", ""), "insight": f"{key}: {value}"})
    
    return insights[:5]  # Return top 5 insights


def _recommend_best_approach(query: str, results: dict) -> dict:
    """Recommend the best reasoning approach based on results"""
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not successful_results:
        return {"recommendation": "standard", "reason": "All reasoning approaches failed"}
    
    # Simple scoring based on query characteristics
    query_lower = query.lower()
    scores = {}
    
    for approach in successful_results.keys():
        score = 0
        
        if approach == "react":
            if len(query.split()) > 15:  # Complex queries
                score += 3
            if any(word in query_lower for word in ["analyze", "suggest", "plan", "find"]):
                score += 2
        
        elif approach == "cot":
            if any(word in query_lower for word in ["why", "how", "compare", "explain"]):
                score += 3
            if "vs" in query_lower or "versus" in query_lower:
                score += 2
        
        elif approach == "standard":
            if len(query.split()) <= 10:  # Simple queries
                score += 2
        
        scores[approach] = score
    
    best_approach = max(scores, key=scores.get)
    
    return {
        "recommendation": best_approach,
        "reason": f"Best suited based on query characteristics (score: {scores[best_approach]})",
        "scores": scores
    }