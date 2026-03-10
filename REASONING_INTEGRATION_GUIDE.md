# Reasoning Integration Guide

This guide explains how to integrate and use the enhanced reasoning capabilities (ReACT, Chain of Thought, and Meta-Reasoning) in your order analysis workflow.

## Overview

The reasoning enhancement adds three main capabilities to your existing system:

1. **ReACT (Reasoning + Acting)**: Multi-step reasoning with tool usage for complex queries
2. **Chain of Thought (CoT)**: Step-by-step analytical reasoning for logical analysis  
3. **Meta-Reasoning**: Automatic selection of the best reasoning approach

## Architecture

```
User Query → Meta-Reasoning → [ReACT | CoT | Standard] → Verification → Enhanced Results
```

## Quick Integration Steps

### 1. Update Main App

Add the reasoning router to your main app:

```python
# app.py
from routes.reasoning import router as reasoning_router

app.include_router(reasoning_router)
```

### 2. Update Existing Query Endpoint

Enhance your existing `/query` endpoint to use reasoning capabilities:

```python
# routes/query.py - Add to existing process_query function

@router.post('/query')
async def process_query(request: QueryRequest):
    # Add reasoning mode parameter to your QueryRequest model
    reasoning_mode = getattr(request, 'reasoning_mode', 'auto')
    
    if reasoning_mode != 'standard':
        # Use enhanced workflow with reasoning
        from enhanced_workflow import build_enhanced_workflow
        from enhanced_planning_llm import reasoning_planning_llm
        
        enhanced_app = build_enhanced_workflow()
        
        # Create enhanced initial state
        enhanced_state = EnhancedAgentState(
            user_query=request.query,
            # ... other fields
            reasoning_mode=reasoning_mode if reasoning_mode != 'auto' else None
        )
        
        result = enhanced_app.invoke(enhanced_state)
        
        # Return enhanced response with reasoning trace
        return {
            "success": True,
            "query": request.query,
            "reasoning_mode": result.get("reasoning_mode"),
            "confidence_score": result.get("confidence_score"),
            "insights": result.get("insights"),
            "reasoning_trace": result.get("reasoning_trace") if request.include_trace else None
        }
    
    # Fall back to existing standard workflow
    # ... existing code
```

### 3. Update Request Models

Add reasoning parameters to your request models:

```python
# models.py
from typing import Literal, Optional

class QueryRequest(BaseModel):
    query: str
    reasoning_mode: Optional[Literal["auto", "react", "cot", "standard"]] = "auto"
    include_trace: Optional[bool] = False
    max_reasoning_steps: Optional[int] = 10
```

## New API Endpoints

### Query Complexity Analysis
```http
POST /reasoning/analyze-complexity
Content-Type: application/json

{
    "query": "Why did our revenue drop last month compared to the previous month?",
    "reasoning_mode": "auto"
}
```

**Response:**
```json
{
    "success": true,
    "complexity_analysis": {
        "recommended_approach": "cot",
        "justification": "Query requires causal analysis with step-by-step reasoning",
        "complexity_score": 7
    }
}
```

### ReACT Reasoning
```http
POST /reasoning/react-reasoning
Content-Type: application/json

{
    "query": "Analyze our top 3 underperforming states and suggest action items",
    "include_trace": true
}
```

### Chain of Thought Reasoning  
```http
POST /reasoning/cot-reasoning
Content-Type: application/json

{
    "query": "Compare prepaid vs COD orders - what factors might cause the differences?",
    "include_trace": true
}
```

### Reasoning Comparison
```http
POST /reasoning/compare-reasoning
Content-Type: application/json

{
    "query": "Why are Flipkart orders performing better than Amazon orders?",
    "include_trace": true
}
```

## When to Use Each Reasoning Approach

### ReACT (Reasoning + Acting)
**Best for:**
- Complex queries requiring multiple tools/data sources
- Exploratory analysis where the solution path isn't clear
- Action-oriented queries ("find", "suggest", "plan")
- Queries that might need iterative refinement

**Examples:**
- "Find the root cause of declining revenues and suggest 3 action items"
- "Analyze market trends and recommend new marketplaces to focus on"

### Chain of Thought (CoT)
**Best for:**  
- Analytical queries requiring logical step-by-step reasoning
- "Why" and "how" questions
- Comparative analysis
- Hypothesis formation and testing

**Examples:**
- "Why do prepaid orders have higher AOV than COD orders?"
- "How do seasonal patterns affect our different product categories?"

### Standard Planning
**Best for:**
- Simple data retrieval queries
- Direct calculations
- Well-defined analysis patterns

**Examples:**
- "Show me orders from last 7 days"
- "Calculate total revenue for Karnataka in October"

## Integration Patterns

### Pattern 1: Auto-Select Reasoning
Let the system automatically choose the best approach:

```python
result = await process_query({
    "query": user_query,
    "reasoning_mode": "auto"
})
```

### Pattern 2: Explicit Reasoning Mode
Force a specific reasoning approach:

```python  
result = await process_query({
    "query": user_query,
    "reasoning_mode": "cot",
    "include_trace": True
})
```

### Pattern 3: Comparison Mode
Compare different approaches for the same query:

```python
comparison = await compare_reasoning_approaches({
    "query": user_query,
    "include_trace": True
})

best_approach = comparison["recommendation"]["recommendation"]
```

### Pattern 4: Progressive Enhancement
Start with standard, upgrade to reasoning if needed:

```python
# Try standard first
standard_result = await process_query({
    "query": user_query,
    "reasoning_mode": "standard"
})

# If insufficient confidence, upgrade to reasoning
if standard_result.get("confidence_score", 1.0) < 0.7:
    enhanced_result = await process_query({
        "query": user_query, 
        "reasoning_mode": "auto"
    })
```

## Configuration Options

### Environment Variables
```bash
# Add to your .env file
REASONING_MAX_ITERATIONS=10
REASONING_TEMPERATURE=0.3
REASONING_VERIFICATION_ENABLED=true
REASONING_CONFIDENCE_THRESHOLD=0.7
```

### Runtime Configuration
```python
# Customize reasoning behavior
reasoning_config = {
    "max_iterations": 8,
    "temperature": 0.2,
    "include_verification": True,
    "confidence_threshold": 0.8
}
```

## Response Format

Enhanced responses include additional reasoning metadata:

```json
{
    "success": true,
    "query": "Original user query",
    "reasoning_mode": "cot",
    "confidence_score": 0.85,
    "answer": "Final answer to the query",
    "insights": "Key business insights",
    "reasoning_trace": [
        {
            "step": "Identify comparison entities",  
            "reasoning": "Step-by-step thought process",
            "insights": {"key": "value"}
        }
    ],
    "verification_results": {
        "consistent": true,
        "issues": [],
        "recommendations": []
    },
    "metadata": {
        "reasoning_steps": 4,
        "execution_time": "3.2s",
        "complexity_score": 7
    }
}
```

## Frontend Integration

### React Example
```typescript
// Enhanced query hook
const useEnhancedQuery = () => {
    const [results, setResults] = useState(null);
    const [reasoning, setReasoning] = useState(null);
    
    const processQuery = async (query: string, mode: 'auto' | 'react' | 'cot' = 'auto') => {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query,
                reasoning_mode: mode,
                include_trace: true
            })
        });
        
        const data = await response.json();
        setResults(data);
        setReasoning(data.reasoning_trace);
    };
    
    return { results, reasoning, processQuery };
};
```

### Reasoning Trace Visualization
```typescript
const ReasoningTrace = ({ trace }: { trace: ReasoningStep[] }) => (
    <div className="reasoning-trace">
        {trace.map((step, index) => (
            <div key={index} className="reasoning-step">
                <h4>Step {index + 1}: {step.step}</h4>
                <p>{step.reasoning}</p>
                {step.insights && (
                    <div className="insights">
                        {Object.entries(step.insights).map(([key, value]) => (
                            <span key={key} className="insight-tag">
                                {key}: {value}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        ))}
    </div>
);
```

## Testing and Validation

### Unit Tests
```python
# test_reasoning.py
def test_complexity_analysis():
    result = analyze_query_complexity("Why did revenue drop?")
    assert result["recommended_approach"] in ["react", "cot", "standard"]
    assert "justification" in result

def test_react_reasoning():
    result = react_reasoning("Find underperforming products and suggest fixes")
    assert result["success"] is True
    assert len(result["reasoning_trace"]) > 0

def test_cot_reasoning():
    result = cot_reasoning("Compare payment modes - what drives the differences?")
    assert result["analysis_type"] == "comparison"
    assert len(result["reasoning_steps"]) >= 3
```

### Integration Tests
```python
def test_enhanced_workflow():
    state = EnhancedAgentState(
        user_query="Complex analytical query",
        reasoning_mode="auto"
    )
    
    result = enhanced_workflow.invoke(state)
    assert result["reasoning_mode"] in ["react", "cot", "standard"]
    assert result["confidence_score"] >= 0.0
```

## Performance Considerations

### Optimization Strategies
1. **Caching**: Cache reasoning results for similar queries
2. **Parallel Processing**: Run verification in parallel with main reasoning
3. **Early Termination**: Stop reasoning when confidence threshold is reached
4. **Tool Optimization**: Optimize individual tools for reasoning workloads

### Monitoring
```python
# Add metrics to track reasoning performance
metrics = {
    "reasoning_mode_distribution": Counter(),
    "average_reasoning_steps": Histogram(),
    "confidence_score_distribution": Histogram(),
    "reasoning_execution_time": Histogram()
}
```

## Troubleshooting

### Common Issues

1. **High Reasoning Overhead**
   - Solution: Use complexity analysis to avoid unnecessary reasoning
   - Set appropriate thresholds for auto-mode

2. **Inconsistent Results**
   - Solution: Enable verification and increase confidence thresholds
   - Review reasoning traces for logical gaps

3. **Memory Usage**
   - Solution: Implement proper caching with TTL
   - Clear reasoning traces for completed requests

### Debug Mode
```python
# Enable detailed logging for reasoning debugging
REASONING_DEBUG = True

if REASONING_DEBUG:
    # Log each reasoning step
    # Save intermediate results
    # Track decision points
```

## Future Enhancements

### Planned Features
1. **Learning from Feedback**: Improve reasoning based on user feedback
2. **Domain-Specific Reasoning**: E-commerce specific reasoning patterns
3. **Multi-Modal Reasoning**: Incorporate charts and visualizations
4. **Collaborative Reasoning**: Multiple agents working together

### Extension Points
```python
# Custom reasoning agents
class CustomBusinessReasoningAgent(ReasoningAgent):
    def invoke(self, params):
        # Domain-specific reasoning logic
        pass

# Custom verification methods  
class BusinessMetricVerifier:
    def verify_business_logic(self, results):
        # Custom business rule validation
        pass
```

This enhanced reasoning system transforms your order analysis workflow from simple data retrieval to intelligent analytical reasoning, providing deeper insights and more comprehensive answers to complex business questions.