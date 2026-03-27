# CustomCalculationLLM Integration Guide

## Overview

The `CustomCalculationLLM` is a new ReAct (Reasoning + Acting) based LLM layer that generates and iteratively refines Python code for custom metric calculations. It works in tandem with the planning phase to handle complex metric generation requests that require dynamic Python code execution.

## Architecture

### Flow Diagram

```
User Query
    ↓
Planning LLM
    ↓
[Identifies query_type as "custom_metric_generation"]
    ↓
Data Extraction & Tool Execution
    ↓
Metric Processing Node
    ↓
Route Metric Processing (Decision Point)
    ├─→ [If custom_metric_generation] CustomCalculationLLM Node
    │                                        ↓
    │                                   ReAct Loop (1-3 iterations)
    │                                        ↓
    │                                   THOUGHT: Analyze intent
    │                                        ↓
    │                                   ACTION: Generate Python code
    │                                        ↓
    │                                   OBSERVATION: Execute & validate
    │                                        ↓
    │                                   REFLECTION: Check validity
    │                                        ↓
    │                                   [If valid] Return results
    │                                        ↓
    └─→ [Otherwise] Insight Generation
            ↓
        Final Analysis Output
```

## Components

### 1. CustomCalculationLLM Class

**Location:** `backend/llm_providers.py`

**Inheritance:** Extends `OpenRouterLLM`

#### Key Methods

- **`invoke(params: dict) -> dict`**
  - Main entry point for custom calculation
  - Orchestrates the ReAct loop
  - Parameters:
    - `query`: Original user query
    - `intent`: Intent from planning LLM (e.g., "Calculate net revenue per order")
    - `data`: DataFrame or raw data to operate on
    - `schema`: Available fields and their types
    - `date_range`: Time range context {start_date, end_date}
  - Returns:
    ```python
    {
        "success": bool,
        "final_result": any,
        "calculation_code": str,
        "iterations": int,
        "intent": str,
        "metadata": dict,
        "reasoning_history": list  # Full ReAct history
    }
    ```

#### ReAct Phases

1. **THOUGHT** - `_generate_thought()`
   - Analyzes the metric calculation request
   - Plans the calculation approach
   - Considers available columns, transformations, edge cases
   - Uses low temperature (0.3) for consistency

2. **ACTION** - `_generate_python_code()`
   - Generates executable Python code
   - Takes DataFrame `df` as input
   - Returns result in `result` variable
   - Includes error handling for edge cases
   - Uses temperature 0.2 for precision

3. **OBSERVATION** - `_execute_and_observe()`
   - Executes generated code
   - Collects results and metadata
   - Captures execution errors
   - Returns structured observation

4. **REFLECTION** - `_validate_result()`
   - Validates result meaningfulness
   - Type checking based on intent
   - NaN/Inf detection
   - Determines if iteration is needed

### 2. Workflow Integration

**Location:** `backend/workflow.py`

#### New Components Added

1. **Import Statement**
   ```python
   from llm_providers import planning_llm, filtering_llm, grouping_llm, insight_llm, custom_calculation_llm
   ```

2. **Node: `custom_calculation_node()`**
   - Async function that orchestrates custom metric calculation
   - Handles state management and caching
   - Emits step events for SSE updates
   - Calls `custom_calculation_llm.invoke()`

3. **Router: `route_metric_processing()`**
   - Decides routing after metric processing
   - Routes `custom_metric_generation` to `custom_calculation_node`
   - Routes other metric types to `insight_generation`

4. **Graph Updates**
   - Added node: `workflow.add_node("custom_calculation", custom_calculation_node)`
   - Added conditional edges: `metric_processing` → `route_metric_processing`
   - Result routing: `custom_calculation` → `insight_generation`

## Query Type Routing

### Planning Phase Identifications

The `PlanningLLM` identifies query types and routes them accordingly:

```python
query_type_mapping = {
    "schema_discovery": "Access data structure",
    "metric_analysis": "Calculate standard metrics (AOV, revenue, etc.)",
    "custom_metric_generation": "Calculate custom user-defined metrics",
    "comparison": "Compare groups or segments",
    "standard": "Generic data fetch"
}
```

### How Custom Metrics Are Identified

The planning prompt includes guidance:
```
- custom_metric_generation → pass intent to CustomCalculationLLM
```

Examples of triggers:
- "Custom metric: (total revenue - refunds) / order count"
- "Calculate net revenue per order"
- "What's the ratio of successful orders to attempts?"
- Any metric that requires custom Python logic

## Example Usage

### User Query
```
"Calculate the metric: (total revenue - refunds) / total orders for last 30 days"
```

### Planning Output
```json
{
    "summarized_query": "Net revenue per order last 30 days",
    "query_type": "custom_metric_generation",
    "steps": [
        {
            "id": "step1",
            "tool": "get_all_orders",
            "params": {
                "start_date": "2026-02-14 00:00:00",
                "end_date": "2026-03-15 23:59:59"
            },
            "depends_on": [],
            "save_as": "orders_raw"
        },
        {
            "id": "step2",
            "tool": "convert_to_df",
            "params": {"raw": "{{orders_raw}}"},
            "depends_on": ["step1"],
            "save_as": "orders_df"
        }
    ],
    "intent": "Net revenue per order last 30 days"
}
```

### CustomCalculationLLM ReAct Loop

**Iteration 1: THOUGHT**
```
"The calculation requires: (1) Sum of revenue column, (2) Sum of refund_amount 
column, (3) Total count of orders. I'll need to handle null values and ensure 
data is not empty before division to avoid ZeroDivisionError."
```

**Iteration 1: ACTION** (Generated Code)
```python
if df.empty:
    result = None
else:
    total_revenue = df['total_amount'].sum()
    total_refunds = df.get('refund_amount', pd.Series(0)).sum()
    net_revenue = total_revenue - total_refunds
    total_orders = df.shape[0]
    
    if total_orders > 0:
        result = net_revenue / total_orders
    else:
        result = None
```

**Iteration 1: OBSERVATION**
```
{
    "status": "executed",
    "result": 1245.50,
    "error": null,
    "metadata": {
        "execution_time_ms": 45,
        "memory_used_mb": 12
    }
}
```

**Iteration 1: REFLECTION**
```
Valid: True
Message: "Calculation valid and complete"
→ Exit loop, return result
```

### Final Output

```python
{
    "success": True,
    "final_result": 1245.50,
    "calculation_code": "if df.empty:\n    result = None\nelse:\n    ...",
    "iterations": 1,
    "intent": "Net revenue per order last 30 days",
    "metadata": {
        "execution_time_ms": 45,
        "memory_used_mb": 12
    },
    "reasoning_history": [
        {
            "iteration": 1,
            "thought": "The calculation requires...",
            "code": "if df.empty:\n    result = None...",
            "observation": {...}
        }
    ]
}
```

## Error Handling

### Iteration-Based Refinement

If a calculation fails or returns invalid results, the LLM automatically:
1. Reviews the error from previous iteration
2. Generates corrected code
3. Attempts again (up to 3 iterations max)

### Common Scenarios Handled

| Scenario | Handling |
|----------|----------|
| Empty DataFrame | Returns None |
| Missing Column | Error captured in OBSERVATION, code refined |
| Type Mismatch | Original code includes type conversions |
| NaN Result | Detected in REFLECTION, iteration continues |
| Division by Zero | Handled in generated code logic |

## Integration Points with Other Components

### Planning LLM
- Provides `intent` for CustomCalculationLLM
- Identifies `custom_metric_generation` query type
- Provides execution steps for data fetching

### Tool Registry
- Fetches data via `get_all_orders`, `convert_to_df`, etc.
- Provides DataFrame to CustomCalculationLLM

### Insight Generation LLM
- Receives metric results from CustomCalculationLLM
- Generates natural language analysis
- Provides business context and recommendations

### State Management
- Caches results in `RESULT_CACHE`
- Tracks iteration history
- Maintains state through SSE updates

## Configuration

### Environment Variables
```bash
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
OPENROUTER_SITE_URL=www.engineermonke.space
OPENROUTER_SITE_NAME=Order Analysis Workflow
```

### Tuning Parameters

In `CustomCalculationLLM.__init__()`:
```python
self.max_iterations = 3  # Maximum ReAct loop iterations
```

Adjust temperature for different behavior:
- `_generate_thought()`: 0.3 (low - consistent planning)
- `_generate_python_code()`: 0.2 (very low - precise code)
- `_validate_result()`: Fixed logic (no temperature)

## Testing

### Basic Test
```python
from llm_providers import custom_calculation_llm

result = custom_calculation_llm.invoke({
    "query": "Calculate average order value minus discount",
    "intent": "AOV minus average discount",
    "data": df,  # Your DataFrame
    "schema": {
        "total_amount": {"type": "float", "example": 1000.0},
        "discount": {"type": "float", "example": 50.0}
    },
    "date_range": {
        "start_date": "2026-03-01 00:00:00",
        "end_date": "2026-03-15 23:59:59"
    }
})

print(f"Success: {result['success']}")
print(f"Result: {result['final_result']}")
print(f"Iterations: {result['iterations']}")
```

## Performance Considerations

### Optimization Strategies

1. **Cache DataFrame operations** - Reuse processed data
2. **Limit iterations** - Currently max 3 iterations
3. **Temperature settings** - Lower temp = faster convergence
4. **Schema hints** - Comprehensive schema reduces code generation errors

### Monitoring

Track in logs:
- `[ReAct Iteration X/Y]` - Iteration progress
- `[VALIDATION]` - Validation results
- `[SUCCESS]` - Final success/failure status

## Future Enhancements

1. **Tool Integration** - Execute code via actual `execute_custom_calculation` tool
2. **Caching** - Store successful calculation templates
3. **Multi-metric** - Calculate multiple custom metrics in sequence
4. **Visualization** - Auto-generate charts for results
5. **Version Control** - Track calculation code versions
6. **A/B Testing** - Compare different calculation approaches

## Troubleshooting

### Issue: LLM keeps regenerating the same code

**Solution:**
- Review error messages in OBSERVATION phase
- Provide more detailed schema information
- Increase iteration limit temporarily for debugging

### Issue: Results seem incorrect

**Solution:**
- Check the generated code in `calculation_code` field
- Verify DataFrame columns match schema
- Review `reasoning_history` for thought process

### Issue: Timeout from OpenRouter

**Solution:**
- Check API key validity
- Rate limit may be reached - implement retry logic
- Review prompt size - large schemas increase latency

## API Behavior Notes

- **Max iterations**: 3 (prevents infinite loops)
- **Default temperature**: Varies by phase (0.2-0.3)
- **Response format**: Always returns structured dict
- **Caching**: Results cached with `cache_result()` utility
- **Async support**: Works in async workflow context

## Related Files

- `llm_providers.py` - CustomCalculationLLM implementation
- `workflow.py` - Node and routing logic
- `tools.py` - Tool registry and execution
- `app.py` - API endpoints and request handling
