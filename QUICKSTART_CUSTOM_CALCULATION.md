# CustomCalculationLLM - Quick Start Guide

## Overview

The CustomCalculationLLM enables users to request complex, custom metric calculations through natural language queries. The system automatically:

1. **Plans** the calculation based on user intent
2. **Generates** Python code using ReAct pattern
3. **Executes** and validates the calculation
4. **Refines** if needed (up to 3 iterations)
5. **Analyzes** results with business context

## User-Facing Examples

### Example 1: Net Revenue Per Order

**User Query:**
```
"Calculate net revenue per order (total revenue minus refunds) for last 30 days"
```

**System Flow:**
1. Planning LLM identifies: `query_type = "custom_metric_generation"`
2. Execution plan fetches orders for last 30 days
3. CustomCalculationLLM ReAct Loop:
   - **THOUGHT**: "Need to sum revenue and refunds separately, then divide by order count"
   - **ACTION**: Generates code with null handling
   - **OBSERVATION**: Executes successfully  
   - **REFLECTION**: Validates result is numeric and non-negative
4. Returns metric: `₹1,245.50 per order`

**Response Structure:**
```json
{
    "success": true,
    "final_result": 1245.50,
    "calculation_code": "...",
    "iterations": 1,
    "insights": "Net revenue per order increased 15% compared to previous month..."
}
```

### Example 2: Custom Discount Impact

**User Query:**
```
"What's the average effective discount percentage? (total_discount / total_amount) * 100"
```

**Flow:**
1. Identifies custom metric needing formula execution
2. Generates Python code: `(df['discount'].sum() / df['total_amount'].sum()) * 100`
3. Executes: Result = `8.5%`
4. Analysis: "Effective discount rate of 8.5% impacts margin by approximately..."

### Example 3: Complex Ratio Metric

**User Query:**
```
"Calculate conversion efficiency: successful orders / total order attempts"
```

**Flow:**
1. Code generates order status comparison
2. Divides successful (Delivered/Completed) by all orders
3. Returns: `0.78` (78% conversion efficiency)
4. Provides insight about operational efficiency

## How Planning LLM Routes to CustomCalculationLLM

The Planning LLM detects custom metric requests by looking for:

### Trigger Patterns
- "Custom metric:" prefix
- Mathematical formulas with operators: `/`, `*`, `-`, `+`, `(`, `)`
- "Calculate X as formula" patterns
- Multi-step calculations not covered by standard metrics
- "Ratio", "proportion", "percentage", "average of" calculations

### Query Type Decision Tree
```
User Query
    ↓
Does it ask for aggregated metric (AOV, revenue, count)?
    ├─ Yes → query_type = "metric_analysis"
    └─ No ↓
Does it involve comparison between groups?
    ├─ Yes → query_type = "comparison"
    └─ No ↓
Does it ask for data structure/schema?
    ├─ Yes → query_type = "schema_discovery"
    └─ No ↓
Does it require custom Python logic?
    ├─ Yes → query_type = "custom_metric_generation" ✓
    └─ No → query_type = "standard"
```

## Integration with Existing Tools

### Built-in Support
The CustomCalculationLLM works seamlessly with:

- **Data Fetching**: `get_all_orders`, `convert_to_df`, filters
- **Tool Execution**: `execute_custom_calculation` for actual computation
- **Result Caching**: Stores results for insight generation
- **Insight Generation**: Provides metric context to InsightLLM

### Available DataFrame Columns

When CustomCalculationLLM generates code, it has access to these columns from orders data:

```python
# Financial Columns
'total_amount'       # Order total (float)
'discount'          # Discount amount (float)
'tax'               # Tax amount (float)
'shipping_charge'   # Shipping cost (float)
'refund_amount'     # Refund amount (float)

# Operational Columns
'order_id'          # Unique identifier (int)
'order_status'      # Status (string): Delivered, Cancelled, Open, etc.
'payment_mode'      # Payment type (string): PrePaid, COD
'marketplace'       # Platform (string): Shopify13, Flipkart, Amazon, etc.
'order_date'        # Date (timestamp)
'sku'              # Product SKU (string)

# Geographic Columns
'state'            # State (string): Karnataka, Maharashtra, etc.
'city'             # City name (string)
'country'          # Country (string)

# Courier Columns
'courier'          # Courier name (string)
```

## Expected Metrics from Calculations

### Financial Metrics
- Net revenue, COGS, profit margin ratios
- Average discount per order
- Effective tax rate
- Shipping cost as % of revenue

### Operational Metrics
- Order success rate / conversion rate
- Refund rate / return rate
- COD vs Prepaid ratio
- Average orders per day/month

### Geographic Metrics
- Revenue concentration by state/city
- Market share percentage
- Geographic growth rates

## Error Scenarios & Recovery

### Scenario: "Revenue Divided by Zero"

**Generation 1 (Fails):**
```python
result = df['total_amount'].sum() / df.shape[0]
# Error: Expected column not found, wrong denominator  
```

**LLM Learns & Refines:**
- Observes: "Column mismatch, need to check available columns"
- Iteration 2 generates better code with checks

**Generation 2 (Success):**
```python
if df.empty:
    result = None
else:
    result = df['total_amount'].sum() / df.shape[0]
```

### Scenario: "No Data for Date Range"

**LLM Handles:**
- Detects empty DataFrame
- Returns meaningful `None` or summary statistic
- Insight generator provides context: "No data available for this period"

## Workflow State Handling

### Input State from Workflow
```python
{
    "user_query": "Calculate net revenue per order",
    "plan": {
        "query_type": "custom_metric_generation",
        "steps": [...],
        "summarized_query": "Net revenue per order"
    },
    "tool_result_refs": {...},  # Data from execution
    "final_result_ref": "cache_key_to_df"
}
```

### Output State to Next Node
```python
{
    "metric_results": {
        "custom_metric": 1245.50,
        "calculation_code": "...",
        "iterations": 1
    },
    "aggregated_metrics": {
        "overall": {
            "custom_metric": 1245.50,
            ...
        }
    },
    "metrics_calculated": ["custom_metric"]
}
```

### Next Node Processing
InsightLLM receives the metric and generates:
- Business interpretation
- Performance context  
- Actionable recommendations
- Market analysis (if applicable)

## Common Patterns & Best Practices

### Pattern 1: Ratio Calculations
```
User: "What's the return rate?"
Generated: total_refunds / total_orders
Result: 0.12 (12%)
```

### Pattern 2: Weighted Averages
```
User: "Average order value weighted by marketplace volume"
Generated: (revenue_per_market * volume) / total_volume
```

### Pattern 3: Period-over-Period
```
User: "Growth rate vs last month"
Generated: (current_period - previous_period) / previous_period * 100
```

### Pattern 4: Segmented Analysis
```
User: "AOV by payment mode"
Generated: Multiple calculations with filtering
Result: {PrePaid: X, COD: Y}
```

## Performance Expectations

| Metric | Typical Value | Range |
|--------|--------------|-------|
| Code Generation Time | 1-2s | 0.5-3s |
| Code Execution Time | 0.5-1s | 0.1-5s |
| Validation Time | <100ms | <500ms |
| Total Query Time | 2-4s | 1-8s |
| Iterations Needed | 1-1.5 | 1-3 |
| Success Rate | 95%+ | 80-99% |

## Limitations & Known Issues

### Current Limitations
1. **Single DataFrame Only** - Works with one input DataFrame
2. **Python-only** - Cannot use external libraries beyond pandas/numpy
3. **Real-time** - Cannot do time-series calculations across dates
4. **Complex Logic** - Very complex business logic may need refinement

### Workarounds
- For multiple sources: Use filtering on single DataFrame
- For advanced libraries: Pre-process in tools, pass to LLM
- For time series: Fetch separate periods, calculate separately

## Debugging

### Enable Debug Logs
```python
# In custom_calculation_node
print(f"[DEBUG] Metric calculation result: {result}")
print(f"[DEBUG] Iterations performed: {result.get('iterations')}")
print(f"[DEBUG] Reasoning history: {result.get('reasoning_history')}")
```

### Review Reasoning History
```json
{
    "reasoning_history": [
        {
            "iteration": 1,
            "thought": "Analysis of calculation needs",
            "code": "Generated Python code",
            "observation": {
                "status": "executed",
                "result": 1245.50,
                "error": null
            }
        }
    ]
}
```

### Common Issues

**Issue**: Metric returns None
**Debug**: Check if data is empty, columns exist
**Fix**: Ensure date range has data, columns are named correctly

**Issue**: Wrong type of result (int vs float)
**Debug**: Check calculation logic
**Fix**: Add explicit type conversion in regenerated code

**Issue**: Multiple iterations needed (>2)
**Debug**: Complex calculation or data validation needs
**Fix**: Simplify request or provide better schema info

## API Example

### Request
```curl
POST /api/analyze
Content-Type: application/json

{
    "query": "Calculate net revenue per order (total - refunds) for last 30 days",
    "mode": "custom_metric"
}
```

### Response
```json
{
    "success": true,
    "metric_type": "custom_metric_generation",
    "result": {
        "final_result": 1245.50,
        "unit": "INR",
        "iterations": 1,
        "calculation_code": "if df.empty:\n    result = None\nelse:\n    ...",
        "reasoning_history": [...]
    },
    "insights": {
        "analysis": "Net revenue per order of ₹1,245.50 represents...",
        "recommendations": "..."
    }
}
```

## See Also

- [CUSTOM_CALCULATION_LLM_GUIDE.md](CUSTOM_CALCULATION_LLM_GUIDE.md) - Detailed technical documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [backend/llm_providers.py](llm_providers.py) - CustomCalculationLLM source code
- [backend/workflow.py](workflow.py) - Workflow integration code
