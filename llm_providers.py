"""
LLM functions using OpenRouter API with Llama 3.1 70B Instruct
"""
import os
import requests
import json
from datetime import datetime, timedelta
import re


class OpenRouterLLM:
    """Base class for OpenRouter API calls"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "Order Analysis Workflow")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in .env file")
    
    def _call_api(self, messages: list, temperature: float = 0.7) -> str:
        """Make API call to OpenRouter"""
        try:
            response = requests.post(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                })
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API error: {e}")
            raise


class PlanningLLM(OpenRouterLLM):
    """Planning LLM - generates execution plan from natural language query"""
    
    def invoke(self, query: str) -> dict:
        """Generate execution plan from query"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""You are a query planning assistant for an e-commerce order management system.
Today's date is {current_date}.

User Query: "{query}"

Available Tools:
1. get_all_orders - Fetch order data for a date range
2. get_schema_info - Get schema/metadata about available fields and constraints
   - Optional param "field": Specify a field name (e.g., "payment_mode") to get only that field's info
   - Without "field": Returns complete schema for all fields

Your task is to create an execution plan in JSON format. Analyze the query and determine:
1. Query Type:
   - "schema_discovery": If asking about available fields, data structure, allowed values, date ranges, etc.
   - "comparison": If comparing two or more groups (marketplaces, payment modes, states, etc.)
   - "standard": Regular data fetch with optional filtering

2. Tool Selection:
   - Use "get_schema_info" for questions about data structure, available fields, enum values, constraints
     * If asking about a SPECIFIC field (e.g., "what are allowed values for payment_mode"), include "field" parameter
     * If asking generally (e.g., "what fields are available"), omit "field" parameter
   - Use "get_all_orders" for actual data queries

3. Extract date range (for data queries only - convert relative dates like "last 5 days" to absolute dates)

4. Determine if filtering/manipulation is needed after fetching data

Return ONLY a valid JSON object with this structure:

For schema discovery queries (specific field):
{{
  "query_type": "schema_discovery",
  "steps": [
    {{
      "id": "step1",
      "tool": "get_schema_info",
      "params": {{
        "entity": "orders",
        "field": "payment_mode"
      }},
      "depends_on": [],
      "save_as": "schema_info"
    }}
  ],
  "manipulation": {{
    "required": false,
    "type": null
  }},
  "base_params": {{}},
  "tool": "get_schema_info"
}}

For schema discovery queries (all fields):
{{
  "query_type": "schema_discovery",
  "steps": [
    {{
      "id": "step1",
      "tool": "get_schema_info",
      "params": {{
        "entity": "orders"
      }},
      "depends_on": [],
      "save_as": "schema_info"
    }}
  ],
  "manipulation": {{
    "required": false,
    "type": null
  }},
  "base_params": {{}},
  "tool": "get_schema_info"
}}

For data queries (standard or comparison):
{{
  "query_type": "standard" or "comparison",
  "steps": [
    {{
      "id": "step1",
      "tool": "get_all_orders",
      "params": {{
        "start_date": "YYYY-MM-DD HH:MM:SS",
        "end_date": "YYYY-MM-DD HH:MM:SS"
      }},
      "depends_on": [],
      "save_as": "orders_data"
    }}
  ],
  "manipulation": {{
    "required": true or false,
    "type": "filter"
  }},
  "base_params": {{
    "start_date": "YYYY-MM-DD HH:MM:SS",
    "end_date": "YYYY-MM-DD HH:MM:SS"
  }},
  "tool": "get_all_orders"
}}

Examples:
- "last 5 days" = start_date: 5 days ago at 00:00:00, end_date: today at 23:59:59
- "last week" = start_date: 7 days ago, end_date: today
- "yesterday" = start_date: yesterday 00:00:00, end_date: yesterday 23:59:59

Return ONLY the JSON, no other text."""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.3)
        
        try:
            # Extract JSON from response (in case LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response = response[json_start:json_end]
            
            plan = json.loads(response)
            return {
                "success": True,
                "plan": plan
            }
        except json.JSONDecodeError as e:
            print(f"Failed to parse plan JSON: {e}\nResponse: {response}")
            return {
                "success": False,
                "error": f"Failed to parse plan: {str(e)}"
            }


class FilteringLLM(OpenRouterLLM):
    """Filtering LLM - extracts filter parameters from natural language"""
    
    def invoke(self, params: dict) -> dict:
        """Generate filter parameters"""
        query = params["query"]
        schema = params.get("schema", {})
        
        # Build schema description with enum values
        schema_desc = "Available fields and their possible values:\n"
        for field, field_info in schema.items():
            schema_desc += f"- {field} ({field_info['type']})"
            if field_info.get('is_categorical') and field_info.get('enum'):
                schema_desc += f": Possible values = {field_info['enum'][:10]}"  # Show first 10 values
            schema_desc += f" (example: {field_info.get('example')})\n"
        
        prompt = f"""You are a filter extraction assistant for an e-commerce order system.

User Query: "{query}"

Data Schema:
{schema_desc}

Extract filter conditions from the query. Return ONLY a valid JSON object with this structure:
{{
  "filters": [
    {{
      "field": "field_name",
      "operator": "eq|ne|gt|lt|gte|lte|contains|in",
      "value": "value or [list of values] for 'in' operator"
    }}
  ]
}}

CRITICAL RULE: DO NOT create filters for date-related fields (order_date, created_at, etc.)
Date filtering is already handled by the API call parameters (start_date/end_date).
Only extract filters for non-date fields like:
- sku
- payment_mode
- marketplace
- order_status
- state
- city
- customer_name
- customer_email
- etc.

Operators:
- eq: equals
- ne: not equals
- gt/lt/gte/lte: greater than, less than, etc.
- contains: string contains
- in: value in list

Important: Use the EXACT field names and values from the schema above. Pay attention to capitalization and enum values. And, for SKU related queries: An sku 10510-455-7 means, sku 10510-455 of size 7. Hence, if size not mentioned in the sku, use "contains" operator on the sku field always.

Examples:
- "prepaid orders" → {{"field": "payment_mode", "operator": "eq", "value": "PrePaid"}}
- "open status" → {{"field": "order_status", "operator": "eq", "value": "Open"}}
- "from Karnataka" → {{"field": "state", "operator": "eq", "value": "Karnataka"}}
- "sku 12345" → {{"field": "sku", "operator": "eq", "value": "12345"}}

DO NOT INCLUDE:
- order_date filters (already handled by API)
- Any temporal filters (dates, times, etc.)

Return ONLY the JSON, no other text."""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.2)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response = response[json_start:json_end]
            
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse filter JSON: {e}\nResponse: {response}")
            return {"filters": []}


class GroupingLLM(OpenRouterLLM):
    """Grouping LLM - extracts comparison groups from query"""
    
    def invoke(self, params: dict) -> dict:
        """Identify comparison groups"""
        query = params["query"]
        
        prompt = f"""You are a comparison group extraction assistant for an e-commerce order system.

User Query: "{query}"

Your task is to identify what groups are being compared. 

IMPORTANT: The "filters" field contains POST-FETCH filters that will be applied AFTER data is fetched.
Do NOT include date range parameters (start_date, end_date) in filters - those are fetched first.
Only include categorical filters like payment_mode, marketplace, state, order_status, etc.

Return ONLY a valid JSON object with this structure:
{{
  "groups": [
    {{
      "group_id": "descriptive_id",
      "filters": {{
        "field_name": "value"
      }}
    }}
  ]
}}

Common comparison dimensions (for filters field):
- Marketplaces: Shopify13, Flipkart, Amazon, Myntra, etc.
- Payment modes: PrePaid, COD
- States: Karnataka, Maharashtra, Delhi, Tamil Nadu, etc.
- Order status: Open, Cancelled, Delivered, etc.

Examples:
Query: "Compare Shopify13 and Flipkart"
Response: {{
  "groups": [
    {{"group_id": "shopify", "filters": {{"marketplace": "Shopify13"}}}},
    {{"group_id": "flipkart", "filters": {{"marketplace": "Flipkart"}}}}
  ]
}}

Query: "Compare prepaid vs COD orders"
Response: {{
  "groups": [
    {{"group_id": "prepaid", "filters": {{"payment_mode": "PrePaid"}}}},
    {{"group_id": "cod", "filters": {{"payment_mode": "COD"}}}}
  ]
}}

Return ONLY the JSON, no other text."""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.3)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response = response[json_start:json_end]
            
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse grouping JSON: {e}\nResponse: {response}")
            return {"groups": []}


class InsightLLM(OpenRouterLLM):
    """Insight generation LLM - creates natural language summaries"""
    
    def invoke(self, params: dict) -> dict:
        """Generate natural language insights from comparison"""
        query = params.get("query", "")
        comparison = params.get("comparison", {})
        metrics = params.get("metrics", {})
        
        prompt = f"""You are a data insights analyst for an e-commerce order system.

User Query: "{query}"

Comparison Data:
{json.dumps(comparison, indent=2)}

Detailed Metrics by Group:
{json.dumps(metrics, indent=2)}

Generate a comprehensive, natural language analysis of this comparison. 

IMPORTANT: Format your response as numbered points ONLY. Each point should be on a new line.
Strictly use this exact format:
Point 1: [First insight with specific numbers and percentages]
Point 2: [Second insight...]
...and so on.

The comparison data may be either:
- "pairwise" (2 groups): Direct A vs B comparison with specific differences and percentages
- "multi_group" (3+ groups): Multiple groups with a baseline comparison and overall winners

Include:
- Clear comparison of order volumes with specific numbers and percentages
- Revenue comparison with specific amounts and growth/decline percentages
- Average order value analysis
- Key insights and trends
- Winner by different metrics (volume, revenue, avg value)
- For multi-group comparisons: highlight how each group compares to the baseline and overall rankings
- Additional interesting findings from the detailed metrics (payment mode distribution, order status, cities, etc.)

Make each point conversational, data-driven, and actionable.

Return your insights in the numbered point format described above."""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.7)
        
        return {"insights": response.strip()}


# Singleton instances
planning_llm = PlanningLLM()
filtering_llm = FilteringLLM()
grouping_llm = GroupingLLM()
insight_llm = InsightLLM()
