"""
LLM functions using OpenRouter API with Llama 3.1 70B Instruct
"""
import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field  # Optional but recommended for schema clarity

from google import genai
from google.genai import types



class OpenRouterLLM:
    """Base class for OpenRouter API calls"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "www.engineermonke.space")
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
    
    def _call_api_with_tools(self, messages: list, tools: list, temperature: float = 0.7) -> dict:
        """Make API call to OpenRouter with tool calling support"""
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
                    "reasoning": {"enabled": True},
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": "auto"
                })
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]
        
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API error: {e}")
            raise

class ExecutionPlan(BaseModel):
    """Structured schema for the planning output"""
    summarized_query: str = Field(..., description="4-5 word summary of the user query")
    query_type: str = Field(..., description="One of: metric_analysis|custom_metric_generation|schema_discovery|standard|comparison")
    steps: list[Dict[str, Any]] = Field(default_factory=list, description="List of execution steps with tool, params, depends_on, save_as")
    manipulation: Dict[str, Any] = Field(default_factory=dict, description="Whether manipulation (filter etc.) is required")
    base_params: Dict[str, str] = Field(default_factory=dict, description="Base start_date and end_date")
    tool: str = Field(..., description="Primary base tool (usually get_all_orders)")


class GeminiLLM:
    """Base class for Gemini calls using the modern Google GenAI SDK"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-1.5-flash if preferred

        if not self.api_key:
            raise ValueError("GEMINI_KEY environment variable is missing.")

        self.client = genai.Client(api_key=self.api_key)

    def _generate_content(
        self,
        prompt: str,
        response_schema: Optional[dict] = None,
        temperature: float = 0.2,
        response_mime_type: Optional[str] = "application/json",
    ) -> str:
        """Core method to generate content with optional structured JSON output"""
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=4096,
        )

        if response_mime_type:
            config.response_mime_type = response_mime_type

        if response_schema:
            config.response_schema = response_schema

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )

            if not response.text:
                raise ValueError("Empty response from Gemini")

            return response.text.strip()

        except Exception as e:
            print(f"[ERROR] Gemini API error: {e}")
            raise


class PlanningLLM(GeminiLLM):
    """Planning LLM - generates execution plan from natural language query"""

    def _generate_fallback_summary(self, query: str) -> str:
        """Simple keyword-based fallback summary"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return "Compare data groups"
        elif any(word in query_lower for word in ['aov', 'average order', 'order value']):
            return "Calculate average order value"
        elif any(word in query_lower for word in ['revenue', 'sales', 'total amount']):
            return "Calculate total revenue"
        elif any(word in query_lower for word in ['distribution', 'breakdown', 'split']):
            return "Analyze data distribution"
        elif any(word in query_lower for word in ['schema', 'fields', 'structure']):
            return "Explore data schema"
        elif any(word in query_lower for word in ['sku', 'product']):
            return "Analyze product data"
        else:
            return "Analyze orders data"

    def _get_execution_plan_schema(self) -> dict:
        """Return JSON schema for structured output (can also use ExecutionPlan.model_json_schema())"""
        return ExecutionPlan.model_json_schema()

    def invoke(self, query: str) -> dict:
        """Generate execution plan from natural language query"""
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        yesterday = today - timedelta(days=1)
        five_days_ago = today - timedelta(days=5)
        thirty_days_ago = today - timedelta(days=30)

        prompt = f""" You are an expert query planning assistant for an e-commerce order management system.
                    
                    Today's date is {current_date}.
                    
                    CRITICAL: Always use actual dates in YYYY-MM-DD HH:MM:SS format, never placeholders.
                    - "last 5 days": "{five_days_ago.strftime('%Y-%m-%d')} 00:00:00" to "{today.strftime('%Y-%m-%d')} 23:59:59"
                    - "yesterday": "{yesterday.strftime('%Y-%m-%d')} 00:00:00" to "{yesterday.strftime('%Y-%m-%d')} 23:59:59"
                    - For chupps.com → "marketplace": "Shopify"

                    User Query: "{query}"

                    Available tools (use these names only):
                    get_all_orders, apply_filters, get_schema_info, convert_to_df, execute_custom_calculation,
                    get_aov, get_total_revenue, get_order_count, get_order_status_distribution,
                    get_payment_mode_distribution, get_marketplace_distribution, get_state_wise_distribution,
                    get_city_wise_distribution, get_courier_distribution, get_average_discount,
                    get_average_shipping_charge, get_average_tax, get_statistical_summary,
                    get_percentile, get_top_percentile, get_bottom_percentile, get_correlation_matrix,
                    get_conversion_rate, get_cod_vs_prepaid_metrics, get_geographic_insights, get_common_metrics

                    QUERY TYPE GUIDELINES:
                    - schema_discovery → use get_schema_info
                    - metric_analysis → use specific metric tools + convert_to_df when needed
                    - custom_metric_generation → pass intent to CustomCalculationLLM
                    - comparison → often use get_cod_vs_prepaid_metrics or multiple metric calls
                    - standard → simple fetch + optional filters

                    Prefer built-in metric tools over custom calculation when possible.
                    Use apply_filters early for optimization when specific conditions are mentioned.

                    Template for generated plan:
                    {{
                    "summarized_query": "4-5 word summary",
                        "query_type": "metric_analysis|custom_metric_generation|schema_discovery|standard|comparison",
                    "steps": [
                        {{
                        "id": "step1",
                        "tool": "get_all_orders",
                        "params": {{
                            "start_date": "2026-03-05 00:00:00",
                            "end_date": "2026-03-10 23:59:59"
                        }},
                        "depends_on": [],
                        "save_as": "orders_raw"
                        }},
                        {{
                        "id": "step2", 
                        "tool": "convert_to_df",
                        "params": {{
                            "raw": "{{{{orders_raw}}}}"
                        }},
                        "depends_on": ["step1"],
                        "save_as": "orders_df"
                        }},
                        {{
                        "id": "step3",
                        "tool": "get_aov",
                        "params": {{
                            "table": "{{{{orders_df}}}}"
                        }},
                        "depends_on": ["step2"],
                        "save_as": "aov_result"
                        }}
                    ],
                    "manipulation": {{
                        "required": true|false,
                        "type": "filter"|null
                    }},
                    "base_params": {{
                        "start_date": "actual date",
                        "end_date": "actual date"
                    }},
                    "tool": "get_all_orders"
                    }}

                    FEW-SHOT EXAMPLES:
                    These show exactly how to chain tools for different query types. Copy the structure exactly.

                    Query: "What is the AOV for last 5 days?"
                    {{
                        "summarized_query": "AOV last 5 days",
                        "query_type": "metric_analysis",
                        "steps": [
                            {{"id": "step1", "tool": "get_all_orders", "params": {{"start_date": "2026-03-11 00:00:00", "end_date": "2026-03-15 23:59:59"}}, "depends_on": [], "save_as": "orders_raw"}},
                            {{"id": "step2", "tool": "convert_to_df", "params": {{"raw": "{{{{orders_raw}}}}"}}, "depends_on": ["step1"], "save_as": "orders_df"}},
                            {{"id": "step3", "tool": "get_aov", "params": {{"table": "{{{{orders_df}}}}"}}, "depends_on": ["step2"], "save_as": "aov_result"}}
                        ],
                        "manipulation": {{"required": false, "type": null}},
                        "base_params": {{"start_date": "2026-03-11 00:00:00", "end_date": "2026-03-15 23:59:59"}},
                        "tool": "get_all_orders"
                    }}

                    Query: "Show only COD orders from Shopify yesterday and calculate their AOV"
                    {{
                        "summarized_query": "COD Shopify AOV yesterday",
                        "query_type": "metric_analysis",
                        "steps": [
                            {{"id": "step1", "tool": "get_all_orders", "params": {{"start_date": "2026-03-15 00:00:00", "end_date": "2026-03-15 23:59:59"}}, "depends_on": [], "save_as": "orders_raw"}},
                            {{"id": "step2", "tool": "apply_filters", "params": {{"table": "{{{{orders_raw}}}}", "filters": [{{"field": "marketplace", "value": "Shopify"}}, {{"field": "payment_mode", "value": "COD"}}]}}, "depends_on": ["step1"], "save_as": "filtered_raw"}},
                            {{"id": "step3", "tool": "convert_to_df", "params": {{"raw": "{{{{filtered_raw}}}}"}}, "depends_on": ["step2"], "save_as": "orders_df"}},
                            {{"id": "step4", "tool": "get_aov", "params": {{"table": "{{{{orders_df}}}}"}}, "depends_on": ["step3"], "save_as": "aov_result"}}
                        ],
                        "manipulation": {{"required": true, "type": "filter"}},
                        "base_params": {{"start_date": "2026-03-15 00:00:00", "end_date": "2026-03-15 23:59:59"}},
                        "tool": "get_all_orders"
                    }}

                    Query: "Revenue vs last week for prepaid vs COD"
                    {{
                        "summarized_query": "Revenue prepaid vs COD comparison",
                        "query_type": "comparison",
                        "steps": [
                            {{"id": "step1", "tool": "get_all_orders", "params": {{"start_date": "2026-03-09 00:00:00", "end_date": "2026-03-15 23:59:59"}}, "depends_on": [], "save_as": "orders_raw"}},
                            {{"id": "step2", "tool": "apply_filters", "params": {{"table": "{{{{orders_raw}}}}", "filters": []}}, "depends_on": ["step1"], "save_as": "filtered_raw"}},
                            {{"id": "step3", "tool": "convert_to_df", "params": {{"raw": "{{{{filtered_raw}}}}"}}, "depends_on": ["step2"], "save_as": "orders_df"}},
                            {{"id": "step4", "tool": "get_cod_vs_prepaid_metrics", "params": {{"table": "{{{{orders_df}}}}"}}, "depends_on": ["step3"], "save_as": "comparison_result"}}
                        ],
                        "manipulation": {{"required": false, "type": null}},
                        "base_params": {{"start_date": "2026-03-09 00:00:00", "end_date": "2026-03-15 23:59:59"}},
                        "tool": "get_all_orders"
                    }}

                    Query: "Custom metric: (total revenue - refunds) / order count last 30 days"
                    {{
                        "summarized_query": "Net revenue per order last 30 days",
                        "query_type": "custom_metric_generation",
                        "steps": [
                            {{"id": "step1", "tool": "get_all_orders", "params": {{"start_date": "2026-02-14 00:00:00", "end_date": "2026-03-15 23:59:59"}}, "depends_on": [], "save_as": "orders_raw"}},
                            {{"id": "step2", "tool": "convert_to_df", "params": {{"raw": "{{{{orders_raw}}}}"}}, "depends_on": ["step1"], "save_as": "orders_df"}},
                            {{"id": "step3", "tool": "execute_custom_calculation", "params": {{"table": "{{{{orders_df}}}}", "calculation_code": "result = (df['revenue'].sum() - df['refund_amount'].sum()) / df.shape[0]"}}, "depends_on": ["step2"], "save_as": "custom_result"}}
                        ],
                        "manipulation": {{"required": false, "type": null}},
                        "base_params": {{"start_date": "2026-02-14 00:00:00", "end_date": "2026-03-15 23:59:59"}},
                        "tool": "get_all_orders"
                    }}

                    Return *only* the JSON object with the structure based on query type.
"""

        try:
            print(f"[DEBUG] Planning query: {query}")

            response_text = self._generate_content(
                prompt=prompt,
                # Gemini structured output rejects schemas that include additionalProperties
                # (generated by Dict[...] fields), so we parse JSON manually and validate locally.
                response_schema=None,
                temperature=0.1,   # Low temperature for consistency
            )

            plan_data = json.loads(response_text)

            # Validate with Pydantic (optional but recommended)
            validated_plan = ExecutionPlan.model_validate(plan_data)

            summarized_query = validated_plan.summarized_query
            if not summarized_query:
                summarized_query = self._generate_fallback_summary(query)

            return {
                "success": True,
                "plan": validated_plan.model_dump(),           # or validated_plan.model_dump()
                "summarized_query": summarized_query
            }

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON from Gemini: {e}")
            print(f"[DEBUG] Raw response: {response_text[:500]}...")
            return {"success": False, "error": f"JSON parse error: {str(e)}"}
        except Exception as e:
            print(f"[ERROR] Planning failed: {e}")
            return {"success": False, "error": str(e)}

class FilteringLLM(GeminiLLM):
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

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.2,
            response_mime_type="text/plain",
        )
        
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


class GroupingLLM(GeminiLLM):
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

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.3,
            response_mime_type="text/plain",
        )
        
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


class MetricLLM(GeminiLLM):
    """Metric analysis LLM - generates insights from calculated metrics"""
    
    def invoke(self, params: dict) -> dict:
        """Generate metric analysis and insights"""
        query = params.get("query", "")
        metrics = params.get("metrics", {})
        raw_data_summary = params.get("data_summary", "")
        
        prompt = f"""You are a data analyst for an e-commerce order management system.

User Query: "{query}"

Calculated Metrics:
{json.dumps(metrics, indent=2)}

Data Summary:
{raw_data_summary}

Analyze the provided metrics and generate comprehensive insights. Your response should include:

1. **Key Performance Indicators**: Highlight the most important metrics and their significance
2. **Trends and Patterns**: Identify notable trends in the data
3. **Comparative Analysis**: Compare different segments (payment modes, marketplaces, regions) if applicable
4. **Statistical Insights**: Interpret percentiles, distributions, and correlations
5. **Business Recommendations**: Provide actionable insights based on the data
6. **Anomalies or Noteworthy Findings**: Point out any unusual patterns or outliers

Use specific numbers and percentages from the metrics. All values in INR.

If the metrics include distributions, highlight the top performers and underperformers.
If statistical summaries are provided, explain what the quartiles and standard deviation indicate.
If geographic data is available, provide location-based insights.
If conversion rates are included, comment on business performance.

IMPORTANT: Format your response in Markdown. Use bullet points for each insight.
Strictly use this exact format:

- **Bold title/metric:** [detailed explanation with specific numbers]
- **Bold title/metric:** [detailed explanation with specific numbers]
- **Bold title/metric:** [detailed explanation with specific numbers]

Example:
- **Order Volume Leadership:** Maharashtra recorded 505 orders, significantly outperforming Telangana's 154 orders by 69.5%
- **Revenue Performance:** Maharashtra generated ₹282,699 total revenue compared to Telangana's ₹106,043 
"""

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.5,
            response_mime_type="text/plain",
        )
        
        return {
            "analysis": response.strip(),
            "metrics_used": list(metrics.keys()) if isinstance(metrics, dict) else []
        }


class NewsRetrievalLLM(GeminiLLM):
    """News retrieval LLM - fetches and analyzes relevant news for market context"""
    
    def __init__(self):
        super().__init__()
        # You'll need API keys for news services
        self.news_api_key = os.getenv("NEWS_API_KEY")  # newsapi.org
        self.gnews_api_key = os.getenv("GNEWS_API_KEY")  # gnews.io
    
    def fetch_news(self, keywords: list, date_range: dict, sources: list = None) -> list:
        """Fetch news articles for given keywords and date range"""
        try:
            import requests
            from datetime import datetime
            
            articles = []
            
            # NewsAPI.org implementation
            if self.news_api_key:
                articles.extend(self._fetch_from_newsapi(keywords, date_range, sources))
            
            # GNews.io implementation  
            if self.gnews_api_key:
                articles.extend(self._fetch_from_gnews(keywords, date_range))
                
            return articles[:20]  # Limit to 20 most relevant articles
            
        except Exception as e:
            print(f"News fetch error: {e}")
            return []
    
    def _fetch_from_newsapi(self, keywords: list, date_range: dict, sources: list = None) -> list:
        """Fetch from NewsAPI.org"""
        import requests
        query = " OR ".join(keywords)
        
        params = {
            "q": query,
            "from": date_range["start_date"][:10],  # YYYY-MM-DD format
            "to": date_range["end_date"][:10],
            "language": "en",
            "sortBy": "relevancy",
            "apiKey": self.news_api_key
        }
        
        if sources:
            params["sources"] = ",".join(sources)
        
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("articles", [])
    
    def _fetch_from_gnews(self, keywords: list, date_range: dict) -> list:
        """Fetch from GNews.io"""
        import requests
        from urllib.parse import quote
        
        query = quote(" OR ".join(keywords))
        
        params = {
            "q": query,
            "from": date_range["start_date"][:10],
            "to": date_range["end_date"][:10],
            "lang": "en",
            "country": "in",  # India-specific news
            "max": 10,
            "apikey": self.gnews_api_key
        }
        
        response = requests.get("https://gnews.io/api/v4/search", params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("articles", [])
    
    def invoke(self, params: dict) -> dict:
        """Generate market insights from news data"""
        query = params.get("query", "")
        date_range = params.get("date_range", {})
        business_metrics = params.get("metrics", {})
        
        # Extract keywords for news search based on your footwear business
        keywords = self._extract_news_keywords(query)
        
        # Fetch relevant news from all available sources
        news_articles = self.fetch_news(keywords, date_range)
        
        if not news_articles:
            return {"market_context": "No relevant news found for the specified period."}
        
        # Analyze news impact
        market_analysis = self._analyze_market_impact(news_articles, business_metrics, query)
        
        return {
            "market_context": market_analysis,
            "news_articles_count": len(news_articles),
            "keywords_used": keywords
        }
    
    def _extract_news_keywords(self, query: str) -> list:
        """Extract relevant keywords for news search based on footwear business"""
        base_keywords = [
            "footwear industry", "shoe sales", "fashion retail", 
            "e-commerce fashion", "online shopping trends",
            "consumer spending", "retail sales India", "open footwear india", "sliders india"
        ]
        
        query_lower = query.lower()
        
        # Add specific keywords based on query context
        if any(word in query_lower for word in ['festival', 'diwali', 'eid', 'christmas']):
            base_keywords.extend(["festival shopping", "seasonal sales", "festive discounts"])
        
        if any(word in query_lower for word in ['monsoon', 'summer', 'winter']):
            base_keywords.extend(["seasonal footwear", "weather impact retail"])
            
        if any(word in query_lower for word in ['compare', 'vs', 'marketplace']):
            base_keywords.extend(["flipkart", "amazon", "myntra", "marketplace competition", "birkenstock", "asian shoes", ""])
        
        print("[NEWS]: base keywords -", base_keywords, flush=True)
        return base_keywords
    
    def _analyze_market_impact(self, articles: list, metrics: dict, query: str) -> str:
        """Analyze news impact on business metrics"""
        # Prepare news summary for LLM analysis
        news_summary = []
        for article in articles[:10]:  # Top 10 articles
            news_summary.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "published_date": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "")
            })
        
        prompt = f"""You are a market analyst specializing in e-commerce and footwear retail.

User Query: "{query}"

Business Metrics:
{json.dumps(metrics, indent=2)}

Recent Market News:
{json.dumps(news_summary, indent=2)}

Analyze how the market events and news might have influenced the business metrics shown above.

Focus on:
- **Economic Factors:** Interest rates, inflation, consumer confidence affecting purchasing power
- **Industry Trends:** Fashion trends, seasonal patterns, competitor activities
- **Market Events:** Sales events, festivals, policy changes, supply chain issues
- **Consumer Behavior:** Shift in shopping patterns, payment preferences, regional preferences
- **Platform Competition:** Marketplace dynamics, commission changes, policy updates
- **Seasonal Trends:** Trends occuring due to seasonal variety

Provide insights in this format:
- **[Market Factor]:** [How it likely impacted your metrics with specific references]

Be specific about correlations between news events and your sales data. All monetary values in INR.
"""

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.6,
            response_mime_type="text/plain",
        )
        return response.strip()


class InsightLLM(GeminiLLM):
    """Insight generation LLM - creates natural language summaries with market context"""
    
    def invoke(self, params: dict) -> dict:
        """Generate natural language insights for comparison and metric analysis with market context."""
        query = params.get("query", "")
        comparison = params.get("comparison", {})
        metrics = params.get("metrics", {})
        raw_metrics = params.get("raw_metrics", {})
        date_range = params.get("date_range", {})
        analysis_mode = params.get("analysis_mode", "comparison")
        
        # Get market context from news
        market_context = ""
        print(f"DEBUG: date_range = {date_range}", flush=True)
        print(f"DEBUG: should_include_news_context = {self._should_include_news_context(query)}", flush=True)
        print(f"DEBUG: condition check = {date_range and self._should_include_news_context(query)}", flush=True)
        
        if date_range and self._should_include_news_context(query):
            print("DEBUG: Entering news context fetch block", flush=True)
            try:
                print("DEBUG: Creating NewsRetrievalLLM instance", flush=True)
                news_llm = NewsRetrievalLLM()
                print("DEBUG: Calling news_llm.invoke", flush=True)
                news_result = news_llm.invoke({
                    "query": query,
                    "date_range": date_range,
                    "metrics": metrics
                })
                market_context = news_result.get("market_context", "")
                print("NEWS RESULT: ", news_result, flush=True)
                
            except Exception as e:
                print(f"News context fetch failed: {e}", flush=True)
                market_context = ""
        else:
            print("DEBUG: Skipping news context fetch due to condition check", flush=True)
        
        if analysis_mode == "metric_analysis":
            prompt = f"""You are a data insights analyst for an e-commerce footwear order system.

User Query: "{query}"

Normalized Metrics:
{json.dumps(metrics, indent=2)}

Raw Metric Results:
{json.dumps(raw_metrics, indent=2)}

Market Context from News Analysis:
{market_context}

Generate a comprehensive metric-focused analysis with clear business implications.

IMPORTANT: Format your response in Markdown. Use bullet points only.
Strictly use this exact format:

- **Bold title/metric:** [detailed explanation with specific numbers]
- **Bold title/metric:** [detailed explanation with specific numbers]

Include both DATA-DRIVEN insights and MARKET CONTEXT insights (if available):
- KPI interpretation and performance significance
- Trends and anomalies in metric values
- Distribution and segment analysis (payment modes, geography, status)
- Practical recommendations with expected business impact
- External factors from market context that may explain metric movement

All monetary values should be in INR and every point should reference concrete numbers from the provided metrics.
"""
        else:
            #comparison
            prompt = f"""You are a data insights analyst for an e-commerce footwear order system.

User Query: "{query}"

Comparison Data:
{json.dumps(comparison, indent=2)}

Detailed Metrics by Group:
{json.dumps(metrics, indent=2)}

Market Context from News Analysis:
{market_context}

Generate a comprehensive analysis combining your sales data insights with market context.

IMPORTANT: Format your response in Markdown. Use bullet points for each insight.
Strictly use this exact format:

- **Bold title/metric:** [detailed explanation with specific numbers]
- **Bold title/metric:** [detailed explanation with specific numbers]

Example:
- **Order Volume Performance:** Maharashtra leads with 505 orders vs Telangana's 154 orders (69.5% higher)
- **Revenue Analysis:** Total revenue shows Maharashtra at ₹282,699 compared to Telangana's ₹106,043

The comparison data may be either:
- "pairwise" (2 groups): Direct A vs B comparison with specific differences and percentages
- "multi_group" (3+ groups): Multiple groups with a baseline comparison and overall winners

Include both DATA-DRIVEN insights and MARKET CONTEXT insights:

DATA INSIGHTS:
- Order volume comparisons with percentages
- Revenue analysis with specific amounts
- AOV trends and patterns
- Geographic and demographic patterns
- Payment mode and marketplace performance differences

MARKET CONTEXT INSIGHTS (if available):
- How external market factors may have influenced your metrics
- Industry trends affecting your footwear performance
- Competitive landscape impact on sales
- Economic factors correlation with sales patterns
- Seasonal and event-driven factors

Make insights actionable and business-focused for footwear e-commerce.
"""

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.7,
            response_mime_type="text/plain",
        )
        
        return {
            "insights": response.strip(),
            "analysis": response.strip(),
            "metrics_used": list(raw_metrics.keys()) if isinstance(raw_metrics, dict) else list(metrics.keys()) if isinstance(metrics, dict) else [],
            "analysis_mode": analysis_mode,
            "market_context_included": bool(market_context)
        }
    
    def _should_include_news_context(self, query: str) -> bool:
        """Determine if news context would be valuable for this query"""
        context_keywords = [
            "compare", "performance", "trends", "analysis", 
            "why", "reason", "factor", "impact", "decline", "growth",
            "market", "industry", "competition", "seasonal"
        ]
        return any(keyword in query.lower() for keyword in context_keywords)


class CustomCalculationLLM(GeminiLLM):
    """Custom Calculation LLM - generates and iteratively refines Python code for custom metrics using ReAct pattern"""
    
    def __init__(self):
        super().__init__()
        self.max_iterations = 3  # Maximum ReAct loop iterations
        self.iteration_count = 0
        self.thought_action_history = []
        self.executor = None  # Will be set to execute_custom_calculation tool
    
    def invoke(self, params: dict) -> dict:
        """
        Generate and execute custom Python calculations using ReAct pattern.
        
        Args:
            params: {
                "query": str,  # Original user query
                "intent": str,  # Intent from planning LLM (e.g., "Calculate net revenue per order")
                "data": DataFrame,  # The DataFrame to operate on
                "schema": dict,  # Available fields and their types
                "date_range": dict,  # start_date and end_date for context
                "executor": callable  # Optional: execute_custom_calculation tool function
            }
        """
        user_query = params.get("query", "")
        intent = params.get("intent", "")
        data = params.get("data")
        schema = params.get("schema", {})
        date_range = params.get("date_range", {})
        executor = params.get("executor")  # Tool function if available

        # Sanity checks to avoid silent placeholder results.
        if data is None:
            return {
                "success": False,
                "error": "CustomCalculationLLM invoked without `data`",
                "reasoning_history": []
            }

        if executor is None:
            return {
                "success": False,
                "error": "CustomCalculationLLM invoked without `executor`",
                "reasoning_history": []
            }

        data_type = type(data).__name__
        data_shape = getattr(data, "shape", None)

        # print(f"[CustomCalculationLLM] ")
        print(f"[CustomCalculationLLM] schema fields: ", list(schema), flush=True)
        print(f"[CustomCalculationLLM] data received: type={data_type}, shape={data_shape}", flush=True)

        self.iteration_count = 0
        self.thought_action_history = []
        self.executor = executor
        
        print(f"[CustomCalculationLLM] Starting ReAct loop for: {intent}", flush=True)
        print(f"[CustomCalculationLLM] Executor available: {executor is not None}", flush=True)
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n[ReAct Iteration {self.iteration_count}/{self.max_iterations}]", flush=True)
            
            # THOUGHT phase
            thought = self._generate_thought(user_query, intent, schema, date_range)
            print(f"[THOUGHT] {thought}", flush=True)
            
            # ACTION phase - Generate Python code
            code = self._generate_python_code(user_query, intent, schema, thought, self.thought_action_history)
            print(f"[ACTION] Generated code:\n{code}", flush=True)
            
            # OBSERVATION phase - Execute code and observe results
            observation = self._execute_and_observe(code)
            print(f"[OBSERVATION] {observation}", flush=True)
            
            # Record in history for next iteration context
            self.thought_action_history.append({
                "iteration": self.iteration_count,
                "thought": thought,
                "code": code,
                "observation": observation
            })
            
            # REFLECTION phase - Check if calculation is valid
            is_valid, validation_message = self._validate_result(observation, intent)
            print(f"[VALIDATION] Valid={is_valid}, Message={validation_message}", flush=True)
            
            if is_valid:
                print(f"[SUCCESS] Calculation completed successfully in iteration {self.iteration_count}", flush=True)
                return {
                    "success": True,
                    "final_result": observation.get("result"),
                    "calculation_code": code,
                    "iterations": self.iteration_count,
                    "intent": intent,
                    "metadata": observation.get("metadata", {}),
                    "reasoning_history": self.thought_action_history
                }
            
            # If not valid, refine for next iteration
            if self.iteration_count < self.max_iterations:
                print(f"[REFINE] Will attempt refinement in next iteration", flush=True)
        
        # Max iterations reached without success
        print(f"[FAILURE] Max iterations ({self.max_iterations}) reached without valid result", flush=True)
        return {
            "success": False,
            "error": f"Could not generate valid calculation after {self.max_iterations} iterations",
            "last_observation": self.thought_action_history[-1].get("observation") if self.thought_action_history else None,
            "reasoning_history": self.thought_action_history
        }
    
    def _execute_and_observe(self, code: str) -> dict:
        """
        OBSERVATION phase: Execute the generated code and observe results.
        Uses the executor tool if available, otherwise returns placeholder.
        """
        if not self.executor:
            # Placeholder for when executor is not available
            print(f"[EXECUTE] Code execution placeholder (no executor provided)", flush=True)
            return {
                "status": "executed",
                "result": None,
                "error": None,
                "metadata": {
                    "execution_time_ms": 0,
                    "memory_used_mb": 0,
                    "executor_available": False
                }
            }
        
        try:
            # Execute code using the provided executor tool
            print(f"[EXECUTE] Executing code via provided executor", flush=True)
            
            import time
            start_time = time.time()
            
            result = self.executor(
                calculation_code=code,
                metric_name=f"custom_metric_iter{self.iteration_count}"
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Transform tool result into observation format
            if result.get("success"):
                observation = {
                    "status": "executed",
                    "result": result.get("result"),  # execute_custom_calculation returns `result`
                    "error": None,
                    "metadata": {
                        "execution_time_ms": execution_time,
                        "memory_used_mb": 0,
                        "executor_available": True,
                        "tool_result": result
                    }
                }
            else:
                observation = {
                    "status": "error",
                    "result": None,
                    "error": result.get("error", "Unknown error"),
                    "metadata": {
                        "execution_time_ms": execution_time,
                        "memory_used_mb": 0,
                        "executor_available": True,
                        "tool_result": result
                    }
                }
            
            return observation
            
        except Exception as e:
            print(f"[EXECUTE] Error during execution: {str(e)}", flush=True)
            return {
                "status": "error",
                "result": None,
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "executor_available": True
                }
            }
    
    def _validate_result(self, observation: dict, intent: str) -> tuple:
        """
        REFLECTION phase: Validate if the calculation result is meaningful and valid.
        
        Returns:
            (is_valid: bool, message: str)
        """
        if observation.get("error"):
            return False, f"Execution error: {observation['error']}"
        
        if observation.get("status") != "executed":
            # Allow "executed" status - executor may not be available yet
            if observation.get("status") != "executed" and observation.get("metadata", {}).get("executor_available"):
                return False, f"Unexpected status: {observation.get('status')}"
        
        result = observation.get("result")
        
        # Check if result exists
        if result is None:
            # Only fail if executor was available
            if observation.get("metadata", {}).get("executor_available"):
                return False, "Result is None - calculation may be incomplete"
            else:
                # Placeholder mode - accept None result
                return True, "Placeholder result accepted (executor pending)"
        
        # Type validations based on intent
        if any(word in intent.lower() for word in ["count", "number", "total"]):
            if not isinstance(result, (int, float)):
                return False, f"Expected numeric result, got {type(result)}"
        
        # Check for NaN or inf
        if isinstance(result, float):
            import math
            if math.isnan(result):
                return False, "Result is NaN - invalid calculation"
        
        return True, "Calculation valid and complete"
    
    def _format_schema(self, schema: dict) -> str:
        """Format schema information for prompt."""
        if not schema:
            return "No schema information available"
        
        schema_lines = []
        for field, info in schema.items():
            field_type = info.get('type', 'unknown')
            example = info.get('example', 'N/A')
            is_categorical = info.get('is_categorical', False)
            
            schema_lines.append(f"- {field} ({field_type}): example={example}")
            
            if is_categorical and info.get('enum'):
                enum_values = info['enum'][:5]
                schema_lines.append(f"  Possible values: {enum_values}")
        
        return "\n".join(schema_lines)
    
    def _generate_thought(self, user_query: str, intent: str, schema: dict, date_range: dict) -> str:
        """
        THOUGHT phase: Generate reasoning about the calculation approach.
        """
        schema_desc = self._format_schema(schema)
        
        history_context = ""
        if self.thought_action_history:
            history_context = "\n\nPrevious Iteration(s):\n"
            for entry in self.thought_action_history:
                history_context += f"- Iteration {entry['iteration']}: {entry['observation'].get('status', 'unknown')}\n"
                if entry['observation'].get('error'):
                    history_context += f"  Error was: {entry['observation']['error']}\n"
        
        prompt = f"""You are a data analysis expert. Analyze the following custom metric request and plan your Python implementation.

USER QUERY: "{user_query}"
CALCULATION INTENT: "{intent}"

Time Range: {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}

Available DataFrame Columns:
{schema_desc}

{history_context}

TASK: Break down the calculation into steps and explain your approach.

Think about:
1. Which columns from the DataFrame are needed?
2. What transformations or aggregations are required?
3. What edge cases might exist (nulls, empty data, etc.)?
4. What should the expected output look like?

Provide a concise thought process (3-4 sentences max)."""

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.3,
            response_mime_type="text/plain",
        )
        return response.strip()
    
    def _generate_python_code(self, user_query: str, intent: str, schema: dict, thought: str, history: list) -> str:
        """
        ACTION phase: Generate Python code for the calculation.
        """
        schema_desc = self._format_schema(schema)
        
        history_context = ""
        if history:
            history_context = "\n\nPrevious Attempts:\n"
            for entry in history:
                history_context += f"\nIteration {entry['iteration']}:\n"
                history_context += f"Code tried:\n```python\n{entry['code']}\n```\n"
                if entry['observation'].get('error'):
                    history_context += f"Error: {entry['observation']['error']}\n"
                    history_context += f"Feedback: Learn from this error and provide corrected code.\n"
        
        prompt = f"""You are an expert Python data analyst. Generate Python code to calculate the requested metric.

USER QUERY: "{user_query}"
CALCULATION INTENT: "{intent}"

YOUR REASONING FROM PREVIOUS STEP: {thought}

Available DataFrame Columns and Sample Data:
{schema_desc}

{history_context}

CRITICAL REQUIREMENTS:
1. Input variable MUST be 'df' (pandas DataFrame). Remember to load input database as pandas df first.
2. Final result MUST be stored in variable 'result'
3. Handle edge cases: empty data, null values, type conversions
4. Use pandas/numpy operations (no loops for large data)
5. Include data validation (check if df is empty, required columns exist)
6. Provide clear variable names and comments
7. Return the calculated metric as a single value or simple dict
8. DO NOT write any import statements (`import ...` or `from ... import ...`)
9. Assume `pd`, `np`, `math`, and `datetime` are already available

CODE STRUCTURE TEMPLATE:
```python
# Result calculation
if df.empty:
    result = None  # or handle appropriately
else:
    # Your calculation logic here
    result = <calculated_value>
```

Generate ONLY the Python code. Start with triple backticks [python]. No explanation needed."""

        response = self._generate_content(
            prompt=prompt,
            response_schema=None,
            temperature=0.2,
            response_mime_type="text/plain",
        )
        
        # Extract code from markdown if wrapped
        code = response.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        # Tool-side validator blocks imports; strip them defensively.
        sanitized_lines = []
        removed_imports = 0
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                removed_imports += 1
                continue
            sanitized_lines.append(line)

        if removed_imports > 0:
            print(f"[CustomCalculationLLM] Removed {removed_imports} import line(s) from generated code", flush=True)

        code = "\n".join(sanitized_lines)
        
        return code.strip()


#provide RAG business logic to both - Metric & Insight LLM
planning_llm = PlanningLLM()
filtering_llm = FilteringLLM()
grouping_llm = GroupingLLM()
insight_llm = InsightLLM()
metric_llm = MetricLLM()
news_llm = NewsRetrievalLLM()
custom_calculation_llm = CustomCalculationLLM()
