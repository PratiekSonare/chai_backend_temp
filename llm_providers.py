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


class PlanningLLM(OpenRouterLLM):
    """Planning LLM - generates execution plan from natural language query using tool calling"""
    
    def _generate_fallback_summary(self, query: str) -> str:
        """Generate a simple fallback summary when LLM doesn't provide one"""
        # Simple keyword-based summary generation
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return "Compare data groups"
        elif any(word in query_lower for word in ['aov', 'average order', 'order value']):
            return "Calculate average order value"
        elif any(word in query_lower for word in ['revenue', 'sales', 'total amount']):
            return "Calculate total revenue"
        elif any(word in query_lower for word in ['distribution', 'breakdown', 'split']):
            return "Analyze data distribution"
        elif any(word in query_lower for word in ['schema', 'fields', 'structure', 'available']):
            return "Explore data schema"
        elif any(word in query_lower for word in ['sku', 'product']):
            return "Analyze product data"
        else:
            return "Analyze orders data"
    
    def _get_tool_definitions(self):
        """Define available tools for the planning LLM"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_all_orders",
                    "description": "Fetch order data for a date range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD HH:MM:SS format"},
                            "end_date": {"type": "string", "description": "End date in YYYY-MM-DD HH:MM:SS format"}
                        },
                        "required": ["start_date", "end_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_filters",
                    "description": "Filter data by conditions (use early for optimization)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filters": {"type": "array", "description": "List of filter conditions"}
                        },
                        "required": ["filters"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_schema_info",
                    "description": "Get schema/metadata about available fields and constraints",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "Entity name (e.g., 'orders')"},
                            "field": {"type": "string", "description": "Optional: specific field name to get info for"}
                        },
                        "required": ["entity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_to_df",
                    "description": "Convert fetched JSON data to pandas DataFrame (REQUIRED before any metric calculation)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_custom_calculation",
                    "description": "Execute custom Python calculations for complex business logic, customer lifetime value, retention rates, growth calculations, time-based metrics, custom ratios, percentages, or derived metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "object", "description": "DataFrame to operate on"},
                            "calculation_code": {"type": "string", "description": "Python code string that operates on 'df' variable and assigns result to 'result' variable"},
                            "metric_name": {"type": "string", "description": "Name for the resulting metric", "default": "custom_metric"}
                        },
                        "required": ["table", "calculation_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_aov",
                    "description": "Calculate Average Order Value",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_total_revenue",
                    "description": "Calculate total revenue",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_count",
                    "description": "Get total number of orders",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_status_distribution",
                    "description": "Get distribution of order statuses",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_payment_mode_distribution",
                    "description": "Distribution of payment modes",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_marketplace_distribution",
                    "description": "Distribution of orders by marketplace",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_state_wise_distribution",
                    "description": "Distribution by state",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_city_wise_distribution",
                    "description": "Distribution by city (top N)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {"type": "integer", "description": "Number of top cities to show", "default": 10}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_courier_distribution",
                    "description": "Distribution of orders by courier service",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_average_discount",
                    "description": "Calculate average discount amount",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_average_shipping_charge",
                    "description": "Calculate average shipping charges",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_average_tax",
                    "description": "Calculate average tax amount",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_statistical_summary",
                    "description": "Get comprehensive statistical summary (mean, median, std, quartiles) for a numeric field",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Numeric field name to analyze"}
                        },
                        "required": ["field"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_percentile",
                    "description": "Get specific percentile for a numeric field",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Numeric field name"},
                            "percentile": {"type": "number", "description": "Percentile value (0-100)"}
                        },
                        "required": ["field", "percentile"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_percentile",
                    "description": "Get records in top percentile for a field and their metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Field name to analyze"},
                            "percentile": {"type": "number", "description": "Percentile threshold (default: 95)", "default": 95}
                        },
                        "required": ["field"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bottom_percentile",
                    "description": "Get records in bottom percentile for a field and their metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Field name to analyze"},
                            "percentile": {"type": "number", "description": "Percentile threshold (default: 5)", "default": 5}
                        },
                        "required": ["field"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_correlation_matrix",
                    "description": "Calculate correlation matrix between numeric fields",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fields": {"type": "array", "items": {"type": "string"}, "description": "List of numeric field names to correlate"}
                        },
                        "required": ["fields"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conversion_rate",
                    "description": "Calculate order conversion/delivery success rate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success_status": {"type": "string", "description": "Status considered successful (default: 'Delivered')", "default": "Delivered"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cod_vs_prepaid_metrics",
                    "description": "Compare COD vs PrePaid performance with counts, revenue, and AOV",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_geographic_insights",
                    "description": "Get geographic distribution insights including top states/cities by orders and revenue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {"type": "integer", "description": "Number of top regions to show (default: 5)", "default": 5}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_common_metrics",
                    "description": "Calculate standard business metrics (AOV, revenue, count, distributions) when no specific metrics are requested",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }
        ]
        
        # Debug: Print tool definitions
        print(f"[DEBUG] Tool definitions count: {len(tools)}")
        print(f"[DEBUG] Tool names: {[tool['function']['name'] for tool in tools]}")
        
        return tools

    def invoke(self, query: str) -> dict:
        """Generate execution plan from query with enhanced debugging"""
        from datetime import datetime, timedelta
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate date examples for the LLM
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        five_days_ago = today - timedelta(days=5)
        
        # Enhanced prompt with FORMAT EXAMPLES for tool calling
        prompt = f""" You are a query planning assistant for an e-commerce order management system.
Today's date is {current_date}.
CRITICAL: Always use actual dates in YYYY-MM-DD HH:MM:SS format, never placeholders.
- "last 5 days": "{five_days_ago.strftime('%Y-%m-%d')} 00:00:00" to "{today.strftime('%Y-%m-%d')} 23:59:59"
- "yesterday": "{yesterday.strftime('%Y-%m-%d')} 00:00:00" to "{yesterday.strftime('%Y-%m-%d')} 23:59:59"
- For chupps.com → "marketplace": "Shopify"

User Query: "{query}"

REQUIRED JSON OUTPUT FORMAT:
{{
  "summarized_query": "4-5 word summary",
  "query_type": "metric_analysis|schema_discovery|standard|comparison",
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

OPTIMIZATION STRATEGIES:
1. **COMPREHENSIVE ANALYSIS**: Include related metrics (COD→get_cod_vs_prepaid_metrics, Revenue→get_aov+get_order_count+get_total_revenue)
2. **EARLY FILTERING**: Use apply_filters before convert_to_df for large datasets  
3. **CUSTOM METRICS**: Use execute_custom_calculation for complex business logic not available in standard tools

Query Types:
- "schema_discovery": Data structure/field questions
- "comparison": Comparing groups (marketplaces, payment modes, etc.)  
- "metric_analysis": Specific metrics (AOV, revenue, distributions)
- "standard": Regular data fetch with optional filtering

IMPORTANT RULES: 
1. You MUST respond with a JSON object, NOT function calls. The available functions are for reference only. 
2. Every response MUST include "summarized_query" field with 4-5 word summary.
3. Use convert_to_df only when calculation / manipulation involved. If simple fetching of orders asked, then no need.

Return ONLY the JSON object with the structure based on query type.
"""

        tools = self._get_tool_definitions()
        
        # Debug: Print request details
        print(f"\\n[DEBUG] ===== TOOL CALLING PLANNING LLM DEBUG =====")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Tools being passed: {len(tools)} tools")
        print(f"[DEBUG] Calling API with tools...")
        
        response = self._call_api_with_tools([{"role": "user", "content": prompt}], tools, temperature=0.3)
        
        # Debug: Print raw LLM response
        print(f"[DEBUG] Raw LLM response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        print(f"[DEBUG] Raw LLM response: {str(response)[:300]}...")
        
        try:
            # Handle tool calling response format
            response_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            print(f"[DEBUG] Response content length: {len(response_content) if response_content else 0}")
            print(f"[DEBUG] Tool calls count: {len(tool_calls)}")
            
            if response_content:
                # Extract JSON from content response
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                else:
                    json_content = response_content
                
                print(f"[DEBUG] Extracted JSON (first 200 chars): {json_content[:200]}...")
                
                plan_data = json.loads(json_content)
                print(f"[DEBUG] Successfully parsed JSON with keys: {list(plan_data.keys())}")
                
                # Extract summarized_query and remove it from plan_data
                summarized_query = plan_data.pop("summarized_query", "")
                print(f"[DEBUG] Extracted summarized_query: '{summarized_query}'")
                
                # If summarized_query is empty, generate a fallback
                if not summarized_query:
                    summarized_query = self._generate_fallback_summary(query)
                    print(f"[DEBUG] Generated fallback summary: '{summarized_query}'")
                
                return {
                    "success": True,
                    "plan": plan_data,
                    "summarized_query": summarized_query
                }
            elif tool_calls:
                print(f"[DEBUG] Unexpected tool calls in response: {tool_calls}")
                return {
                    "success": False,
                    "error": "LLM tried to call tools instead of returning JSON"
                }
            else:
                print(f"[DEBUG] No content or tool calls found in response")
                return {
                    "success": False,
                    "error": "No content or tool calls in response"
                }
                    
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse plan JSON: {e}")
            print(f"[ERROR] Response content: {response}")
            return {
                "success": False,
                "error": f"Failed to parse plan: {str(e)}"
            }
        except Exception as e:
            print(f"[ERROR] Error processing response: {e}")
            return {
                "success": False,
                "error": f"Error processing response: {str(e)}"
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


class MetricLLM(OpenRouterLLM):
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

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.5)
        
        return {
            "analysis": response.strip(),
            "metrics_used": list(metrics.keys()) if isinstance(metrics, dict) else []
        }


class NewsRetrievalLLM(OpenRouterLLM):
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

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.6)
        return response.strip()


class InsightLLM(OpenRouterLLM):
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

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.7)
        
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


#provide RAG business logic to both - Metric & Insight LLM
planning_llm = PlanningLLM()
filtering_llm = FilteringLLM()
grouping_llm = GroupingLLM()
insight_llm = InsightLLM()
metric_llm = MetricLLM()
news_llm = NewsRetrievalLLM()
