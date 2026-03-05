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
    
    def invoke(self, query: str) -> dict:
        """Generate execution plan from query"""
        from datetime import datetime, timedelta
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate some date examples for the LLM
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        five_days_ago = today - timedelta(days=5)
        
        prompt = f"""You are a query planning assistant for an e-commerce order management system.
Today's date is {current_date}.

CRITICAL: When generating dates in JSON responses, always use actual date values in YYYY-MM-DD HH:MM:SS format.
- For "last 5 days": Calculate actual dates like "{five_days_ago.strftime('%Y-%m-%d %H:%M:%S')}" to "{today.strftime('%Y-%m-%d')} 23:59:59"
- For "yesterday": Use actual yesterday date like "{yesterday.strftime('%Y-%m-%d')} 00:00:00" to "{yesterday.strftime('%Y-%m-%d')} 23:59:59"
- NEVER use placeholder text like 'YYYY-MM-DD HH:MM:SS' - always calculate real dates

User Query: "{query}"

Available Tools:
1. get_all_orders_recent - Fetch order data for a date range
2. get_schema_info - Get schema/metadata about available fields and constraints
   - Optional param "field": Specify a field name (e.g., "payment_mode") to get only that field's info
   - Without "field": Returns complete schema for all fields
3. convert_to_df - Convert fetched JSON data to pandas DataFrame (REQUIRED before any metric calculation)
4. METRIC TOOLS (use after convert_to_df):
   - get_aov - Calculate Average Order Value
   - get_total_revenue - Calculate total revenue 
   - get_order_count - Get total number of orders
   - get_order_status_distribution - Distribution of order statuses
   - get_payment_mode_distribution - Distribution of payment modes (COD vs PrePaid)
   - get_marketplace_distribution - Distribution by marketplace
   - get_state_wise_distribution - Distribution by state
   - get_city_wise_distribution - Distribution by city (top N)
   - get_courier_distribution - Distribution by courier service
   - get_average_discount - Average discount amount
   - get_average_shipping_charge - Average shipping charges
   - get_average_tax - Average tax amount
5. STATISTICAL TOOLS:
   - get_statistical_summary - Comprehensive stats (mean, median, std, quartiles) for numeric field
   - get_percentile - Get specific percentile for a field
   - get_top_percentile - Get top N% records and their metrics
   - get_bottom_percentile - Get bottom N% records and their metrics
   - get_correlation_matrix - Calculate correlation between numeric fields
6. BUSINESS INTELLIGENCE TOOLS:
   - get_conversion_rate - Calculate delivery success rate
   - get_cod_vs_prepaid_metrics - Compare COD vs PrePaid performance
   - get_geographic_insights - Get geographic distribution insights
   - get_common_metrics - Calculate standard business metrics when no specific metrics are requested

Your task is to create an execution plan in JSON format. Analyze the query and determine:
1. Query Type:
   - "schema_discovery": If asking about available fields, data structure, allowed values, date ranges, etc.
   - "comparison": If comparing two or more groups (marketplaces, payment modes, states, etc.)
   - "metric_analysis": If asking for specific metrics like AOV, revenue, distributions, statistical analysis
   - "standard": Regular data fetch with optional filtering

2. Tool Selection:
   - Use "get_schema_info" for questions about data structure, available fields, enum values, constraints
     * If asking about a SPECIFIC field (e.g., "what are allowed values for payment_mode"), include "field" parameter
     * If asking generally (e.g., "what fields are available"), omit "field" parameter
   - Use "get_all_orders" for actual data queries
   - ALWAYS use "convert_to_df" after getting data and before any metric calculations
   - Use appropriate metric/statistical tools based on the query

3. Extract date range (for data queries only - convert relative dates like "last 5 days" to absolute dates)

4. Determine if filtering/manipulation is needed after fetching data

IMPORTANT WORKFLOW FOR METRIC QUERIES:
1. get_all_orders (with date range)
2. convert_to_df (convert to DataFrame) 
3. Apply filtering if needed
4. Use appropriate metric tools

IMPORTANT: Always use actual dates in YYYY-MM-DD HH:MM:SS format, never use placeholder text like 'YYYY-MM-DD HH:MM:SS'. When asked for chupps.com -> "marketplace": "Shopify",
CRITICAL REQUIREMENT: EVERY JSON response MUST include a "summarized_query" field at the top level with 4-5 words summarizing the user query.

Return ONLY a valid JSON object with this structure:

For metric analysis queries:
{{
  "summarized_query": "Calculate metrics for recent orders",
  "query_type": "metric_analysis",
  "steps": [
    {{
      "id": "step1",
      "tool": "get_all_orders",
      "params": {{
        "start_date": "2026-02-28 00:00:00",
        "end_date": "2026-03-05 23:59:59"
      }},
      "depends_on": [],
      "save_as": "orders_data"
    }},
    {{
      "id": "step2",
      "tool": "convert_to_df",
      "params": {{
        "raw": "{{orders_data}}"
      }},
      "depends_on": ["step1"],
      "save_as": "orders_df"
    }},
    {{
      "id": "step3",
      "tool": "get_aov",
      "params": {{
        "table": "{{orders_df}}"
      }},
      "depends_on": ["step2"],
      "save_as": "aov_result"
    }}
  ],
  "manipulation": {{
    "required": false,
    "type": null
  }},
  "base_params": {{
    "start_date": "2026-02-28 00:00:00",
    "end_date": "2026-03-05 23:59:59"
  }},
  "tool": "get_all_orders"
}}

For schema discovery queries (specific field):
{{
  "summarized_query": "Get field constraints and allowed values",
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
  "summarized_query": "Explore available data fields and structure",
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
  "summarized_query": "Retrieve orders data with optional filtering",
  "query_type": "standard" or "comparison",
  "steps": [
    {{
      "id": "step1",
      "tool": "get_all_orders",
      "params": {{
        "start_date": "2026-02-23 00:00:00",
        "end_date": "2026-03-05 23:59:59"
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
    "start_date": "2026-02-23 00:00:00",
    "end_date": "2026-03-05 23:59:59"
  }},
  "tool": "get_all_orders"
}}

Examples:
- "last 5 days" = start_date: "2026-02-28 00:00:00", end_date: "2026-03-05 23:59:59"
- "last week" = start_date: "2026-02-26 00:00:00", end_date: "2026-03-05 23:59:59"
- "yesterday" = start_date: "2026-03-04 00:00:00", end_date: "2026-03-04 23:59:59"
- "today" = start_date: "2026-03-05 00:00:00", end_date: "2026-03-05 23:59:59"
"""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.3)
        
        # Debug: Print raw LLM response
        print(f"[DEBUG] Raw LLM response: {response[:200]}...")
        
        try:
            # Extract JSON from response (in case LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response = response[json_start:json_end]
            
            print(f"[DEBUG] Extracted JSON: {response[:200]}...")
            
            plan_data = json.loads(response)
            print(f"[DEBUG] Parsed plan_data keys: {list(plan_data.keys())}")
            
            # Extract summarized_query and remove it from plan_data
            summarized_query = plan_data.pop("summarized_query", "")
            print(f"[DEBUG] Extracted summarized_query: '{summarized_query}'")
            
            # If summarized_query is empty, try to generate a fallback
            if not summarized_query:
                # Generate a simple fallback from the user query
                summarized_query = self._generate_fallback_summary(query)
                print(f"[DEBUG] Generated fallback: '{summarized_query}'", flush=True)
            
            return {
                "success": True,
                "plan": plan_data,
                "summarized_query": summarized_query
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

- **[Bold title/metric]:** [detailed explanation with specific numbers]
- **[Bold title/metric]:** [detailed explanation with specific numbers]
- **[Bold title/metric]:** [detailed explanation with specific numbers]

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
        """Generate natural language insights from comparison with market context"""
        query = params.get("query", "")
        comparison = params.get("comparison", {})
        metrics = params.get("metrics", {})
        date_range = params.get("date_range", {})
        
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

        Return your insights in the numbered point format described above."""

        response = self._call_api([{"role": "user", "content": prompt}], temperature=0.7)
        
        return {
            "insights": response.strip(),
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


# Singleton instances
planning_llm = PlanningLLM()
filtering_llm = FilteringLLM()
grouping_llm = GroupingLLM()
insight_llm = InsightLLM()
metric_llm = MetricLLM()
news_llm = NewsRetrievalLLM()
