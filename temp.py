from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import hashlib
import requests
import json

def get_embedding(text: str) -> list:
    """Get embedding using OpenRouter API"""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "input": text,
                "encoding_format": "float"
            })
        )
        
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            # Fallback to hash-based embedding if API fails
            hash_obj = hashlib.sha256(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            return [(hash_int >> i) % 2 - 0.5 for i in range(384)]
            
    except Exception as e:
        # Fallback embedding for any errors
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return [(hash_int >> i) % 2 - 0.5 for i in range(384)]
# === ALL PLANNER RULES & TOOL DEFINITIONS AS CLEAN CHUNKS ===

chunks = [
    # ================== TOOL DEFINITIONS (structured) ==================
    {"id": "tool_get_all_orders", "text": "Tool: get_all_orders\nDescription: Fetch raw order data for a date range.\nParams: start_date, end_date (YYYY-MM-DD HH:MM:SS)\nWhen to use: Always first step for any data query.", "category": "tool_definition", "topic": "data_fetch"},
    {"id": "tool_apply_filters", "text": "Tool: apply_filters\nDescription: Filter data early using field/operator/value.\nExample: {\"data\": \"{{orders_raw}}\", \"filters\": [{\"field\": \"payment_mode\", \"operator\": \"eq\", \"value\": \"COD\"}]}\nWhen to use: Before convert_to_df for optimization (COD, returned, marketplace, etc.).", "category": "tool_definition", "topic": "optimization"},
    {"id": "tool_convert_to_df", "text": "Tool: convert_to_df\nDescription: Convert JSON to pandas DataFrame. REQUIRED before any metric tool.\nParams: {\"raw\": \"{{variable_name}}\"}", "category": "tool_definition", "topic": "core"},
    {"id": "tool_execute_custom_calculation", "text": "Tool: execute_custom_calculation\nDescription: Run custom Python code when standard metrics are insufficient.\nMust assign to 'result' variable. Use pandas, numpy, datetime.\nUse for: CLV, retention, growth rates, delivery days, custom ratios.", "category": "tool_definition", "topic": "custom"},
    {"id": "tool_get_schema_info", "text": "Tool: get_schema_info\nDescription: Get schema/metadata about available fields and constraints.\nParams: entity='orders', optional field parameter for specific field info\nUse for: Questions about data structure, field values, constraints.", "category": "tool_definition", "topic": "schema"},
    {"id": "tool_get_aov", "text": "Tool: get_aov\nDescription: Calculate Average Order Value.\nParams: table={{dataframe_variable}}\nUse after: convert_to_df", "category": "tool_definition", "topic": "metrics"},
    {"id": "tool_get_total_revenue", "text": "Tool: get_total_revenue\nDescription: Calculate total revenue sum.\nParams: table={{dataframe_variable}}\nUse for: Revenue analysis queries.", "category": "tool_definition", "topic": "metrics"},
    {"id": "tool_get_order_count", "text": "Tool: get_order_count\nDescription: Get total number of orders.\nParams: table={{dataframe_variable}}\nUse for: Volume analysis.", "category": "tool_definition", "topic": "metrics"},
    {"id": "tool_get_payment_mode_distribution", "text": "Tool: get_payment_mode_distribution\nDescription: Distribution of payment modes (COD vs PrePaid).\nParams: table={{dataframe_variable}}\nUse for: Payment analysis.", "category": "tool_definition", "topic": "distribution"},
    {"id": "tool_get_marketplace_distribution", "text": "Tool: get_marketplace_distribution\nDescription: Distribution by marketplace.\nParams: table={{dataframe_variable}}\nUse for: Marketplace comparison.", "category": "tool_definition", "topic": "distribution"},
    {"id": "tool_get_state_wise_distribution", "text": "Tool: get_state_wise_distribution\nDescription: Distribution by state.\nParams: table={{dataframe_variable}}\nUse for: Geographic analysis.", "category": "tool_definition", "topic": "geographic"},
    {"id": "tool_get_city_wise_distribution", "text": "Tool: get_city_wise_distribution\nDescription: Distribution by city (top N).\nParams: table={{dataframe_variable}}\nUse for: City-level analysis.", "category": "tool_definition", "topic": "geographic"},
    {"id": "tool_get_order_status_distribution", "text": "Tool: get_order_status_distribution\nDescription: Distribution of order statuses.\nParams: table={{dataframe_variable}}\nUse for: Order status analysis.", "category": "tool_definition", "topic": "distribution"},
    {"id": "tool_get_cod_vs_prepaid_metrics", "text": "Tool: get_cod_vs_prepaid_metrics\nDescription: Compare COD vs PrePaid performance metrics.\nParams: table={{dataframe_variable}}\nCRITICAL: MUST use for COD-related queries.", "category": "tool_definition", "topic": "cod"},
    {"id": "tool_get_statistical_summary", "text": "Tool: get_statistical_summary\nDescription: Comprehensive stats (mean, median, std, quartiles) for numeric field.\nParams: table, field\nUse for: Statistical analysis.", "category": "tool_definition", "topic": "statistics"},
    {"id": "tool_get_conversion_rate", "text": "Tool: get_conversion_rate\nDescription: Calculate delivery success rate.\nParams: table={{dataframe_variable}}\nUse for: Performance analysis.", "category": "tool_definition", "topic": "performance"},
    {"id": "tool_get_courier_distribution", "text": "Tool: get_courier_distribution\nDescription: Distribution by courier service.\nParams: table={{dataframe_variable}}\nUse for: Logistics analysis.", "category": "tool_definition", "topic": "distribution"},
    {"id": "tool_get_average_discount", "text": "Tool: get_average_discount\nDescription: Average discount amount.\nParams: table={{dataframe_variable}}\nUse for: Pricing analysis.", "category": "tool_definition", "topic": "metrics"},
    {"id": "tool_get_common_metrics", "text": "Tool: get_common_metrics\nDescription: Calculate standard business metrics when no specific metrics requested.\nParams: table={{dataframe_variable}}\nUse for: General analysis.", "category": "tool_definition", "topic": "metrics"},

    # ================== OPTIMIZATION RULES ==================
    {"id": "rule_early_filtering", "text": "EARLY FILTERING RULE: Always use apply_filters BEFORE convert_to_df for large reductions.\nExamples: Returned orders → filter order_status=Returned first; COD queries → filter payment_mode=COD first.", "category": "optimization_rule", "topic": "optimization"},
    {"id": "rule_comprehensive_analysis", "text": "COMPREHENSIVE ANALYSIS: COD queries → MUST include get_cod_vs_prepaid_metrics + get_payment_mode_distribution\nRevenue queries → include get_aov + get_order_count + get_total_revenue\nReturns → filter first then multiple metrics\nRegional → state + city distributions", "category": "optimization_rule", "topic": "cod,revenue,returns"},
    {"id": "rule_custom_metrics", "text": "CUSTOM METRICS RULE: Use execute_custom_calculation for anything not in standard tools (CLV, retention, weekly growth, delivery time, profit margins, etc.).", "category": "optimization_rule", "topic": "custom"},
    {"id": "rule_parallel_execution", "text": "PARALLEL EXECUTION: When possible, run metric tools in parallel by using same depends_on. Example: step3a, step3b, step3c all depend on step2.", "category": "optimization_rule", "topic": "performance"},

    # ================== WORKFLOW PATTERNS ==================
    {"id": "pattern_simple_metric", "text": "WORKFLOW PATTERN 1 (Simple Metric Analysis):\n{\n  \"summarized_query\": \"Calculate metrics for recent orders\",\n  \"query_type\": \"metric_analysis\",\n  \"steps\": [\n    {\n      \"id\": \"step1\",\n      \"tool\": \"get_all_orders\",\n      \"params\": {\n        \"start_date\": \"2026-02-28 00:00:00\",\n        \"end_date\": \"2026-03-05 23:59:59\"\n      },\n      \"depends_on\": [],\n      \"save_as\": \"orders_data\"\n    },\n    {\n      \"id\": \"step2\",\n      \"tool\": \"convert_to_df\",\n      \"params\": {\n        \"raw\": \"{{orders_data}}\"\n      },\n      \"depends_on\": [\"step1\"],\n      \"save_as\": \"orders_df\"\n    },\n    {\n      \"id\": \"step3\",\n      \"tool\": \"get_aov\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"aov_result\"\n    }\n  ]\n}", "category": "workflow_pattern", "topic": "metric_analysis"},
    {"id": "pattern_cod_revenue", "text": "WORKFLOW PATTERN 2 (COD Revenue Analysis - RECOMMENDED FOR COD QUERIES):\n{\n  \"summarized_query\": \"Revenue loss from COD returns\",\n  \"query_type\": \"metric_analysis\",\n  \"steps\": [\n    {\n      \"id\": \"step1\",\n      \"tool\": \"get_all_orders\",\n      \"params\": {\n        \"start_date\": \"2026-03-06 00:00:00\",\n        \"end_date\": \"2026-03-06 23:59:59\"\n      },\n      \"depends_on\": [],\n      \"save_as\": \"orders_raw\"\n    },\n    {\n      \"id\": \"step2\",\n      \"tool\": \"apply_filters\",\n      \"params\": {\n        \"data\": \"{{orders_raw}}\",\n        \"filters\": [\n          {\"field\": \"payment_mode\", \"operator\": \"eq\", \"value\": \"COD\"},\n          {\"field\": \"order_status\", \"operator\": \"eq\", \"value\": \"Returned\"}\n        ]\n      },\n      \"depends_on\": [\"step1\"],\n      \"save_as\": \"cod_returned_raw\"\n    },\n    {\n      \"id\": \"step3\",\n      \"tool\": \"convert_to_df\",\n      \"params\": {\n        \"raw\": \"{{orders_raw}}\"\n      },\n      \"depends_on\": [\"step1\"],\n      \"save_as\": \"orders_df\"\n    },\n    {\n      \"id\": \"step6a\",\n      \"tool\": \"get_cod_vs_prepaid_metrics\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step3\"],\n      \"save_as\": \"cod_prepaid_comparison\"\n    },\n    {\n      \"id\": \"step6b\",\n      \"tool\": \"get_payment_mode_distribution\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step3\"],\n      \"save_as\": \"payment_distribution\"\n    }\n  ]\n}", "category": "workflow_pattern", "topic": "cod"},
    {"id": "pattern_comprehensive_revenue", "text": "WORKFLOW PATTERN 3 (Comprehensive Revenue Analysis with Parallel Execution):\n{\n  \"summarized_query\": \"Comprehensive revenue analysis\",\n  \"query_type\": \"metric_analysis\",\n  \"steps\": [\n    {\n      \"id\": \"step1\",\n      \"tool\": \"get_all_orders\",\n      \"params\": {\n        \"start_date\": \"2026-02-28 00:00:00\",\n        \"end_date\": \"2026-03-05 23:59:59\"\n      },\n      \"depends_on\": [],\n      \"save_as\": \"orders_raw\"\n    },\n    {\n      \"id\": \"step2\",\n      \"tool\": \"convert_to_df\",\n      \"params\": {\n        \"raw\": \"{{orders_raw}}\"\n      },\n      \"depends_on\": [\"step1\"],\n      \"save_as\": \"orders_df\"\n    },\n    {\n      \"id\": \"step3a\",\n      \"tool\": \"get_total_revenue\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"total_revenue\"\n    },\n    {\n      \"id\": \"step3b\",\n      \"tool\": \"get_aov\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"average_order_value\"\n    },\n    {\n      \"id\": \"step3c\",\n      \"tool\": \"get_order_count\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"order_count\"\n    },\n    {\n      \"id\": \"step3d\",\n      \"tool\": \"get_marketplace_distribution\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"marketplace_breakdown\"\n    }\n  ]\n}", "category": "workflow_pattern", "topic": "revenue"},
    {"id": "pattern_custom_metrics", "text": "WORKFLOW PATTERN 4 (Custom Metrics with Dynamic Code Generation):\n{\n  \"summarized_query\": \"Customer lifetime value analysis\",\n  \"query_type\": \"metric_analysis\",\n  \"steps\": [\n    {\n      \"id\": \"step1\",\n      \"tool\": \"get_all_orders\",\n      \"params\": {\n        \"start_date\": \"2025-01-01 00:00:00\",\n        \"end_date\": \"2026-03-07 23:59:59\"\n      },\n      \"depends_on\": [],\n      \"save_as\": \"orders_raw\"\n    },\n    {\n      \"id\": \"step2\",\n      \"tool\": \"convert_to_df\",\n      \"params\": {\n        \"raw\": \"{{orders_raw}}\"\n      },\n      \"depends_on\": [\"step1\"],\n      \"save_as\": \"orders_df\"\n    },\n    {\n      \"id\": \"step3a\",\n      \"tool\": \"execute_custom_calculation\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\",\n        \"calculation_code\": \"clv_stats = df.groupby('customer_email')['total_amount'].sum(); result = clv_stats.describe().to_dict()\",\n        \"metric_name\": \"customer_lifetime_value_stats\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"clv_statistics\"\n    },\n    {\n      \"id\": \"step3b\",\n      \"tool\": \"execute_custom_calculation\",\n      \"params\": {\n        \"table\": \"{{orders_df}}\",\n        \"calculation_code\": \"retention = df.groupby('customer_email')['order_date'].nunique(); result = (retention > 1).mean() * 100\",\n        \"metric_name\": \"customer_retention_rate\"\n      },\n      \"depends_on\": [\"step2\"],\n      \"save_as\": \"retention_rate\"\n    }\n  ]\n}", "category": "workflow_pattern", "topic": "custom"},

    # ================== DATE HANDLING & CRITICAL RULES ==================
    {"id": "rule_date_handling", "text": "DATE RULE: Always use REAL dates in YYYY-MM-DD HH:MM:SS. Never placeholders. For 'last 5 days', 'yesterday', 'today' calculate actual values in the plan.", "category": "critical_rule", "topic": "date"},
    {"id": "rule_output_structure", "text": "OUTPUT REQUIREMENT: Every JSON must have \"summarized_query\" (4-5 words) at top level. Follow the exact JSON structures shown in patterns.", "category": "critical_rule", "topic": "output"},
    {"id": "rule_query_types", "text": "QUERY TYPES: schema_discovery, comparison, metric_analysis, standard. Use get_schema_info for field questions.", "category": "critical_rule", "topic": "classification"},
    {"id": "rule_cod_specific", "text": "COD QUERY RULE: For any COD-related query, MUST include get_cod_vs_prepaid_metrics tool. This is critical for COD analysis.", "category": "critical_rule", "topic": "cod"},
    {"id": "rule_schema_discovery", "text": "SCHEMA DISCOVERY: For field-specific questions use get_schema_info with field parameter. For general structure questions omit field parameter.", "category": "critical_rule", "topic": "schema"},
    
    # ================== BUSINESS CONTEXT ==================
    {"id": "business_context", "text": "BUSINESS CONTEXT: This is an e-commerce footwear business. Marketplace mapping: chupps.com = Shopify. Key metrics: AOV, COD vs PrePaid performance, geographic distribution, return rates.", "category": "business_context", "topic": "domain"},
    {"id": "common_queries", "text": "COMMON QUERY PATTERNS: Revenue analysis → AOV + total + count in parallel. COD analysis → cod_vs_prepaid + payment_distribution. Geographic → state + city distribution. Returns → filter by status first.", "category": "business_context", "topic": "patterns"},
]

# === Ingest ===
points = []
for i, chunk in enumerate(chunks):
    # Get embedding for the chunk text
    vector = get_embedding(chunk["text"])
    
    point = PointStruct(
        id=i + 1,
        vector=vector,
        payload={
            "text": chunk["text"],
            "category": chunk["category"],
            "topic": chunk["topic"],
            "chunk_id": chunk["id"]
        }
    )
    points.append(point)