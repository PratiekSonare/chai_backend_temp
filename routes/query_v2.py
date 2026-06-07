"""
Query V2 Route - Multi-source dual categorization (Query Type + Data Source)
Uses Gemini API with dynamic tool selection for Orders, Profit, and Payment Cycle data
"""

import os
import json
import uuid
import warnings
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from google import genai
from google.genai import types
import requests as http_requests

# Suppress Supabase initialization warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from tools import (
        ORDERS_TOOL_REGISTRY,
        PROFIT_TOOL_REGISTRY,
        PAYMENT_CYCLE_TOOL_REGISTRY,
        PRODUCT_TOOL_REGISTRY,
        INVENTORY_TOOL_REGISTRY,
        list_sku_files,
        get_sku_metrics_json,
        get_insights_json,
        get_metrics_presets,
        build_sku_index_summary,
    )
    from data_schema import (
        validate_filter_field, 
        validate_filter_list,
        get_schema_info,
        get_available_fields,
        get_schema_prompt
    )
    from generated_tools import ALL_GENERATED_TOOLS
    try:
        from utils.request_log_store import append_request_log, read_request_logs, get_latest_sequence
    except Exception:
        # Fallback if request_log_store doesn't exist
        def append_request_log(**kwargs): pass
        def read_request_logs(*args, **kwargs): return []
        def get_latest_sequence(*args, **kwargs): return 0

# Initialize Gemini client
client = genai.Client(api_key=os.getenv('GEMINI_KEY'))

# OpenRouter config
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'moonshotai/kimi-k2.6:free')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')


# ===================================================================
# OPENROUTER HELPERS (OpenAI-compatible format conversion)
# ===================================================================

def _convert_gemini_tools_to_openai(tools: list) -> list:
    """Convert Gemini tool format to OpenAI function-calling format."""
    openai_tools = []
    for tool_group in tools:
        declarations = tool_group.get("function_declarations", [])
        for decl in declarations:
            params = decl.get("parameters", {})
            # Convert properties to OpenAI format
            properties = {}
            for prop_name, prop_def in params.get("properties", {}).items():
                openai_prop = {
                    "type": prop_def.get("type", "string"),
                    "description": prop_def.get("description", ""),
                }
                if "enum" in prop_def:
                    openai_prop["enum"] = prop_def["enum"]
                if "items" in prop_def:
                    openai_prop["items"] = prop_def["items"]
                properties[prop_name] = openai_prop

            openai_tool = {
                "type": "function",
                "function": {
                    "name": decl.get("name", ""),
                    "description": decl.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": params.get("required", []),
                    },
                },
            }
            openai_tools.append(openai_tool)
    return openai_tools


def _convert_gemini_messages_to_openai(messages: list, system_instruction: str = None) -> list:
    """Convert Gemini Content list to OpenAI chat messages format."""
    openai_msgs = []
    if system_instruction:
        openai_msgs.append({"role": "system", "content": system_instruction})

    for msg in messages:
        role = msg.role
        if role == "model":
            role = "assistant"
        for part in msg.parts:
            if part.text:
                openai_msgs.append({"role": role, "content": part.text})
            elif part.function_call:
                openai_msgs.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{part.function_call.name}_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args)),
                        },
                    }],
                })
            elif part.function_response:
                openai_msgs.append({
                    "role": "tool",
                    "tool_call_id": f"call_{part.function_response.name}_fallback",
                    "content": json.dumps(part.function_response.response),
                })
    return openai_msgs


def _convert_openai_response_to_gemini(openai_response: dict) -> object:
    """Convert OpenAI chat completion response to a Gemini-like response object."""
    class GeminiPart:
        def __init__(self, **kwargs):
            self.text = kwargs.get("text")
            self.function_call = kwargs.get("function_call")
            self.function_response = kwargs.get("function_response")

    class GeminiResponse:
        def __init__(self, parts, text=""):
            self.parts = parts
            self.text = text

    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})

    parts = []
    text_content = message.get("content", "") or ""

    # Handle tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        args_raw = func.get("arguments", "{}")
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        fc = type("FunctionCall", (), {"name": func.get("name"), "args": args})()
        parts.append(GeminiPart(function_call=fc))

    if text_content:
        parts.append(GeminiPart(text=text_content))

    return GeminiResponse(parts=parts, text=text_content)


def generate_content_with_openrouter(
    messages: list,
    openai_tools: list,
    system_instruction: str = None,
    model: str = None,
) -> object:
    """
    Call OpenRouter (OpenAI-compatible) API with tool support.
    Returns a Gemini-like response object.
    """
    api_key = OPENROUTER_API_KEY
    base_url = OPENROUTER_BASE_URL
    model = model or OPENROUTER_MODEL

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "OPENROUTER_API_KEY not set"}
        )

    openai_msgs = _convert_gemini_messages_to_openai(messages, system_instruction)

    payload = {
        "model": model,
        "messages": openai_msgs,
        "tools": openai_tools if openai_tools else None,
        "tool_choice": "auto" if openai_tools else None,
        "temperature": 0.2,
    }
    # Remove None tools/tool_choice
    payload = {k: v for k, v in payload.items() if v is not None}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chupps.ai",
        "X-Title": "Chupps Query V2",
    }

    print(f"🌐 Calling OpenRouter: {model} ...")
    resp = http_requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        error_detail = resp.text[:300]
        print(f"❌ OpenRouter {model} failed ({resp.status_code}): {error_detail}")
        raise Exception(f"OpenRouter {model} returned {resp.status_code}: {error_detail}")

    data = resp.json()
    return _convert_openai_response_to_gemini(data)


# Global state management
MEMORY_STORE = {}
_QUERY_CANCEL_LOCK = threading.Lock()
_QUERY_CANCEL_REGISTRY: Dict[str, Dict] = {}

# Session conversation history (session_id -> list of messages)
SESSION_STORE: Dict[str, List[types.Content]] = {}
SESSION_MAX_HISTORY = 20  # max messages to keep per session

# Categorization cache (query_hash -> (query_type, data_source, timestamp))
_CATEGORIZATION_CACHE: Dict[str, Tuple[str, str, datetime]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour

# SKU index cache (avoids re-fetching from S3 on every request)
_SKU_INDEX_CACHE: Dict[str, Any] = {"data": None, "timestamp": None}
SKU_INDEX_TTL_SECONDS = 3600  # 1 hour


def get_cached_sku_index() -> str:
    """Return the cached SKU index summary, refreshing from S3 if stale."""
    now = datetime.now()
    if _SKU_INDEX_CACHE["data"] and _SKU_INDEX_CACHE["timestamp"]:
        if (now - _SKU_INDEX_CACHE["timestamp"]).seconds < SKU_INDEX_TTL_SECONDS:
            return _SKU_INDEX_CACHE["data"]
    try:
        _SKU_INDEX_CACHE["data"] = build_sku_index_summary()
        _SKU_INDEX_CACHE["timestamp"] = now
    except Exception as e:
        print(f"Warning: Could not build SKU index: {e}")
        if not _SKU_INDEX_CACHE["data"]:
            _SKU_INDEX_CACHE["data"] = "(SKU metrics unavailable)"
    return _SKU_INDEX_CACHE["data"]

router = APIRouter()

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_session_history(session_id: str) -> List[types.Content]:
    """Retrieve conversation history for a session"""
    return SESSION_STORE.get(session_id, [])


def save_to_session_history(session_id: str, messages: List[types.Content]) -> None:
    """Save messages to session history, trimming to max length"""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []
    
    SESSION_STORE[session_id].extend(messages)
    
    # Trim to max history (keep most recent messages)
    if len(SESSION_STORE[session_id]) > SESSION_MAX_HISTORY:
        SESSION_STORE[session_id] = SESSION_STORE[session_id][-SESSION_MAX_HISTORY:]


def clear_session_history(session_id: str) -> None:
    """Clear conversation history for a session"""
    SESSION_STORE.pop(session_id, None)


def get_current_date_str() -> str:
    """Get current date and time as formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_date_range_instruction() -> str:
    """Generate date range examples based on today's date for LLM context"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    return f"""DATE HANDLING GUIDE (Today is {today.strftime('%Y-%m-%d')}):
When user says:
- "yesterday" → start_date="{yesterday.strftime('%Y-%m-%d')}", end_date="{today.strftime('%Y-%m-%d')}"
- "today" → start_date="{today.strftime('%Y-%m-%d')}", end_date="{today.strftime('%Y-%m-%d')}"
- "last 7 days" / "past week" → start_date="{week_ago.strftime('%Y-%m-%d')}", end_date="{today.strftime('%Y-%m-%d')}"
- "last 30 days" / "past month" → start_date="{month_ago.strftime('%Y-%m-%d')}", end_date="{today.strftime('%Y-%m-%d')}"
- "last N days" → subtract N days from today to get start_date
- Specific date like "2026-01-15" → use that date as start_date and end_date (or date range if specified)

ALWAYS convert relative dates to the exact format YYYY-MM-DD before calling get_all_orders."""


def generate_content_with_fallback(contents, config=None, initial_model: str = "gemini-2.5-flash", openai_tools: list = None, system_instruction: str = None) -> Any:
    """
    Call Gemini generate_content with fallbacks:
    1. Try gemini-2.5-flash (or initial_model)
    2. If 503/UNAVAILABLE, try gemini-2.5-pro
    3. If 503/UNAVAILABLE, try gemini-2.5-flash-lite
    4. If all Gemini models fail with 503, try OpenRouter (moonshotai/kimi-k2.6:free)
    Max 3 retries (total attempts of the fallback chain) if all respond with 503.
    """
    import time
    fallback_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    
    # Ensure initial_model is placed first in the sequence
    if initial_model in fallback_models:
        fallback_models.remove(initial_model)
    fallback_models = [initial_model] + fallback_models

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        for model in fallback_models:
            try:
                print(f"🤖 Calling Gemini LLM: {model} (Attempt {attempt + 1}/{max_retries})...")
                if config is not None:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                else:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents
                    )
                return response
            except Exception as e:
                err_str = str(e)
                last_error = e
                # Check for 503 UNAVAILABLE or overloading
                is_503 = (
                    "503" in err_str or 
                    "UNAVAILABLE" in err_str or 
                    "high demand" in err_str or 
                    "temporary" in err_str or
                    "resource_exhausted" in err_str.lower()
                )
                
                if is_503:
                    print(f"⚠️  Model {model} failed with 503 (UNAVAILABLE/High Demand): {err_str}. Trying fallback...")
                    continue
                else:
                    # Non-503 error, raise immediately
                    print(f"❌ Model {model} failed with non-503 error: {err_str}")
                    raise e
        
        # If we reached here, all models in fallback_models failed with 503 in this attempt
        if attempt < max_retries - 1:
            wait_time = 2 * (attempt + 1)  # 2s, 4s delay
            print(f"⏳ All models returned 503. Waiting {wait_time}s before retry attempt {attempt + 2}...")
            time.sleep(wait_time)

    # All Gemini attempts exhausted - try OpenRouter fallback
    print("⚡ All Gemini models exhausted. Trying OpenRouter fallback...")
    try:
        return generate_content_with_openrouter(
            messages=contents,
            openai_tools=openai_tools or [],
            system_instruction=system_instruction,
        )
    except Exception as or_error:
        print(f"❌ OpenRouter fallback also failed: {str(or_error)}")
        # Raise the original Gemini error since both providers failed
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": f"All upstream AI services unavailable. Gemini: {str(last_error)} | OpenRouter: {str(or_error)}"
            }
        )


# ===================================================================
# REQUEST/RESPONSE MODELS
# ===================================================================

class QueryV2Request(BaseModel):
    query: str
    session_id: Optional[str] = None


class PlanV2Response(BaseModel):
    success: bool
    query: str
    query_type: str
    data_source: str
    plan: Dict[str, Any]


class QueryV2Response(BaseModel):
    success: bool
    request_id: str
    session_id: Optional[str] = None
    query: str
    query_type: str
    data_source: str
    answer: str
    thinking: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    logs: Optional[List[Dict]] = None


# ===================================================================
# CATEGORIZATION LOGIC
# ===================================================================

CATEGORIZATION_PROMPT = """You are a query classifier. Classify the user query into exactly TWO fields:
1. Query Type: SCHEMA_DISCOVERY | STANDARD_QUERY | COMPARISON_QUERY | METRIC_ANALYSIS
2. Data Source: orders | profit | payment_cycle | inventory

Respond with ONLY: "QUERY_TYPE,DATA_SOURCE" (nothing else).

QUERY TYPE RULES:
- SCHEMA_DISCOVERY: User asks what fields/data is available ("what fields", "what data do we have", "schema")
- STANDARD_QUERY: User asks for a single metric, value, or fact ("what is the AOV", "show me damage rate", "how many units")
- COMPARISON_QUERY: User compares two or more groups ("compare X vs Y", "which location has more", "difference between")
- METRIC_ANALYSIS: User asks for patterns, rankings, distributions, breakdowns ("top 10", "which SKUs", "distribution by", "trends")

DATA SOURCE KEYWORDS (use these to determine the source):
- orders: order, order_id, AOV, revenue, sales amount, payment mode, COD, prepaid, marketplace order, shipping, delivery, cancelled, returned order, customer, pincode, state, city, courier
- profit: margin, profit, cost, MRP, selling price, cost price, markup, vendor cost, gross profit
- payment_cycle: distributor, payment cycle, payment terms, cash discount, CD, payment delay, margin exposure
- inventory: inventory, stock, available quantity, damaged, dead stock, QC, quality check, expiry, near expiry, warehouse, location stock, channel stock, marketplace available, website inventory, reserved, quarantine, repair, discard, overstock, understock, reorder, stock health, SKU stock, unit count, product quantity, fulfillment stock

EXAMPLES (4 per source, balanced):
"What fields are in the order data?" → SCHEMA_DISCOVERY,orders
"What was the AOV last week?" → STANDARD_QUERY,orders
"Compare COD vs prepaid orders" → COMPARISON_QUERY,orders
"Top cities by order volume" → METRIC_ANALYSIS,orders
"What fields are in the profit data?" → SCHEMA_DISCOVERY,profit
"What is the margin on SKU-123?" → STANDARD_QUERY,profit
"Compare margins across vendors" → COMPARISON_QUERY,profit
"Which SKUs have the highest markup?" → METRIC_ANALYSIS,profit
"What fields are in payment cycle data?" → SCHEMA_DISCOVERY,payment_cycle
"What is the average payment cycle?" → STANDARD_QUERY,payment_cycle
"Compare payment terms across distributors" → COMPARISON_QUERY,payment_cycle
"Which distributors have the longest payment cycles?" → METRIC_ANALYSIS,payment_cycle
"What fields are in the inventory data?" → SCHEMA_DISCOVERY,inventory
"What is the current stock health?" → STANDARD_QUERY,inventory
"Compare stock levels across locations" → COMPARISON_QUERY,inventory
"Which SKUs have the most dead stock?" → METRIC_ANALYSIS,inventory"""


def categorize_query(user_query: str) -> tuple:
    """Categorize query type and data source with caching"""
    # Create cache key
    query_hash = hashlib.md5(user_query.encode()).hexdigest()
    
    # Check cache
    if query_hash in _CATEGORIZATION_CACHE:
        cached_type, cached_source, cached_time = _CATEGORIZATION_CACHE[query_hash]
        if datetime.now() - cached_time < timedelta(seconds=_CACHE_TTL_SECONDS):
            print(f"📦 Cache HIT: {user_query[:40]}... → {cached_type}, {cached_source}")
            return cached_type, cached_source
        else:
            # Expired, remove from cache
            del _CATEGORIZATION_CACHE[query_hash]
    
    # Cache miss - call LLM
    try:
        response = generate_content_with_fallback(
            contents=[types.Content(role="user", parts=[types.Part(text=f"{CATEGORIZATION_PROMPT}\n\nUser Query: {user_query}")])],
            initial_model="gemini-2.5-flash"
        )
        raw_result = response.text.strip()
        # Strip quotes, periods, and extra whitespace
        cleaned = raw_result.strip('"\'.,;:').strip()
        result = cleaned.upper()
        parts = result.split(',')
        
        query_type = parts[0].strip() if len(parts) > 0 else "STANDARD_QUERY"
        data_source = parts[1].strip() if len(parts) > 1 else ""
        
        valid_types = ["SCHEMA_DISCOVERY", "STANDARD_QUERY", "COMPARISON_QUERY", "METRIC_ANALYSIS"]
        valid_sources = ["orders", "profit", "payment_cycle", "inventory"]
        
        query_type = query_type if query_type in valid_types else "STANDARD_QUERY"
        
        # Try exact match first, then partial match, then fallback to orders
        if data_source in valid_sources:
            pass  # exact match
        else:
            # Partial match: check if any valid source is contained in the response
            matched = False
            for src in valid_sources:
                if src in data_source.lower():
                    data_source = src
                    matched = True
                    break
            if not matched:
                print(f"⚠️  Unrecognized data source '{data_source}' from LLM, defaulting to 'orders'. Raw: {raw_result}")
                data_source = "orders"
        
        # Store in cache
        _CATEGORIZATION_CACHE[query_hash] = (query_type, data_source, datetime.now())
        
        print(f"🏷️  Categorize: '{user_query[:50]}' → {query_type}, {data_source} (raw: {raw_result})")
        return query_type, data_source
    except Exception as e:
        print(f"⚠️  Categorization error: {str(e)}")
        return "STANDARD_QUERY", "orders"


# ===================================================================
# TOOL DEFINITIONS & PROCESSING
# ===================================================================

def fetch_orders(start_date: str, end_date: str) -> str:
    try:
        raw_orders = ORDERS_TOOL_REGISTRY['get_all_orders'](start_date, end_date)
        ref_key = f"orders_{start_date[:10]}"
        MEMORY_STORE[ref_key] = raw_orders
        return f"SUCCESS: Fetched {len(raw_orders)} orders. Reference: '{ref_key}'."
    except Exception as e:
        return f"ERROR: {str(e)}"


def fetch_profit_data() -> str:
    try:
        df = PROFIT_TOOL_REGISTRY['get_vendor_cost_sheet']()
        ref_key = "profit_vendor_cost_sheet"
        MEMORY_STORE[ref_key] = df
        return f"SUCCESS: Fetched vendor cost sheet with {len(df)} SKUs. Reference: '{ref_key}'."
    except Exception as e:
        return f"ERROR: {str(e)}"


def fetch_payment_cycle_data(distributor_name: str = None) -> str:
    try:
        data = PAYMENT_CYCLE_TOOL_REGISTRY['get_payment_cycle_data'](distributor_name)
        ref_key = f"payment_cycle_{distributor_name or 'all'}"
        MEMORY_STORE[ref_key] = data
        return f"SUCCESS: Fetched {len(data)} distributor records. Reference: '{ref_key}'."
    except Exception as e:
        return f"ERROR: {str(e)}"


def fetch_inventory_snapshot(start_date: str, end_date: str) -> str:
    try:
        df = INVENTORY_TOOL_REGISTRY['get_inventory_snapshot'](start_date, end_date)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return "ERROR: No inventory data returned for the given date range."
        ref_key = f"inventory_{start_date[:10]}_{end_date[:10]}"
        MEMORY_STORE[ref_key] = df
        return f"SUCCESS: Fetched inventory snapshot with {len(df)} SKUs. Reference: '{ref_key}'."
    except Exception as e:
        return f"ERROR: {str(e)}"


def get_schema(data_source: str = "orders") -> str:
    try:
        schema_info = get_schema_info(data_source)
        if "error" in schema_info:
            return f"ERROR: {schema_info['error']}"
        
        formatted = f"📋 SCHEMA FOR '{data_source.upper()}'\n"
        
        if data_source == "orders":
            for field_name, field_info in list(schema_info['available_fields'].items())[:15]:
                formatted += f"\n  • {field_name}: {field_info.get('description', 'N/A')}"
        elif data_source == "profit":
            formatted += "\nFields: MRP, Final price, Cost, Margin%, Gross Profit, Markup\n"
            formatted += "  • get_margin - Calculate profit margin %\n"
            formatted += "  • get_gross_profit - Total profit (MRP - Cost)\n"
        elif data_source == "payment_cycle":
            formatted += "\nFields: PARTY NAME, MARGIN, CD (Cash Discount), PAYMENT CYCLE (days)\n"
            formatted += "  • get_avg_margin - Average margin %\n"
        
        return formatted
    except Exception as e:
        return f"ERROR: {str(e)}"


def filter_and_format_data(data_ref_id: str, filters: list) -> str:
    try:
        if data_ref_id not in MEMORY_STORE:
            return f"ERROR: Data reference '{data_ref_id}' not found."
        
        raw_data = MEMORY_STORE[data_ref_id]
        
        # Validate filters for orders
        if 'orders' in data_ref_id:
            is_valid, validation_errors = validate_filter_list("orders", filters)
            if not is_valid:
                error_msg = "❌ INVALID FILTERS:\n" + "\n".join(f"  • {err}" for err in validation_errors)
                return error_msg
        
        # Apply filters
        filtered_data = PROFIT_TOOL_REGISTRY['apply_filters'](raw_data, filters) if isinstance(raw_data, list) else raw_data
        
        # Convert to DataFrame
        import pandas as pd
        if 'orders' in data_ref_id:
            df = ORDERS_TOOL_REGISTRY['convert_to_df'](filtered_data)
        else:
            df = pd.DataFrame(filtered_data) if isinstance(filtered_data, list) else filtered_data
        
        df_ref_id = f"{data_ref_id}_df"
        MEMORY_STORE[df_ref_id] = df
        
        return f"SUCCESS: Filtered to {len(df)} records. Reference: '{df_ref_id}'."
    except Exception as e:
        return f"ERROR: {str(e)}"


def _get_registry_for_datasource(data_source: str):
    """Map data_source to the correct registry"""
    registry_map = {
        "orders": ORDERS_TOOL_REGISTRY,
        "profit": PROFIT_TOOL_REGISTRY,
        "payment_cycle": PAYMENT_CYCLE_TOOL_REGISTRY,
        "inventory": INVENTORY_TOOL_REGISTRY,
    }
    return registry_map.get(data_source, ORDERS_TOOL_REGISTRY)


def calculate_metric(df_ref_id: str, metric_function_name: str, data_source: str = "orders") -> str:
    try:
        if df_ref_id not in MEMORY_STORE:
            return f"ERROR: DataFrame reference '{df_ref_id}' not found."
        
        df = MEMORY_STORE[df_ref_id]
        
        # Get the correct registry based on data_source
        registry = _get_registry_for_datasource(data_source)
        func = registry.get(metric_function_name)
        
        if func:
            try:
                result = func(df)
                return str(result)
            except Exception as e:
                return f"ERROR: Failed to execute '{metric_function_name}': {str(e)}"
        
        return f"ERROR: Tool '{metric_function_name}' does not exist in {data_source} registry."
    except Exception as e:
        return f"ERROR: {str(e)}"


def get_tool_definitions(tool_names: list) -> list:
    """Generate tool definitions for specified tools"""
    selected_tools = [ALL_GENERATED_TOOLS[name] for name in tool_names if name in ALL_GENERATED_TOOLS]
    
    # Handle legacy tool names mapping to generated ones if necessary
    legacy_map = {
        "fetch_orders": "get_all_orders",
        "fetch_profit_data": "get_vendor_cost_sheet",
        "fetch_payment_cycle_data": "get_payment_cycle_data",
        "fetch_inventory_snapshot": "get_inventory_snapshot",
        "get_schema": "get_schema_info",
        "filter_and_format_data": "apply_filters"
    }
    
    for legacy, current in legacy_map.items():
        if legacy in tool_names and current in ALL_GENERATED_TOOLS:
            selected_tools.append(ALL_GENERATED_TOOLS[current])
            
    return [{"function_declarations": selected_tools}]


def process_tool_call(tool_name: str, tool_args: dict, data_source: str = "orders") -> str:
    """Process any tool call with optional data_source context"""
    try:
        # 1. Handle Fetch Tools (Special cases that create new refs)
        if tool_name in ["fetch_orders", "get_all_orders"]:
            start = tool_args.get("start_date")
            end = tool_args.get("end_date")
            raw_orders = ORDERS_TOOL_REGISTRY['get_all_orders'](start, end)
            ref_key = f"orders_{start[:10]}"
            MEMORY_STORE[ref_key] = raw_orders
            return f"SUCCESS: Fetched {len(raw_orders)} orders. Reference: '{ref_key}'."
            
        if tool_name in ["fetch_profit_data", "get_vendor_cost_sheet"]:
            df = PROFIT_TOOL_REGISTRY['get_vendor_cost_sheet']()
            ref_key = "profit_vendor_cost_sheet"
            MEMORY_STORE[ref_key] = df
            return f"SUCCESS: Fetched vendor cost sheet with {len(df)} SKUs. Reference: '{ref_key}'."
            
        if tool_name in ["fetch_payment_cycle_data", "get_payment_cycle_data"]:
            dist = tool_args.get("distributor_name")
            data = PAYMENT_CYCLE_TOOL_REGISTRY['get_payment_cycle_data'](dist)
            ref_key = f"payment_cycle_{dist or 'all'}"
            MEMORY_STORE[ref_key] = data
            return f"SUCCESS: Fetched {len(data)} distributor records. Reference: '{ref_key}'."

        if tool_name in ["fetch_inventory_snapshot", "get_inventory_snapshot"]:
            start = tool_args.get("start_date")
            end = tool_args.get("end_date")
            if not start or not end:
                from datetime import date
                today = date.today()
                end = today.isoformat()
                start = (today - __import__('datetime').timedelta(days=7)).isoformat()
            df = INVENTORY_TOOL_REGISTRY['get_inventory_snapshot'](start, end)
            if df is None or (hasattr(df, 'empty') and df.empty):
                return "ERROR: No inventory data returned for the given date range."
            ref_key = f"inventory_{start[:10]}_{end[:10]}"
            MEMORY_STORE[ref_key] = df
            return f"SUCCESS: Fetched inventory snapshot with {len(df)} SKUs. Reference: '{ref_key}'."

        # Product / SKU Metrics Tools
        if tool_name in ["get_sku_index", "list_sku_files"]:
            try:
                sku_list = list_sku_files()
                index_summary = get_cached_sku_index()
                return f"SUCCESS: {len(sku_list)} SKUs available.\n\n{index_summary}"
            except Exception as e:
                return f"ERROR: Could not list SKUs: {str(e)}"

        if tool_name == "get_sku_metrics":
            sku = tool_args.get("sku")
            if not sku:
                return "ERROR: 'sku' parameter is required."
            try:
                data = get_sku_metrics_json(sku)
                if "error" in data:
                    return f"ERROR: {data['error']}"
                ref_key = f"sku_metrics_{sku}"
                MEMORY_STORE[ref_key] = data
                return f"SUCCESS: Fetched metrics for SKU '{sku}'. Reference: '{ref_key}'. Data: {json.dumps(data)[:2000]}"
            except Exception as e:
                return f"ERROR: Could not fetch SKU metrics: {str(e)}"

        if tool_name == "get_insights":
            try:
                data = get_insights_json()
                if "error" in data:
                    return f"ERROR: {data['error']}"
                ref_key = "sku_insights"
                MEMORY_STORE[ref_key] = data
                return f"SUCCESS: Fetched SKU insights. Reference: '{ref_key}'. Data: {json.dumps(data)[:3000]}"
            except Exception as e:
                return f"ERROR: Could not fetch insights: {str(e)}"

        if tool_name == "get_metrics_presets":
            time_window = tool_args.get("time_window", "7d")
            try:
                data = get_metrics_presets(time_window)
                if "error" in data:
                    return f"ERROR: {data['error']}"
                ref_key = f"metrics_presets_{time_window}"
                MEMORY_STORE[ref_key] = data
                return f"SUCCESS: Fetched metrics presets for '{time_window}'. Reference: '{ref_key}'. Data: {json.dumps(data)[:3000]}"
            except Exception as e:
                return f"ERROR: Could not fetch metrics presets: {str(e)}"

        # 2. Handle Schema tool
        if tool_name in ["get_schema", "get_schema_info"]:
            ds = tool_args.get("data_source") or tool_args.get("entity") or data_source
            return get_schema(ds)

        # 3. Handle Data Processing Tools (tools that return filtered/transformed data)
        if tool_name in ["filter_and_format_data", "apply_filters", "convert_to_df"]:
            # These tools expect a data reference
            ref_id = tool_args.get("data_ref_id") or tool_args.get("table") or tool_args.get("raw") or tool_args.get("data")
            if not ref_id or ref_id not in MEMORY_STORE:
                return f"ERROR: Data reference '{ref_id}' not found."
            
            raw_data = MEMORY_STORE[ref_id]
            
            if tool_name == "convert_to_df":
                df = ORDERS_TOOL_REGISTRY['convert_to_df'](raw_data)
                new_ref = f"{ref_id}_df"
                MEMORY_STORE[new_ref] = df
                return f"SUCCESS: Converted to DataFrame. Reference: '{new_ref}'."
            
            # apply_filters or filter_and_format_data
            filters = tool_args.get("filters", [])
            filtered = PROFIT_TOOL_REGISTRY['apply_filters'](raw_data, filters)
            
            # Auto-convert orders to DF if it was a filter on raw orders
            if 'orders' in ref_id and not ref_id.endswith('_df'):
                df = ORDERS_TOOL_REGISTRY['convert_to_df'](filtered)
                new_ref = f"{ref_id}_filtered_df"
                MEMORY_STORE[new_ref] = df
                return f"SUCCESS: Filtered and converted to DF. Reference: '{new_ref}'."
            else:
                new_ref = f"{ref_id}_filtered"
                MEMORY_STORE[new_ref] = filtered
                return f"SUCCESS: Applied filters. Reference: '{new_ref}'."

        # 4. Handle Metric/Calculation Tools
        # Check if it's in our registries
        registry = _get_registry_for_datasource(data_source)
        func = registry.get(tool_name)
        
        if func:
            # Prepare arguments by resolving references
            processed_args = {}
            for k, v in tool_args.items():
                if k in ["table", "data", "raw"] and isinstance(v, str) and v in MEMORY_STORE:
                    processed_args[k] = MEMORY_STORE[v]
                else:
                    processed_args[k] = v
            
            try:
                result = func(**processed_args)
                return str(result)
            except Exception as e:
                return f"ERROR executing {tool_name}: {str(e)}"

        return f"ERROR: Unknown tool '{tool_name}'"
    except Exception as e:
        return f"ERROR in process_tool_call: {str(e)}"


# ===================================================================
# RESPONSE FORMATTERS
# ===================================================================

def format_schema_discovery_response(
    request_id: str,
    user_query: str,
    extracted_data: Any,
    plan: Dict[str, Any],
    data_source: str = "orders"
) -> Dict[str, Any]:
    """Format SCHEMA_DISCOVERY response with actual schema data"""
    schema_info = extracted_data if isinstance(extracted_data, dict) else {}
    return {
        "response_type": "schema",
        "data": schema_info,
        "metadata": {
            "entity": data_source,
            "request_id": request_id
        }
    }


def format_standard_query_response(
    request_id: str,
    user_query: str,
    extracted_data: Any,
    data_source: str
) -> Dict[str, Any]:
    """Format STANDARD_QUERY response with actual extracted data"""
    data = []
    if isinstance(extracted_data, list):
        data = extracted_data
    elif isinstance(extracted_data, dict):
        data = [extracted_data]
    else:
        try:
            import pandas as pd
            if isinstance(extracted_data, pd.DataFrame):
                data = extracted_data.to_dict(orient='records')
        except:
            pass
            
    return {
        "response_type": "records",
        "count": len(data),
        "data": data,
        "metadata": {
            "data_source": data_source,
            "request_id": request_id
        }
    }


def format_comparison_response(
    request_id: str,
    user_query: str,
    extracted_data: Any,
    data_source: str
) -> Dict[str, Any]:
    """Format COMPARISON_QUERY response with actual comparison data"""
    comparison_data = extracted_data if isinstance(extracted_data, dict) else {}
    return {
        "response_type": "comparison",
        "data": comparison_data,
        "metadata": {
            "data_source": data_source,
            "request_id": request_id
        }
    }


def format_metric_analysis_response(
    request_id: str,
    user_query: str,
    extracted_data: Any,
    plan: Dict[str, Any],
    data_source: str = "orders"
) -> Dict[str, Any]:
    """Format METRIC_ANALYSIS response with actual metrics data"""
    metrics_data = extracted_data if isinstance(extracted_data, dict) else {}
    return {
        "response_type": "metric_analysis",
        "data": metrics_data,
        "metrics_calculated": list(metrics_data.keys()) if isinstance(metrics_data, dict) else [],
        "metadata": {
            "data_source": data_source,
            "request_id": request_id
        }
    }


def extract_data_from_memory(memory_store: Dict, data_source: str = "orders") -> Any:
    """Extract actual data from MEMORY_STORE after query execution"""
    if not memory_store:
        return None
    
    # Get the last fetched/processed data (usually ends with _df or raw ref)
    df_refs = [k for k in memory_store.keys() if k.endswith('_df')]
    raw_refs = [k for k in memory_store.keys() if not k.endswith('_df')]
    
    # Prefer processed DataFrames, then raw data
    ref_key = df_refs[-1] if df_refs else (raw_refs[-1] if raw_refs else None)
    
    if not ref_key:
        return None
    
    data = memory_store[ref_key]
    
    # Convert DataFrame to records
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
    except:
        pass
    
    return data


def format_query_response(
    query_type: str,
    request_id: str,
    user_query: str,
    extracted_data: Any,
    plan: Dict[str, Any],
    data_source: str = "orders"
) -> Dict[str, Any]:
    """Route to appropriate response formatter based on query_type with actual data"""
    if query_type == "SCHEMA_DISCOVERY":
        return format_schema_discovery_response(request_id, user_query, extracted_data, plan, data_source)
    elif query_type == "STANDARD_QUERY":
        return format_standard_query_response(request_id, user_query, extracted_data, data_source)
    elif query_type == "COMPARISON_QUERY":
        return format_comparison_response(request_id, user_query, extracted_data, data_source)
    elif query_type == "METRIC_ANALYSIS":
        return format_metric_analysis_response(request_id, user_query, extracted_data, plan, data_source)
    else:
        # Fallback to standard query format
        return format_standard_query_response(request_id, user_query, extracted_data, data_source)


# ===================================================================
# PROMPT CONFIGURATIONS
# ===================================================================

PROMPTS = {
    "orders": {
        "SCHEMA_DISCOVERY": {
            "system_instruction": """You are a Data Schema Assistant for Orders.
Help users understand what order data is available and how to query it.

Available fields include: order_id, marketplace, payment_mode, order_status, total_amount,
order_date, state, city, pincode, customer_name, sku, quantity, and more.

WORKFLOW:
1. Call get_schema_info(entity='orders') to show all fields
2. Explain what filters are available
3. Suggest example queries""",
            "tools": ["get_schema_info", "get_all_orders", "get_sku_index", "get_sku_metrics"],
        },
        "STANDARD_QUERY": {
            "system_instruction": """You are a Data Retrieval Agent for Orders.
Answer metric queries on order data using specific tools.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

ALLOWED OPERATORS for filters:
- 'eq': equal to (standard)
- 'ne': not equal to
- 'gt': greater than
- 'lt': less than
- 'gte': greater than or equal to
- 'lte': less than or equal to
- 'contains': string contains
- 'in': value is in list

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_all_orders(start_date, end_date) with the determined date range.
3. Optional: apply_filters(table, filters) -> creates a filtered reference.
   - ALWAYS use 'eq' for exact matches. 
   - Field names must match the schema exactly (e.g., 'state', 'city', 'order_status').
4. Optional: convert_to_df(raw) -> converts raw data to DataFrame reference (needed for most metrics)
5. Call specific metric tools (e.g., get_aov, get_total_revenue) using the DataFrame reference.

Always prefer specific tools over custom calculations if they exist.""",
            "tools": [
                "get_all_orders", "apply_filters", "convert_to_df", "get_aov", 
                "get_total_revenue", "get_order_count", "get_cancelled_count",
                "get_order_status_distribution", "get_payment_mode_distribution",
                "get_marketplace_distribution", "get_state_wise_distribution",
                "get_city_wise_distribution", "get_courier_distribution",
                "get_average_discount", "get_average_shipping_charge", "get_average_tax",
                "get_statistical_summary", "get_percentile", "get_top_percentile",
                "get_bottom_percentile", "get_correlation_matrix", "get_conversion_rate",
                "get_cod_vs_prepaid_metrics", "get_geographic_insights", "get_common_metrics",
                "execute_custom_calculation", "get_schema_info",
                "get_sku_index", "get_sku_metrics", "get_insights", "get_metrics_presets"
            ],
        },
        "COMPARISON_QUERY": {
            "system_instruction": """You are a Comparison Analyst for Orders.
Compare metrics across payment modes, marketplaces, states, or other groups.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_all_orders(start_date, end_date) with the determined date range.
3. For EACH group to compare:
   a. apply_filters(table, [{"field": "...", "operator": "eq", "value": "..."}])
   b. Call metric tools (get_aov, get_total_revenue, etc.) on the filtered data
4. Present side-by-side comparison with key metrics.

If no specific tool compares what you need, first convert to DataFrame using convert_to_df, then use execute_custom_calculation on each filtered group or on the main DataFrame with a groupby.""",
            "tools": [
                "get_all_orders", "apply_filters", "convert_to_df", "get_aov", 
                "get_total_revenue", "get_order_count", "get_common_metrics",
                "get_cod_vs_prepaid_metrics", "get_geographic_insights",
                "execute_custom_calculation",
                "get_sku_index", "get_sku_metrics", "get_insights"
            ],
        },
        "METRIC_ANALYSIS": {
            "system_instruction": """You are a Metric Analyst for Orders.
Find patterns: top cities, top states, distributions, etc.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

FALLBACK FOR COMPLEX QUERIES:
If no specific tool fits (e.g., "highest selling SKU", "orders with > 2 items"), use this REPL workflow:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_all_orders(start_date, end_date) with the determined date range.
3. convert_to_df(raw) -> creates a DataFrame reference (e.g., 'orders_2026-02-01_df')
4. execute_custom_calculation(table, calculation_code, metric_name) -> write Pandas code.
   - USE THE DATAFRAME REFERENCE for the 'table' argument.
   - The DataFrame 'df' has exploded suborders. Fields: suborder_sku, suborder_quantity, suborder_selling_price, total_amount, payment_mode, marketplace, state, city.
   - Example to find top SKU: result = df['suborder_sku'].value_counts().idxmax()
   - Assign final value to 'result' variable.""",
            "tools": [
                "get_all_orders", "apply_filters", "convert_to_df", "get_geographic_insights",
                "get_order_status_distribution", "get_payment_mode_distribution", 
                "get_marketplace_distribution", "get_state_wise_distribution",
                "get_city_wise_distribution", "get_statistical_summary", "get_correlation_matrix",
                "execute_custom_calculation",
                "get_sku_index", "get_sku_metrics", "get_insights", "get_metrics_presets"
            ],
        },
    },
    "profit": {
        "SCHEMA_DISCOVERY": {
            "system_instruction": """You are a Data Schema Assistant for Profit/Cost.
Help users understand profit metrics available.""",
            "tools": ["get_schema_info", "get_vendor_cost_sheet"],
        },
        "STANDARD_QUERY": {
            "system_instruction": """You are a Profit Metrics Agent.
Calculate profit metrics for vendors/SKUs.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_vendor_cost_sheet() -> creates 'profit_vendor_cost_sheet' reference
3. Optional: apply_filters(table, filters)
4. Call specific profit tools (get_margin, get_gross_profit, etc.)""",
            "tools": [
                "get_vendor_cost_sheet", "apply_filters", "get_cost_price",
                "get_selling_price", "get_gross_profit", "get_margin",
                "get_markup", "get_cost_to_price_ratio", "execute_custom_calculation",
                "get_statistical_summary", "get_percentile", "get_top_percentile",
                "get_bottom_percentile", "get_correlation_matrix",
                "get_sku_metrics"
            ],
        },
        "COMPARISON_QUERY": {
            "system_instruction": """You are a Profit Comparison Analyst.
Compare profitability across vendors, categories, etc.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_vendor_cost_sheet() -> creates 'profit_vendor_cost_sheet' reference
3. For EACH group to compare:
   a. apply_filters(table, filters) for each vendor/category
   b. Call metric tools (get_margin, get_gross_profit) on each group
4. Present side-by-side comparison.""",
            "tools": [
                "get_vendor_cost_sheet", "apply_filters", "get_margin", "get_gross_profit",
                "get_sku_metrics"
            ],
        },
        "METRIC_ANALYSIS": {
            "system_instruction": """You are a Profit Analysis Agent.
Find high-margin SKUs, cost outliers, correlation patterns.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_vendor_cost_sheet() -> creates 'profit_vendor_cost_sheet' reference
3. apply_filters(table, filters) for analysis
4. Call specific profit tools or use execute_custom_calculation for deeper analysis.""",
            "tools": [
                "get_vendor_cost_sheet", "apply_filters", "get_margin", 
                "get_statistical_summary", "get_top_percentile", "get_correlation_matrix",
                "get_sku_metrics"
            ],
        },
    },
    "payment_cycle": {
        "SCHEMA_DISCOVERY": {
            "system_instruction": """You are a Data Schema Assistant for Payment Cycles.
Help users understand distributor payment terms and cash discount data.""",
            "tools": ["get_schema_info", "get_payment_cycle_data"],
        },
        "STANDARD_QUERY": {
            "system_instruction": """You are a Payment Cycle Agent.
Answer questions about distributor payment terms and margins.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_payment_cycle_data(distributor_name) -> fetches payment cycle data
3. Optional: apply_filters(table, filters) if needed
4. Call metric tools to analyze payment terms and margins.""",
            "tools": [
                "get_payment_cycle_data", "apply_filters", "get_avg_margin",
                "get_weighted_avg_margin", "get_margin_per_payment_day",
                "get_total_margin_exposure", "get_high_risk_distributors",
                "get_cycle_efficiency_score", "get_payment_cycle_distribution",
                "get_cash_discount_stats", "execute_custom_calculation",
                "get_statistical_summary", "get_percentile"
            ],
        },
        "COMPARISON_QUERY": {
            "system_instruction": """You are a Distributor Comparison Analyst.
Compare payment terms, margins, and cash discounts across distributors.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_payment_cycle_data() -> fetches all distributor data
3. For EACH distributor/group to compare:
   a. apply_filters(table, filters) for each group
   b. Call metric tools on each group
4. Present side-by-side comparison of payment terms and margins.""",
            "tools": [
                "get_payment_cycle_data", "apply_filters", "get_avg_margin", "get_cash_discount_stats"
            ],
        },
        "METRIC_ANALYSIS": {
            "system_instruction": """You are a Payment Cycle Risk Analyst.
Identify high-risk distributors, payment cycle patterns, margin exposure.

DATE RANGE HANDLING:
- If the user specifies a date range (e.g., "in the last 7 days", "since Monday", "from Jan 1"), use that range.
- **IF NO DATE RANGE IS SPECIFIED, DEFAULT TO THE LAST 7 DAYS** (past week).
- Always convert relative dates (yesterday, today, last N days) to concrete YYYY-MM-DD format.

WORKFLOW:
1. Determine the date range: Parse user dates OR use default (last 7 days).
2. get_payment_cycle_data() -> fetches distributor payment cycle data
3. apply_filters(table, filters) if needed for specific analysis
4. Call risk analysis tools to identify patterns and high-risk distributors.""",
            "tools": [
                "get_payment_cycle_data", "apply_filters", "get_high_risk_distributors",
                "get_payment_cycle_distribution", "get_cycle_efficiency_score"
            ],
        },
    },
    "inventory": {
        "SCHEMA_DISCOVERY": {
            "system_instruction": """You are a Data Schema Assistant for Inventory.
Help users understand what inventory data is available and how to query it.

Available fields include: sku, product_name, category, brand, location, available_qty, reserved_picked, damaged, total_lost, qc_passed, qc_failed, marketplace_available, website_inventory, etc.

WORKFLOW:
1. Call get_inventory_snapshot() to load inventory data
2. Explain what fields and filters are available
3. Suggest example queries""",
            "tools": ["get_inventory_snapshot", "get_inventory_summary"],
        },
        "STANDARD_QUERY": {
            "system_instruction": """You are an Inventory Intelligence Agent.
Answer questions about stock levels, damage rates, QC performance, and inventory health.

ALLOWED OPERATORS for filters:
- 'eq': equal to, 'ne': not equal to
- 'gt': greater than, 'lt': less than
- 'gte': greater than or equal to, 'lte': less than or equal to
- 'contains': string contains, 'in': value is in list

WORKFLOW:
1. get_inventory_snapshot(start_date, end_date) to load data.
   - If no dates specified, use last 7 days.
2. Optional: apply_filters(table, filters) to narrow down.
3. Call specific inventory tools (get_stock_health, get_damage_rate, get_qc_performance, etc.).
4. For general queries, use the DataFrame reference with apply_filters, execute_custom_calculation, get_statistical_summary.

Always prefer specific tools over custom calculations if they exist.""",
            "tools": [
                "get_inventory_snapshot", "get_inventory_summary",
                "apply_filters", "execute_custom_calculation",
                "get_stock_health", "get_damage_rate", "get_dead_stock",
                "get_qc_performance", "get_expiry_risk",
                "get_channel_distribution", "get_category_breakdown",
                "get_brand_breakdown", "get_location_breakdown",
                "get_statistical_summary", "get_percentile",
            ],
        },
        "COMPARISON_QUERY": {
            "system_instruction": """You are an Inventory Comparison Analyst.
Compare inventory metrics across locations, categories, brands, or channels.

WORKFLOW:
1. get_inventory_snapshot(start_date, end_date) to load data.
2. For EACH group to compare:
   a. apply_filters(table, [{"field": "...", "operator": "eq", "value": "..."}])
   b. Call metric tools on the filtered data
3. Present side-by-side comparison with key metrics.

If no specific tool fits, use execute_custom_calculation with groupby.""",
            "tools": [
                "get_inventory_snapshot", "apply_filters",
                "get_stock_health", "get_damage_rate", "get_qc_performance",
                "get_channel_distribution", "get_category_breakdown",
                "execute_custom_calculation",
            ],
        },
        "METRIC_ANALYSIS": {
            "system_instruction": """You are an Inventory Analytics Agent.
Find patterns: dead stock, overstock, understock, damage trends, QC failures, channel distribution.

WORKFLOW:
1. get_inventory_snapshot(start_date, end_date) to load data.
2. Call specific analysis tools for the requested metric.
3. For complex analysis, use execute_custom_calculation with the DataFrame.

The DataFrame columns include: sku, product_name, category, brand, location, available_qty, reserved_picked, damaged, total_lost, qc_passed, qc_failed, marketplace_available, website_inventory, etc.""",
            "tools": [
                "get_inventory_snapshot",
                "apply_filters", "execute_custom_calculation",
                "get_stock_health", "get_damage_rate", "get_dead_stock",
                "get_overstock_risk", "get_understock_risk",
                "get_qc_performance", "get_expiry_risk",
                "get_channel_distribution", "get_category_breakdown",
                "get_brand_breakdown", "get_location_breakdown",
                "get_statistical_summary",
            ],
        },
    },
}


# ===================================================================
# QUERY CANCELLATION SUPPORT
# ===================================================================

def register_active_query(request_id: str) -> None:
    with _QUERY_CANCEL_LOCK:
        _QUERY_CANCEL_REGISTRY[request_id] = {"cancelled": False}


def is_query_cancelled(request_id: str | None) -> bool:
    if not request_id:
        return False
    with _QUERY_CANCEL_LOCK:
        return bool(_QUERY_CANCEL_REGISTRY.get(request_id, {}).get("cancelled"))


def clear_query_registry_entry(request_id: str) -> None:
    with _QUERY_CANCEL_LOCK:
        _QUERY_CANCEL_REGISTRY.pop(request_id, None)


# ===================================================================
# ROUTE HANDLERS2
# ===================================================================

@router.post('/query-v2/query', response_model=QueryV2Response)
async def process_query_v2(
    request: QueryV2Request,
    raw_request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID")
):
    """
    Process a natural language query end-to-end.
    Categorizes the query, selects tools, and runs the agentic loop.
    
    Example:
    {
        "query": "What's the AOV for prepaid orders in the last 7 days?"
    }
    
    Response:
    {
        "success": true,
        "request_id": "abc123",
        "query_type": "STANDARD_QUERY",
        "data_source": "orders",
        "answer": "The average order value for prepaid orders is $45.23..."
    }
    """
    user_query = request.query.strip()
    request_id = getattr(raw_request.state, "request_id", None) or x_request_id or str(uuid.uuid4())[:8]
    session_id = request.session_id or str(uuid.uuid4())[:12]
    
    register_active_query(request_id)
    
    try:
        # Log request start
        append_request_log(
            request_id=request_id,
            step_key="QUERY_V2_START",
            summary="Query V2 request started",
            details=user_query[:200],
            status="START"
        )
        
        # Categorize query
        print(f"🔍 Analyzing query: {user_query[:60]}...")
        query_type, data_source = categorize_query(user_query)
        print(f"📌 Query Type: {query_type} | Data Source: {data_source}")
        
        append_request_log(
            request_id=request_id,
            step_key="QUERY_CATEGORIZED",
            summary=f"Query categorized as {query_type}",
            details=f"Data source: {data_source}",
            status="INFO"
        )
        
        # Get prompt config
        prompt_config = PROMPTS.get(data_source, {}).get(query_type)
        if not prompt_config:
            # Fallback: try STANDARD_QUERY for the same data source, then orders
            prompt_config = PROMPTS.get(data_source, {}).get("STANDARD_QUERY")
        if not prompt_config:
            prompt_config = PROMPTS["orders"]["STANDARD_QUERY"]
        
        system_instruction = prompt_config["system_instruction"]
        tool_names = prompt_config["tools"]
        tools = get_tool_definitions(tool_names)
        
        # Prepend current date/time, date range handling, schema, and SKU index to system instruction
        schema_prompt = get_schema_prompt(data_source)
        date_range_guidance = generate_date_range_instruction()
        sku_index = get_cached_sku_index()
        system_instruction_with_date = (
            f"Today's date and time is {get_current_date_str()}.\n\n"
            f"{date_range_guidance}\n\n"
            f"{system_instruction}\n\n"
            f"{schema_prompt}\n\n"
            f"## Available Products (SKU Index)\n{sku_index}\n\n"
            f"When the user asks about specific products, SKUs, or product performance metrics, "
            f"use get_sku_metrics(sku) for detailed per-SKU data, get_insights() for cross-SKU trends and rankings, "
            f"or get_metrics_presets(time_window) for aggregated dashboard KPIs."
        )
        
        print(f"🛠️  Using tools: {', '.join(tool_names)}")
        
        # Pre-convert tools for OpenRouter fallback
        openai_tools = _convert_gemini_tools_to_openai(tools)
        
        # Load session history and initialize conversation
        session_history = get_session_history(session_id)
        messages = list(session_history)  # copy prior messages
        messages.append(types.Content(role="user", parts=[types.Part(text=user_query)]))
        
        max_iterations = 10
        iteration = 0
        final_answer = None
        tool_trace = []
        
        # Agentic loop for tool selection
        while iteration < max_iterations:
            iteration += 1
            
            if is_query_cancelled(request_id):
                append_request_log(
                    request_id=request_id,
                    step_key="QUERY_CANCELLED",
                    summary="Query execution cancelled",
                    status="CANCELLED"
                )
                return {
                    "success": False,
                    "request_id": request_id,
                    "cancelled": True,
                    "error": "Request cancelled by client"
                }
            
            print(f"\n--- Iteration {iteration} ---")
            
            response = generate_content_with_fallback(
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=system_instruction_with_date
                ),
                initial_model="gemini-2.5-flash",
                openai_tools=openai_tools,
                system_instruction=system_instruction_with_date
            )
            
            # Check for tool calls
            tool_calls = [part.function_call for part in response.parts if part.function_call]
            print("#"*18)
            print("tool calls: ", tool_calls, flush=True)

            if not tool_calls:
                # No tool calls - this is the final answer
                final_answer = response.text
                print(f"\n✅ Final Answer:\n{final_answer}")
                
                # Save conversation to session history
                save_to_session_history(session_id, [
                    types.Content(role="user", parts=[types.Part(text=user_query)]),
                    types.Content(role="model", parts=[types.Part(text=final_answer)])
                ])
                
                append_request_log(
                    request_id=request_id,
                    step_key="QUERY_COMPLETE",
                    summary="Query execution completed",
                    status="COMPLETE"
                )
                break
            
            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = dict(tool_call.args)
                print(f"🔧 {tool_name}({tool_args})")
                
                tool_result = process_tool_call(tool_name, tool_args, data_source)
                # print(f"   → {tool_result[:150]}")
                
                # Record tool trace for thinking
                result_preview = tool_result[:200] if tool_result else ""
                is_error = tool_result.startswith("ERROR") if tool_result else False
                tool_trace.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result_preview,
                    "success": not is_error,
                })
                
                append_request_log(
                    request_id=request_id,
                    step_key=f"TOOL_{tool_name.upper()}",
                    summary=f"Tool executed: {tool_name}",
                    details=tool_result[:200],
                    status="INFO"
                )
                
                tool_results.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": tool_result}
                        )
                    )
                )
            
            messages.append(types.Content(role="model", parts=response.parts))
            messages.append(types.Content(role="user", parts=tool_results))
        
        if iteration >= max_iterations:
            print(f"\n⚠️  Max iterations reached")
            append_request_log(
                request_id=request_id,
                step_key="MAX_ITERATIONS",
                summary="Max iterations reached",
                status="WARNING"
            )
        
        # Build thinking trace from tool calls
        thinking = None
        if tool_trace:
            tool_lines = []
            for i, t in enumerate(tool_trace, 1):
                status = "✓" if t["success"] else "✗"
                # Build a short args summary (skip large data refs)
                arg_parts = []
                for k, v in t["args"].items():
                    if isinstance(v, str) and len(v) > 50:
                        arg_parts.append(f"{k}=\"{v[:40]}...\"")
                    else:
                        arg_parts.append(f"{k}={v!r}")
                args_str = ", ".join(arg_parts)
                tool_lines.append(f"{i}. {status} **{t['tool']}**({args_str})")
            
            thinking = f"Used {len(tool_trace)} tool(s):\n" + "\n".join(tool_lines)
        
        # Extract actual data from MEMORY_STORE (not LLM response)
        extracted_data = extract_data_from_memory(MEMORY_STORE, data_source)
        
        # Format results based on query type
        results = format_query_response(
            query_type=query_type,
            request_id=request_id,
            user_query=user_query,
            extracted_data=extracted_data,
            plan={"query_type": query_type, "data_source": data_source, "tools": tool_names},
            data_source=data_source
        )
        
        # Construct final unified response
        final_response = {
            "success": extracted_data is not None,
            "request_id": request_id,
            "session_id": session_id,
            "query": user_query,
            "query_type": query_type,
            "data_source": data_source,
            "answer": final_answer or "Calculation completed successfully.",
            "thinking": thinking,
            "results": results,
            "logs": read_request_logs(request_id)
        }
        
        # Cache extracted data for later retrieval (1 hour TTL)
        _QUERY_RESULTS_CACHE[request_id] = (extracted_data, datetime.now())
        
        return final_response
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        append_request_log(
            request_id=request_id,
            step_key="QUERY_ERROR",
            summary="Query execution failed",
            details=str(e),
            status="ERROR"
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "request_id": request_id,
                "error": str(e)
            }
        )
    
    finally:
        # Clear memory and registry
        MEMORY_STORE.clear()
        clear_query_registry_entry(request_id)


@router.post('/query-v2/{request_id}/cancel')
async def cancel_query_v2(request_id: str):
    """Cancel an ongoing query V2 execution."""
    try:
        with _QUERY_CANCEL_LOCK:
            if request_id not in _QUERY_CANCEL_REGISTRY:
                _QUERY_CANCEL_REGISTRY[request_id] = {"cancelled": True}
            else:
                _QUERY_CANCEL_REGISTRY[request_id]["cancelled"] = True
        
        append_request_log(
            request_id=request_id,
            step_key="CANCEL_REQUESTED",
            summary="Query cancellation requested",
            status="INFO"
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "cancelled": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )


@router.post('/query-v2/{session_id}/reset')
async def reset_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        clear_session_history(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session history cleared"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )


@router.get('/query-v2/logs/{request_id}')
async def get_query_v2_logs(request_id: str, since: int = 0):
    """Poll logs for a running query V2 execution."""
    logs = read_request_logs(request_id, since_sequence=since)
    return {
        "success": True,
        "request_id": request_id,
        "logs": logs,
        "next_sequence": get_latest_sequence(request_id)
    }


# ===================================================================
# QUERY RESULTS PERSISTENCE (Optional: for large datasets)
# ===================================================================

# In-memory result cache (request_id -> extracted data)
# For production: use Redis with TTL for multi-instance support
_QUERY_RESULTS_CACHE: Dict[str, Tuple[Any, datetime]] = {}
_RESULTS_TTL_SECONDS = 3600  # 1 hour


@router.get('/query-v2/{request_id}/results')
async def get_query_results(request_id: str):
    """
    Retrieve cached query results (available for 1 hour after execution).
    Use this to fetch large datasets without storing in initial response.
    """
    try:
        if request_id in _QUERY_RESULTS_CACHE:
            cached_data, cached_time = _QUERY_RESULTS_CACHE[request_id]
            if datetime.now() - cached_time < timedelta(seconds=_RESULTS_TTL_SECONDS):
                ttl_remaining = int(_RESULTS_TTL_SECONDS - (datetime.now() - cached_time).total_seconds())
                print(f"📦 Results CACHE HIT: {request_id}, TTL: {ttl_remaining}s")
                return {
                    "success": True,
                    "request_id": request_id,
                    "source": "cache",
                    "ttl_remaining_seconds": ttl_remaining,
                    "data": cached_data
                }
            else:
                # Expired
                del _QUERY_RESULTS_CACHE[request_id]
        
        return {
            "success": False,
            "request_id": request_id,
            "error": "Results not found or expired (1 hour TTL)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )
