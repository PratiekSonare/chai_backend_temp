# Order Analysis Workflow - Flask Server

A LangGraph-based workflow server for analyzing e-commerce orders with natural language queries.

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Date Windowing**: Automatically handles date ranges >7 days by splitting into multiple API calls
- **Smart Filtering**: Extracts filters from natural language (payment mode, status, location, etc.)
- **Comparison Analysis**: Compare orders across marketplaces, payment modes, states, etc.
- **Schema Learning**: Automatically detects categorical fields and their possible values

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Python 3.14+**: You may see a Pydantic V1 compatibility warning from LangChain. This is a known issue and can be safely ignored - the functionality works correctly.

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```env
# EasyEcom API
EASYECOM_API_KEY=your_actual_api_key
EASYECOM_JWT_TOKEN=your_actual_jwt_token
EASYECOM_BASE_URL=https://api.easyecom.io

# OpenRouter LLM
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
```

Get your OpenRouter API key from: https://openrouter.ai/keys

### 3. Run the Server

```bash
python app.py
```

Server will start on `https://${process.env.NEXT_PUBLIC_API_URL}`

## API Endpoints

### `GET /health`
Health check endpoint

### `GET /examples`
Get example queries you can try

### `POST /query`
Process a natural language query

**Request:**
```json
{
  "query": "Show me orders from last 5 days with payment mode prepaid"
}
```

**Response (Standard Query):**
```json
{
  "success": true,
  "query_type": "standard",
  "count": 42,
  "data": [...],
  "total_records": 42
}
```

**Response (Comparison Query):**
```json
{
  "success": true,
  "query_type": "comparison",
  "insights": "Detailed comparison insights...",
  "comparison_data": {...},
  "detailed_metrics": {...}
}
```

## Example Queries

### Standard Queries
- "Show me orders from last 5 days with payment mode prepaid"
- "Get all open orders from last week"
- "Orders from Karnataka in last 10 days"
- "Show COD orders from last 3 days"

### Comparison Queries
- "Compare orders between Shopify13 and Flipkart from the last 10 days"
- "Compare prepaid vs COD orders from last week"
- "Compare Karnataka vs Maharashtra order volumes in last 15 days"

## Testing with cURL

### Standard Query
```bash
curl -X POST https://${process.env.NEXT_PUBLIC_API_URL}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me orders from last 5 days with payment mode prepaid"
  }'
```

### Comparison Query
```bash
curl -X POST https://${process.env.NEXT_PUBLIC_API_URL}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare orders between Shopify13 and Flipkart from the last 10 days"
  }'
```

## Architecture

### Workflow Nodes

**Standard Flow:**
1. **Planning** → Analyzes query and creates execution plan
2. **Execute Tool** → Fetches data from API with date windowing
3. **Filtering** → Extracts filters from natural language
4. **Apply Filters** → Applies filters to data
5. **Return Result** → Returns filtered data

**Comparison Flow:**
1. **Planning** → Identifies comparison query
2. **Grouping** → Extracts comparison dimensions
3. **Parallel Fetch** → Fetches data for each group
4. **Aggregation** → Calculates metrics (count, revenue, avg order value, etc.)
5. **Comparison** → Computes differences and percentages
6. **Insight Generation** → Generates natural language summary
7. **Retproviders.py**: OpenRouter LLM implementations using Llama 3.1 70B Instruct

## LLM Configuration

This system uses **Llama 3.1 70B Instruct** via OpenRouter for:
- **Planning**: Analyzing queries and creating execution plans
- **Filtering**: Extracting filter conditions from natural language
- **Grouping**: Identifying comparison dimensions
- **Insights**: Generating natural language summaries

The LLM is prompted with structured instructions to return JSON for parsing, ensuring reliable integration.
ate Limiting**: Add rate limiting for API protection
2. **Add Redis**: Replace in-memory cache with Redis for production
3. **Error Handling**: Add comprehensive error handling and logging
4. **Authentication**: Add authentication/authorization
5. **Pagination**: Add pagination for large result sets
6. **Async Processing**: Consider async processing for long-running queries
7. **LLM Fallbacks**: Add fallback logic if LLM API fail)

## Production Considerations

1. **Replace Mock LLMs**: Currently using rule-based mocks. Replace with actual LLM calls (OpenAI, etc.)
2. **Add Redis**: Replace in-memory cache with Redis for production
3. **Error Handling**: Add comprehensive error handling and logging
4. **Rate Limiting**: Add rate limiting for API protection
5. **Authentication**: Add authentication/authorization
6. **Pagproviders.py    # OpenRouterion for large result sets
7. **Async Processing**: Consider async processing for long-running queries

## Celery + Redis Preset KPI Cache

The backend now supports background precomputation for the unified KPI endpoint:

- Presets only: `7d`, `30d`, `all`
- No filters in this background cache flow
- Cache-first path in `POST /history/kpi/all`

### How It Works

1. API request hits `POST /history/kpi/all`.
2. If request is a supported preset with no filters, API checks Redis preset key first.
3. On cache hit, response is returned immediately.
4. On cache miss, API enqueues Celery preset precompute and falls back to live compute.
5. Live result is written back to Redis for subsequent fast reads.

### Cache Refresh / Invalidation

Use:

```bash
POST /history/cache/refresh
```

Body (optional):

```json
{
  "table_name": "history-orders-dev",
  "invalidate": true
}
```

- `invalidate=true` bumps cache version and invalidates old preset keys.
- This endpoint should be called after new order ingestion.

### Run Celery Worker

From `backend/`:

```bash
celery -A celery_app.celery_app worker --loglevel=info
```

Optional beat scheduler (if you later add periodic schedules):

```bash
celery -A celery_app.celery_app beat --loglevel=info
```

## File Structure

```
.
├── app.py              # Flask server
├── workflow.py         # LangGraph workflow
├── tools.py            # Tool functions
├── llm_mocks.py        # Mock LLM implementations
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your actual environment variables (git-ignored)
└── README.md           # This file
```
