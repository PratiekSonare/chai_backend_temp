"""
FastAPI Server for Order Analysis Workflow
"""
import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import defaultdict

# Load environment variables
load_dotenv()

# Import workflow components
from workflow import app as workflow_app, AgentState

# WebSocket connection manager for real-time logging
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.request_logs: Dict[str, List[Dict]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message with connection info
        await self.send_log({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Connected to log stream",
            "request_id": None,
            "type": "connection"
        })
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_log(self, log_data: Dict[str, Any]):
        """Send log to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(log_data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def log_request_start(self, request_id: str, endpoint: str, query: str = None):
        """Log the start of a request"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": f"Request started: {endpoint}",
            "request_id": request_id,
            "endpoint": endpoint,
            "query": query,
            "type": "request_start"
        }
        
        # Store log for this request
        if request_id not in self.request_logs:
            self.request_logs[request_id] = []
        self.request_logs[request_id].append(log_data)
        
        await self.send_log(log_data)
    
    async def log_request_step(self, request_id: str, step: str, details: str = None):
        """Log a step in request processing"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": f"Processing step: {step}",
            "request_id": request_id,
            "step": step,
            "details": details,
            "type": "request_step"
        }
        
        if request_id in self.request_logs:
            self.request_logs[request_id].append(log_data)
        
        await self.send_log(log_data)
    
    async def log_request_end(self, request_id: str, success: bool, result_summary: str = None, error: str = None):
        """Log the end of a request"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO" if success else "ERROR",
            "message": f"Request {'completed successfully' if success else 'failed'}: {result_summary or error}",
            "request_id": request_id,
            "success": success,
            "result_summary": result_summary,
            "error": error,
            "type": "request_end"
        }
        
        if request_id in self.request_logs:
            self.request_logs[request_id].append(log_data)
        
        await self.send_log(log_data)
    
    def get_request_logs(self, request_id: str) -> List[Dict]:
        """Get all logs for a specific request"""
        return self.request_logs.get(request_id, [])

# Global connection manager
manager = ConnectionManager()

app = FastAPI(
    title="Order Analysis Workflow API",
    description="FastAPI server for processing order analysis queries",
    version="1.0.0"
)

# Enable CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class HealthResponse(BaseModel):
    status: str
    service: str

class LogsRequest(BaseModel):
    request_id: str

class OrdersMetricsRequest(BaseModel):
    orders: List[Dict[str, Any]]


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for any client messages
            data = await websocket.receive_text()
            
            # Handle client requests (e.g., request specific logs)
            try:
                message = json.loads(data)
                if message.get("type") == "get_request_logs":
                    request_id = message.get("request_id")
                    if request_id:
                        logs = manager.get_request_logs(request_id)
                        await websocket.send_text(json.dumps({
                            "type": "request_logs",
                            "request_id": request_id,
                            "logs": logs
                        }))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get('/health', response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "order-analysis-workflow"
    }

@app.get('/logs/{request_id}')
def get_request_logs(request_id: str):
    """Get logs for a specific request ID"""
    logs = manager.get_request_logs(request_id)
    return {
        "request_id": request_id,
        "logs": logs,
        "total_logs": len(logs)
    }

@app.post('/plan')
async def generate_plan(request: QueryRequest):
    """
    Generate execution plan for a natural language query
    
    Request body:
    {
        "query": "Show me orders from last 5 days with payment mode prepaid"
    }
    
    Response:
    {
        "success": true,
        "query": "...",
        "plan": {
            "query_type": "standard",
            "steps": [...],
            "manipulation": {...}
        }
    }
    """
    # Generate unique request ID
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        user_query = request.query
        
        # Log request start
        await manager.log_request_start(request_id, "/plan", user_query)
        
        await manager.log_request_step(request_id, "initialize_state", "Setting up planning state")
        
        # Initialize state for planning only
        initial_state = AgentState(
            user_query=user_query,
            summarized_query=None,
            plan=None,
            tool_result_refs={},
            tool_result_schemas={},
            current_step_index=0,
            filters=None,
            final_result_ref=None,
            error=None,
            retry_count=0,
            comparison_mode=False,
            comparison_groups=None,
            group_results=None,
            group_schemas=None,
            current_group_index=0,
            aggregated_metrics=None,
            comparison_results=None,
            insights=None,
            metric_results=None,
            metric_analysis=None
        )
        
        # Run only the planning node
        await manager.log_request_step(request_id, "generate_plan", "Executing planning logic")
        from workflow import planning_node
        result = planning_node(initial_state)
        
        # Check for planning errors
        if result.get("error"):
            await manager.log_request_end(request_id, False, error=result["error"])
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": result["error"],
                    "request_id": request_id
                }
            )
        
        # Return the generated plan
        if result.get("plan"):
            plan_summary = f"Generated plan with {len(result['plan'].get('steps', []))} steps"
            await manager.log_request_end(request_id, True, plan_summary)
            return {
                "success": True,
                "query": user_query,
                "summarized_query": result.get("summarized_query", ""),
                "plan": result["plan"],
                "request_id": request_id
            }
        
        await manager.log_request_end(request_id, False, error="No plan generated")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No plan generated",
                "request_id": request_id
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@app.post('/query')
async def process_query(request: QueryRequest):
    """
    Process a natural language query
    
    Request body:
    {
        "query": "Show me orders from last 5 days with payment mode prepaid"
    }
    
    Response:
    {
        "success": true,
        "data": [...],
        "insights": "...",
        "metadata": {...}
    }
    """
    # Generate unique request ID
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        user_query = request.query
        
        # Log request start
        await manager.log_request_start(request_id, "/query", user_query)
        
        # Determine query type
        await manager.log_request_step(request_id, "analyze_query", "Determining query type")
        is_comparison = any(word in user_query.lower() for word in ["compare", "vs", "versus", "between"])
        is_metric = any(word in user_query.lower() for word in ["aov", "average order", "revenue", "metrics", "calculate", "total"])
        
        # Initialize state
        await manager.log_request_step(request_id, "initialize_workflow", f"Setting up {'comparison' if is_comparison else 'metric' if is_metric else 'standard'} workflow")
        initial_state = AgentState(
            user_query=user_query,
            summarized_query=None,
            plan=None,
            tool_result_refs={},
            tool_result_schemas={},
            current_step_index=0,
            filters=None,
            final_result_ref=None,
            error=None,
            retry_count=0,
            comparison_mode=is_comparison,
            comparison_groups=None,
            group_results=None,
            group_schemas=None,
            current_group_index=0,
            aggregated_metrics=None,
            comparison_results=None,
            insights=None,
            metric_results=None,
            metric_analysis=None
        )
        
        # Run workflow
        await manager.log_request_step(request_id, "execute_workflow", "Running order analysis workflow")
        result = workflow_app.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            await manager.log_request_end(request_id, False, error=result["error"])
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "error": result["error"],
                    "request_id": request_id
                }
            )
        
        # Get final result
        if result.get("final_result_ref"):
            await manager.log_request_step(request_id, "process_results", "Processing final results")
            from workflow import get_cached_result
            final_data = get_cached_result(result["final_result_ref"])
            
            # For comparison queries
            if is_comparison and isinstance(final_data, dict) and "insights" in final_data:
                await manager.log_request_end(request_id, True, "Comparison analysis completed successfully")
                return {
                    "success": True,
                    "query_type": "comparison",
                    "summarized_query": result.get("summarized_query", ""),
                    "insights": final_data["insights"],
                    "comparison_data": final_data.get("comparison_data"),
                    "detailed_metrics": final_data.get("detailed_metrics"),
                    "request_id": request_id
                }
            
            # For metric analysis queries
            elif isinstance(final_data, dict) and "metrics" in final_data and "analysis" in final_data:
                await manager.log_request_end(request_id, True, f"Metric analysis completed - calculated {len(final_data.get('metrics_calculated', []))} metrics")
                return {
                    "success": True,
                    "query_type": "metric_analysis",
                    "summarized_query": result.get("summarized_query", ""),
                    "query": final_data["query"],
                    "metrics": final_data["metrics"],
                    "analysis": final_data["analysis"],
                    "metrics_calculated": final_data.get("metrics_calculated", []),
                    "request_id": request_id
                }
            
            # For standard queries
            else:
                record_count = len(final_data) if isinstance(final_data, list) else 1
                await manager.log_request_end(request_id, True, f"Query completed successfully - returned {record_count} records")
                return {
                    "success": True,
                    "query_type": "standard",
                    "summarized_query": result.get("summarized_query", ""),
                    "count": record_count,
                    "data": final_data,  # Return all records (removed 100 record limit)
                    "total_records": record_count,
                    "request_id": request_id
                }
        
        await manager.log_request_end(request_id, False, error="No result generated")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "No result generated",
                "request_id": request_id
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )

@app.get('/examples')
def get_examples():
    """Get example queries and API usage"""
    return {
        "endpoints": {
            "/plan": "POST - Generate execution plan without running the query",
            "/query": "POST - Execute the full query workflow and return results",
            "/ws/logs": "WebSocket - Real-time log streaming for all requests",
            "/logs/{request_id}": "GET - Retrieve logs for a specific request ID"
        },
        "schema_discovery_queries": [
            "What fields are available in the orders data?",
            "What are the allowed values for payment_mode?",
            "Which date ranges does this API support?",
            "What marketplaces do we have data for?",
            "Show me the complete data schema",
            "What enum values are available for order_status?"
        ],
        "standard_queries": [
            "Show me orders from last 5 days with payment mode prepaid",
            "Get all open orders from last week",
            "Orders from Karnataka in last 10 days",
            "Show COD orders from last 3 days"
        ],
        "metric_analysis_queries": [
            "Calculate AOV from the past 2 days",
            "What is the average order value and revenue from last 7 days?",
            "Calculate total revenue and order count for last month",
            "Show me key metrics for orders from last week"
        ],
        "comparison_queries": [
            "Compare orders between Shopify13 and Flipkart from the last 10 days",
            "Compare prepaid vs COD orders from last week",
            "Compare Karnataka vs Maharashtra order volumes in last 15 days",
            "Compare Shopify13, Flipkart, and Amazon sales from last month"
        ]
    }

@app.post('/orders/metrics')
def calculate_orders_metrics(request: OrdersMetricsRequest):
    """
    Calculate comprehensive order metrics from order data
    
    Request body:
    {
        "orders": [...]  // Array of order objects
    }
    
    Response: Comprehensive metrics including volume, revenue, payment, cancellation, time-based, geographic, and product metrics
    """
    try:
        orders_data = request.orders
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        # Flatten suborders data
        suborders_list = []
        for _, order in df.iterrows():
            if 'suborders' in order and order['suborders']:
                for suborder in order['suborders']:
                    suborder_row = {
                        **order.to_dict(),  # Include all order-level data
                        **suborder  # Add suborder-specific data
                    }
                    suborders_list.append(suborder_row)
        
        suborders_df = pd.DataFrame(suborders_list) if suborders_list else pd.DataFrame()
        
        # Convert date columns to datetime
        date_columns = ['order_date', 'invoice_date', 'import_date', 'last_update_date', 'available_after']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['total_amount', 'total_tax', 'total_shipping_charge', 'order_quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if not suborders_df.empty:
            suborder_numeric_cols = ['selling_price', 'cost', 'mrp', 'item_quantity', 'cancelled_quantity', 'shipped_quantity']
            for col in suborder_numeric_cols:
                if col in suborders_df.columns:
                    suborders_df[col] = pd.to_numeric(suborders_df[col], errors='coerce').fillna(0)
        
        # ============ VOLUME METRICS ============
        total_orders = len(df)
        
        if not suborders_df.empty:
            total_skus_sold = suborders_df['shipped_quantity'].sum()
            net_skus_sold = suborders_df['item_quantity'].sum() - suborders_df['cancelled_quantity'].sum()
            unique_skus_ordered = suborders_df['sku'].nunique() if 'sku' in suborders_df.columns else 0
        else:
            total_skus_sold = 0
            net_skus_sold = 0
            unique_skus_ordered = 0
        
        # State-wise order count
        state_wise_orders = df['state'].value_counts().to_dict() if 'state' in df.columns else {}
        
        volume_metrics = {
            "total_orders": int(total_orders),
            "total_skus_sold": int(total_skus_sold),
            "net_skus_sold": int(net_skus_sold),
            "unique_skus_ordered": int(unique_skus_ordered),
            "state_wise_order_count": state_wise_orders
        }
        
        # ============ REVENUE METRICS ============
        gross_revenue = df['total_amount'].sum()
        
        if not suborders_df.empty:
            gross_cost = suborders_df['cost'].sum() * suborders_df['item_quantity'].sum() / suborders_df['item_quantity'].sum() if suborders_df['item_quantity'].sum() > 0 else 0
            gross_margin = gross_revenue - gross_cost if gross_cost > 0 else 0
            avg_selling_price = suborders_df['selling_price'].mean() if len(suborders_df) > 0 else 0
        else:
            gross_margin = 0
            avg_selling_price = 0
        
        aov = gross_revenue / total_orders if total_orders > 0 else 0
        
        revenue_metrics = {
            "gross_revenue": float(gross_revenue),
            "gross_margin": float(gross_margin),
            "aov": float(aov),
            "asp": float(avg_selling_price)
        }
        
        # ============ PAYMENT METRICS ============
        payment_breakdown = df['payment_mode'].value_counts().to_dict() if 'payment_mode' in df.columns else {}
        cod_orders = payment_breakdown.get('COD', 0)
        prepaid_orders = sum(count for mode, count in payment_breakdown.items() if mode != 'COD')
        
        payment_metrics = {
            "cod_vs_prepaid": {
                "cod_orders": int(cod_orders),
                "prepaid_orders": int(prepaid_orders),
                "cod_percentage": float(cod_orders / total_orders * 100) if total_orders > 0 else 0,
                "prepaid_percentage": float(prepaid_orders / total_orders * 100) if total_orders > 0 else 0
            },
            "payment_mode_breakdown": payment_breakdown
        }
        
        # ============ CANCELLATION METRICS ============
        cancelled_orders = len(df[df['order_status'] == 'Cancelled']) if 'order_status' in df.columns else 0
        cancellation_rate = (cancelled_orders / total_orders * 100) if total_orders > 0 else 0
        
        # RTO Rate calculation (assuming RTO orders have specific status)
        rto_orders = len(df[df['order_status'].str.contains('RTO|Return', case=False, na=False)]) if 'order_status' in df.columns else 0
        rto_rate = (rto_orders / total_orders * 100) if total_orders > 0 else 0
        
        # Highest state RTO
        if 'state' in df.columns and 'order_status' in df.columns:
            state_rto = df[df['order_status'].str.contains('RTO|Return', case=False, na=False)]['state'].value_counts()
            highest_state_rto = state_rto.index[0] if len(state_rto) > 0 else None
        else:
            highest_state_rto = None
        
        cancellation_metrics = {
            "rto_rate": float(rto_rate),
            "cancellation_rate": float(cancellation_rate),
            "cancelled_orders": int(cancelled_orders),
            "rto_orders": int(rto_orders),
            "highest_state_rto": highest_state_rto
        }
        
        # ============ TIME-BASED METRICS ============
        if 'order_date' in df.columns:
            df['order_day'] = df['order_date'].dt.date
            day_wise_frequency = df['order_day'].value_counts().sort_index().to_dict()
            
            # 7-day moving average
            daily_counts = df['order_day'].value_counts().sort_index()
            moving_avg_7d = daily_counts.rolling(window=7, min_periods=1).mean().to_dict()
        else:
            day_wise_frequency = {}
            moving_avg_7d = {}
        
        # Processing times calculation
        processing_times = {}
        if not suborders_df.empty:
            for _, suborder in suborders_df.iterrows():
                if 'suborder_history' in suborder and suborder['suborder_history']:
                    history = suborder['suborder_history']
                    
                    # Inventory assignment time
                    if history.get('inventory_assigned_datetime') and 'order_date' in suborder:
                        inv_time = pd.to_datetime(history['inventory_assigned_datetime']) - pd.to_datetime(suborder['order_date'])
                        processing_times.setdefault('inventory_assignment_times', []).append(inv_time.total_seconds() / 3600)  # hours
                    
                    # QC time
                    if history.get('qc_pass_datetime') and history.get('inventory_assigned_datetime'):
                        qc_time = pd.to_datetime(history['qc_pass_datetime']) - pd.to_datetime(history['inventory_assigned_datetime'])
                        processing_times.setdefault('qc_times', []).append(qc_time.total_seconds() / 3600)  # hours
        
        avg_inventory_time = np.mean(processing_times.get('inventory_assignment_times', [0])) if processing_times.get('inventory_assignment_times') else 0
        avg_qc_time = np.mean(processing_times.get('qc_times', [0])) if processing_times.get('qc_times') else 0
        avg_processing_time = avg_inventory_time + avg_qc_time
        
        time_based_metrics = {
            "day_wise_frequency": {str(k): int(v) for k, v in day_wise_frequency.items()},
            "moving_average_7d": {str(k): float(v) for k, v in moving_avg_7d.items()},
            "avg_inventory_assignment_time_hours": float(avg_inventory_time),
            "avg_qc_time_hours": float(avg_qc_time),
            "avg_order_processing_time_hours": float(avg_processing_time)
        }
        
        # ============ GEOGRAPHIC METRICS ============
        revenue_by_state = df.groupby('state')['total_amount'].sum().to_dict() if 'state' in df.columns else {}
        
        cancellation_by_state = {}
        if 'state' in df.columns and 'order_status' in df.columns:
            cancelled_by_state = df[df['order_status'] == 'Cancelled'].groupby('state').size()
            total_by_state = df.groupby('state').size()
            cancellation_by_state = ((cancelled_by_state / total_by_state) * 100).fillna(0).to_dict()
        
        geographic_metrics = {
            "state_wise_order_count": state_wise_orders,
            "revenue_by_state": {k: float(v) for k, v in revenue_by_state.items()},
            "cancellation_by_state": {k: float(v) for k, v in cancellation_by_state.items()}
        }
        
        # ============ PRODUCT METRICS ============
        if not suborders_df.empty:
            # Top SKUs by quantity
            top_skus_qty = suborders_df.groupby('sku')['item_quantity'].sum().sort_values(ascending=False).head(10).to_dict()
            
            # Top SKUs by revenue
            suborders_df['revenue'] = suborders_df['selling_price'] * suborders_df['item_quantity']
            top_skus_revenue = suborders_df.groupby('sku')['revenue'].sum().sort_values(ascending=False).head(10).to_dict()
            
            # Top products by name
            top_products = suborders_df.groupby('productName')['item_quantity'].sum().sort_values(ascending=False).head(10).to_dict() if 'productName' in suborders_df.columns else {}
        else:
            top_skus_qty = {}
            top_skus_revenue = {}
            top_products = {}
        
        product_metrics = {
            "top_skus_by_quantity": {k: int(v) for k, v in top_skus_qty.items()},
            "top_skus_by_revenue": {k: float(v) for k, v in top_skus_revenue.items()},
            "top_products_by_quantity": {k: int(v) for k, v in top_products.items()}
        }
        
        # ============ SUMMARY RESPONSE ============
        return {
            "success": True,
            "total_orders_analyzed": int(total_orders),
            "analysis_timestamp": datetime.now().isoformat(),
            "metrics": {
                "volume_metrics": volume_metrics,
                "revenue_metrics": revenue_metrics,
                "payment_metrics": payment_metrics,
                "cancellation_metrics": cancellation_metrics,
                "time_based_metrics": time_based_metrics,
                "geographic_metrics": geographic_metrics,
                "product_metrics": product_metrics
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error calculating metrics: {str(e)}"
            }
        )

@app.post('/orders/chart/count')
def order_count(request: OrdersMetricsRequest):
    """
        Depending on date-range, return monthly / weekly / daily count of orders.
        If daily, use monday, tuesday, wednesday...
        If weekly, use date-ranges.
        If monthly, mention months.
    """
    try:
        orders_data = request.orders
        
        if not orders_data:
            raise HTTPException(status_code=400, detail="No orders data provided")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(orders_data)
        
        # Convert order_date to datetime
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.dropna(subset=['order_date'])
        
        if df.empty:
            return {
                "success": True,
                "chart_type": "daily",
                "labels": [],
                "data": []
            }
        
        # Get date range
        min_date = df['order_date'].min()
        max_date = df['order_date'].max()
        date_range_days = (max_date - min_date).days
        
        # Determine chart type based on date range
        if date_range_days <= 7:
            # Daily view
            chart_type = "daily"
            df['date_group'] = df['order_date'].dt.date
            grouped = df.groupby('date_group').size()
            labels = [d.strftime('%A') for d in grouped.index]  # Monday, Tuesday, etc.
        elif date_range_days <= 90:
            # Weekly view
            chart_type = "weekly"
            df['week_start'] = df['order_date'].dt.to_period('W').dt.start_time
            df['date_group'] = df['week_start'].dt.date
            grouped = df.groupby('date_group').size()
            labels = [f"{d.strftime('%b %d')} - {(d + pd.Timedelta(days=6)).strftime('%b %d')}" for d in grouped.index]
        else:
            # Monthly view
            chart_type = "monthly"
            df['date_group'] = df['order_date'].dt.to_period('M').dt.start_time
            grouped = df.groupby('date_group').size()
            labels = [d.strftime('%B %Y') for d in grouped.index]
        
        # Prepare data for chart.js
        data = grouped.values.tolist()
        
        return {
            "success": True,
            "chart_type": chart_type,
            "labels": labels,
            "data": data,
            "total_orders": int(len(df)),
            "date_range_days": int(date_range_days)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error generating chart data: {str(e)}"
            }
        )

if __name__ == '__main__':
    import uvicorn
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 Order Analysis Workflow Server (FastAPI)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"📡 Server: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"🧠 Plan: POST http://{host}:{port}/plan", flush=True)
    print(f"🔍 Query: POST http://{host}:{port}/query", flush=True)
    print(f"📖 Docs: http://{host}:{port}/docs", flush=True)
    print(f"📋 ReDoc: http://{host}:{port}/redoc", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
