"""
Flask API Server for Order Analysis Workflow
"""
import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import workflow components
from workflow import app as workflow_app, AgentState

app = Flask(__name__)

# Disable Flask's default logger to prevent duplicate logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "order-analysis-workflow"
    })

@app.route('/query', methods=['POST'])
def process_query():
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
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' in request body"
            }), 400
        
        user_query = data['query']
        
        # Determine if it's a comparison query
        is_comparison = any(word in user_query.lower() for word in ["compare", "vs", "versus", "between"])
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
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
            insights=None
        )
        
        # Run workflow
        result = workflow_app.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
        
        # Get final result
        if result.get("final_result_ref"):
            from workflow import get_cached_result
            final_data = get_cached_result(result["final_result_ref"])
            
            # For comparison queries
            if is_comparison and isinstance(final_data, dict) and "insights" in final_data:
                return jsonify({
                    "success": True,
                    "query_type": "comparison",
                    "insights": final_data["insights"],
                    "comparison_data": final_data.get("comparison_data"),
                    "detailed_metrics": final_data.get("detailed_metrics")
                })
            
            # For standard queries
            else:
                return jsonify({
                    "success": True,
                    "query_type": "standard",
                    "count": len(final_data) if isinstance(final_data, list) else 1,
                    "data": final_data[:100] if isinstance(final_data, list) else final_data,  # Limit to 100 records
                    "total_records": len(final_data) if isinstance(final_data, list) else 1
                })
        
        return jsonify({
            "success": False,
            "error": "No result generated"
        }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get example queries"""
    return jsonify({
        "standard_queries": [
            "Show me orders from last 5 days with payment mode prepaid",
            "Get all open orders from last week",
            "Orders from Karnataka in last 10 days",
            "Show COD orders from last 3 days"
        ],
        "comparison_queries": [
            "Compare orders between Shopify13 and Flipkart from the last 10 days",
            "Compare prepaid vs COD orders from last week",
            "Compare Karnataka vs Maharashtra order volumes in last 15 days"
        ]
    })

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 Order Analysis Workflow Server", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"📡 Server: http://{host}:{port}", flush=True)
    print(f"❤️  Health: http://{host}:{port}/health", flush=True)
    print(f"📝 Examples: http://{host}:{port}/examples", flush=True)
    print(f"🔍 Query: POST http://{host}:{port}/query", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    app.run(host=host, port=port, debug=debug, use_reloader=False)
