import json
from datetime import datetime
from typing import Dict, List, Any
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.request_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.client_counter = 0
    
    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = f"client_{self.client_counter}"
        self.client_counter += 1
        self.active_connections[client_id] = websocket
        return client_id
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def broadcast_log(self, log_data: Dict[str, Any]):
        """Broadcast log data to all connected WebSocket clients"""
        if self.active_connections:
            message = json.dumps(log_data)
            disconnected_clients = []
            
            for client_id, connection in self.active_connections.items():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"Error sending message to {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Remove disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def log_request_start(self, request_id: str, query: str):
        """Log the start of a request"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "step": "REQUEST_START",
            "message": f"Starting query processing: {query}",
            "query": query
        }
        
        # Store in request logs
        if request_id not in self.request_logs:
            self.request_logs[request_id] = []
        self.request_logs[request_id].append(log_entry)
        
        # Broadcast to WebSocket clients
        await self.broadcast_log(log_entry)
    
    async def log_request_step(self, request_id: str, step: str, message: str, **kwargs):
        """Log an individual step in request processing"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "step": step,
            "message": message,
            **kwargs
        }
        
        # Store in request logs
        if request_id not in self.request_logs:
            self.request_logs[request_id] = []
        self.request_logs[request_id].append(log_entry)
        
        # Broadcast to WebSocket clients
        await self.broadcast_log(log_entry)
    
    async def log_request_end(self, request_id: str, has_error: bool = False, error: str = None):
        """Log the end of a request"""
        level = "ERROR" if has_error else "SUCCESS"
        message = f"Request completed with error: {error}" if has_error else "Request completed successfully"
        
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "step": "REQUEST_END",
            "message": message,
            "has_error": has_error
        }
        
        if error:
            log_entry["error"] = error
        
        # Store in request logs
        if request_id not in self.request_logs:
            self.request_logs[request_id] = []
        self.request_logs[request_id].append(log_entry)
        
        # Broadcast to WebSocket clients
        await self.broadcast_log(log_entry)
    
    def get_request_logs(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific request ID"""
        return self.request_logs.get(request_id, [])

# Global connection manager instance
manager = ConnectionManager()
