from fastapi import APIRouter, WebSocket
from utils.connection_manager import manager

router = APIRouter()

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

@router.get('/logs/{request_id}')
def get_request_logs(request_id: str):
    logs = manager.get_request_logs(request_id)
    return {
        "request_id": request_id,
        "logs": logs
    }
