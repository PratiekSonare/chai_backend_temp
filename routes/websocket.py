import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from utils.connection_manager import manager

router = APIRouter()

@router.get("/sse/logs")
async def sse_logs():
    queue = asyncio.Queue()
    client_id = await manager.connect(queue)
    
    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except Exception:
            pass
        finally:
            manager.disconnect(client_id)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get('/logs/{request_id}')
def get_request_logs(request_id: str):
    logs = manager.get_request_logs(request_id)
    return {
        "request_id": request_id,
        "logs": logs
    }
