
import json
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from app.auth.security import decode_access_token, COOKIE_NAME
from app.core.manager import manager

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New dashboard WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Dashboard WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
            
        # Standardize message to JSON string if it's a dict
        if isinstance(message, dict):
            message_str = json.dumps(message)
        else:
            message_str = str(message)

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# Singleton manager for browser connections
browser_ws_manager = ConnectionManager()

@ws_router.websocket("/ws/dashboard")
async def dashboard_websocket(
    websocket: WebSocket,
    token: str = Query(None)
):
    """
    WebSocket endpoint for real-time dashboard updates.
    Expects JWT token in query string for authentication.
    """
    # 1. Authenticate
    if not token:
        # Check cookies as backup
        token = websocket.cookies.get(COOKIE_NAME)
        
    if not token:
        await websocket.close(code=4001) # Unauthorized
        return

    payload = decode_access_token(token)
    if not payload:
        await websocket.close(code=4001)
        return

    username = payload.get("sub")
    logger.info(f"User {username} connecting to dashboard WebSocket")

    # 2. Connect
    await browser_ws_manager.connect(websocket)
    
    try:
        # Keep connection alive
        while True:
            # We don't expect much from the client, but we need to listen for pings/disconnects
            data = await websocket.receive_text()
            # Handle client-to-server messages if needed
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except:
                pass
    except WebSocketDisconnect:
        browser_ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        browser_ws_manager.disconnect(websocket)
