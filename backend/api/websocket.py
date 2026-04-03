"""WebSocket handler for real-time pipeline status."""
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from pipeline.orchestrator import register_ws_queue, unregister_ws_queue
from utils.logging import get_logger

logger = get_logger(__name__)


async def pipeline_ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    register_ws_queue(run_id, queue)
    logger.info(f"WebSocket connected for run {run_id}")

    try:
        # Send initial connection ack
        await websocket.send_text(json.dumps({
            "run_id": run_id,
            "event": "connected",
            "stage": 0,
            "stage_name": "Connected",
            "progress": 0.0,
            "message": f"Connected to pipeline run {run_id}",
            "timestamp": "",
        }))

        while True:
            try:
                # Wait for messages from the pipeline with a 30s timeout
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(msg)

                # Check if pipeline is done
                try:
                    data = json.loads(msg)
                    if data.get("event") in ("pipeline_complete", "pipeline_error"):
                        # Drain remaining messages then close
                        await asyncio.sleep(0.1)
                        while not queue.empty():
                            remaining = queue.get_nowait()
                            await websocket.send_text(remaining)
                        break
                except Exception:
                    pass
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text(json.dumps({"event": "ping", "run_id": run_id}))
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
    finally:
        unregister_ws_queue(run_id)
