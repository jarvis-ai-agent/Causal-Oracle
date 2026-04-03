"""Causal Oracle — FastAPI application entry point."""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from storage.db import init_db
from api.routes import router
from api.websocket import pipeline_ws
from utils.logging import get_logger
import config as cfg

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Causal Oracle API server")
    await init_db()
    yield
    logger.info("Shutting down Causal Oracle API server")


app = FastAPI(
    title="Causal Oracle",
    description="Causal inference-driven stock prediction engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.websocket("/ws/pipeline/{run_id}")
async def ws_pipeline(websocket: WebSocket, run_id: str):
    await pipeline_ws(websocket, run_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=True,
        log_level=cfg.LOG_LEVEL.lower(),
    )
