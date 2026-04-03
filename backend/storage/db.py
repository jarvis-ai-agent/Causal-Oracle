import aiosqlite
import json
from pathlib import Path
from typing import Optional, List
from config import DB_PATH
from utils.logging import get_logger

logger = get_logger(__name__)


async def get_db():
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                current_stage INTEGER DEFAULT 0,
                stage_statuses TEXT DEFAULT '{}',
                logs TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                results_available INTEGER DEFAULT 0
            )
        """)
        await db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def save_run(run_data: dict):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            INSERT OR REPLACE INTO pipeline_runs
            (id, status, config, current_stage, stage_statuses, logs, created_at, completed_at, results_available)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data["id"],
            run_data["status"],
            json.dumps(run_data["config"]),
            run_data["current_stage"],
            json.dumps(run_data["stage_statuses"]),
            json.dumps(run_data["logs"]),
            run_data["created_at"],
            run_data.get("completed_at"),
            1 if run_data.get("results_available") else 0,
        ))
        await db.commit()


async def load_run(run_id: str) -> Optional[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_dict(row)


async def list_runs() -> List[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_dict(r) for r in rows]


async def delete_run(run_id: str) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        result = await db.execute(
            "DELETE FROM pipeline_runs WHERE id = ?", (run_id,)
        )
        await db.commit()
        return result.rowcount > 0


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["config"] = json.loads(d["config"])
    d["stage_statuses"] = json.loads(d["stage_statuses"])
    d["logs"] = json.loads(d["logs"])
    d["results_available"] = bool(d["results_available"])
    return d
