"""REST API routes."""
from fastapi import APIRouter, HTTPException
from typing import List
from models.schemas import PipelineConfig, PipelineRun, RunListItem
from storage import db as storage_db
from storage import artifacts
from pipeline.orchestrator import start_pipeline
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


@router.post("/pipeline/run")
async def create_run(config: PipelineConfig):
    run_id = await start_pipeline(config)
    return {"run_id": run_id}


@router.get("/pipeline/runs")
async def list_runs():
    runs = await storage_db.list_runs()
    result = []
    for r in runs:
        cfg = r.get("config", {})
        result.append({
            "id": r["id"],
            "status": r["status"],
            "target": cfg.get("target", ""),
            "tickers": cfg.get("tickers", []),
            "created_at": r["created_at"],
            "completed_at": r.get("completed_at"),
            "current_stage": r.get("current_stage", 0),
        })
    return result


@router.get("/pipeline/runs/{run_id}")
async def get_run(run_id: str):
    run = await storage_db.load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.delete("/pipeline/runs/{run_id}")
async def delete_run(run_id: str):
    deleted = await storage_db.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted"}


@router.get("/pipeline/runs/{run_id}/graph")
async def get_graph(run_id: str):
    data = artifacts.load_artifact(run_id, "stage04_graph")
    if data is None:
        raise HTTPException(status_code=404, detail="Causal graph not available")
    # Enrich with validation status
    validated = artifacts.load_artifact(run_id, "stage05_validated_parents") or {}
    dropped = artifacts.load_artifact(run_id, "stage05_dropped_parents") or {}

    validated_names = set()
    for parents in validated.values():
        for p in parents:
            validated_names.add(p["name"])
    dropped_names = set()
    for parents in dropped.values():
        for p in parents:
            dropped_names.add(p["name"])

    for edge in data.get("edges", []):
        src = edge.get("source", "")
        if src in validated_names:
            edge["validated"] = True
        elif src in dropped_names:
            edge["validated"] = False

    return data


@router.get("/pipeline/runs/{run_id}/forecast")
async def get_forecast(run_id: str):
    data = artifacts.load_artifact(run_id, "stage07_forecasts")
    if data is None:
        raise HTTPException(status_code=404, detail="Forecast not available")
    # Include features index for dates
    try:
        import pandas as pd
        from pathlib import Path
        from config import ARTIFACTS_DIR
        feat_path = ARTIFACTS_DIR / run_id / "stage02_features.parquet"
        if feat_path.exists():
            df = pd.read_parquet(str(feat_path))
            dates = [str(d)[:10] for d in df.index]
            data["dates"] = dates[-512:]
    except Exception:
        pass
    return data


@router.get("/pipeline/runs/{run_id}/backtest")
async def get_backtest(run_id: str):
    equity = artifacts.load_artifact(run_id, "stage08_equity_curve")
    trades = artifacts.load_artifact(run_id, "stage08_trades")
    metrics = artifacts.load_artifact(run_id, "stage08_metrics")
    signal_decay = artifacts.load_artifact(run_id, "stage08_signal_decay")
    regime_perf = artifacts.load_artifact(run_id, "stage08_regime_performance")
    comparison = artifacts.load_artifact(run_id, "stage08_comparison")

    if metrics is None:
        raise HTTPException(status_code=404, detail="Backtest results not available")

    return {
        "equity_curve": equity or {},
        "trades": trades or [],
        "metrics": metrics or {},
        "signal_decay": signal_decay or {},
        "regime_performance": regime_perf or {},
        "comparison": comparison or {},
    }


@router.get("/pipeline/runs/{run_id}/regimes")
async def get_regimes(run_id: str):
    regime_map = artifacts.load_artifact(run_id, "stage06_regime_map")
    transition = artifacts.load_artifact(run_id, "stage06_transition_matrix")
    distribution = artifacts.load_artifact(run_id, "stage06_distribution")
    current = artifacts.load_artifact(run_id, "stage06_current_regime")
    labels = artifacts.load_artifact(run_id, "stage06_regime_labels")

    if regime_map is None:
        raise HTTPException(status_code=404, detail="Regime data not available")

    return {
        "regime_map": regime_map,
        "transition_matrix": transition,
        "regime_distribution": distribution,
        "current_regime": current,
        "regime_labels": labels or {},
    }


@router.get("/pipeline/runs/{run_id}/validation")
async def get_validation(run_id: str):
    validated = artifacts.load_artifact(run_id, "stage05_validated_parents")
    dropped = artifacts.load_artifact(run_id, "stage05_dropped_parents")
    report = artifacts.load_artifact(run_id, "stage05_refutation_report")
    metadata = artifacts.load_artifact(run_id, "stage05_metadata")

    if validated is None:
        raise HTTPException(status_code=404, detail="Validation results not available")

    return {
        "validated_parents": validated,
        "dropped_parents": dropped or {},
        "refutation_report": report or [],
        "validation_metadata": metadata or {},
    }


@router.get("/pipeline/runs/{run_id}/logs")
async def get_logs(run_id: str):
    run = await storage_db.load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.get("logs", [])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "causal-oracle"}
