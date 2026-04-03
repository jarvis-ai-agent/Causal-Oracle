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
            "start_date": cfg.get("start_date"),
            "end_date": cfg.get("end_date"),
            "created_at": r["created_at"],
            "started_at": r.get("started_at"),
            "completed_at": r.get("completed_at"),
            "total_duration_sec": r.get("total_duration_sec"),
            "current_stage": r.get("current_stage", 0),
            "results_available": r.get("results_available", False),
            "stage_statuses": r.get("stage_statuses", {}),
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


@router.get("/pipeline/runs/{run_id}/export")
async def export_run(run_id: str):
    """Export a complete, well-organised JSON snapshot of every pipeline stage."""
    run = await storage_db.load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    cfg = run.get("config", {})
    stage_statuses = run.get("stage_statuses", {})

    # Compute total duration from stage timings
    total_sec = sum(
        (v.get("duration_sec") or 0)
        for v in stage_statuses.values()
    )

    export = {
        "export_version": "1.0",
        "exported_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "run": {
            "id": run_id,
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "completed_at": run.get("completed_at"),
            "total_duration_sec": round(total_sec, 2),
        },
        "configuration": {
            "tickers": cfg.get("tickers", []),
            "target": cfg.get("target"),
            "date_range": {
                "start": cfg.get("start_date"),
                "end": cfg.get("end_date"),
            },
            "model_params": {
                "max_lag": cfg.get("max_lag"),
                "alpha": cfg.get("alpha"),
                "independence_test": cfg.get("indep_test"),
                "n_regimes": cfg.get("n_regimes"),
                "forecast_horizon": cfg.get("horizon"),
                "context_length": cfg.get("context_length"),
            },
            "backtest_params": {
                "initial_capital": cfg.get("initial_capital"),
                "causal_retrain_interval": cfg.get("causal_retrain_interval"),
                "forecast_retrain_interval": cfg.get("forecast_retrain_interval"),
            },
            "data_sources": {
                "include_macro": cfg.get("include_macro"),
                "include_fama_french": cfg.get("include_factors"),
            },
        },
        "stages": {},
        "results": {},
        "logs": run.get("logs", []),
    }

    # Stage metadata & timings
    stage_names = {
        "1": "data_ingestion", "2": "feature_engineering", "3": "lag_matrix",
        "4": "causal_discovery", "5": "causal_validation", "6": "regime_detection",
        "7": "forecasting", "8": "backtesting",
    }
    for k, name in stage_names.items():
        st = stage_statuses.get(k, stage_statuses.get(int(k), {}))
        export["stages"][name] = {
            "status": st.get("status", "pending"),
            "duration_sec": st.get("duration_sec"),
            "summary": st.get("summary"),
            "error": st.get("error"),
        }

    # Stage 01 — ingest metadata
    ingest_meta = artifacts.load_artifact(run_id, "stage01_metadata")
    if ingest_meta:
        export["stages"]["data_ingestion"]["metadata"] = ingest_meta

    # Stage 02 — features
    feat_names = artifacts.load_artifact(run_id, "stage02_feature_names")
    stationarity = artifacts.load_artifact(run_id, "stage02_stationarity")
    if feat_names:
        export["stages"]["feature_engineering"]["feature_names"] = feat_names
    if stationarity:
        export["stages"]["feature_engineering"]["stationarity_report"] = stationarity

    # Stage 03 — lag matrix config
    lag_config = artifacts.load_artifact(run_id, "stage03_lag_config")
    lag_cols = artifacts.load_artifact(run_id, "stage03_column_names")
    if lag_config:
        export["stages"]["lag_matrix"]["config"] = lag_config
    if lag_cols:
        export["stages"]["lag_matrix"]["columns"] = lag_cols

    # Stage 04 — causal discovery
    graph = artifacts.load_artifact(run_id, "stage04_graph")
    causal_parents = artifacts.load_artifact(run_id, "stage04_causal_parents")
    discovery_meta = artifacts.load_artifact(run_id, "stage04_metadata")
    if graph:
        export["stages"]["causal_discovery"]["graph"] = graph
    if causal_parents:
        export["stages"]["causal_discovery"]["causal_parents"] = causal_parents
    if discovery_meta:
        export["stages"]["causal_discovery"]["metadata"] = discovery_meta

    # Stage 05 — validation
    validated = artifacts.load_artifact(run_id, "stage05_validated_parents")
    dropped = artifacts.load_artifact(run_id, "stage05_dropped_parents")
    val_meta = artifacts.load_artifact(run_id, "stage05_metadata")
    refutation = artifacts.load_artifact(run_id, "stage05_refutation_report")
    if validated:
        export["stages"]["causal_validation"]["validated_parents"] = validated
    if dropped:
        export["stages"]["causal_validation"]["dropped_parents"] = dropped
    if val_meta:
        export["stages"]["causal_validation"]["metadata"] = val_meta
    if refutation:
        export["stages"]["causal_validation"]["refutation_report"] = refutation

    # Stage 06 — regimes
    regime_map = artifacts.load_artifact(run_id, "stage06_regime_map")
    transition = artifacts.load_artifact(run_id, "stage06_transition_matrix")
    distribution = artifacts.load_artifact(run_id, "stage06_distribution")
    current_regime = artifacts.load_artifact(run_id, "stage06_current_regime")
    regime_labels = artifacts.load_artifact(run_id, "stage06_regime_labels")
    if regime_map:
        export["stages"]["regime_detection"]["regime_map"] = regime_map
    if transition:
        export["stages"]["regime_detection"]["transition_matrix"] = transition
    if distribution:
        export["stages"]["regime_detection"]["distribution"] = distribution
    if current_regime:
        export["stages"]["regime_detection"]["current_regime"] = current_regime
    if regime_labels:
        export["stages"]["regime_detection"]["regime_labels"] = regime_labels

    # Stage 07 — forecasts
    forecasts = artifacts.load_artifact(run_id, "stage07_forecasts")
    if forecasts:
        export["stages"]["forecasting"]["forecasts"] = forecasts

    # Stage 08 — backtest
    metrics = artifacts.load_artifact(run_id, "stage08_metrics")
    equity = artifacts.load_artifact(run_id, "stage08_equity_curve")
    trades = artifacts.load_artifact(run_id, "stage08_trades")
    signal_decay = artifacts.load_artifact(run_id, "stage08_signal_decay")
    regime_perf = artifacts.load_artifact(run_id, "stage08_regime_performance")
    comparison = artifacts.load_artifact(run_id, "stage08_comparison")
    if metrics:
        export["stages"]["backtesting"]["metrics"] = metrics
    if equity:
        export["stages"]["backtesting"]["equity_curve"] = equity
    if trades:
        export["stages"]["backtesting"]["trades"] = trades
    if signal_decay:
        export["stages"]["backtesting"]["signal_decay"] = signal_decay
    if regime_perf:
        export["stages"]["backtesting"]["regime_performance"] = regime_perf
    if comparison:
        export["stages"]["backtesting"]["benchmark_comparison"] = comparison

    # Top-level results summary
    if metrics:
        export["results"]["backtest_summary"] = metrics
    if current_regime:
        export["results"]["current_regime"] = current_regime
    if forecasts:
        export["results"]["forecast_assets"] = list(forecasts.keys())
    if discovery_meta:
        export["results"]["causal_edges"] = discovery_meta.get("n_directed")

    from fastapi.responses import JSONResponse
    from utils.serialization import NumpyEncoder
    import json as _json

    # Safe serialize — replace any unserializable values with an error note
    try:
        content = _json.dumps(export, cls=NumpyEncoder, indent=2)
    except Exception as e:
        logger.error(f"Export serialization error for {run_id}: {e}")
        # Strip problematic keys and try again
        for stage_key in export.get("stages", {}).values():
            for k in list(stage_key.keys()):
                try:
                    _json.dumps(stage_key[k], cls=NumpyEncoder)
                except Exception:
                    stage_key[k] = f"[unserializable: {type(stage_key[k]).__name__}]"
        content = _json.dumps(export, cls=NumpyEncoder, indent=2)

    return JSONResponse(
        content=_json.loads(content),
        headers={"Content-Disposition": f'attachment; filename="causal-oracle-{run_id}.json"'}
    )


_AV_KEY = "P2UE8R8FJ5DD89DN"

@router.get("/search")
async def search_symbols(q: str):
    """Search for stock symbols using AlphaVantage SYMBOL_SEARCH."""
    if not q or len(q.strip()) < 1:
        return []
    try:
        import httpx
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=SYMBOL_SEARCH&keywords={q.strip()}&apikey={_AV_KEY}"
        )
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            data = resp.json()
        matches = data.get("bestMatches", [])
        return [
            {
                "symbol":   m.get("1. symbol", ""),
                "name":     m.get("2. name", ""),
                "type":     m.get("3. type", ""),
                "region":   m.get("4. region", ""),
                "currency": m.get("8. currency", ""),
            }
            for m in matches
            if m.get("1. symbol")
        ]
    except Exception as e:
        logger.warning(f"Symbol search failed for '{q}': {e}")
        return []


@router.get("/health")
async def health():
    return {"status": "ok", "service": "causal-oracle"}
