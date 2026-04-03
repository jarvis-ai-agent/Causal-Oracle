"""Pipeline Orchestrator — manages stage execution, status tracking, and WebSocket events."""
import asyncio
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from utils.logging import get_logger, iso_now
from storage import db as storage_db
from storage import artifacts
from models.schemas import PipelineConfig
from utils.serialization import NumpyEncoder, series_to_dict, df_to_records
import json
import numpy as np
import pandas as pd

logger = get_logger(__name__)

STAGE_NAMES = {
    1: "Data Ingestion",
    2: "Feature Engineering",
    3: "Lag Matrix",
    4: "Causal Discovery",
    5: "Causal Validation",
    6: "Regime Detection",
    7: "Forecasting",
    8: "Backtesting",
}

# Global registry of active runs and their websocket queues
_active_runs: Dict[str, dict] = {}
_ws_queues: Dict[str, asyncio.Queue] = {}


def register_ws_queue(run_id: str, queue: asyncio.Queue):
    _ws_queues[run_id] = queue


def unregister_ws_queue(run_id: str):
    _ws_queues.pop(run_id, None)


async def _emit_ws(run_id: str, event: dict):
    queue = _ws_queues.get(run_id)
    if queue:
        try:
            await queue.put(json.dumps(event, cls=NumpyEncoder))
        except Exception as e:
            logger.debug(f"WS emit error: {e}")


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig):
        self.run_id = str(uuid.uuid4())[:8]
        self.config = config
        self.status = "pending"
        self.current_stage = 0
        self.started_at = None
        self.stage_statuses: Dict[int, dict] = {
            i: {"status": "pending", "duration_sec": None, "output_path": None, "summary": None, "error": None}
            for i in range(1, 9)
        }
        self.logs = []
        self.created_at = iso_now()
        self.completed_at = None
        self._stage_data: Dict[str, Any] = {}

    def _log(self, msg: str, level: str = "INFO", stage: Optional[int] = None):
        entry = {"timestamp": iso_now(), "level": level, "stage": stage, "message": msg}
        self.logs.append(entry)
        logger.info(f"[run:{self.run_id}] {msg}")

    async def _emit(self, event: str, stage: int, progress: float, message: str, data: Any = None):
        payload = {
            "run_id": self.run_id,
            "event": event,
            "stage": stage,
            "stage_name": STAGE_NAMES.get(stage, f"Stage {stage}"),
            "progress": progress,
            "message": message,
            "timestamp": iso_now(),
        }
        if data is not None:
            payload["data"] = data
        await _emit_ws(self.run_id, payload)

    async def _save(self):
        run_data = {
            "id": self.run_id,
            "status": self.status,
            "config": self.config.model_dump(),
            "current_stage": self.current_stage,
            "stage_statuses": self.stage_statuses,
            "logs": self.logs[-200:],
            "created_at": self.created_at,
            "started_at": getattr(self, "started_at", None),
            "completed_at": self.completed_at,
            "total_duration_sec": getattr(self, "total_duration_sec", None),
            "results_available": self.status == "completed",
        }
        await storage_db.save_run(run_data)

    def _make_progress_cb(self, stage: int) -> Callable:
        """Create a progress callback that emits WS events for a stage."""
        loop = asyncio.get_event_loop()

        def cb(pct: float, msg: str):
            self._log(msg, stage=stage)
            asyncio.run_coroutine_threadsafe(
                self._emit("stage_progress", stage, pct, msg),
                loop,
            )

        return cb

    async def run(self):
        import time as _time
        self.status = "running"
        self.started_at = iso_now()
        _pipeline_start = _time.time()
        self._log(f"Pipeline started: tickers={self.config.tickers}, target={self.config.target}")
        await self._save()
        await self._emit("stage_start", 0, 0.0, "Pipeline starting")

        try:
            await self._run_stage_01()
            await self._run_stage_02()
            await self._run_stage_03()
            await self._run_stage_04()
            await self._run_stage_05()
            await self._run_stage_06()
            await self._run_stage_07()
            await self._run_stage_08()

            self.status = "completed"
            self.completed_at = iso_now()
            self.total_duration_sec = round(_time.time() - _pipeline_start, 2)
            self._log(f"Pipeline completed successfully in {self.total_duration_sec}s")
            await self._emit("pipeline_complete", 8, 1.0, f"Pipeline completed in {self.total_duration_sec}s")

        except Exception as e:
            self.status = "failed"
            self.completed_at = iso_now()
            self.total_duration_sec = round(_time.time() - _pipeline_start, 2)
            err_msg = f"Pipeline failed: {e}\n{traceback.format_exc()}"
            self._log(err_msg, level="ERROR")
            await self._emit("pipeline_error", self.current_stage, 0.0, str(e))

        await self._save()

    async def _run_stage(self, stage_num: int, fn, **kwargs):
        """Generic stage runner with timing, error handling, and WS events."""
        import time
        self.current_stage = stage_num
        self.stage_statuses[stage_num]["status"] = "running"
        await self._emit("stage_start", stage_num, 0.0, f"Starting {STAGE_NAMES[stage_num]}")
        await self._save()
        t0 = time.time()
        try:
            progress_cb = self._make_progress_cb(stage_num)
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fn(progress_cb=progress_cb, **kwargs)
            )
            duration = time.time() - t0
            self.stage_statuses[stage_num]["status"] = "completed"
            self.stage_statuses[stage_num]["duration_sec"] = round(duration, 2)
            await self._emit("stage_complete", stage_num, 1.0, f"{STAGE_NAMES[stage_num]} complete ({duration:.1f}s)")
            await self._save()
            return result
        except Exception as e:
            duration = time.time() - t0
            self.stage_statuses[stage_num]["status"] = "failed"
            self.stage_statuses[stage_num]["duration_sec"] = round(duration, 2)
            self.stage_statuses[stage_num]["error"] = str(e)
            await self._emit("stage_error", stage_num, 0.0, f"{STAGE_NAMES[stage_num]} failed: {e}")
            await self._save()
            raise

    async def _run_stage_01(self):
        from pipeline import stage_01_ingest
        result = await self._run_stage(
            1, stage_01_ingest.run,
            tickers=self.config.tickers,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            include_macro=self.config.include_macro,
            include_factors=self.config.include_factors,
        )
        self._stage_data["ingest"] = result
        summary = f"Loaded {result.metadata['T']} rows × {result.metadata['N']} columns"
        self.stage_statuses[1]["summary"] = summary
        # Save serializable metadata
        artifacts.save_artifact(self.run_id, "stage01_metadata", result.metadata)
        artifacts.save_artifact(self.run_id, "stage01_column_map", result.column_map)
        # Save dataframe as parquet for later
        try:
            result.raw_df.to_parquet(str(artifacts.run_dir(self.run_id) / "stage01_raw.parquet"))
        except Exception:
            pass
        await self._save()

    async def _run_stage_02(self):
        from pipeline import stage_02_features
        ingest = self._stage_data["ingest"]
        result = await self._run_stage(
            2, stage_02_features.run,
            raw_df=ingest.raw_df,
            column_map=ingest.column_map,
        )
        self._stage_data["features"] = result
        n_diff = sum(1 for v in result.stationarity_report.values() if v.get("differenced"))
        summary = f"Engineered {len(result.feature_names)} features, {n_diff} required differencing"
        self.stage_statuses[2]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage02_feature_names", result.feature_names)
        artifacts.save_artifact(self.run_id, "stage02_stationarity", result.stationarity_report)
        try:
            result.features_df.to_parquet(str(artifacts.run_dir(self.run_id) / "stage02_features.parquet"))
        except Exception:
            pass
        await self._save()

    async def _run_stage_03(self):
        from pipeline import stage_03_lagmatrix
        features = self._stage_data["features"]
        result = await self._run_stage(
            3, stage_03_lagmatrix.run,
            features_df=features.features_df,
            max_lag=self.config.max_lag,
            target_col=self.config.target,
        )
        self._stage_data["lagmatrix"] = result
        summary = f"Lag matrix: {result.lag_matrix.shape[0]} rows × {result.lag_matrix.shape[1]} features"
        self.stage_statuses[3]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage03_lag_config", result.lag_config)
        artifacts.save_artifact(self.run_id, "stage03_column_names", result.column_names)
        await self._save()

    async def _run_stage_04(self):
        from pipeline import stage_04_causal
        lm = self._stage_data["lagmatrix"]
        result = await self._run_stage(
            4, stage_04_causal.run,
            lag_matrix=lm.lag_matrix,
            c_indx=lm.c_indx,
            column_names=lm.column_names,
            target_col_index=lm.target_col_index,
            alpha=self.config.alpha,
            indep_test=self.config.indep_test,
        )
        self._stage_data["causal"] = result
        n_parents = sum(len(v) for v in result.causal_parents.values())
        summary = (f"Discovered {result.discovery_metadata['n_directed']} directed edges, "
                   f"{n_parents} causal parents")
        self.stage_statuses[4]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage04_graph", result.graph_json)
        artifacts.save_artifact(self.run_id, "stage04_causal_parents", result.causal_parents)
        artifacts.save_artifact(self.run_id, "stage04_metadata", result.discovery_metadata)
        artifacts.save_artifact(self.run_id, "stage04_nonstationary", result.nonstationary_vars)
        await self._save()

    async def _run_stage_05(self):
        from pipeline import stage_05_validate
        causal = self._stage_data["causal"]
        features = self._stage_data["features"]
        result = await self._run_stage(
            5, stage_05_validate.run,
            causal_parents=causal.causal_parents,
            features_df=features.features_df,
            target_col=self.config.target,
        )
        self._stage_data["validation"] = result
        passed = result.validation_metadata["passed"]
        total = result.validation_metadata["total_tested"]
        summary = f"{passed}/{total} causal parents passed validation"
        self.stage_statuses[5]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage05_validated_parents", result.validated_parents)
        artifacts.save_artifact(self.run_id, "stage05_dropped_parents", result.dropped_parents)
        artifacts.save_artifact(self.run_id, "stage05_metadata", result.validation_metadata)
        # Save refutation report
        try:
            report_records = df_to_records(result.refutation_report)
            artifacts.save_artifact(self.run_id, "stage05_refutation_report", report_records)
        except Exception:
            pass
        await self._save()

    async def _run_stage_06(self):
        from pipeline import stage_06_regime
        features = self._stage_data["features"]
        result = await self._run_stage(
            6, stage_06_regime.run,
            features_df=features.features_df,
            n_regimes=self.config.n_regimes,
        )
        self._stage_data["regime"] = result
        summary = f"Current regime: {result.current_regime}, distribution: {result.regime_distribution}"
        self.stage_statuses[6]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage06_regime_map", result.regime_map)
        artifacts.save_artifact(self.run_id, "stage06_transition_matrix", result.transition_matrix.tolist())
        artifacts.save_artifact(self.run_id, "stage06_distribution", result.regime_distribution)
        artifacts.save_artifact(self.run_id, "stage06_current_regime", result.current_regime)
        # Save regime labels with dates
        try:
            features_df = features.features_df
            regime_series = {
                str(features_df.index[i]): int(result.regime_labels[i])
                for i in range(min(len(features_df), len(result.regime_labels)))
            }
            artifacts.save_artifact(self.run_id, "stage06_regime_labels", regime_series)
        except Exception as e:
            logger.warning(f"Failed to save regime labels: {e}")
        artifacts.save_joblib(self.run_id, "stage06_hmm_model", result.hmm_model)
        await self._save()

    async def _run_stage_07(self):
        from pipeline import stage_07_forecast
        validation = self._stage_data["validation"]
        features = self._stage_data["features"]
        regime = self._stage_data["regime"]
        result = await self._run_stage(
            7, stage_07_forecast.run,
            validated_parents=validation.validated_parents,
            features_df=features.features_df,
            regime_labels=regime.regime_labels,
            target_col=self.config.target,
            horizon=self.config.horizon,
            context_length=self.config.context_length,
        )
        self._stage_data["forecast"] = result
        # Serialize forecasts
        fc_serializable = {}
        for asset, fc in result.forecasts.items():
            fc_serializable[asset] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in fc.items()
            }
        summary = f"Forecasts generated for {len(result.forecasts)} asset(s)"
        self.stage_statuses[7]["summary"] = summary
        artifacts.save_artifact(self.run_id, "stage07_forecasts", fc_serializable)
        await self._save()

    async def _run_stage_08(self):
        from pipeline import stage_08_backtest
        features = self._stage_data["features"]
        validation = self._stage_data["validation"]
        regime = self._stage_data["regime"]
        result = await self._run_stage(
            8, stage_08_backtest.run,
            features_df=features.features_df,
            raw_returns_df=features.raw_returns_df,
            validated_parents=validation.validated_parents,
            regime_labels=regime.regime_labels,
            regime_map=regime.regime_map,
            target_col=self.config.target,
            initial_capital=self.config.initial_capital,
            horizon=self.config.horizon,
            causal_retrain_interval=self.config.causal_retrain_interval,
            forecast_retrain_interval=self.config.forecast_retrain_interval,
            signal_direction=getattr(self.config, "signal_direction", "both"),
            min_expected_return=getattr(self.config, "min_expected_return", 0.0015),
        )
        self._stage_data["backtest"] = result

        metrics = result.metrics
        summary = (f"Sharpe: {metrics.get('sharpe', 0):.2f}, "
                   f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%, "
                   f"Trades: {metrics.get('total_trades', 0)}")
        self.stage_statuses[8]["summary"] = summary

        # Serialize and save
        try:
            equity_dict = series_to_dict(result.equity_curve)
            artifacts.save_artifact(self.run_id, "stage08_equity_curve", equity_dict)
        except Exception:
            pass
        try:
            trades_records = df_to_records(result.trades)
            artifacts.save_artifact(self.run_id, "stage08_trades", trades_records)
        except Exception:
            pass
        artifacts.save_artifact(self.run_id, "stage08_metrics", result.metrics)
        try:
            signal_decay_dict = series_to_dict(result.signal_decay)
            artifacts.save_artifact(self.run_id, "stage08_signal_decay", signal_decay_dict)
        except Exception:
            pass
        artifacts.save_artifact(self.run_id, "stage08_regime_performance", result.regime_performance)
        artifacts.save_artifact(self.run_id, "stage08_comparison", result.comparison)
        await self._save()


async def start_pipeline(config: PipelineConfig) -> str:
    orch = PipelineOrchestrator(config)
    _active_runs[orch.run_id] = orch
    # Save initial state
    await orch._save()
    # Launch as background task
    asyncio.create_task(orch.run())
    return orch.run_id
