"""Stage 07 — TimesFM Forecasting with ARIMA ensemble"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastOutput:
    forecasts: Dict[str, Dict]


def run(
    validated_parents: Dict[str, List[Dict]],
    features_df: pd.DataFrame,
    regime_labels: np.ndarray,
    target_col: str,
    horizon: int = 5,
    context_length: int = 512,
    progress_cb: Optional[Callable] = None,
) -> ForecastOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 07] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting forecasting", 0.0)

    if target_col not in features_df.columns:
        candidates = [c for c in features_df.columns if "_ret" in c]
        target_col = candidates[0] if candidates else features_df.columns[0]

    target_series = features_df[target_col].dropna()
    context_length = min(context_length, len(target_series))
    ctx = target_series.iloc[-context_length:].values.astype(np.float64)

    # Current regime
    if regime_labels is not None and len(regime_labels) > 0:
        current_regime_idx = int(regime_labels[-1])
    else:
        current_regime_idx = 0

    regime_name = "unknown"

    emit("Loading TimesFM model", 0.1)
    point_forecast, quantile_forecast = _run_timesfm(ctx, horizon, emit)

    emit("Running ARIMA ensemble", 0.7)
    arima_forecast = _run_arima(ctx, horizon)

    # Check ensemble agreement (sign agreement on majority of steps)
    if point_forecast is not None and arima_forecast is not None:
        agreements = [np.sign(p) == np.sign(a) for p, a in zip(point_forecast, arima_forecast)]
        ensemble_agreement = sum(agreements) > horizon / 2
    else:
        ensemble_agreement = False

    if quantile_forecast is not None:
        q10 = quantile_forecast[:, 0] if quantile_forecast.ndim == 2 else quantile_forecast
        q90 = quantile_forecast[:, -1] if quantile_forecast.ndim == 2 else quantile_forecast
        confidence = float(np.mean(q90 - q10))
    else:
        confidence = 0.0
        q10 = point_forecast - 0.01 if point_forecast is not None else np.zeros(horizon)
        q90 = point_forecast + 0.01 if point_forecast is not None else np.zeros(horizon)
        quantile_forecast = np.column_stack([q10 + i * (q90 - q10) / 9 for i in range(10)])

    if point_forecast is None:
        point_forecast = np.zeros(horizon)
    if arima_forecast is None:
        arima_forecast = np.zeros(horizon)

    emit("Forecast complete", 1.0)
    logger.info(
        f"[Stage 07] Point forecast: {point_forecast}, ensemble_agreement: {ensemble_agreement}"
    )

    forecasts = {
        target_col: {
            "point": point_forecast,
            "quantiles": quantile_forecast,
            "arima_point": arima_forecast,
            "ensemble_agreement": ensemble_agreement,
            "regime_at_forecast": regime_name,
            "forecast_date": str(target_series.index[-1].date() if hasattr(target_series.index[-1], "date") else ""),
            "confidence": round(confidence, 6),
            "target_col": target_col,
            "horizon": horizon,
        }
    }

    return ForecastOutput(forecasts=forecasts)


def _run_timesfm(ctx: np.ndarray, horizon: int, emit: Callable):
    """TimesFM 2.5 inference with statistical fallback."""
    try:
        import timesfm
        import torch
        emit("Loading TimesFM 2.5 model weights (first run downloads ~800MB)", 0.2)

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=min(1024, len(ctx)),
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,     # returns can go negative
                fix_quantile_crossing=True,
            )
        )

        emit("Running TimesFM 2.5 inference", 0.5)
        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[ctx],
        )
        emit("TimesFM 2.5 inference complete", 0.65)

        # point_forecast shape: (1, horizon), quantile_forecast: (1, horizon, n_quantiles)
        pf = np.array(point_forecast[0][:horizon])
        qf = np.array(quantile_forecast[0][:horizon])  # (horizon, n_quantiles)
        if qf.ndim == 1:
            qf = qf[:, np.newaxis]
        return pf, qf

    except Exception as e:
        logger.warning(f"TimesFM unavailable ({e}), using statistical fallback")
        return _statistical_forecast(ctx, horizon)


def _statistical_forecast(ctx: np.ndarray, horizon: int):
    """Exponential smoothing forecast with Gaussian quantile bands."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from scipy import stats as scipy_stats
        data = ctx[-min(200, len(ctx)):]
        # trend="add" requires at least 2 non-nan points
        model = ExponentialSmoothing(data, trend="add", seasonal=None).fit(optimized=True)
        point = np.array(model.forecast(horizon))
        std = float(np.std(np.diff(data))) if len(data) > 1 else 1e-4
        q_levels = np.linspace(0.1, 0.9, 10)
        quantiles = np.array([
            [float(point[h] + scipy_stats.norm.ppf(q) * std * np.sqrt(h + 1)) for q in q_levels]
            for h in range(horizon)
        ])
        return point, quantiles
    except Exception as e:
        logger.warning(f"Statistical forecast failed: {e}")
        # Last resort: naive forecast (last observed value, expanding uncertainty)
        last_val = float(ctx[-1]) if len(ctx) > 0 else 0.0
        std = float(np.std(np.diff(ctx[-50:]))) if len(ctx) > 50 else 1e-4
        point = np.full(horizon, last_val)
        q_levels = np.linspace(0.1, 0.9, 10)
        from scipy import stats as scipy_stats
        quantiles = np.array([
            [float(last_val + scipy_stats.norm.ppf(q) * std * np.sqrt(h + 1)) for q in q_levels]
            for h in range(horizon)
        ])
        return point, quantiles


def _run_arima(ctx: np.ndarray, horizon: int) -> np.ndarray:
    """Run ARIMA(5,1,2) ensemble forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        data = ctx[-min(500, len(ctx)):]
        model = ARIMA(data, order=(2, 1, 1))
        result = model.fit()
        forecast = result.forecast(steps=horizon)
        return np.array(forecast)
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        return np.zeros(horizon)
