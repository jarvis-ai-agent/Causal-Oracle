"""Stage 07 — Regime-aware Ensemble Forecasting: TimesFM + Chronos + ARIMA

Ensemble strategy:
  - trending regime    → TimesFM ONLY (Chronos degrades trending WR from 64% → 44%)
  - mean_reverting     → Chronos primary (60%+ WR; TimesFM blended on agreement)
  - crisis / unknown   → statistical fallback only (ARIMA/ExponentialSmoothing)

Both models are loaded once and reused. Chronos uses the small variant (710M params)
for speed; can be upgraded to chronos-t5-large for more accuracy at cost of inference time.
"""
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

    # Determine current regime name
    if regime_labels is not None and len(regime_labels) > 0:
        current_regime_idx = int(regime_labels[-1])
    else:
        current_regime_idx = 0

    # Map integer → regime name (trending / mean_reverting / crisis)
    # Mirrors stage_06 logic: sorted by vol mean, lowest=trending, highest=crisis
    # We receive regime_labels as integers; regime_name is derived from distribution
    unique_labels = np.unique(regime_labels[regime_labels >= 0]) if regime_labels is not None else np.array([0])
    n_reg = len(unique_labels)
    if n_reg == 3:
        regime_name_map = {int(unique_labels[0]): "trending",
                           int(unique_labels[1]): "mean_reverting",
                           int(unique_labels[2]): "crisis"}
    elif n_reg == 2:
        regime_name_map = {int(unique_labels[0]): "trending",
                           int(unique_labels[1]): "crisis"}
    else:
        regime_name_map = {int(l): f"regime_{l}" for l in unique_labels}
    regime_name = regime_name_map.get(current_regime_idx, "unknown")

    emit(f"Current regime: {regime_name} — selecting ensemble strategy", 0.05)

    # ── Load models ──────────────────────────────────────────────────────────
    emit("Loading TimesFM 2.5", 0.1)
    timesfm_point, timesfm_quantiles = _run_timesfm(ctx, horizon, emit)

    chronos_point, chronos_quantiles = None, None
    # Only load Chronos for mean_reverting — it hurts in trending regime
    if regime_name == "mean_reverting":
        emit(f"Loading Chronos (regime={regime_name})", 0.55)
        chronos_point, chronos_quantiles = _run_chronos(ctx, horizon, emit)

    emit("Running ARIMA", 0.75)
    arima_forecast = _run_arima(ctx, horizon)

    # ── Ensemble logic ────────────────────────────────────────────────────────
    if regime_name == "trending":
        # TimesFM ONLY — Chronos is a mean-reversion model and degrades trend signals
        if timesfm_point is not None:
            point_forecast, quantile_forecast = timesfm_point, timesfm_quantiles
            ensemble_method = "timesfm_only"
        else:
            point_forecast, quantile_forecast = _statistical_forecast(ctx, horizon)
            ensemble_method = "statistical_fallback"

    elif regime_name == "mean_reverting":
        # Primary: Chronos standalone (better for mean-reversion)
        # Cross-check: if TimesFM and Chronos strongly disagree, reduce confidence
        if chronos_point is not None:
            point_forecast, quantile_forecast = chronos_point, chronos_quantiles
            ensemble_method = "chronos_primary"
            # If TimesFM available and agrees directionally, blend slightly
            if timesfm_point is not None:
                tf_dir = np.sign(np.mean(timesfm_point))
                ch_dir = np.sign(np.mean(chronos_point))
                if tf_dir == ch_dir:
                    # Agreement — blend 30% TimesFM, 70% Chronos
                    point_forecast = 0.7 * chronos_point + 0.3 * timesfm_point
                    ensemble_method = "chronos70_timesfm30"
                else:
                    # Disagreement — use Chronos only, widen quantile bands
                    if quantile_forecast is not None:
                        quantile_forecast = quantile_forecast * 1.5
                    ensemble_method = "chronos_only_disagreement"
        elif timesfm_point is not None:
            point_forecast, quantile_forecast = timesfm_point, timesfm_quantiles
            ensemble_method = "timesfm_fallback"
        else:
            point_forecast, quantile_forecast = _statistical_forecast(ctx, horizon)
            ensemble_method = "statistical_fallback"

    else:
        # crisis / unknown — statistical only, neural models unreliable in tail events
        point_forecast, quantile_forecast = _statistical_forecast(ctx, horizon)
        ensemble_method = "statistical_crisis"

    # Safety fallbacks
    if point_forecast is None:
        point_forecast = np.zeros(horizon)
    if arima_forecast is None:
        arima_forecast = np.zeros(horizon)
    if quantile_forecast is None:
        quantile_forecast, _ = _statistical_forecast(ctx, horizon)
        quantile_forecast = quantile_forecast[1] if isinstance(quantile_forecast, tuple) else quantile_forecast

    # Ensure quantile_forecast is 2D (horizon × n_quantiles)
    if isinstance(quantile_forecast, np.ndarray) and quantile_forecast.ndim == 1:
        quantile_forecast = quantile_forecast[:, np.newaxis]

    # Ensemble agreement check (neural ensemble vs ARIMA)
    agreements = [np.sign(p) == np.sign(a) for p, a in zip(point_forecast, arima_forecast)]
    ensemble_agreement = sum(agreements) > horizon / 2

    q10 = quantile_forecast[:, 0]  if quantile_forecast.ndim == 2 and quantile_forecast.shape[1] > 1 else quantile_forecast.flatten()
    q90 = quantile_forecast[:, -1] if quantile_forecast.ndim == 2 and quantile_forecast.shape[1] > 1 else quantile_forecast.flatten()
    confidence = float(np.mean(q90 - q10))

    emit(f"Ensemble complete: method={ensemble_method} regime={regime_name} agreement={ensemble_agreement}", 1.0)
    logger.info(f"[Stage 07] method={ensemble_method} regime={regime_name} point={point_forecast[:3]}")

    forecasts = {
        target_col: {
            "point": point_forecast,
            "quantiles": quantile_forecast,
            "arima_point": arima_forecast,
            "ensemble_agreement": ensemble_agreement,
            "ensemble_method": ensemble_method,
            "regime_at_forecast": regime_name,
            "timesfm_available": timesfm_point is not None,
            "chronos_available": chronos_point is not None,
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


def _run_chronos(ctx: np.ndarray, horizon: int, emit: Callable):
    """
    Chronos probabilistic forecaster (Amazon, 2024).
    Uses chronos-t5-small for speed; upgrade to chronos-t5-large for accuracy.
    Returns (point_forecast, quantile_forecast) matching TimesFM format.
    """
    try:
        import torch
        from chronos import BaseChronosPipeline

        emit("Chronos: loading model weights (first run downloads ~250MB)", 0.57)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device,
            dtype="float32",
        )

        emit("Chronos: running inference", 0.62)
        context_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        forecast_tensor = pipeline.predict(
            inputs=context_tensor,
            prediction_length=horizon,
            num_samples=50,
            limit_prediction_length=False,
        )
        # forecast_tensor shape: (batch=1, num_samples, horizon)
        samples = forecast_tensor[0].numpy()  # (num_samples, horizon)

        point_forecast = samples.mean(axis=0)  # (horizon,)

        # Build quantile matrix (horizon × 10 quantiles) matching TimesFM format
        q_levels = np.linspace(0.1, 0.9, 10)
        quantile_forecast = np.array([
            np.quantile(samples[:, h], q_levels) for h in range(horizon)
        ])  # (horizon, 10)

        emit("Chronos: inference complete", 0.70)
        logger.info(f"[Stage 07] Chronos point: {point_forecast[:3]}")
        return point_forecast, quantile_forecast

    except Exception as e:
        logger.warning(f"[Stage 07] Chronos unavailable ({e}), skipping")
        return None, None


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
