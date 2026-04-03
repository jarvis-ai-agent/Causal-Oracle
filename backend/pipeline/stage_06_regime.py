"""Stage 06 — Regime Detection (Gaussian HMM)"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeOutput:
    regime_labels: np.ndarray
    regime_map: Dict[int, str]
    transition_matrix: np.ndarray
    regime_distribution: Dict[str, float]
    current_regime: str
    hmm_model: object  # GaussianHMM


def run(
    features_df: pd.DataFrame,
    n_regimes: int = 3,
    regime_features: Optional[List[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> RegimeOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 06] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting regime detection", 0.0)

    # Default regime features
    if regime_features is None:
        # Find available SPY volatility and return columns
        candidates = []
        for col in features_df.columns:
            if "SPY_vol" in col or "vol20" in col:
                candidates.append(col)
            elif "SPY_ret" in col:
                candidates.append(col)
            elif "VIX" in col and "level" in col:
                candidates.append(col)
        regime_features = candidates[:3]

    # Fall back to any available columns
    available = [f for f in regime_features if f in features_df.columns]
    if not available:
        # Use any volatility columns
        available = [c for c in features_df.columns if "vol" in c.lower()][:3]
    if not available:
        available = features_df.columns[:3].tolist()

    emit(f"Using regime features: {available}", 0.2)

    regime_df = features_df[available].dropna()
    X = regime_df.values

    emit(f"Fitting HMM with {n_regimes} regimes on {X.shape} data", 0.3)

    try:
        from hmmlearn import hmm

        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        labels = model.predict(X)
        emit("HMM fitted successfully", 0.7)
    except Exception as e:
        logger.error(f"HMM fitting failed: {e}")
        # Fallback: simple volatility-based regime classification
        labels, model = _fallback_regime(X, n_regimes)
        emit("Using volatility-based fallback regime detection", 0.7)

    # Label regimes by volatility level
    emit("Labeling regimes by volatility", 0.8)
    vol_col_idx = 0  # first feature assumed to be volatility
    regime_means = []
    for r in range(n_regimes):
        mask = labels == r
        if mask.sum() > 0:
            regime_means.append((r, float(X[mask, vol_col_idx].mean())))
        else:
            regime_means.append((r, 0.0))

    regime_means.sort(key=lambda x: x[1])
    regime_names = ["trending", "mean_reverting", "crisis"]
    if n_regimes == 2:
        regime_names = ["trending", "crisis"]
    elif n_regimes > 3:
        regime_names = [f"regime_{i}" for i in range(n_regimes)]
        regime_names[0] = "trending"
        regime_names[-1] = "crisis"

    regime_map = {}
    for rank, (r_idx, _) in enumerate(regime_means):
        name = regime_names[rank] if rank < len(regime_names) else f"regime_{rank}"
        regime_map[r_idx] = name

    # Compute transition matrix
    try:
        trans = model.transmat_
    except Exception:
        trans = _compute_transition_matrix(labels, n_regimes)

    # Regime distribution
    counts = {regime_map[r]: 0 for r in range(n_regimes)}
    for lbl in labels:
        name = regime_map.get(int(lbl), f"regime_{lbl}")
        counts[name] = counts.get(name, 0) + 1
    total = len(labels)
    regime_distribution = {k: round(v / total, 4) for k, v in counts.items()}

    current_regime = regime_map.get(int(labels[-1]), "unknown")
    emit(f"Current regime: {current_regime}", 0.95)

    # Reindex labels to full features_df length
    full_labels = np.full(len(features_df), -1, dtype=np.int64)
    regime_df_idx = features_df.index.get_indexer(regime_df.index)
    for i, orig_i in enumerate(regime_df_idx):
        if orig_i >= 0:
            full_labels[orig_i] = labels[i]

    # Forward-fill missing regime labels
    last = -1
    for i in range(len(full_labels)):
        if full_labels[i] == -1:
            full_labels[i] = last if last >= 0 else 0
        else:
            last = full_labels[i]

    emit("Regime detection complete", 1.0)
    logger.info(f"[Stage 06] Regime distribution: {regime_distribution}, current: {current_regime}")

    return RegimeOutput(
        regime_labels=full_labels,
        regime_map=regime_map,
        transition_matrix=trans,
        regime_distribution=regime_distribution,
        current_regime=current_regime,
        hmm_model=model,
    )


def _fallback_regime(X: np.ndarray, n_regimes: int):
    """Simple volatility-based regime classification as fallback."""
    logger.warning("Using fallback regime detection")
    vol = X[:, 0]
    thresholds = np.percentile(vol, [100 * i / n_regimes for i in range(1, n_regimes)])

    labels = np.zeros(len(vol), dtype=int)
    for i, v in enumerate(vol):
        for j, t in enumerate(thresholds):
            if v > t:
                labels[i] = j + 1

    class FakeHMM:
        def __init__(self):
            self.transmat_ = np.eye(n_regimes)

    return labels, FakeHMM()


def _compute_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    trans = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return trans / row_sums
