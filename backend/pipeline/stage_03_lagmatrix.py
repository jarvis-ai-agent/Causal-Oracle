"""Stage 03 — Lag Matrix Construction"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LagMatrixOutput:
    lag_matrix: np.ndarray
    c_indx: np.ndarray
    column_names: List[str]
    target_col_index: int
    lag_config: Dict


def run(
    features_df: pd.DataFrame,
    max_lag: int = 5,
    target_col: str = "AAPL_ret",
    progress_cb: Optional[Callable] = None,
) -> LagMatrixOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 03] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Building lag matrix", 0.0)

    # If target col not in features, find closest
    if target_col not in features_df.columns:
        candidates = [c for c in features_df.columns if "_ret" in c and not c.endswith("_ret_")]
        if candidates:
            target_col = candidates[0]
            logger.warning(f"Target column not found, using {target_col}")
        else:
            target_col = features_df.columns[0]
            logger.warning(f"No return columns found, using {target_col}")

    all_frames = []
    col_names = []

    n_features = len(features_df.columns)
    emit(f"Creating lags 0..{max_lag} for {n_features} features", 0.2)

    for lag in range(0, max_lag + 1):
        shifted = features_df.shift(lag)
        suffix = f"_t0" if lag == 0 else f"_t-{lag}"
        shifted.columns = [f"{c}{suffix}" for c in features_df.columns]
        all_frames.append(shifted)
        col_names.extend(shifted.columns.tolist())

    emit("Concatenating lag matrix", 0.7)
    lag_df = pd.concat(all_frames, axis=1)
    lag_df = lag_df.dropna()

    lag_matrix = lag_df.values.astype(np.float64)
    c_indx = np.arange(len(lag_matrix), dtype=np.int64)

    # Find target column index
    target_t0 = f"{target_col}_t0"
    if target_t0 in col_names:
        target_col_index = col_names.index(target_t0)
    else:
        target_col_index = 0
        logger.warning(f"Target column {target_t0} not in lag matrix, using index 0")

    emit("Lag matrix ready", 1.0)
    logger.info(
        f"[Stage 03] Lag matrix shape: {lag_matrix.shape}, "
        f"target_col_index: {target_col_index} ({col_names[target_col_index]})"
    )

    return LagMatrixOutput(
        lag_matrix=lag_matrix,
        c_indx=c_indx,
        column_names=col_names,
        target_col_index=target_col_index,
        lag_config={
            "max_lag": max_lag,
            "n_features": n_features,
            "n_lagged_features": len(col_names),
            "target_col": target_col,
        },
    )
