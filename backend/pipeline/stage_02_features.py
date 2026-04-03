"""Stage 02 — Feature Engineering"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from statsmodels.tsa.stattools import adfuller
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureOutput:
    features_df: pd.DataFrame       # normalized feature matrix (for causal/ML)
    raw_returns_df: pd.DataFrame    # un-normalized log returns (for backtest P&L)
    feature_names: List[str]
    stationarity_report: Dict[str, Dict]
    warmup_rows_dropped: int


def _rolling_beta(asset_ret: pd.Series, spy_ret: pd.Series, window: int = 60) -> pd.Series:
    """Rolling OLS beta of asset vs SPY."""
    betas = []
    idx = []
    for i in range(window, len(asset_ret) + 1):
        y = asset_ret.iloc[i - window:i].values
        x = spy_ret.iloc[i - window:i].values
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < window // 2:
            betas.append(np.nan)
        else:
            cov = np.cov(x[valid], y[valid])
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else np.nan
            betas.append(beta)
        idx.append(asset_ret.index[i - 1])
    return pd.Series(betas, index=idx, name=f"{asset_ret.name}_beta")


def _adf_test(series: pd.Series):
    try:
        clean = series.dropna()
        if len(clean) < 20:
            return {"adf_stat": np.nan, "p": 1.0, "differenced": False}
        result = adfuller(clean, autolag="AIC")
        return {"adf_stat": float(result[0]), "p": float(result[1]), "differenced": False}
    except Exception:
        return {"adf_stat": np.nan, "p": 1.0, "differenced": False}


def run(
    raw_df: pd.DataFrame,
    column_map: Dict[str, str],
    vol_window: int = 20,
    rsi_period: int = 14,
    normalize: bool = True,
    progress_cb: Optional[Callable] = None,
) -> FeatureOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 02] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting feature engineering", 0.0)

    price_cols = [c for c in raw_df.columns if c.endswith("_close")]
    vol_cols = [c for c in raw_df.columns if c.endswith("_vol")]
    macro_cols = [c for c in raw_df.columns if column_map.get(c) == "macro"]

    feature_frames = []
    stationarity_report = {}
    initial_len = len(raw_df)

    # ---- Log returns ----
    emit("Computing log returns", 0.1)
    for col in price_cols:
        ticker = col.replace("_close", "")
        ret = np.log(raw_df[col] / raw_df[col].shift(1))
        ret.name = f"{ticker}_ret"
        feature_frames.append(ret)

    # ---- Macro log returns ----
    for col in macro_cols:
        macro_ret = np.log(raw_df[col] / raw_df[col].shift(1))
        macro_ret.name = f"{col}_ret"
        feature_frames.append(macro_ret)
        # Also include level for VIX
        lvl = raw_df[col].copy()
        lvl.name = f"{col}_level"
        feature_frames.append(lvl)

    # ---- TA features per ticker ----
    emit("Computing technical indicators", 0.3)
    spy_ret = None
    for col in price_cols:
        ticker = col.replace("_close", "")
        prices = raw_df[col]
        ret_series = np.log(prices / prices.shift(1))

        # Realized volatility
        rvol = ret_series.rolling(vol_window).std()
        rvol.name = f"{ticker}_vol{vol_window}"
        feature_frames.append(rvol)

        try:
            import ta
            # RSI
            rsi = ta.momentum.RSIIndicator(prices, window=rsi_period).rsi()
            rsi.name = f"{ticker}_rsi"
            feature_frames.append(rsi)

            # MACD signal
            macd = ta.trend.MACD(prices)
            macd_sig = macd.macd_signal()
            macd_sig.name = f"{ticker}_macd_sig"
            feature_frames.append(macd_sig)

            # Bollinger Band width
            bb = ta.volatility.BollingerBands(prices, window=20)
            bbw = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            bbw.name = f"{ticker}_bbw"
            feature_frames.append(bbw)
        except Exception as e:
            logger.warning(f"TA error for {ticker}: {e}")

        if ticker == "SPY":
            spy_ret = ret_series.copy()

    # ---- Volume z-score ----
    emit("Computing volume z-scores", 0.5)
    for col in vol_cols:
        ticker = col.replace("_vol", "")
        v = raw_df[col].astype(float)
        rolling_mean = v.rolling(vol_window).mean()
        rolling_std = v.rolling(vol_window).std()
        vz = (v - rolling_mean) / (rolling_std + 1e-8)
        vz.name = f"{ticker}_vol_zscore"
        feature_frames.append(vz)

    # ---- Rolling beta ----
    emit("Computing rolling beta vs SPY", 0.65)
    if spy_ret is not None:
        for col in price_cols:
            ticker = col.replace("_close", "")
            if ticker == "SPY":
                continue
            ret_s = np.log(raw_df[col] / raw_df[col].shift(1))
            ret_s.name = ticker
            try:
                beta_s = _rolling_beta(ret_s, spy_ret, window=60)
                feature_frames.append(beta_s)
            except Exception as e:
                logger.warning(f"Beta calc failed for {ticker}: {e}")

    # ---- Combine ----
    emit("Combining features", 0.75)
    features_df = pd.concat(feature_frames, axis=1)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    # Drop leading NaN rows (due to rolling windows)
    features_df = features_df.dropna(how="all")
    warmup = initial_len - len(features_df)

    # Keep rows where target column has valid values
    ret_cols = [c for c in features_df.columns if c.endswith("_ret") and not c.endswith("_level")]
    if ret_cols:
        features_df = features_df.dropna(subset=[ret_cols[0]])

    # ---- Stationarity check + differencing ----
    emit("Running stationarity tests (ADF)", 0.85)
    for col in features_df.columns:
        stat = _adf_test(features_df[col])
        if stat["p"] > 0.05 and not col.endswith("_ret"):
            # Difference the series
            features_df[col] = features_df[col].diff()
            stat["differenced"] = True
        stationarity_report[col] = stat

    features_df = features_df.dropna(how="all")

    # Save raw (un-normalized) returns BEFORE normalization — backtest needs real return values
    raw_ret_cols = [c for c in features_df.columns if c.endswith("_ret")]
    raw_returns_df = features_df[raw_ret_cols].copy()

    # ---- Normalization — only non-return columns ----
    # Returns are NOT normalized: their scale carries real financial meaning (P&L)
    if normalize:
        emit("Normalizing features (non-return columns only)", 0.92)
        non_ret_cols = [c for c in features_df.columns if not c.endswith("_ret")]
        for col in non_ret_cols:
            mu = features_df[col].mean()
            sigma = features_df[col].std()
            if sigma > 1e-8:
                features_df[col] = (features_df[col] - mu) / sigma

    emit("Feature engineering complete", 1.0)
    logger.info(f"[Stage 02] Features shape: {features_df.shape}, warmup dropped: {warmup}")

    return FeatureOutput(
        features_df=features_df,
        raw_returns_df=raw_returns_df,
        feature_names=list(features_df.columns),
        stationarity_report=stationarity_report,
        warmup_rows_dropped=warmup,
    )
