"""Stage 01 — Data Ingestion"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestOutput:
    raw_df: pd.DataFrame
    column_map: Dict[str, str]
    metadata: Dict


def run(
    tickers: List[str],
    start_date: str,
    end_date: str,
    include_macro: bool = True,
    include_factors: bool = False,
    progress_cb: Optional[Callable] = None,
) -> IngestOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 01] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting data ingestion", 0.0)
    all_dfs = []
    column_map = {}

    # ---- Fetch price data ----
    emit(f"Downloading OHLCV for {tickers}", 0.1)
    try:
        price_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(price_data.columns, pd.MultiIndex):
            close_df = price_data["Close"]
            volume_df = price_data["Volume"]
        else:
            close_df = price_data[["Close"]].rename(columns={"Close": tickers[0]})
            volume_df = price_data[["Volume"]].rename(columns={"Volume": tickers[0]})

        close_df.columns = [f"{t}_close" for t in close_df.columns]
        volume_df.columns = [f"{t}_vol" for t in volume_df.columns]
        for col in close_df.columns:
            column_map[col] = "price"
        for col in volume_df.columns:
            column_map[col] = "volume"
        all_dfs.extend([close_df, volume_df])
        emit(f"Downloaded {len(tickers)} tickers", 0.3)
    except Exception as e:
        logger.error(f"Error downloading price data: {e}")
        raise

    # ---- Fetch macro data ----
    if include_macro:
        macro_tickers = {"^VIX": "VIX", "DX-Y.NYB": "DXY", "^TNX": "TNX"}
        emit("Downloading macro series (VIX, DXY, TNX)", 0.5)
        macro_frames = []
        for yticker, name in macro_tickers.items():
            try:
                s = yf.download(yticker, start=start_date, end=end_date,
                                auto_adjust=True, progress=False)["Close"]
                s.name = name
                macro_frames.append(s)
                column_map[name] = "macro"
            except Exception as e:
                logger.warning(f"Could not download {yticker}: {e}")
        if macro_frames:
            macro_df = pd.concat(macro_frames, axis=1)
            all_dfs.append(macro_df)

    # ---- Fama-French factors ----
    if include_factors:
        emit("Downloading Fama-French 5-factor data", 0.7)
        try:
            import pandas_datareader.data as web
            ff_data = web.DataReader(
                "F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                start=start_date, end=end_date
            )
            ff_df = ff_data[0] / 100.0  # convert from % to decimal
            ff_df.index = pd.to_datetime(ff_df.index, format="%Y%m%d")
            for col in ff_df.columns:
                column_map[col] = "factor"
            all_dfs.append(ff_df)
        except Exception as e:
            logger.warning(f"Could not download Fama-French data: {e}")

    # ---- Combine and align ----
    emit("Aligning and cleaning data", 0.85)
    combined = pd.concat(all_dfs, axis=1)
    combined = combined.sort_index()

    # Forward-fill up to 5 days (market holidays), then drop leading NaNs
    combined = combined.ffill(limit=5)
    combined = combined.dropna(how="all")

    # Drop rows at start that still have NaN in key price cols
    price_cols = [c for c in combined.columns if c.endswith("_close")]
    if price_cols:
        combined = combined.dropna(subset=price_cols[:1])

    emit("Data ingestion complete", 1.0)
    logger.info(f"[Stage 01] Shape: {combined.shape}")

    return IngestOutput(
        raw_df=combined,
        column_map=column_map,
        metadata={
            "start": str(combined.index[0].date()),
            "end": str(combined.index[-1].date()),
            "T": len(combined),
            "N": len(combined.columns),
            "tickers": tickers,
            "columns": list(combined.columns),
        },
    )
