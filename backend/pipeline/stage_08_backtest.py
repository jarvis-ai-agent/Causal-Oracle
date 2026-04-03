"""Stage 08 — Signal Generation & Walk-Forward Backtesting"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from utils.logging import get_logger
from pipeline import stage_07_forecast

logger = get_logger(__name__)


@dataclass
class BacktestOutput:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    signal_decay: pd.Series
    regime_performance: Dict[str, Dict[str, float]]
    comparison: Dict[str, float]


def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    k = win_rate / abs(avg_loss) - (1 - win_rate) / abs(avg_win) if avg_win != 0 else 0.0
    return max(0.0, min(k, 0.20))


def run(
    features_df: pd.DataFrame,
    validated_parents: Dict[str, List[Dict]],
    regime_labels: np.ndarray,
    target_col: str,
    initial_capital: float = 100000.0,
    horizon: int = 5,
    causal_retrain_interval: int = 60,
    forecast_retrain_interval: int = 5,
    progress_cb: Optional[Callable] = None,
) -> BacktestOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 08] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting walk-forward backtest", 0.0)

    if target_col not in features_df.columns:
        candidates = [c for c in features_df.columns if "_ret" in c]
        target_col = candidates[0] if candidates else features_df.columns[0]

    target_series = features_df[target_col].dropna()
    n = len(target_series)

    # We need at least 200 rows for meaningful backtest
    min_train = min(200, n // 2)
    if n < min_train + horizon:
        logger.warning("Not enough data for backtest")
        return _empty_backtest(initial_capital)

    equity = initial_capital
    equity_curve = []
    equity_dates = []
    trades_list = []
    wins = []
    losses = []

    step = max(horizon, 1)
    t_range = range(min_train, n - horizon, step)
    total_steps = len(list(t_range))

    last_forecast_t = -999
    cached_point = np.zeros(horizon)
    cached_q10 = np.full(horizon, -0.01)
    cached_q90 = np.full(horizon, 0.01)
    cached_arima = np.zeros(horizon)
    cached_agreement = False
    cached_quantiles = np.zeros((horizon, 10))

    emit(f"Running {total_steps} backtest steps (horizon={horizon})", 0.1)

    for step_idx, t in enumerate(t_range):
        pct = 0.1 + 0.85 * (step_idx / max(total_steps, 1))
        if step_idx % 20 == 0:
            emit(f"Backtest step {step_idx}/{total_steps}, equity: ${equity:,.0f}", pct)

        train_data = target_series.iloc[max(0, t - 512):t]
        ctx = train_data.values.astype(np.float64)

        # Re-forecast according to retrain interval
        if t - last_forecast_t >= forecast_retrain_interval:
            try:
                from pipeline.stage_07_forecast import _statistical_forecast, _run_arima
                point_fc, quant_fc = _statistical_forecast(ctx, horizon)
                arima_fc = _run_arima(ctx, horizon)

                cached_point = point_fc
                cached_quantiles = quant_fc
                cached_arima = arima_fc
                cached_q10 = quant_fc[:, 0] if quant_fc.ndim == 2 else quant_fc
                cached_q90 = quant_fc[:, -1] if quant_fc.ndim == 2 else quant_fc

                agreements = [np.sign(p) == np.sign(a) for p, a in zip(point_fc, arima_fc)]
                cached_agreement = sum(agreements) > horizon / 2
                last_forecast_t = t
            except Exception as e:
                logger.debug(f"Forecast failed at t={t}: {e}")

        # Generate signal for next horizon steps
        point_1 = cached_point[0] if len(cached_point) > 0 else 0.0
        q10_1 = cached_q10[0] if len(cached_q10) > 0 else -0.01
        q90_1 = cached_q90[0] if len(cached_q90) > 0 else 0.01

        uncertainty = float(np.mean(cached_q90 - cached_q10))

        # Signal logic
        signal = "FLAT"
        if point_1 > 0 and cached_agreement and q10_1 > -0.02:
            signal = "LONG"
        elif point_1 < 0 and cached_agreement and q90_1 < 0.02:
            signal = "SHORT"
        if uncertainty > 0.05:
            signal = "FLAT"

        # Determine regime at this point
        regime_idx = int(regime_labels[t]) if t < len(regime_labels) else 0
        regime_name = f"regime_{regime_idx}"

        if signal != "FLAT":
            # Compute position size using simple Kelly fraction
            win_rate = max(0.3, min(0.7, (sum(1 for w in wins if w > 0) + 1) / (len(wins) + 2)))
            avg_win = np.mean([w for w in wins if w > 0]) if any(w > 0 for w in wins) else 0.01
            avg_loss = abs(np.mean([l for l in losses if l < 0])) if any(l < 0 for l in losses) else 0.01
            kelly = _kelly_fraction(win_rate, avg_win, avg_loss)
            position_size = kelly * equity

            # Entry and exit
            entry_ret = float(target_series.iloc[t])
            exit_t = min(t + horizon, n - 1)
            exit_ret = float(target_series.iloc[exit_t])

            # P&L
            direction = 1 if signal == "LONG" else -1
            pnl = direction * exit_ret * position_size
            equity += pnl

            trade = {
                "date": target_series.index[t].isoformat() if hasattr(target_series.index[t], "isoformat") else str(target_series.index[t]),
                "asset": target_col,
                "side": signal,
                "size": round(position_size, 2),
                "entry": round(entry_ret, 6),
                "exit": round(exit_ret, 6),
                "pnl": round(pnl, 2),
                "regime": regime_name,
                "horizon": horizon,
            }
            trades_list.append(trade)
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)

        equity_curve.append(equity)
        equity_dates.append(target_series.index[t])

    emit("Computing performance metrics", 0.97)

    equity_series = pd.Series(equity_curve, index=equity_dates)
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["date", "asset", "side", "size", "entry", "exit", "pnl", "regime", "horizon"]
    )

    metrics = _compute_metrics(equity_series, trades_df, initial_capital)

    # Signal decay: 30-day rolling Sharpe
    signal_decay = _rolling_sharpe(equity_series, window=30)

    # Regime performance
    regime_performance = _regime_metrics(trades_df, equity_series)

    # Buy and hold comparison
    bh_returns = target_series.iloc[min_train:].values
    bh_equity = initial_capital * np.cumprod(1 + np.clip(bh_returns, -0.5, 0.5))
    bh_series = pd.Series(bh_equity, index=target_series.index[min_train:min_train + len(bh_equity)])
    comparison = _compute_metrics(bh_series, pd.DataFrame(), initial_capital)
    comparison["strategy"] = "buy_and_hold"

    emit(f"Backtest complete: Sharpe={metrics.get('sharpe', 0):.2f}, "
         f"Trades={metrics.get('total_trades', 0)}", 1.0)

    return BacktestOutput(
        equity_curve=equity_series,
        trades=trades_df,
        metrics=metrics,
        signal_decay=signal_decay,
        regime_performance=regime_performance,
        comparison=comparison,
    )


def _compute_metrics(equity: pd.Series, trades: pd.DataFrame, initial: float) -> Dict:
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "total_trades": 0, "avg_hold_days": 0.0,
                "total_return": 0.0}

    returns = equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    total_return = float((equity.iloc[-1] - initial) / initial)

    if len(trades) > 0 and "pnl" in trades.columns:
        profitable = (trades["pnl"] > 0).sum()
        total_t = len(trades)
        win_rate = float(profitable / total_t) if total_t > 0 else 0.0
        gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit)
        avg_hold = float(trades["horizon"].mean()) if "horizon" in trades.columns else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        total_t = 0
        avg_hold = 0.0

    return {
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": int(len(trades)) if len(trades) > 0 else 0,
        "avg_hold_days": round(avg_hold, 2),
        "total_return": round(total_return, 4),
    }


def _rolling_sharpe(equity: pd.Series, window: int = 30) -> pd.Series:
    returns = equity.pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std.clip(lower=1e-8)) * np.sqrt(252)
    return sharpe.fillna(0.0)


def _regime_metrics(trades: pd.DataFrame, equity: pd.Series) -> Dict:
    if len(trades) == 0 or "regime" not in trades.columns:
        return {}
    result = {}
    for regime in trades["regime"].unique():
        mask = trades["regime"] == regime
        sub = trades[mask]
        total = len(sub)
        wins = (sub["pnl"] > 0).sum()
        result[regime] = {
            "total_trades": int(total),
            "win_rate": round(float(wins / total), 4) if total > 0 else 0.0,
            "total_pnl": round(float(sub["pnl"].sum()), 2),
            "avg_pnl": round(float(sub["pnl"].mean()), 2) if total > 0 else 0.0,
        }
    return result


def _empty_backtest(initial_capital: float) -> BacktestOutput:
    return BacktestOutput(
        equity_curve=pd.Series([initial_capital], name="equity"),
        trades=pd.DataFrame(
            columns=["date", "asset", "side", "size", "entry", "exit", "pnl", "regime", "horizon"]
        ),
        metrics={"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                 "profit_factor": 0.0, "total_trades": 0, "avg_hold_days": 0.0,
                 "total_return": 0.0},
        signal_decay=pd.Series([0.0]),
        regime_performance={},
        comparison={"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0},
    )
