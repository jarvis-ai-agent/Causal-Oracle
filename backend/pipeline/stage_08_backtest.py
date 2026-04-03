"""Stage 08 — Signal Generation & Walk-Forward Backtesting

Design principles (to avoid overfitting and lookahead bias):
- Only lag≥1 causal parents used for signal (t0 = contemporaneous, untradeable)
- Regime filter uses ONLY past rolling volatility — no HMM labels in signal path
  (HMM is fit on full history → lookahead bias; rolling vol is strictly backward-looking)
- Kelly sizing uses per-volatility-regime history (not global) to prevent miscalibration
- Payoff asymmetry uses forecast quantile bands (available at decision time only)
- All of the above are theoretically motivated, not data-mined from this backtest run
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from utils.logging import get_logger

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
    """
    Kelly criterion: f* = (W * b - (1-W)) / b  where b = avg_win / avg_loss
    Capped at 20% for risk management.
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    k = (win_rate * b - (1 - win_rate)) / b
    return max(0.0, min(k, 0.20))


def _vol_regime(rolling_vol: float, median_vol: float) -> str:
    """
    Classify current market environment from past rolling volatility only.
    No lookahead bias — uses only data available at decision time.

    ratio < 0.8  → low-vol trending environment  → trade normally
    0.8–1.5      → normal environment             → trade normally
    > 1.5        → elevated vol / choppy          → reduce size
    > 2.5        → crisis / tail event            → go flat

    Thresholds are theoretically motivated (2.5x = ~2 std above median vol),
    not tuned to any specific asset or backtest period.
    """
    if median_vol <= 0:
        return "normal"
    ratio = rolling_vol / median_vol
    if ratio > 2.5:
        return "crisis"
    elif ratio > 1.5:
        return "elevated"
    else:
        return "normal"


def run(
    features_df: pd.DataFrame,
    validated_parents: Dict[str, List[Dict]],
    regime_labels: np.ndarray,
    target_col: str,
    raw_returns_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 100000.0,
    horizon: int = 5,
    causal_retrain_interval: int = 60,
    forecast_retrain_interval: int = 3,
    signal_direction: str = "both",        # "both" | "long_only" | "short_only"
    progress_cb: Optional[Callable] = None,
) -> BacktestOutput:

    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 08] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit(f"Starting walk-forward backtest (signal_direction={signal_direction})", 0.0)

    # ── 1. Filter to lag≥1 (predictive) parents only ──────────────────────────
    # t0 parents are contemporaneous correlations — untradeable (you'd need
    # to know t's value to predict t). This is a structural fix, not data-mined.
    predictive_parents = {
        target: [p for p in parents if p.get("lag", 0) >= 1]
        for target, parents in validated_parents.items()
    }
    n_t0  = sum(len(validated_parents[k]) - len(predictive_parents[k]) for k in validated_parents)
    n_lag = sum(len(v) for v in predictive_parents.values())
    emit(f"Parents: {n_lag} predictive (lag≥1), {n_t0} contemporaneous dropped", 0.02)

    # ── 2. Return series setup ─────────────────────────────────────────────────
    returns_source = raw_returns_df if raw_returns_df is not None else features_df
    if target_col not in returns_source.columns:
        candidates = [c for c in returns_source.columns if "_ret" in c]
        target_col = candidates[0] if candidates else returns_source.columns[0]
        logger.warning(f"Target col not in returns_source, falling back to {target_col}")

    target_series = returns_source[target_col].dropna()
    logger.info(
        f"[Stage 08] Returns: mean={target_series.mean():.6f} "
        f"std={target_series.std():.6f} n={len(target_series)}"
    )
    n = len(target_series)
    min_train = min(252, n // 2)   # at least 1 year of warm-up
    if n < min_train + horizon:
        logger.warning("Not enough data for backtest")
        return _empty_backtest(initial_capital)

    # Pre-compute 252-day rolling vol on full series for median benchmark
    # This is used by the vol-regime filter — computed from past data only at each step
    full_vol_series = target_series.rolling(20).std()

    # ── 3. Load TimesFM once, reuse across all backtest steps ─────────────────
    # Loading takes ~4s but inference is ~0.1s/call once loaded.
    # Loading once and reusing: 450 steps = ~45s vs re-loading each time = 1800s.
    timesfm_model = None
    try:
        import timesfm
        emit("Loading TimesFM 2.5 for backtest (loads once, reused each step)...", 0.06)
        timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        timesfm_model.compile(timesfm.ForecastConfig(
            max_context=min(512, min_train),
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        ))
        emit("TimesFM loaded — using for all backtest steps", 0.09)
    except Exception as e:
        logger.warning(f"TimesFM not available for backtest ({e}), using statistical fallback")
        timesfm_model = None

    # ── 4. Walk-forward loop ───────────────────────────────────────────────────
    equity = initial_capital
    equity_curve, equity_dates = [], []
    trades_list = []

    # Per-vol-regime win/loss history for Kelly sizing
    # Keys: "normal", "elevated", "crisis"
    regime_history: Dict[str, Dict] = {
        r: {"wins": [], "losses": []} for r in ("normal", "elevated", "crisis")
    }

    step = max(horizon, 1)
    t_range = list(range(min_train, n - horizon, step))
    total_steps = len(t_range)

    # Forecast cache (retrain every N steps to balance freshness vs compute)
    # With TimesFM at ~0.1s/call, every step is fine; with fallback, every 3 steps
    retrain_interval = 1 if timesfm_model is not None else forecast_retrain_interval
    last_forecast_t = -999
    cached_point    = np.zeros(horizon)
    cached_q10      = np.full(horizon, -0.01)
    cached_q90      = np.full(horizon, 0.01)

    emit(f"Running {total_steps} steps, horizon={horizon}", 0.1)

    for step_idx, t in enumerate(t_range):
        pct = 0.1 + 0.85 * (step_idx / max(total_steps, 1))
        if step_idx % 20 == 0:
            emit(f"Step {step_idx}/{total_steps}  equity=${equity:,.0f}", pct)

        train_data = target_series.iloc[max(0, t - 512):t]
        ctx = train_data.values.astype(np.float64)

        # ── 3a. Vol-regime filter (strictly backward-looking) ─────────────────
        # rolling_vol: 20-day realised vol using only past data up to t-1
        rolling_vol = float(full_vol_series.iloc[t - 1]) if t > 0 else float(target_series.std())
        # median_vol: median of past vol observations (up to t) — dynamic baseline
        past_vols = full_vol_series.iloc[max(0, t - 252):t].dropna()
        median_vol = float(past_vols.median()) if len(past_vols) > 20 else float(rolling_vol)
        vol_state = _vol_regime(rolling_vol, median_vol)

        # Crisis → go completely flat, no positions
        if vol_state == "crisis":
            equity_curve.append(equity)
            equity_dates.append(target_series.index[t])
            continue

        # ── 4b. Forecast (every step with TimesFM, every N steps with fallback) ─
        if t - last_forecast_t >= retrain_interval:
            try:
                if timesfm_model is not None:
                    # TimesFM: fast enough (~0.1s) to run every step
                    point_arr, quant_arr = timesfm_model.forecast(
                        horizon=horizon, inputs=[ctx]
                    )
                    point_fc = np.array(point_arr[0][:horizon])
                    quant_fc = np.array(quant_arr[0][:horizon])   # (horizon, n_quantiles)
                    if quant_fc.ndim == 1:
                        quant_fc = quant_fc[:, np.newaxis]
                else:
                    from pipeline.stage_07_forecast import _statistical_forecast
                    point_fc, quant_fc = _statistical_forecast(ctx, horizon)

                cached_point = point_fc
                cached_q10   = quant_fc[:, 0]  if quant_fc.ndim == 2 and quant_fc.shape[1] > 1 else quant_fc.flatten()
                cached_q90   = quant_fc[:, -1] if quant_fc.ndim == 2 and quant_fc.shape[1] > 1 else quant_fc.flatten()
                last_forecast_t = t
            except Exception as e:
                logger.debug(f"Forecast failed at t={t}: {e}")

        # ── 4c. Signal generation ──────────────────────────────────────────────
        # Direction consistency: ≥60% of horizon steps must agree
        signs   = np.sign(cached_point)
        n_pos   = int(np.sum(signs > 0))
        n_neg   = int(np.sum(signs < 0))
        consistency = max(n_pos, n_neg) / max(len(cached_point), 1)

        signal = "FLAT"
        if consistency >= 0.6:
            if n_pos > n_neg and cached_point[0] >= 0:
                signal = "LONG"
            elif n_neg > n_pos and cached_point[0] <= 0:
                signal = "SHORT"

        # Elevated vol → allow only LONG (buy dips), reduce size later
        if vol_state == "elevated" and signal == "SHORT":
            signal = "FLAT"

        # Apply user-selected direction filter
        if signal_direction == "long_only"  and signal == "SHORT": signal = "FLAT"
        if signal_direction == "short_only" and signal == "LONG":  signal = "FLAT"

        # ── 4d. Conviction pre-filter ─────────────────────────────────────────
        # Compute conviction before the parent gate so we can skip low-conviction
        # setups entirely. Conviction = how much probability mass is on signal side.
        # Only trade conviction >= 0.65 — below this, quantile bands straddle zero
        # too symmetrically to have real edge.
        if signal != "FLAT":
            q10_pre = float(cached_q10[0]) if len(cached_q10) > 0 else -0.01
            q90_pre = float(cached_q90[0]) if len(cached_q90) > 0 else 0.01
            band_pre = max(q90_pre - q10_pre, 1e-8)
            if signal == "LONG":
                pre_conviction = max(0.0, min(1.0, (q90_pre - 0.0) / band_pre))
            else:
                pre_conviction = max(0.0, min(1.0, (0.0 - q10_pre) / band_pre))

            if pre_conviction < 0.65:
                signal = "FLAT"

        # ── 3d. Lagged parent confirmation gate ───────────────────────────────
        # If causal parents (lag≥1) oppose the forecast direction → veto to FLAT
        target_key = list(predictive_parents.keys())[0] if predictive_parents else None
        if signal != "FLAT" and target_key and predictive_parents[target_key]:
            votes = []
            for p in predictive_parents[target_key]:
                col = p["name"]
                if col in features_df.columns and t < len(features_df):
                    val = float(features_df[col].iloc[t])
                    if not np.isnan(val) and p["strength"] >= 0.15:
                        votes.append(np.sign(val))
            if votes:
                parent_dir  = np.sign(np.mean(votes))
                signal_dir  = 1.0 if signal == "LONG" else -1.0
                if parent_dir != 0 and parent_dir != signal_dir:
                    signal = "FLAT"

        if signal == "FLAT":
            equity_curve.append(equity)
            equity_dates.append(target_series.index[t])
            continue

        # ── 3e. Kelly sizing ───────────────────────────────────────────────────
        # Use per-vol-regime history to avoid global miscalibration.
        # When vol is elevated, we've already filtered SHORTs; use smaller floor.
        hist = regime_history[vol_state]
        n_w = len([w for w in hist["wins"]  if w > 0])
        n_l = len([l for l in hist["losses"] if l < 0])
        total_h = n_w + n_l
        wr = (n_w + 1) / (total_h + 2)  # Laplace-smoothed

        pct_wins   = [w / equity for w in hist["wins"]   if w > 0] or [0.01]
        pct_losses = [abs(l)/equity for l in hist["losses"] if l < 0] or [0.01]
        avg_win  = float(np.mean(pct_wins))
        avg_loss = float(np.mean(pct_losses))

        kelly = _kelly_fraction(wr, avg_win, avg_loss)

        # ── 3f. Asymmetric sizing via quantile conviction ──────────────────────
        # Conviction score based on what the forecast quantile bands say:
        #   q10 > 0 → even pessimistic scenario is positive → strong conviction → full size
        #   q90 < 0 → even optimistic scenario is negative → strong conviction → full size
        #   Otherwise → partial conviction based on how far q10/q90 are from zero
        # This uses ONLY information available at decision time (the forecast).
        q10_now = float(cached_q10[0])
        q90_now = float(cached_q90[0])
        if signal == "LONG":
            # How much of the probability mass is positive?
            band = max(q90_now - q10_now, 1e-8)
            conviction = max(0.0, min(1.0, (q90_now - 0.0) / band))
        else:  # SHORT
            band = max(q90_now - q10_now, 1e-8)
            conviction = max(0.0, min(1.0, (0.0 - q10_now) / band))

        # Scale Kelly by conviction: low conviction → reduce size to 40% of Kelly
        size_scale = 0.4 + 0.6 * conviction

        # Vol-adjusted floor: lower floor in elevated vol (more caution)
        floor = 0.02 if vol_state == "normal" else 0.01
        kelly = max(kelly * size_scale, floor)

        if equity <= 0:
            logger.warning(f"[Stage 08] Equity depleted at t={t}, stopping")
            break

        position_size = kelly * equity

        # ── 3g. Execute trade ─────────────────────────────────────────────────
        exit_t   = min(t + horizon, n - 1)
        hold_rets = target_series.iloc[t:exit_t].values
        cum_ret   = float(np.expm1(np.sum(hold_rets)))  # correct: exp(sum(log_rets))-1

        direction = 1 if signal == "LONG" else -1
        pnl       = direction * cum_ret * position_size
        equity   += pnl

        entry_ret = float(target_series.iloc[t])
        exit_ret  = float(target_series.iloc[exit_t])

        trade = {
            "date":     target_series.index[t].isoformat() if hasattr(target_series.index[t], "isoformat") else str(target_series.index[t]),
            "asset":    target_col,
            "side":     signal,
            "size":     round(position_size, 2),
            "kelly":    round(kelly, 4),
            "conviction": round(conviction, 3),
            "vol_state":  vol_state,
            "entry":    round(entry_ret, 6),
            "exit":     round(exit_ret, 6),
            "cum_ret":  round(cum_ret, 6),
            "pnl":      round(pnl, 2),
            "regime":   f"regime_{int(regime_labels[t]) if t < len(regime_labels) else 0}",
            "horizon":  horizon,
        }
        trades_list.append(trade)

        # Update per-regime history
        if pnl > 0:
            regime_history[vol_state]["wins"].append(pnl)
        else:
            regime_history[vol_state]["losses"].append(pnl)

        equity_curve.append(equity)
        equity_dates.append(target_series.index[t])

    # ── 4. Metrics ─────────────────────────────────────────────────────────────
    emit("Computing performance metrics", 0.97)

    equity_series = pd.Series(equity_curve, index=equity_dates)
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["date", "asset", "side", "size", "kelly", "conviction",
                 "vol_state", "entry", "exit", "cum_ret", "pnl", "regime", "horizon"]
    )

    metrics = _compute_metrics(equity_series, trades_df, initial_capital)

    # Augment metrics with vol-state breakdown
    if len(trades_df) > 0 and "vol_state" in trades_df.columns:
        for vs in ("normal", "elevated"):
            sub = trades_df[trades_df["vol_state"] == vs]
            if len(sub) > 0:
                wr = float((sub["pnl"] > 0).sum() / len(sub))
                metrics[f"win_rate_{vs}"] = round(wr, 4)
                metrics[f"trades_{vs}"] = len(sub)

    signal_decay      = _rolling_sharpe(equity_series, window=30)
    regime_performance = _regime_metrics(trades_df, equity_series)

    bh_rets   = target_series.iloc[min_train:].values
    bh_equity = initial_capital * np.cumprod(1 + np.clip(bh_rets, -0.5, 0.5))
    bh_series = pd.Series(bh_equity, index=target_series.index[min_train:min_train + len(bh_equity)])
    comparison = _compute_metrics(bh_series, pd.DataFrame(), initial_capital)
    comparison["strategy"] = "buy_and_hold"

    # Cross-fill B&H comparison into strategy metrics
    metrics["bh_return"]              = comparison.get("total_return", 0.0)
    metrics["bh_pct_gain"]            = comparison.get("total_return", 0.0)
    metrics["strategy_outperformance"] = round(
        metrics.get("total_return", 0.0) - comparison.get("total_return", 0.0), 4
    )

    emit(
        f"Backtest complete: Sharpe={metrics.get('sharpe',0):.2f}  "
        f"WR={metrics.get('win_rate',0)*100:.1f}%  "
        f"Trades={metrics.get('total_trades',0)}  "
        f"Return={metrics.get('total_return',0)*100:.2f}%",
        1.0
    )

    return BacktestOutput(
        equity_curve=equity_series,
        trades=trades_df,
        metrics=metrics,
        signal_decay=signal_decay,
        regime_performance=regime_performance,
        comparison=comparison,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_metrics(equity: pd.Series, trades: pd.DataFrame, initial: float) -> Dict:
    if len(equity) < 2:
        return {
            "initial_capital": round(initial, 2),
            "open_pnl": 0.0, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": 0.0, "commission_paid": 0.0, "expected_payoff": 0.0,
            "bh_return": 0.0, "bh_pct_gain": 0.0, "strategy_outperformance": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
            "total_trades": 0, "total_open_trades": 0, "winning_trades": 0,
            "losing_trades": 0, "avg_pnl": 0.0, "avg_winning_trade": 0.0,
            "avg_losing_trade": 0.0, "ratio_avg_win_loss": 0.0,
            "largest_winning_trade": 0.0, "largest_winning_trade_pct": 0.0,
            "largest_winner_pct_of_gross_profit": 0.0,
            "largest_losing_trade": 0.0, "largest_losing_trade_pct": 0.0,
            "largest_loser_pct_of_gross_loss": 0.0,
            "avg_bars_in_trades": 0.0, "avg_bars_in_winning_trades": 0.0,
            "avg_bars_in_losing_trades": 0.0,
            "cagr": 0.0, "return_on_initial_capital": 0.0,
            "account_size_required": 0.0, "return_on_account_size_required": 0.0,
            "net_profit_pct_of_largest_loss": 0.0,
            "avg_margin_used": 0.0, "max_margin_used": 0.0,
            "margin_efficiency": 0.0, "margin_calls": 0,
            "avg_equity_runup_duration": 0.0, "avg_equity_runup": 0.0,
            "max_equity_runup_close": 0.0, "max_equity_runup_intrabar": 0.0,
            "max_equity_runup_pct_initial": 0.0,
            "avg_equity_drawdown_duration": 0.0, "avg_equity_drawdown_close": 0.0,
            "max_equity_drawdown_close": 0.0, "max_equity_drawdown_intrabar": 0.0,
            "max_equity_drawdown_pct_initial": 0.0,
            "return_of_max_drawdown": 0.0,
            "avg_hold_days": 0.0, "total_return": 0.0,
        }

    # ── Returns & equity basics ───────────────────────────────────────────────
    final_equity  = float(equity.iloc[-1])
    net_pnl       = final_equity - initial
    total_return  = net_pnl / initial if initial > 0 else 0.0

    returns = equity.pct_change().dropna()

    # ── Sharpe ────────────────────────────────────────────────────────────────
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # ── Sortino (downside-only std) ───────────────────────────────────────────
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # ── Drawdown (close-to-close) ─────────────────────────────────────────────
    peak     = equity.cummax()
    drawdown_series = (equity - peak) / peak
    max_dd   = float(drawdown_series.min())                   # most negative
    max_dd_abs = float((peak - equity).max())                 # dollar peak-to-trough

    # Drawdown segments: duration and average
    in_dd    = drawdown_series < 0
    dd_durations, dd_depths = [], []
    run_len, run_depth = 0, 0.0
    for in_d, depth in zip(in_dd, drawdown_series):
        if in_d:
            run_len += 1
            run_depth = min(run_depth, depth)
        else:
            if run_len > 0:
                dd_durations.append(run_len)
                dd_depths.append(run_depth)
            run_len, run_depth = 0, 0.0
    if run_len > 0:
        dd_durations.append(run_len)
        dd_depths.append(run_depth)

    avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0.0
    avg_dd_close    = float(np.mean(dd_depths))    if dd_depths    else 0.0

    # Intrabar approximation: use the worst-case daily close as intrabar proxy
    max_dd_intrabar = max_dd
    max_dd_intrabar_pct_initial = abs(max_dd_intrabar) * float(equity.max()) / initial if initial > 0 else 0.0

    # ── Run-up (close-to-close) ────────────────────────────────────────────────
    trough   = equity.cummin()
    runup_series = (equity - trough) / trough.clip(lower=1e-8)
    max_runup = float(runup_series.max())

    in_ru    = runup_series > 0
    ru_durations, ru_depths = [], []
    run_len, run_depth = 0, 0.0
    for in_r, depth in zip(in_ru, runup_series):
        if in_r:
            run_len += 1
            run_depth = max(run_depth, depth)
        else:
            if run_len > 0:
                ru_durations.append(run_len)
                ru_depths.append(run_depth)
            run_len, run_depth = 0, 0.0
    if run_len > 0:
        ru_durations.append(run_len)
        ru_depths.append(run_depth)

    avg_ru_duration = float(np.mean(ru_durations)) if ru_durations else 0.0
    avg_ru_close    = float(np.mean(ru_depths))    if ru_depths    else 0.0
    max_ru_intrabar = max_runup
    max_ru_pct_initial = max_runup * float(equity.min()) / initial if initial > 0 else 0.0

    # ── CAGR ─────────────────────────────────────────────────────────────────
    n_periods = len(equity)
    years = n_periods / 252.0
    cagr = float((final_equity / initial) ** (1 / years) - 1) if years > 0 and initial > 0 else 0.0

    # ── Trade-level metrics ───────────────────────────────────────────────────
    if len(trades) > 0 and "pnl" in trades.columns:
        pnls         = trades["pnl"].values
        winning      = trades[trades["pnl"] > 0]
        losing       = trades[trades["pnl"] < 0]
        n_total      = len(trades)
        n_winning    = len(winning)
        n_losing     = len(losing)
        win_rate     = float(n_winning / n_total) if n_total > 0 else 0.0

        gross_profit = float(winning["pnl"].sum()) if n_winning > 0 else 0.0
        gross_loss   = float(abs(losing["pnl"].sum())) if n_losing > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        avg_pnl          = float(np.mean(pnls))
        avg_win          = float(winning["pnl"].mean()) if n_winning > 0 else 0.0
        avg_loss_val     = float(losing["pnl"].mean())  if n_losing  > 0 else 0.0
        ratio_win_loss   = abs(avg_win / avg_loss_val) if avg_loss_val != 0 else 0.0
        expected_payoff  = float(win_rate * avg_win + (1 - win_rate) * avg_loss_val)

        largest_win      = float(winning["pnl"].max()) if n_winning > 0 else 0.0
        largest_loss     = float(losing["pnl"].min())  if n_losing  > 0 else 0.0

        # Percent-based per-trade: pnl / size
        if "size" in trades.columns:
            winning_pct_col = winning["pnl"] / winning["size"].clip(lower=1e-8)
            losing_pct_col  = losing["pnl"]  / losing["size"].clip(lower=1e-8)
            largest_win_pct  = float(winning_pct_col.max()) if n_winning > 0 else 0.0
            largest_loss_pct = float(losing_pct_col.min())  if n_losing  > 0 else 0.0
        else:
            largest_win_pct = largest_loss_pct = 0.0

        largest_win_pct_of_gp  = (largest_win  / gross_profit * 100) if gross_profit > 0 else 0.0
        largest_loss_pct_of_gl = (abs(largest_loss) / gross_loss * 100) if gross_loss > 0 else 0.0

        avg_hold = float(trades["horizon"].mean()) if "horizon" in trades.columns else 0.0
        avg_win_hold  = float(winning["horizon"].mean()) if n_winning > 0 and "horizon" in winning.columns else avg_hold
        avg_loss_hold = float(losing["horizon"].mean())  if n_losing  > 0 and "horizon" in losing.columns  else avg_hold

        # Margin: use position size as proxy for margin used
        if "size" in trades.columns:
            avg_margin = float(trades["size"].mean())
            max_margin = float(trades["size"].max())
        else:
            avg_margin = max_margin = 0.0
        margin_efficiency = net_pnl / avg_margin if avg_margin > 0 else 0.0
        margin_calls = 0  # No leverage model yet; placeholder

        net_profit_pct_largest_loss = (net_pnl / abs(largest_loss) * 100) if largest_loss != 0 else 0.0
    else:
        n_total = n_winning = n_losing = 0
        win_rate = profit_factor = avg_pnl = avg_win = avg_loss_val = 0.0
        ratio_win_loss = expected_payoff = 0.0
        gross_profit = gross_loss = 0.0
        largest_win = largest_loss = largest_win_pct = largest_loss_pct = 0.0
        largest_win_pct_of_gp = largest_loss_pct_of_gl = 0.0
        avg_hold = avg_win_hold = avg_loss_hold = 0.0
        avg_margin = max_margin = margin_efficiency = 0.0
        margin_calls = 0
        net_profit_pct_largest_loss = 0.0

    # ── Account size required (initial capital + max drawdown buffer) ─────────
    account_size_required = initial + abs(max_dd * initial)
    return_on_account_size = net_pnl / account_size_required if account_size_required > 0 else 0.0

    # ── Return of max drawdown (recovery) ─────────────────────────────────────
    return_of_max_dd = total_return / abs(max_dd) if max_dd != 0 else 0.0

    def r(v, d=4):
        try:
            return round(float(v), d)
        except Exception:
            return 0.0

    return {
        # Capital
        "initial_capital":               r(initial, 2),
        "open_pnl":                      0.0,   # live trading only; always 0 in backtest
        "net_pnl":                       r(net_pnl, 2),
        "gross_profit":                  r(gross_profit, 2),
        "gross_loss":                    r(gross_loss, 2),
        "profit_factor":                 r(profit_factor),
        "commission_paid":               0.0,   # placeholder — no commission model yet
        "expected_payoff":               r(expected_payoff, 2),
        # B&H / outperformance (populated at call site)
        "bh_return":                     0.0,
        "bh_pct_gain":                   0.0,
        "strategy_outperformance":       0.0,
        # Risk-adjusted
        "sharpe":                        r(sharpe),
        "sortino":                       r(sortino),
        # Trade counts
        "total_trades":                  int(n_total),
        "total_open_trades":             0,     # closed-only backtest
        "winning_trades":                int(n_winning),
        "losing_trades":                 int(n_losing),
        "win_rate":                      r(win_rate),
        # P&L averages
        "avg_pnl":                       r(avg_pnl, 2),
        "avg_winning_trade":             r(avg_win, 2),
        "avg_losing_trade":              r(avg_loss_val, 2),
        "ratio_avg_win_loss":            r(ratio_win_loss),
        # Largest trades
        "largest_winning_trade":         r(largest_win, 2),
        "largest_winning_trade_pct":     r(largest_win_pct * 100, 4),
        "largest_winner_pct_of_gross_profit": r(largest_win_pct_of_gp, 2),
        "largest_losing_trade":          r(largest_loss, 2),
        "largest_losing_trade_pct":      r(largest_loss_pct * 100, 4),
        "largest_loser_pct_of_gross_loss": r(largest_loss_pct_of_gl, 2),
        # Bar counts
        "avg_bars_in_trades":            r(avg_hold, 2),
        "avg_bars_in_winning_trades":    r(avg_win_hold, 2),
        "avg_bars_in_losing_trades":     r(avg_loss_hold, 2),
        # Returns / sizing
        "cagr":                          r(cagr),
        "return_on_initial_capital":     r(total_return),
        "account_size_required":         r(account_size_required, 2),
        "return_on_account_size_required": r(return_on_account_size),
        "net_profit_pct_of_largest_loss": r(net_profit_pct_largest_loss, 2),
        # Margin
        "avg_margin_used":               r(avg_margin, 2),
        "max_margin_used":               r(max_margin, 2),
        "margin_efficiency":             r(margin_efficiency),
        "margin_calls":                  int(margin_calls),
        # Run-up
        "avg_equity_runup_duration":     r(avg_ru_duration, 2),
        "avg_equity_runup":              r(avg_ru_close, 4),
        "max_equity_runup_close":        r(max_runup, 4),
        "max_equity_runup_intrabar":     r(max_ru_intrabar, 4),
        "max_equity_runup_pct_initial":  r(max_ru_pct_initial * 100, 2),
        # Drawdown
        "avg_equity_drawdown_duration":  r(avg_dd_duration, 2),
        "avg_equity_drawdown_close":     r(avg_dd_close, 4),
        "max_equity_drawdown_close":     r(max_dd, 4),
        "max_equity_drawdown_intrabar":  r(max_dd_intrabar, 4),
        "max_equity_drawdown_pct_initial": r(max_dd_intrabar_pct_initial * 100, 2),
        "return_of_max_drawdown":        r(return_of_max_dd, 4),
        # Legacy keys (kept for backward compat with existing charts)
        "max_drawdown":                  r(max_dd),
        "total_return":                  r(total_return),
        "avg_hold_days":                 r(avg_hold, 2),
    }


def _rolling_sharpe(equity: pd.Series, window: int = 30) -> pd.Series:
    returns      = equity.pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    sharpe       = (rolling_mean / rolling_std.clip(lower=1e-8)) * np.sqrt(252)
    return sharpe.fillna(0.0)


def _regime_metrics(trades: pd.DataFrame, equity: pd.Series) -> Dict:
    if len(trades) == 0 or "regime" not in trades.columns:
        return {}
    result = {}
    for regime in trades["regime"].unique():
        sub   = trades[trades["regime"] == regime]
        total = len(sub)
        wins  = (sub["pnl"] > 0).sum()
        result[regime] = {
            "total_trades": int(total),
            "win_rate":     round(float(wins / total), 4) if total > 0 else 0.0,
            "total_pnl":    round(float(sub["pnl"].sum()), 2),
            "avg_pnl":      round(float(sub["pnl"].mean()), 2) if total > 0 else 0.0,
        }
    return result


def _empty_backtest(initial_capital: float) -> BacktestOutput:
    return BacktestOutput(
        equity_curve=pd.Series([initial_capital], name="equity"),
        trades=pd.DataFrame(
            columns=["date", "asset", "side", "size", "kelly", "conviction",
                     "vol_state", "entry", "exit", "cum_ret", "pnl", "regime", "horizon"]
        ),
        metrics={"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                 "profit_factor": 0.0, "total_trades": 0, "avg_hold_days": 0.0,
                 "total_return": 0.0},
        signal_decay=pd.Series([0.0]),
        regime_performance={},
        comparison={"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0},
    )
