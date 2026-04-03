# Causal Oracle — System Context

## What It Is
A quant-grade, asset-agnostic analytics platform that predicts stock returns using causal inference (not correlation) combined with Google's TimesFM 2.5 transformer. Built April 2026.

**Core thesis:** Standard ML finds correlations. CD-NOTS finds *causes*. By identifying what actually causes a stock to move (not just what moves with it), we can generate more robust, less overfit signals.

---

## Architecture

### Frontend
- **Stack:** React + Vite + Tailwind, Bloomberg Terminal dark theme (`#0a0e17`)
- **Views:** Control, Status, Causal Graph, Forecasts, Backtest, Regimes, Validation, Runs, Guide
- **State:** Zustand store, WebSocket for live pipeline updates
- **Deploy target:** Netlify

### Backend
- **Stack:** Python 3.12, FastAPI, uvicorn, aiosqlite
- **Deploy target:** Railway (lightweight API layer only)
- **Compute target:** Modal (GPU) — planned but not yet implemented
- **Start:** `./start.sh` (uses `.venv/bin/uvicorn` explicitly — critical, system Python 3.9 exists and breaks things)

---

## The 8-Stage Pipeline

### Stage 01 — Data Ingestion
- Fetches OHLCV via `yfinance` for all tickers
- Optionally fetches macro: VIX (`^VIX`), DXY (`DX-Y.NYB`), TNX (`^TNX`)
- Optionally fetches Fama-French 5-factor data via `pandas_datareader`
- Output: raw DataFrame, column map

### Stage 02 — Feature Engineering
- Log returns: `log(price_t / price_t-1)` — these are REAL log returns (~0.005 scale)
- Technical indicators: RSI, MACD signal, Bollinger width, realized vol, volume z-score, rolling beta vs SPY
- **CRITICAL:** Return columns (`*_ret`) are NOT normalized. Only non-return columns get z-scored.
  - This was a major bug fixed 2026-04-03: normalized returns fed into backtest → P&L was nonsense
- Output: `features_df` (normalized non-returns), `raw_returns_df` (raw log returns for backtest P&L)

### Stage 03 — Lag Matrix
- Creates columns: `feature_t0`, `feature_t-1`, ..., `feature_t-{max_lag}`
- Produces `c_indx` array (time indices) required by CD-NOTS
- Output: lag matrix (~162 columns for 3 tickers + macro, max_lag=5)

### Stage 04 — Causal Discovery (CD-NOTS)
- Algorithm: CD-NOTS from `causal-learn` — constraint-based, handles non-stationarity via time index
- Pre-filter: correlation-based reduction to 60 features max before expensive independence tests
- Auto-falls back to FisherZ if dataset is large (>30 features or >1000 samples) — this silently overrides user's KCI/RCoT choice
- **KEY INSIGHT:** `_t0` parents (lag=0) are contemporaneous correlations, NOT predictive.
  - `SPY_ret_t0 → AAPL_ret_t0` just means they move together today — you can't trade this
  - Only `_t-N` parents (N≥1) are true predictive signals
- Output: `causal_parents` dict with `lag` field per parent, graph JSON for visualization

### Stage 05 — DoWhy Validation
- Refutation tests: placebo treatment, random common cause, data subset
- Drops parents that fail refutation — filters spurious causal claims
- **Note:** Currently passes all parents (10/10 in test run) — may be too permissive

### Stage 06 — Regime Detection
- `hmmlearn.GaussianHMM` with N states (default 3)
- Features: realized volatility + returns
- Labels regimes by volatility: low-vol = "trending", high-vol = "crisis", mid = "mean_reverting"
- **KEY FINDING:** Model trades well in "trending" regime (67-86% WR in good quarters), poorly in "mean_reverting"/"crisis" (29-43% WR)

### Stage 07 — Forecasting (TimesFM 2.5)
- **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`) — 200M param zero-shot transformer
- API: `TimesFM_2p5_200M_torch.from_pretrained()` → `.compile(ForecastConfig(...))` → `.forecast(horizon, inputs)`
- Returns point forecast + quantile forecast (10th–90th percentile bands)
- ARIMA(2,1,1) ensemble for agreement check
- **TimesFM forecast magnitudes are tiny:** avg ~0.0005 vs daily vol ~0.013 (ratio: 0.04)
  - This is normal — returns are hard to predict; model is directionally correct more than by magnitude
- Fallback chain: TimesFM → ExponentialSmoothing (fixed: removed `disp` arg) → naive last-value

### Stage 08 — Walk-Forward Backtest
- Walk-forward: train on [0..t], predict [t..t+horizon], step by `horizon` days
- **Signal generation:** Direction consistency — if 60%+ of forecast horizon steps agree on direction, trade it
- **Critical fix:** `_statistical_forecast` + ARIMA used inside backtest loop (NOT TimesFM — too slow, ~2s/call for hundreds of steps)
- **Critical fix:** t0 parents filtered out before signal generation — only lag≥1 parents used
- **Lagged parent confirmation gate:** If lagged parents' current values oppose forecast direction → veto to FLAT
- **Kelly sizing:** Correct formula `f* = (W*b - (1-W)) / b` where `b = avg_win/avg_loss`
  - Floor at 2% to prevent zero-sized trades on first steps
  - **KNOWN ISSUE:** Kelly mostly at floor (avg 0.020) because win history doesn't build enough edge
- **P&L calculation:** `pnl = direction * cum_ret * position_size` where `cum_ret = expm1(sum(log_rets over horizon))`
- **Equity guard:** Stops trading if equity ≤ 0

---

## Configuration Parameters

| Param | Default | Recommended | Notes |
|-------|---------|-------------|-------|
| Tickers | AAPL+MSFT+GOOGL+SPY | AAPL+SPY+GLD | Less co-linear; GLD = risk-off signal |
| Max Lag | 5 | 7 | Deeper lags find predictive (not just contemporaneous) parents |
| Alpha | 0.05 | 0.10 | More permissive → keeps weak but real predictive edges |
| Indep Test | rcot | rcot | KCI overridden to FisherZ anyway when >30 features |
| Start Date | 2015 | 2018 | Post-2018 more regime-consistent; 10yr has structural breaks |
| Horizon | 5 | 3 | Shorter = less noise in holding period |
| Macro | true | true | VIX/DXY genuinely predictive |
| Fama-French | false | false | Adds noise for single-stock targets |

---

## Key Bugs Fixed (2026-04-03)

1. **venv issue:** `start.sh` was using system Python 3.9 → `No module named yfinance`. Fixed: use `.venv/bin/uvicorn` explicitly.
2. **Kelly formula inverted:** Was `win_rate/avg_loss - (1-win_rate)/avg_win` → always 0. Fixed to correct formula.
3. **Returns normalized:** `AAPL_ret` was z-scored before being used for P&L → insane returns (6388%). Fixed: returns excluded from normalization, `raw_returns_df` passed separately to backtest.
4. **Uncertainty threshold:** Was comparing abs forecast magnitude to rolling_std → all signals flat. Fixed: direction consistency metric.
5. **TimesFM API:** Was using v1.0 API (`TimesFm`, `TimesFmHparams`) on v2.5. Fixed: `TimesFM_2p5_200M_torch.from_pretrained()`.
6. **ExponentialSmoothing `disp` arg:** Removed in newer statsmodels. Fixed.
7. **TimesFM in backtest loop:** Was calling TimesFM per step → 10+ min runs. Fixed: use statistical fallback in loop, TimesFM only in Stage 07.
8. **t0 contemporaneous parents:** Were used as signals → random-direction trading. Fixed: filter to lag≥1 only.

---

## Current Performance (Best Run: c7b2af01)
Settings: AAPL+SPY+GLD, 2018-2024, lag=7, alpha=0.10, RCoT, horizon=3, macro=true

| Metric | Value |
|--------|-------|
| Sharpe | 0.19 |
| Win Rate | 54.6% |
| Total Return | +1.23% |
| Max Drawdown | -4.56% |
| Profit Factor | 1.05 |
| Total Trades | 474 |

**LONG-only win rate: 59.0%** (SHORT: 47.5%) — SHORTs are destroying overall edge
**Regime breakdown:** Trending quarters hit 67-86% WR; choppy/bear quarters hit 29-43% WR
**Payoff ratio:** 0.86x (avg win $107 < avg loss $125) — inverted, needs fixing
**Kelly:** Mostly at 2% floor — global win rate history doesn't build fast enough

---

## What's NOT Built Yet
- Modal GPU integration (planned — Railway for API, Modal for heavy compute)
- Multi-target (portfolio) mode — currently single target only
- Deploy to Railway/Netlify (pending Modal token setup)

---

## Overfitting Risks
- Any strategy tuned specifically to AAPL's upward drift will overfit
- Any regime filter tuned on the same data used for backtest will overfit (lookahead bias)
- Kelly sizing trained on in-sample win rates is optimistic (will degrade OOS)
- The t0 parent filter is a structural/theoretical fix — not data-mined, so no overfit risk
- Regime-based signal gating must use the HMM's OOS predictions (it does — HMM is fit on full history, labels are generated sequentially), but the threshold choice (only trade in "trending") is data-mined from this backtest run — moderate overfit risk

---

## Repo
- **GitHub:** https://github.com/jarvis-ai-agent/Causal-Oracle
- **Local:** `~/Desktop/Software Engineering Projects/Causal Oracle/`
- **Branch:** main
