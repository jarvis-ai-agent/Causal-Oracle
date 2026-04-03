# Causal Stock Prediction Engine — Build Specification

## For: Claude Code Implementation

## Version: 1.0

## Date: April 2026

## Table of Contents

[Project Overview](#1-project-overview)

- [Research Context & Decision Log](#2-research-context--decision-log)

- [System Architecture](#3-system-architecture)

- [Technical Stack](#4-technical-stack)

- [Backend Pipeline Specification](#5-backend-pipeline-specification)

- [Frontend Specification](#6-frontend-specification)

- [API Contract](#7-api-contract)

- [Data Structures Reference](#8-data-structures-reference)

- [Build Order & Milestones](#9-build-order--milestones)

- [Configuration & Environment](#10-configuration--environment)

## 1. Project Overview

### What We Are Building

A production-ready prototype of a causal inference–driven stock prediction system that combines Google's TimesFM (time-series foundation model) with causal discovery algorithms to identify why assets move — not just that they correlate — and generate tradeable forecasting signals with full backtesting.

### Why This Approach

Standard ML models for stock prediction find correlations that are often spurious and decay quickly. This system adds a causal discovery layer that:

- Identifies which variables actually cause price movements (not just correlate)

- Validates those causal relationships with statistical refutation tests

- Only feeds validated causal features into the forecasting model

- Detects market regime changes that would invalidate the causal graph

- Produces uncertainty-quantified forecasts (quantile bands, not just point estimates)

### Core Thesis

Most prediction systems fail because they train on spurious correlations. By inserting a causal discovery step (CD-NOTS algorithm) and a causal validation step (DoWhy refutation tests) between feature engineering and forecasting, we filter out features that happen to correlate but have no structural relationship to the target. This should produce signals that are more robust across market regimes and decay more slowly.

### What Success Looks Like

- A working React dashboard where a user can select assets, trigger pipeline runs, monitor training status, view causal graphs, inspect forecasts, and review backtest performance

- A Python backend that executes the full 8-stage pipeline from data ingestion through backtesting

- Walk-forward backtests showing whether causal feature selection outperforms naive feature selection

- Clear visualization of discovered causal relationships and their validation status

## 2. Research Context & Decision Log

This section documents the research and decisions made during the design phase so the builder has full context on why each component was chosen.

### 2.1 TimesFM (Google Research)

Repository: https://github.com/google-research/timesfm

TimesFM 2.5 is a 200M-parameter pretrained time-series foundation model from Google Research. Key properties:

- Decoder-only transformer architecture trained on massive time-series corpora

- Zero-shot forecasting — no fine-tuning required to get started

- Supports up to 16k context length (roughly 60+ years of daily data)

- Outputs continuous quantile forecasts (uncertainty bands) via an optional 30M quantile head

- Supports external regressors (XReg covariates) — this is how we feed in causal features

- No frequency indicator needed in v2.5

- Apache 2.0 license

Why chosen: Zero-shot capability means fast iteration. Quantile output gives uncertainty bounds critical for position sizing. XReg support lets us condition forecasts on validated causal parents rather than just price history.

Installation:

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
uv venv && source .venv/bin/activate
uv pip install -e .[torch]  # PyTorch backend
uv pip install -e .[xreg]   # XReg covariate support
```

### 2.2 CD-NOTS (Causal Discovery from Nonstationary Time Series)

Paper: "Causal Discovery in Financial Markets: A Framework for Nonstationary Time-Series Data" — Sadeghi, Gopal, Fesanghary (2024). Published in International Journal of Data Science and Analytics, Vol 19, pp 33–59.

arXiv: https://arxiv.org/abs/2312.17375

Why this algorithm over alternatives:

| Algorithm | Handles Nonstationarity | Handles Lags | Nonparametric | Handles Latent Confounders |
|-----------|------------------------|--------------|---------------|---------------------------|
| Granger Causality | No | Yes | No (linear only) | No |
| PCMCI (Tigramite) | No | Yes | Yes | Partial (LPCMCI) |
| CD-NOD | Yes | No | Yes | Pseudo-confounders |
| CD-NOTS | Yes | Yes | Yes | Pseudo-confounders |

Financial data is neither linear, Gaussian, nor stationary. The CD-NOTS paper tested this empirically and showed standard algorithms produce spurious causal links on financial data. CD-NOTS was the only algorithm in the comparison that handles all three properties simultaneously.

The 4-phase algorithm:

- Add time-indexed node — explicitly models nonstationarity by connecting a time surrogate node to all variables. If a variable's distribution shifts over time, the algorithm detects it.

- Skeleton discovery — uses conditional independence (CI) tests to remove spurious edges. Starts fully connected, prunes edges where X ⊥ Y | Z. Supports kernel-based (KCIT), approximate kernel (RCoT), k-nearest neighbor mutual information (CMIknn), and partial correlation (ParCorr) tests.

- Orient edges — time-lagged edges are auto-oriented (cause precedes effect). V-structures (colliders) orient additional edges.

- Independent change principle — for remaining undirected edges, tests whether P(A) and P(B|A) change independently. If they do, A → B. Uses Hilbert-Schmidt Independence Criterion on nonstationary conditional distributions.

Implementation: The base CD-NOD algorithm is implemented in the causal-learn Python package:

```python
from causallearn.search.ConstraintBased.CDNOD import cdnod
cg = cdnod(data, c_indx, alpha=0.05, indep_test='kci')
```

The CD-NOTS extension (handling lagged time-series) requires wrapping this with lag matrix construction and time-lag-aware edge orientation. This wrapper is what we build in Stage 03–04 of the pipeline.

Performance findings from the paper:

- KCIT gives best accuracy but is O(n³) in sample size — slow for large datasets

- RCoT is the recommended tradeoff between accuracy and speed for real-world use

- CD-NOTS consistently outperformed PCMCI on financial data in F-score evaluations

- Applied successfully to: Fama-French factor attribution, cross-country macro variable causality, Price-to-Book → stock returns prediction

### 2.3 DoWhy (Causal Validation)

Repository: https://github.com/py-why/dowhy

Microsoft's causal inference library. We use it as a validation/refutation layer — not for discovery, but to stress-test the relationships CD-NOTS finds.

Key capabilities used:

- Build causal model from discovered graph structure

- Identify estimands (backdoor criterion, instrumental variables)

- Estimate causal effects

- Refutation tests (this is the critical part):

  - Placebo treatment: replace real treatment with random noise, expect null effect

  - Random common cause: add random confounders, effect should remain stable

  - Data subset: estimate on random subsets, effect should be consistent

Any causal parent that fails ≥2 of 3 refutation tests gets dropped before forecasting.

### 2.4 Regime Detection (HMM)

Markets cycle between trending, mean-reverting, and crisis regimes. No single model works across all three. We use a Hidden Markov Model to classify the current regime and gate downstream behavior.

Package: hmmlearn.GaussianHMM

### 2.5 Key Financial Application Examples from CD-NOTS Paper

These are the real-world applications the researchers validated, which serve as templates for our implementation:

Application 1 — Factor Attribution: Ran CD-NOTS on Fama-French factors + Apple's returns. Found that nonstationarity in Apple's returns can be explained by Fama-French factors in shorter windows, but over 22 years, Apple's exposure to factors itself changed (business model shift). The algorithm detected this automatically.

Application 2 — Price-to-Book Signal: Discovered that normalized log Price-to-Book ratio causally predicts next-quarter returns for S&P 500 financial companies. Built a simple strategy from this — invest in securities whose P/B exceeds a threshold — which outperformed market-cap weighting.

## 3. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          REACT FRONTEND                             │
│   Dashboard │ Causal Graph Viz │ Forecast Charts │ Backtest         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  REST API + WebSocket (status updates)
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                   PYTHON API SERVER (FastAPI)                       │
│     Orchestrator │ Pipeline Manager │ Status Tracker │ Results      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                      PIPELINE WORKERS (Python)                      │
│                                                                     │
│  ┌──────┐  ┌──────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐   │
│  │Ingest│→ │FeatureEng│→ │Lag Build│→ │CD-NOTS │→ │DoWhy Val │   │
│  │  01  │  │    02    │  │   03    │  │   04   │  │    05    │   │
│  └──────┘  └──────────┘  └─────────┘  └────────┘  └──────────┘   │
│                                                                     │
│  ┌──────┐  ┌──────────┐  ┌──────────────────────┐                 │
│  │Regime│→ │ TimesFM  │→ │Signal Gen & Backtest  │                 │
│  │  06  │  │    07    │  │         08            │                 │
│  └──────┘  └──────────┘  └──────────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                  STORAGE (SQLite + Filesystem)                      │
│    Pipeline runs │ Causal graphs │ Forecasts │ Backtest results     │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Pattern

- Frontend → Backend: REST API for triggering pipeline runs, fetching results, configuring parameters

- Backend → Frontend: WebSocket for real-time pipeline status updates (stage progress, logs, errors)

- Pipeline stages: Sequential execution within a single pipeline run, managed by an orchestrator class that tracks state and emits status events

## 4. Technical Stack

### Backend (Python)

| Component | Package | Version | Purpose |
|-----------|---------|---------|---------|
| API Server | fastapi + uvicorn | Latest | REST API + WebSocket |
| Task Queue | asyncio (or celery if scaling) | — | Pipeline orchestration |
| Data Ingestion | yfinance | Latest | Market data |
| Data Processing | pandas, numpy, scipy | Latest | Core computation |
| Feature Engineering | ta (technical analysis) | Latest | RSI, MACD, Bollinger |
| Stationarity Tests | statsmodels | Latest | ADF test, ARIMA ensemble |
| Causal Discovery | causal-learn | >=0.1.3.6 | CD-NOD/CD-NOTS algorithm |
| Causal Discovery Alt | tigramite | >=5.2 | PCMCI (fallback/comparison) |
| Causal Validation | dowhy | Latest | Refutation tests |
| Regime Detection | hmmlearn | Latest | Gaussian HMM |
| Forecasting | timesfm | Latest (2.5) | TimesFM 2.5 200M |
| ML Framework | torch | >=2.0 | TimesFM backend |
| Visualization Data | networkx | Latest | Causal graph serialization |

### Frontend (React)

| Component | Package | Purpose |
|-----------|---------|---------|
| Framework | React 18+ (Vite) | SPA |
| Styling | Tailwind CSS | Utility-first CSS |
| Charts | Recharts or Plotly.js | Forecast + equity curve visualization |
| Graph Viz | react-force-graph-2d or @antv/g6 | Interactive causal graph |
| State | Zustand or React Context | Global state management |
| API Client | axios or fetch | REST calls |
| WebSocket | Native WebSocket API | Real-time status |
| Tables | @tanstack/react-table | Trade log, refutation results |

### Infrastructure

- Development: Single machine, SQLite for storage, filesystem for artifacts

- Python version: 3.10+ (required by causal-learn and timesfm)

- GPU: Optional but recommended for TimesFM inference (CUDA) — CPU works but slower

- OS: Linux or macOS (timesfm has best support on Linux)

## 5. Backend Pipeline Specification

### 5.0 Project Structure

```
causal-oracle/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Global configuration
│   ├── models/
│   │   └── schemas.py           # Pydantic models for API
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   └── websocket.py         # WebSocket status handler
│   ├── pipeline/
│   │   ├── orchestrator.py      # Pipeline runner + status tracking
│   │   ├── stage_01_ingest.py   # Data ingestion
│   │   ├── stage_02_features.py # Feature engineering
│   │   ├── stage_03_lagmatrix.py# Lag matrix construction
│   │   ├── stage_04_causal.py   # CD-NOTS causal discovery
│   │   ├── stage_05_validate.py # DoWhy validation
│   │   ├── stage_06_regime.py   # HMM regime detection
│   │   ├── stage_07_forecast.py # TimesFM forecasting
│   │   └── stage_08_backtest.py # Signal generation + backtesting
│   ├── storage/
│   │   ├── db.py                # SQLite connection + models
│   │   └── artifacts.py         # File-based artifact storage
│   └── utils/
│       ├── logging.py           # Structured logging
│       └── serialization.py     # Graph/array serialization helpers
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Dashboard.jsx        # Main layout
│   │   │   ├── PipelineControl.jsx  # Start/stop/configure runs
│   │   │   ├── PipelineStatus.jsx   # Live stage progress
│   │   │   ├── CausalGraph.jsx      # Interactive graph visualization
│   │   │   ├── ForecastChart.jsx    # Point + quantile forecasts
│   │   │   ├── BacktestResults.jsx  # Equity curve + metrics
│   │   │   ├── RegimeTimeline.jsx   # Regime labels over time
│   │   │   ├── RefutationTable.jsx  # DoWhy validation results
│   │   │   └── TradeLog.jsx         # Individual trades table
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js      # WebSocket connection
│   │   │   └── useApi.js            # REST API hooks
│   │   ├── store/
│   │   │   └── pipelineStore.js     # Zustand store
│   │   └── utils/
│   │       └── formatting.js        # Number/date formatters
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── data/          # Local data cache
├── artifacts/     # Pipeline output artifacts
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 5.1 Stage 01 — Data Ingestion

File: `stage_01_ingest.py`

Inputs:

| Parameter | Type | Example |
|-----------|------|---------|
| tickers | List[str] | ['AAPL', 'MSFT', 'GOOGL', 'SPY'] |
| start_date | str (ISO) | '2015-01-01' |
| end_date | str (ISO) | '2025-12-31' |
| include_macro | bool | True |
| include_factors | bool | True |

Process:

- Fetch OHLCV per ticker via `yfinance.download(tickers, start, end)`

- Fetch Fama-French 5 factors via `pandas_datareader.data.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')`

- Fetch macro series: ^VIX, DX-Y.NYB (DXY), ^TNX (US 10Y yield) via yfinance

- Align all series to common DatetimeIndex using `pd.DataFrame.reindex`

- Forward-fill missing values (market holidays), then drop any remaining NaN rows at the start

Output:

```python
class IngestOutput:
    raw_df: pd.DataFrame        # shape (T, N), float64, DatetimeIndex
    column_map: Dict[str, str]  # {'AAPL_close': 'price', 'VIX': 'macro', ...}
    metadata: Dict              # {'start': '2015-01-02', 'end': '2025-12-31', 'T': 2768, 'N': 22}
```

### 5.2 Stage 02 — Feature Engineering

File: `stage_02_features.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| raw_df | pd.DataFrame | Stage 01 output |
| vol_window | int | 20 (trading days) |
| rsi_period | int | 14 |
| normalize | bool | True |

Process:

- Log returns per price column: `r_t = np.log(P_t / P_{t-1})`

- Realized volatility: `rolling(vol_window).std()` on returns

- RSI (14-day) via `ta.momentum.RSIIndicator`

- MACD signal line via `ta.trend.MACD`

- Bollinger Band width: `(upper - lower) / middle`

- Volume z-score: `(V_t - rolling_mean) / rolling_std` over 20-day window

- Cross-asset beta: rolling OLS regression of each asset return vs SPY return, 60-day window

- Macro returns: log returns on VIX, DXY, US10Y

- Normalization: `(X - mean) / std` per column (fit on training window only in walk-forward)

- Stationarity check: `statsmodels.tsa.stattools.adfuller` per column, difference any column with p > 0.05

Output:

```python
class FeatureOutput:
    features_df: pd.DataFrame             # shape (T-warmup, M), float64, M ≈ 30-50
    feature_names: List[str]              # column names
    stationarity_report: Dict[str, Dict]  # {col: {'adf_stat': -4.2, 'p': 0.001, 'differenced': False}}
    warmup_rows_dropped: int              # number of rows lost to rolling windows
```

### 5.3 Stage 03 — Lag Matrix Construction

File: `stage_03_lagmatrix.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| features_df | pd.DataFrame | Stage 02 output |
| max_lag | int | 5 (trading days) |
| target_col | str | 'AAPL_ret' |

Process:

- For each column in features_df, create shifted copies: `col_t-1, col_t-2, ..., col_t-L`

- Rename columns with lag suffix: `AAPL_ret_t0, AAPL_ret_t-1, ..., AAPL_ret_t-5`

- Concatenate all into single `np.ndarray`

- Create `c_indx = np.arange(0, len(lag_matrix))` as time surrogate

- Drop NaN rows from lag creation

- Store column name mapping for interpreting causal graph output

Output:

```python
class LagMatrixOutput:
    lag_matrix: np.ndarray      # shape (T', K) where K = M × (L+1), float64
    c_indx: np.ndarray          # shape (T',), int64 — time index surrogate
    column_names: List[str]     # ['AAPL_ret_t0', 'AAPL_vol_t0', ..., 'AAPL_ret_t-5', ...]
    target_col_index: int       # index of target variable in lag_matrix columns
    lag_config: Dict            # {'max_lag': 5, 'n_features': 42, 'n_lagged_features': 252}
```

### 5.4 Stage 04 — Causal Discovery (CD-NOTS)

File: `stage_04_causal.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| lag_matrix | np.ndarray | Stage 03 output |
| c_indx | np.ndarray | Stage 03 output |
| column_names | List[str] | Stage 03 output |
| target_col_index | int | Stage 03 output |
| alpha | float | 0.05 |
| indep_test | str | 'rcot' (or 'kci' for accuracy) |

Process:

- Run `cdnod(lag_matrix, c_indx, alpha=alpha, indep_test=indep_test)` from causal-learn

- Extract adjacency matrix from `cg.G.graph` — this is a (K+1) × (K+1) matrix (K features + time node)

- Parse edge types: 1 = directed →, -1 = undirected —, 0 = no edge

- Identify causal parents of target: all nodes with a directed edge pointing to `target_col_index`

- Extract MCI test statistic values as causal strength measure

- Identify nonstationary variables: those connected to the time-indexed node (last node)

- Serialize causal graph as node-link JSON for frontend visualization

**Critical implementation note:** For datasets with >50 features (common after lag expansion), pre-filter using partial correlation (`indep_test='fisherz'`) at a liberal alpha (0.1) to remove obviously independent pairs, then run the full kernel test (RCoT or KCI) on the remaining skeleton. This reduces compute from hours to minutes.

Output:

```python
class CausalOutput:
    adjacency_matrix: np.ndarray                   # shape (K+1, K+1), int
    causal_parents: Dict[str, List[Dict]]          # {target: [{'name': str, 'strength': float, 'p_value': float, 'lag': int}]}
    nonstationary_vars: List[str]                  # variables connected to time node
    graph_json: Dict                               # node-link format for frontend viz
    discovery_metadata: Dict                       # {'n_edges': 47, 'n_directed': 38, 'n_undirected': 9, 'runtime_sec': 342}
```

Example graph_json structure:

```json
{
  "nodes": [
    {"id": "AAPL_ret_t0", "type": "target", "nonstationary": true},
    {"id": "VIX_ret_t-1", "type": "feature", "nonstationary": false},
    {"id": "Mkt-RF_t0", "type": "feature", "nonstationary": true},
    {"id": "_time_", "type": "time_node", "nonstationary": false}
  ],
  "edges": [
    {"source": "VIX_ret_t-1", "target": "AAPL_ret_t0", "directed": true, "strength": 0.34, "p_value": 0.002},
    {"source": "Mkt-RF_t0", "target": "AAPL_ret_t0", "directed": true, "strength": 0.71, "p_value": 0.000},
    {"source": "_time_", "target": "AAPL_ret_t0", "directed": true, "strength": 0.22, "p_value": 0.011}
  ]
}
```

### 5.5 Stage 05 — Causal Validation (DoWhy)

File: `stage_05_validate.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| causal_parents | Dict | Stage 04 output |
| features_df | pd.DataFrame | Stage 02 output |
| target_col | str | Config |
| refutation_runs | int | 100 (per test) |

Process:

For each discovered parent → target link:

- Build DoWhy CausalModel:

```python
import dowhy
model = dowhy.CausalModel(
    data=df,
    treatment=parent_col,
    outcome=target_col,
    graph=dot_graph_from_discovered_structure
)
```

- Identify estimand: `model.identify_effect(proceed_when_unidentifiable=True)`

- Estimate effect: `model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')`

- Refutation 1 — Placebo treatment: `model.refute_estimate(method_name='placebo_treatment_refuter', placebo_type='permute', num_simulations=100)`

- Refutation 2 — Random common cause: `model.refute_estimate(method_name='random_common_cause', num_simulations=100)`

- Refutation 3 — Data subset: `model.refute_estimate(method_name='data_subset_refuter', subset_fraction=0.8, num_simulations=100)`

- Score each parent: pass if placebo p > 0.05 AND random common cause p > 0.05 AND subset is stable

- Drop parents that fail ≥2 of 3 refutation tests

Output:

```python
class ValidationOutput:
    validated_parents: Dict[str, List[Dict]]   # same structure as causal_parents, filtered
    dropped_parents: Dict[str, List[Dict]]     # parents that failed validation, with reasons
    refutation_report: pd.DataFrame            # columns: parent, effect, placebo_p, random_cause_p, subset_stable, verdict
    validation_metadata: Dict                  # {'total_tested': 4, 'passed': 3, 'failed': 1}
```

### 5.6 Stage 06 — Regime Detection

File: `stage_06_regime.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| features_df | pd.DataFrame | Stage 02 output |
| n_regimes | int | 3 |
| regime_features | List[str] | ['SPY_vol20', 'SPY_ret_20d', 'VIX'] |

Process:

- Extract regime feature columns from features_df

- Fit `hmmlearn.GaussianHMM(n_components=n_regimes, covariance_type='full', n_iter=200)`

- Predict regime labels for all rows

- Compute transition matrix from label sequence

- Label regimes by characteristic: sort by mean volatility, assign trending (lowest vol), mean_reverting (mid), crisis (highest)

- Store model for inference on new data

Output:

```python
class RegimeOutput:
    regime_labels: np.ndarray             # shape (T',), int64
    regime_map: Dict[int, str]            # {0: 'trending', 1: 'mean_reverting', 2: 'crisis'}
    transition_matrix: np.ndarray         # shape (n_regimes, n_regimes), float64
    regime_distribution: Dict[str, float] # {'trending': 0.58, 'mean_reverting': 0.30, 'crisis': 0.12}
    current_regime: str                   # regime at most recent data point
    hmm_model: GaussianHMM               # fitted model object (serialized via joblib)
```

### 5.7 Stage 07 — TimesFM Forecasting

File: `stage_07_forecast.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| validated_parents | Dict | Stage 05 output |
| features_df | pd.DataFrame | Stage 02 output |
| regime_labels | np.ndarray | Stage 06 output |
| target_col | str | Config |
| horizon | int | 5 (days ahead) |
| context_length | int | 1024 |

Process:

- Extract target series and validated covariate columns from features_df

- Take last `context_length` rows as input window

- Load model:

```python
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
```

- Compile with forecast config:

```python
model.compile(timesfm.ForecastConfig(
    max_context=context_length,
    max_horizon=horizon,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=False,  # returns can be negative
    fix_quantile_crossing=True,
))
```

- Run forecast: `point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[target_series])`

**Note:** XReg covariate integration depends on timesfm XReg API — check latest docs. If XReg not stable, feed covariates as additional input channels.

- Run ARIMA ensemble: `statsmodels.tsa.arima.model.ARIMA(target_series, order=(5,1,2)).fit().forecast(horizon)`

- Check ensemble agreement: do TimesFM and ARIMA agree on sign of prediction for majority of horizon steps?

Output:

```python
class ForecastOutput:
    forecasts: Dict[str, Dict]   # per-asset forecast bundle
    # Each asset dict contains:
    # 'point': np.ndarray (H,)         — point forecast per horizon step
    # 'quantiles': np.ndarray (H, 10)  — 10th through 90th percentile
    # 'arima_point': np.ndarray (H,)   — ARIMA comparison forecast
    # 'ensemble_agreement': bool       — sign agreement on majority of steps
    # 'regime_at_forecast': str        — current regime label
    # 'forecast_date': str             — date forecast was generated
    # 'confidence': float              — mean quantile spread (lower = more confident)
```

### 5.8 Stage 08 — Signal Generation & Backtesting

File: `stage_08_backtest.py`

Inputs:

| Parameter | Type | Source |
|-----------|------|--------|
| features_df | pd.DataFrame | Stage 02 output |
| validated_parents | Dict | Stage 05 output |
| regime_labels | np.ndarray | Stage 06 output |
| target_col | str | Config |
| initial_capital | float | 100000.0 |
| horizon | int | 5 |
| causal_retrain_interval | int | 60 (trading days) |
| forecast_retrain_interval | int | 5 (trading days) |

Process — Walk-Forward Backtest:

```python
for t in range(train_end, len(data), step=horizon):
    train_data = data[max(0, t-context_length) : t]

    if t % causal_retrain_interval == 0:
        re-run stages 03-05 on train_data → updated validated_parents

    if t % forecast_retrain_interval == 0:
        re-run stage 06 on train_data → updated regime

    run stage 07 on train_data → forecast for [t : t+horizon]

    generate signal:
        LONG  if point > 0 AND ensemble_agreement AND quantile_10 > -threshold
        SHORT if point < 0 AND ensemble_agreement AND quantile_90 < threshold
        FLAT  otherwise

    position_size = kelly_criterion(quantile_spread, win_rate) capped at 0.20

    record trade: entry at data[t], exit at data[t+horizon]
    update equity curve
```

Signal rules detail:

| Condition | Signal | Position Size |
|-----------|--------|---------------|
| point > 0 AND ensemble_agree AND q10 > -0.02 | LONG | Kelly × capital, max 20% |
| point < 0 AND ensemble_agree AND q90 < 0.02 | SHORT | Kelly × capital, max 20% |
| No agreement OR q90 - q10 > 0.05 | FLAT | 0% |

Output:

```python
class BacktestOutput:
    equity_curve: pd.Series            # daily NAV, DatetimeIndex
    trades: pd.DataFrame               # columns: date, asset, side, size, entry, exit, pnl, regime, horizon
    metrics: Dict[str, float]          # sharpe, max_drawdown, win_rate, profit_factor, total_trades, avg_hold_days
    signal_decay: pd.Series            # 30-day rolling Sharpe ratio
    regime_performance: Dict[str, Dict[str, float]]  # metrics broken out by regime
    comparison: Dict[str, float]       # buy-and-hold benchmark metrics for comparison
```

### 5.9 Pipeline Orchestrator

File: `orchestrator.py`

The orchestrator manages pipeline execution, status tracking, and inter-stage data flow.

```python
class PipelineRun:
    id: str                         # UUID
    status: str                     # 'pending' | 'running' | 'completed' | 'failed'
    config: PipelineConfig          # all user-specified parameters
    current_stage: int              # 1-8
    stage_statuses: Dict[int, Dict] # {1: {'status': 'completed', 'duration_sec': 12, 'output_path': '...'}, ...}
    logs: List[Dict]                # timestamped log entries
    created_at: str                 # ISO timestamp
    completed_at: Optional[str]     # ISO timestamp

class PipelineConfig:
    tickers: List[str]
    target: str
    start_date: str
    end_date: str
    max_lag: int
    alpha: float
    indep_test: str                 # 'rcot' | 'kci' | 'fisherz'
    n_regimes: int
    horizon: int
    context_length: int
    initial_capital: float
    causal_retrain_interval: int
    forecast_retrain_interval: int
```

Status emission — The orchestrator emits status events over WebSocket at each stage transition and on key milestones within stages:

```json
{
  "run_id": "abc-123",
  "event": "stage_progress",
  "stage": 4,
  "stage_name": "Causal Discovery",
  "progress": 0.65,
  "message": "Testing conditional independence: edge 31/47",
  "timestamp": "2026-04-03T14:22:31Z"
}
```

## 6. Frontend Specification

### 6.1 Dashboard Layout

The frontend is a single-page React application with a sidebar navigation and main content area.

Views:

- **Pipeline Control** — Configure and launch pipeline runs

- **Pipeline Status** — Live progress tracking with stage-by-stage updates

- **Causal Graph** — Interactive force-directed graph of discovered causal relationships

- **Forecasts** — Point forecasts with quantile bands, ensemble comparison

- **Backtest Results** — Equity curve, metrics summary, trade log

- **Regime Timeline** — Color-coded regime labels over time with transition overlay

### 6.2 Pipeline Control View

Purpose: Configure parameters and start a pipeline run.

UI Elements:

- Ticker multi-select input (searchable, supports adding custom tickers)

- Target asset dropdown (populated from selected tickers)

- Date range picker (start/end)

- Advanced settings collapsible panel:

  - Max lag slider (1–10, default 5)

  - Significance level dropdown (0.01, 0.05, 0.10)

  - Independence test selector (RCoT recommended, KCI for accuracy)

  - Number of regimes slider (2–5, default 3)

  - Forecast horizon slider (1–20 days, default 5)

  - Initial capital input

- "Run Pipeline" primary action button

- Previous runs list with status badges

### 6.3 Pipeline Status View

Purpose: Real-time monitoring of a running pipeline.

UI Elements:

- Horizontal stage progress bar showing 8 stages with status icons:

  - Grey circle = pending
  - Spinning blue = running
  - Green checkmark = complete
  - Red X = failed

- Current stage detail card:

  - Stage name and description
  - Progress bar (0-100%)
  - Live log stream (last 20 lines, auto-scroll)
  - Duration timer

- Stage results summary cards (populated as each stage completes):

  - Stage 01: "Loaded 2,768 rows × 22 columns"
  - Stage 02: "Engineered 47 features, 3 required differencing"
  - Stage 04: "Discovered 38 directed edges, 4 causal parents of AAPL_ret"
  - Stage 05: "3/4 parents passed validation"
  - Stage 06: "Current regime: trending (58% historical)"
  - Stage 08: "Sharpe: 1.43, Win rate: 57.3%"

### 6.4 Causal Graph View

Purpose: Visualize and explore the discovered causal structure.

UI Elements:

- Interactive force-directed graph (react-force-graph-2d or similar):

  - Nodes colored by type: target (gold), feature (teal), time node (orange)
  - Node size proportional to number of connections
  - Edge thickness proportional to causal strength
  - Edge color: green for validated parents, red for failed validation, grey for non-target edges
  - Directed edges shown with arrowheads
  - Hover on node: show variable name, stationarity status, regime sensitivity
  - Hover on edge: show strength, p-value, validation status
  - Click on node: highlight all connected edges

- Sidebar panel showing:

  - Target variable's causal parents (sorted by strength)
  - Nonstationary variables list
  - Validation status per parent (pass/fail with refutation scores)

- Filter controls:

  - Show/hide non-target edges
  - Minimum strength threshold slider
  - Show only validated parents toggle

### 6.5 Forecast View

Purpose: Display current and historical forecasts.

UI Elements:

- Main chart (Recharts or Plotly):

  - X-axis: date
  - Historical price line (actual)
  - Point forecast line extending beyond last actual date
  - Shaded quantile bands (10th-90th percentile fan)
  - ARIMA comparison forecast overlay (dashed line)
  - Regime color bands in background

- Signal indicator: current signal (LONG / SHORT / FLAT) with confidence score

- Ensemble agreement badge (green checkmark or red warning)

- Forecast metadata: regime at forecast time, number of causal parents used, context length

### 6.6 Backtest Results View

Purpose: Evaluate strategy performance.

UI Elements:

- Equity curve chart:

  - Strategy NAV line vs buy-and-hold benchmark
  - Drawdown area chart below
  - Regime color bands in background

- Metrics summary cards (2×3 grid):

  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Total Trades
  - Avg Holding Period

- Regime breakdown table: same metrics split by regime (trending, mean-reverting, crisis)

- Signal decay chart: rolling 30-day Sharpe over time

- Trade log table (sortable, filterable):

  - Date, Asset, Side, Size, Entry Price, Exit Price, P&L, Regime, Duration

### 6.7 Regime Timeline View

Purpose: Visualize market regime classifications over time.

UI Elements:

- Full-width timeline bar color-coded by regime (green=trending, yellow=mean_reverting, red=crisis)

- Overlay: SPY price line on top of regime bands

- Transition matrix heatmap (3×3 grid)

- Current regime badge with probability scores

- Regime distribution pie chart

## 7. API Contract

### REST Endpoints

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| POST | /api/pipeline/run | Start new pipeline run | PipelineConfig | {run_id: str} |
| GET | /api/pipeline/runs | List all runs | — | List[PipelineRun] |
| GET | /api/pipeline/runs/{id} | Get run details + results | — | PipelineRun with stage outputs |
| DELETE | /api/pipeline/runs/{id} | Cancel/delete run | — | {status: 'deleted'} |
| GET | /api/pipeline/runs/{id}/graph | Get causal graph JSON | — | graph_json from Stage 04 |
| GET | /api/pipeline/runs/{id}/forecast | Get forecast data | — | ForecastOutput serialized |
| GET | /api/pipeline/runs/{id}/backtest | Get backtest results | — | BacktestOutput serialized |
| GET | /api/pipeline/runs/{id}/regimes | Get regime data | — | RegimeOutput serialized |
| GET | /api/pipeline/runs/{id}/validation | Get refutation report | — | ValidationOutput serialized |
| GET | /api/pipeline/runs/{id}/logs | Get pipeline logs | — | List[LogEntry] |

### WebSocket

Endpoint: `ws://localhost:8000/ws/pipeline/{run_id}`

Events (server → client):

```typescript
type WSEvent = {
  run_id: string;
  event: 'stage_start' | 'stage_progress' | 'stage_complete' | 'stage_error' | 'pipeline_complete' | 'pipeline_error';
  stage: number;        // 1-8
  stage_name: string;
  progress: number;     // 0.0 - 1.0
  message: string;
  timestamp: string;    // ISO
  data?: any;           // stage-specific payload on completion
}
```

## 8. Data Structures Reference

```
Stage 01 Output:
  raw_df       pd.DataFrame (T, N)     float64  DatetimeIndex
  column_map   Dict[str, str]
  metadata     Dict

       │
       ▼

Stage 02 Output:
  features_df     pd.DataFrame (T-20, M)  float64  DatetimeIndex
  feature_names   List[str]  length M
  stationarity_report  Dict[str, Dict]

       │
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼

Stage 03 Output:                          Stage 06 Input (parallel)
  lag_matrix     np.ndarray (T', K)  float64  K = M × (L+1)
  c_indx         np.ndarray (T',)    int64
  column_names   List[str]  length K
  target_col_index  int

       │
       ▼

Stage 04 Output:
  adjacency_matrix  np.ndarray (K+1, K+1)  int
  causal_parents    Dict[str, List[Dict]]
  nonstationary_vars  List[str]
  graph_json        Dict (JSON-serializable)

       │
       ▼

Stage 05 Output:
  validated_parents  Dict[str, List[Dict]]  (filtered subset of causal_parents)
  dropped_parents    Dict[str, List[Dict]]
  refutation_report  pd.DataFrame

       │                            │
       ▼                            ▼

Stage 06 Output:              (fed into Stage 07)
  regime_labels       np.ndarray (T',)    int64
  regime_map          Dict[int, str]
  transition_matrix   np.ndarray (R, R)   float64
  current_regime      str

       │
       ▼

Stage 07 Output:
  forecasts  Dict[str, Dict]
  per asset:
    point             np.ndarray (H,)      float64
    quantiles         np.ndarray (H, 10)   float64
    arima_point       np.ndarray (H,)      float64
    ensemble_agreement  bool
    regime_at_forecast  str
    confidence          float

       │
       ▼

Stage 08 Output:
  equity_curve         pd.Series (T_backtest,)  float64  DatetimeIndex
  trades               pd.DataFrame (N_trades, 9)
  metrics              Dict[str, float]
  signal_decay         pd.Series (T_backtest,)  float64
  regime_performance   Dict[str, Dict[str, float]]
  comparison           Dict[str, float]  (buy-and-hold benchmark)
```

## 9. Build Order & Milestones

Build in this order to have a working system at each milestone:

### Milestone 1 — Skeleton (Day 1-2)

- Set up project structure (backend + frontend)

- FastAPI server with health check endpoint

- React app with Vite, Tailwind, sidebar navigation shell

- WebSocket connection between frontend and backend

- SQLite database for pipeline runs

### Milestone 2 — Data Pipeline (Day 2-3)

- Stage 01: Data ingestion (yfinance)

- Stage 02: Feature engineering

- Stage 03: Lag matrix construction

- Pipeline orchestrator with status tracking

- Frontend: Pipeline Control view (configure + launch)

- Frontend: Pipeline Status view (live progress)

### Milestone 3 — Causal Discovery (Day 3-5)

- Stage 04: CD-NOTS wrapper around causal-learn cdnod

- Stage 05: DoWhy validation loop

- Frontend: Causal Graph visualization

- Frontend: Refutation results table

- API endpoints for graph and validation data

### Milestone 4 — Forecasting (Day 5-6)

- Stage 06: HMM regime detection

- Stage 07: TimesFM forecasting with XReg

- Frontend: Forecast chart with quantile bands

- Frontend: Regime timeline

- API endpoints for forecast and regime data

### Milestone 5 — Backtesting (Day 6-7)

- Stage 08: Walk-forward backtester

- Frontend: Backtest results view (equity curve, metrics, trade log)

- Comparison against buy-and-hold benchmark

- Signal decay monitoring

### Milestone 6 — Polish (Day 7-8)

- Error handling across all stages

- Loading states and error states in frontend

- Pipeline re-run capability

- Export results (CSV download for trades, PDF for report)

- README with setup instructions

## 10. Configuration & Environment

### Environment Variables

```bash
# .env
PIPELINE_DATA_DIR=./data
PIPELINE_ARTIFACTS_DIR=./artifacts
PIPELINE_DB_PATH=./pipeline.db
TIMESFM_MODEL_ID=google/timesfm-2.5-200m-pytorch
TIMESFM_DEVICE=cuda  # or 'cpu'
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=5173
LOG_LEVEL=INFO
```

### Default Pipeline Configuration

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "SPY"],
  "target": "AAPL_ret",
  "start_date": "2015-01-01",
  "end_date": "2025-12-31",
  "max_lag": 5,
  "alpha": 0.05,
  "indep_test": "rcot",
  "n_regimes": 3,
  "horizon": 5,
  "context_length": 1024,
  "initial_capital": 100000,
  "causal_retrain_interval": 60,
  "forecast_retrain_interval": 5
}
```

### Setup Commands

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn yfinance pandas numpy scipy statsmodels ta hmmlearn dowhy causal-learn networkx joblib

# TimesFM (separate due to torch dependency)
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .[torch,xreg]
cd ..

# Frontend
cd frontend
npm create vite@latest . -- --template react
npm install tailwindcss @tailwindcss/vite axios recharts react-force-graph-2d @tanstack/react-table zustand
```

## End of Specification

This document contains everything needed to build the system. Every data structure, array shape, function signature, API endpoint, and UI component is specified. Build in milestone order so there is a working demo at each checkpoint.
