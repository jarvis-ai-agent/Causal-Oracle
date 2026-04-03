import React, { useState } from 'react'

const Section = ({ icon, title, children }) => (
  <div className="panel mb-4">
    <div className="panel-header flex items-center gap-2">
      <span>{icon}</span>
      <span>{title}</span>
    </div>
    <div className="p-4 flex flex-col gap-4">{children}</div>
  </div>
)

const Control = ({ name, type, range, default: def, recommended, impact, children }) => (
  <div className="border border-terminal-border rounded p-4 hover:border-terminal-accent/50 transition-colors">
    <div className="flex items-start justify-between gap-4 mb-2">
      <div>
        <span className="text-terminal-accent font-bold text-sm">{name}</span>
        <span className="ml-2 px-1.5 py-0.5 bg-terminal-surface border border-terminal-border rounded text-xs text-terminal-muted uppercase">{type}</span>
      </div>
      {def && (
        <span className="text-xs text-terminal-muted whitespace-nowrap">
          Default: <span className="text-terminal-text">{def}</span>
        </span>
      )}
    </div>
    <p className="text-xs text-terminal-text leading-relaxed mb-3">{children}</p>
    {range && (
      <div className="flex flex-wrap gap-3 text-xs mb-3">
        <span className="text-terminal-muted">Range: <span className="text-terminal-text font-mono">{range}</span></span>
      </div>
    )}
    <div className="grid grid-cols-2 gap-3">
      {recommended && (
        <div className="bg-terminal-surface rounded p-2.5">
          <div className="text-terminal-green text-xs font-bold uppercase tracking-wider mb-1">✓ Recommended</div>
          <div className="text-xs text-terminal-text">{recommended}</div>
        </div>
      )}
      {impact && (
        <div className="bg-terminal-surface rounded p-2.5">
          <div className="text-terminal-gold text-xs font-bold uppercase tracking-wider mb-1">⚡ Effect on Output</div>
          <div className="text-xs text-terminal-text">{impact}</div>
        </div>
      )}
    </div>
  </div>
)

const Callout = ({ type, children }) => {
  const styles = {
    tip: 'border-terminal-accent/40 bg-terminal-accent/5 text-terminal-accent',
    warning: 'border-terminal-gold/40 bg-terminal-gold/5 text-terminal-gold',
    info: 'border-terminal-muted/40 bg-terminal-surface text-terminal-muted',
  }
  const icons = { tip: '💡', warning: '⚠', info: 'ℹ' }
  return (
    <div className={`border rounded p-3 text-xs leading-relaxed ${styles[type]}`}>
      <span className="mr-2">{icons[type]}</span>{children}
    </div>
  )
}

const pages = [
  { id: 'control', label: '⚙ Control Page', icon: '⚙' },
]

export default function Guide() {
  const [activePage, setActivePage] = useState('control')

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sub-nav */}
      <aside className="w-44 flex-none bg-terminal-surface border-r border-terminal-border py-3">
        <div className="px-4 mb-3 text-xs text-terminal-muted uppercase tracking-wider">Pages</div>
        {pages.map(p => (
          <button
            key={p.id}
            onClick={() => setActivePage(p.id)}
            className={`w-full text-left px-4 py-2 text-xs transition-colors ${
              activePage === p.id
                ? 'text-terminal-accent bg-terminal-accent/10 border-r-2 border-terminal-accent'
                : 'text-terminal-muted hover:text-terminal-text'
            }`}
          >
            {p.label}
          </button>
        ))}
        <div className="px-4 mt-4 text-xs text-terminal-muted/50 italic leading-relaxed">
          More pages coming soon — Status, Forecasts, Backtest, Causal Graph, Regimes, Validation
        </div>
      </aside>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {activePage === 'control' && <ControlGuide />}
      </div>
    </div>
  )
}

function ControlGuide() {
  return (
    <div className="max-w-3xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-terminal-accent text-xl font-bold tracking-wider uppercase mb-1">Control Page</h1>
        <p className="text-terminal-muted text-xs leading-relaxed">
          The Control page is where you configure and launch a pipeline run. Every setting here shapes what data gets fetched,
          how the causal model is built, and what the forecast looks like. Understanding each input helps you get better, more reliable predictions.
        </p>
      </div>

      <Callout type="tip">
        Start simple: one ticker, 3–5 years of data, default advanced settings. Once you understand the outputs, layer in complexity.
      </Callout>

      <div className="mt-6" />

      {/* ASSETS */}
      <Section icon="◈" title="Assets (Tickers)">
        <Control
          name="Assets"
          type="multi-select"
          default="AAPL"
          recommended="1–3 tickers for your first run. Add related assets (e.g. AAPL + QQQ + VIX) to let the causal engine find cross-asset relationships."
          impact="More tickers = richer causal graph but exponentially longer Stage 4 (Causal Discovery) runtime. Each added ticker multiplies the lag matrix width."
        >
          The stocks, ETFs, or indices you want the pipeline to analyse. You can type any valid ticker symbol and press Enter, or click the preset buttons (+AAPL, +SPY etc.).
          All tickers are fetched together so the causal model can discover cross-asset cause-and-effect relationships.
        </Control>

        <Callout type="warning">
          Adding more than 4–5 tickers will make Stage 4 (Causal Discovery) very slow on CPU — think 15–30+ minutes. Keep it lean locally.
        </Callout>

        <div className="text-xs text-terminal-muted mt-1">
          <span className="text-terminal-text font-bold">Preset suggestions:</span>
          <ul className="mt-1 space-y-1 ml-3">
            <li><span className="text-terminal-accent">AAPL + QQQ</span> — single stock vs its index, good for beta analysis</li>
            <li><span className="text-terminal-accent">SPY + TLT + GLD</span> — classic risk-on/risk-off macro basket</li>
            <li><span className="text-terminal-accent">MSFT + GOOGL + NVDA</span> — tech sector causal dynamics</li>
          </ul>
        </div>
      </Section>

      {/* TARGET VARIABLE */}
      <Section icon="◎" title="Target Variable">
        <Control
          name="Target Variable"
          type="dropdown"
          default="First ticker's daily return"
          recommended="Pick the asset you want to predict. Usually the primary ticker you care about most."
          impact="This is the variable the causal graph tries to explain and the time series TimesFM will forecast. Everything else becomes a potential cause."
        >
          The specific variable the pipeline will treat as the prediction target. It's always expressed as a daily log return (e.g. <span className="text-terminal-accent font-mono">AAPL_ret</span>).
          The causal discovery stage will find which other variables in your dataset causally influence this one.
        </Control>
      </Section>

      {/* DATE RANGE */}
      <Section icon="◬" title="Date Range">
        <Control
          name="Start Date"
          type="date"
          default="~3 years ago"
          recommended="3–7 years of data. More data improves causal discovery and regime detection accuracy."
          impact="Too short (<1 year): not enough samples for reliable causal inference. Too long (>10 years): regime shifts make older data misleading."
        >
          The beginning of the historical window to fetch. Data is sourced from Yahoo Finance via yfinance. Going further back gives the model more training samples.
        </Control>

        <Control
          name="End Date"
          type="date"
          default="Today"
          recommended="Leave as today unless you want to backtest against a specific historical period."
          impact="Setting an earlier end date lets you hold out recent data to manually validate how the model would have performed."
        >
          The end of the historical window. The pipeline fetches daily OHLCV data from Start Date to End Date inclusive.
        </Control>

        <Callout type="info">
          The pipeline automatically handles weekends and holidays — you'll get trading days only, so don't worry about aligning to market calendars.
        </Callout>
      </Section>

      {/* ADVANCED SETTINGS */}
      <Section icon="⬡" title="Advanced Settings">
        <Callout type="tip">
          Click "Advanced Settings" on the Control page to reveal these. They're hidden by default because the defaults are solid for most use cases.
        </Callout>

        <Control
          name="Max Lag"
          type="slider"
          range="1 – 10 days"
          default="5"
          recommended="5 for most assets. Increase to 7–10 if you suspect slow-moving macro effects. Decrease to 2–3 for high-frequency, reactive assets."
          impact="Controls how far back in time the model looks for causal relationships. Max Lag = 5 means the model checks if today's return was caused by anything in the past 5 days. Higher values = exponentially larger lag matrix = slower causal discovery."
        >
          The maximum number of past time steps included in the lag matrix (Stage 3). For each feature (RSI, volume, return etc.) the model creates copies lagged by 1 day, 2 days… up to Max Lag days. These lagged copies are what CD-NOTS searches for causal arrows.
        </Control>

        <Control
          name="Significance Level (Alpha)"
          type="dropdown"
          default="0.05"
          recommended="0.05 (standard) for most runs. Use 0.01 if you want only very strong, confident causal edges. Use 0.10 if your graph looks too sparse."
          impact="The p-value threshold for the independence test. Lower alpha = fewer but more reliable causal edges. Higher alpha = denser graph but more false positives. Affects Stage 4 and Stage 5."
        >
          The statistical significance threshold used by the causal discovery algorithm (CD-NOTS) to decide whether a relationship between two variables is real or noise.
          Think of it as how much proof the algorithm needs before drawing an arrow in the causal graph.
        </Control>

        <Control
          name="Independence Test"
          type="dropdown"
          default="RCoT"
          recommended="RCoT for balanced speed and accuracy. KCI if you need maximum accuracy and can wait. Fisher Z only if you're doing a quick prototype and know the data is roughly linear."
          impact="Determines how Stage 4 tests whether two variables are causally related. Affects both speed and the quality of the causal graph."
        >
          The statistical test CD-NOTS uses to determine if two variables are conditionally independent (i.e. no causal link) or dependent (potential causal link).
          <ul className="mt-2 space-y-1 ml-3 text-terminal-muted">
            <li><span className="text-terminal-accent">RCoT</span> — Randomised Conditional Correlation Test. Fast, handles non-linear relationships. Best default.</li>
            <li><span className="text-terminal-accent">KCI</span> — Kernel-based Conditional Independence. Most accurate, captures complex non-linear dependencies, but 3–5× slower than RCoT.</li>
            <li><span className="text-terminal-accent">Fisher Z</span> — Assumes linear Gaussian relationships. Very fast, but misses non-linear structure. Only use for sanity checks.</li>
          </ul>
        </Control>

        <Control
          name="Regimes"
          type="slider"
          range="2 – 5"
          default="3"
          recommended="3 for most markets (bull / bear / sideways). Use 4 if you want to distinguish high-vol bear from low-vol bear. Rarely need 5."
          impact="Sets the number of hidden market states for the HMM (Stage 6). More regimes = finer-grained market segmentation, but each regime needs enough data samples to be statistically meaningful."
        >
          The number of hidden market regimes the Gaussian HMM (Hidden Markov Model) will try to identify in Stage 6. Each regime represents a distinct market environment — typically characterised by different return distributions and volatility levels (e.g. trending bull, high-volatility bear, choppy sideways).
        </Control>

        <Control
          name="Forecast Horizon"
          type="slider"
          range="1 – 20 days"
          default="10"
          recommended="5–10 days for actionable short-term predictions. Beyond 15 days, uncertainty bands widen dramatically and the forecast becomes directional guidance at best."
          impact="How many trading days ahead TimesFM predicts. Longer horizons produce wider quantile bands (more uncertainty). Shorter horizons are more precise. Affects Stage 7 only."
        >
          The number of future trading days TimesFM will generate predictions for. The Forecast view will show a point estimate plus 10th–90th percentile quantile bands for each day in the horizon.
        </Control>

        <Control
          name="Initial Capital"
          type="number"
          default="$100,000"
          recommended="Use your actual intended capital or $100,000 as a benchmark. The absolute value doesn't affect strategy quality — only dollar P&L figures and Kelly sizing."
          impact="Used only in Stage 8 (Backtest). Sets the starting portfolio value for the walk-forward backtest. Affects the equity curve scale, P&L in dollars, and Kelly position sizing in dollar terms."
        >
          The hypothetical starting capital for the walk-forward backtest. The backtest simulates LONG / SHORT / FLAT signals using Kelly criterion position sizing, and tracks how this capital would have grown or shrunk over the historical test period.
        </Control>

        <Control
          name="Include Macro (VIX, DXY, TNX)"
          type="checkbox"
          default="Enabled"
          recommended="Keep enabled. Macro variables are often the hidden common causes that explain why assets move together. Disabling can cause spurious causal edges."
          impact="Adds 3 extra time series to the feature set — VIX (volatility index), DXY (US Dollar index), TNX (10-year Treasury yield). These become potential causal nodes in the graph. Disabling speeds up causal discovery slightly but risks missing macro-driven relationships."
        >
          When enabled, the pipeline fetches three macro variables alongside your tickers: VIX (CBOE Volatility Index), DXY (US Dollar Index), and TNX (10-Year Treasury Yield). These are among the most powerful drivers of equity returns and including them lets the causal model attribute moves correctly instead of creating false stock-to-stock edges.
        </Control>

        <Control
          name="Include Fama-French Factors"
          type="checkbox"
          default="Disabled"
          recommended="Enable if you're analysing individual stocks and want to control for well-known risk premia (size, value, momentum). Disable for ETF/index analysis — the factors are already embedded."
          impact="Adds Mkt-RF, SMB, HML, RMW, CMA, and MOM factors to the feature set. These are academic risk factors that explain a large portion of stock returns. More features = slower causal discovery."
        >
          When enabled, the pipeline downloads the Fama-French 5-factor + momentum data from Kenneth French's data library. These factors (Market excess return, Small-minus-Big, High-minus-Low, Robust-minus-Weak, Conservative-minus-Aggressive, Momentum) are standard controls in academic finance and help the causal model separate genuine stock-specific causation from factor exposure.
        </Control>
      </Section>

      {/* PIPELINE BUTTON */}
      <Section icon="▶" title="Run Pipeline Button">
        <div className="text-xs text-terminal-text leading-relaxed">
          Clicking <span className="text-terminal-accent font-bold">▶ Run Pipeline</span> does the following in sequence:
        </div>
        <ol className="text-xs text-terminal-text space-y-2 ml-4 list-decimal leading-relaxed">
          <li>Sends your configuration to the FastAPI backend and receives a unique <span className="font-mono text-terminal-accent">run_id</span></li>
          <li>Opens a WebSocket connection to stream real-time stage updates</li>
          <li>Automatically navigates you to the <span className="text-terminal-accent">Status</span> view where you can watch each stage progress</li>
          <li>Results populate each view (Causal Graph, Forecasts, Backtest etc.) as stages complete</li>
        </ol>
        <Callout type="warning">
          The first run after installing TimesFM will download ~800MB of model weights from HuggingFace. This happens once — all subsequent runs use the cached weights and are much faster.
        </Callout>
        <Callout type="info">
          A typical full run takes 5–15 minutes locally on CPU depending on tickers, date range, and independence test choice. Stage 4 (Causal Discovery) is usually the longest.
        </Callout>
      </Section>

      {/* PREVIOUS RUNS */}
      <Section icon="◉" title="Previous Runs Panel">
        <div className="text-xs text-terminal-text leading-relaxed">
          The right-hand panel lists all historical pipeline runs stored in the local SQLite database. Each card shows:
        </div>
        <ul className="text-xs text-terminal-muted space-y-1 ml-3 leading-relaxed">
          <li><span className="text-terminal-accent">Run ID</span> — unique identifier (click to reload that run's results)</li>
          <li><span className="text-terminal-accent">Status badge</span> — pending / running / completed / failed</li>
          <li><span className="text-terminal-accent">Target</span> — the variable that was predicted</li>
          <li><span className="text-terminal-accent">Tickers</span> — assets included in that run</li>
          <li><span className="text-terminal-accent">Timestamp</span> — when the run was started</li>
          <li><span className="text-terminal-accent">✕ button</span> — permanently deletes the run and its artifacts</li>
        </ul>
        <Callout type="tip">
          Click any previous run card to load its cached results into all views — you don't need to re-run the pipeline. Results persist across browser refreshes.
        </Callout>
      </Section>
    </div>
  )
}
