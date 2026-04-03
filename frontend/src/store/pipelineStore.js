import { create } from 'zustand'

const DEFAULT_CONFIG = {
  tickers: ['AAPL', 'MSFT', 'GOOGL', 'SPY'],
  target: 'AAPL_ret',
  start_date: '2015-01-01',
  end_date: '2025-12-31',
  max_lag: 5,
  alpha: 0.05,
  indep_test: 'rcot',
  n_regimes: 3,
  horizon: 5,
  context_length: 1024,
  initial_capital: 100000,
  causal_retrain_interval: 60,
  forecast_retrain_interval: 5,
  include_macro: true,
  include_factors: false,
  signal_direction: 'both',
}

export const usePipelineStore = create((set, get) => ({
  // UI state
  activeView: 'control',
  setActiveView: (view) => set({ activeView: view }),

  // Config
  config: { ...DEFAULT_CONFIG },
  setConfig: (updates) => set((s) => ({ config: { ...s.config, ...updates } })),
  resetConfig: () => set({ config: { ...DEFAULT_CONFIG } }),

  // Active run
  activeRunId: null,
  setActiveRunId: (id) => set({ activeRunId: id }),

  // Run list
  runs: [],
  setRuns: (runs) => set({ runs }),

  // Current run state
  currentRun: null,
  setCurrentRun: (run) => set({ currentRun: run }),

  // WS events for active run
  wsEvents: [],
  appendWsEvent: (evt) => set((s) => ({
    wsEvents: [...s.wsEvents.slice(-200), evt]
  })),
  clearWsEvents: () => set({ wsEvents: [] }),

  // Stage progress
  stageProgress: {},
  setStageProgress: (stage, pct) => set((s) => ({
    stageProgress: { ...s.stageProgress, [stage]: pct }
  })),
  resetStageProgress: () => set({ stageProgress: {} }),

  // Results
  graphData: null,
  setGraphData: (data) => set({ graphData: data }),

  forecastData: null,
  setForecastData: (data) => set({ forecastData: data }),

  backtestData: null,
  setBacktestData: (data) => set({ backtestData: data }),

  regimeData: null,
  setRegimeData: (data) => set({ regimeData: data }),

  validationData: null,
  setValidationData: (data) => set({ validationData: data }),

  clearResults: () => set({
    graphData: null, forecastData: null, backtestData: null,
    regimeData: null, validationData: null,
  }),
}))
