import React, { useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { useApi } from '../hooks/useApi'
import { fmtDate, fmtDateTime, fmtDuration } from '../utils/formatting'

const TICKERS_PRESETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'GLD', 'TLT']

function Badge({ status }) {
  const colors = {
    pending: 'bg-terminal-muted text-terminal-bg',
    running: 'bg-terminal-accent text-terminal-bg pulse-accent',
    completed: 'bg-terminal-green text-terminal-bg',
    failed: 'bg-terminal-red text-terminal-bg',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${colors[status] || colors.pending}`}>
      {status}
    </span>
  )
}

export default function PipelineControl() {
  const { config, setConfig, runs, setRuns, setActiveRunId, setActiveView, clearResults, clearWsEvents, resetStageProgress } = usePipelineStore()
  const { startRun, listRuns, deleteRun, exportRun } = useApi()
  const [exporting, setExporting] = useState(null)
  const [advanced, setAdvanced] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [tickerInput, setTickerInput] = useState('')

  const addTicker = (t) => {
    const ticker = t.trim().toUpperCase()
    if (ticker && !config.tickers.includes(ticker)) {
      setConfig({ tickers: [...config.tickers, ticker] })
    }
    setTickerInput('')
  }

  const removeTicker = (t) => {
    setConfig({ tickers: config.tickers.filter(x => x !== t) })
  }

  const handleRun = async () => {
    setError(null)
    setLoading(true)
    clearResults()
    clearWsEvents()
    resetStageProgress()
    try {
      const { run_id } = await startRun(config)
      setActiveRunId(run_id)
      // Refresh runs list
      const updated = await listRuns()
      setRuns(updated)
      setActiveView('status')
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Failed to start pipeline')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteRun = async (id) => {
    try {
      await deleteRun(id)
      const updated = await listRuns()
      setRuns(updated)
    } catch (e) {
      console.error('Delete failed:', e)
    }
  }

  const handleExport = async (e, id) => {
    e.stopPropagation()
    setExporting(id)
    try {
      const data = await exportRun(id)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `causal-oracle-${id}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    } finally {
      setExporting(null)
    }
  }

  return (
    <div className="flex gap-4 h-full p-4 overflow-auto">
      {/* Config panel */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        <div className="panel">
          <div className="panel-header">Pipeline Configuration</div>
          <div className="p-4 flex flex-col gap-4">

            {/* Tickers */}
            <div>
              <label className="block text-xs text-terminal-muted mb-2 uppercase tracking-wider">Assets</label>
              <div className="flex flex-wrap gap-1 mb-2">
                {config.tickers.map(t => (
                  <span key={t} className="flex items-center gap-1 px-2 py-1 bg-terminal-surface border border-terminal-border rounded text-xs">
                    <span className="text-terminal-accent font-bold">{t}</span>
                    <button onClick={() => removeTicker(t)} className="text-terminal-muted hover:text-terminal-red ml-1">×</button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={tickerInput}
                  onChange={e => setTickerInput(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === 'Enter' && addTicker(tickerInput)}
                  placeholder="Add ticker..."
                  className="bg-terminal-surface border border-terminal-border rounded px-3 py-1.5 text-xs text-terminal-text w-28 focus:outline-none focus:border-terminal-accent"
                />
                <button onClick={() => addTicker(tickerInput)} className="px-3 py-1.5 bg-terminal-surface border border-terminal-border rounded text-xs hover:border-terminal-accent">
                  Add
                </button>
                {TICKERS_PRESETS.filter(t => !config.tickers.includes(t)).slice(0, 4).map(t => (
                  <button key={t} onClick={() => addTicker(t)} className="px-2 py-1.5 bg-terminal-surface border border-terminal-border rounded text-xs text-terminal-muted hover:text-terminal-text hover:border-terminal-accent">
                    +{t}
                  </button>
                ))}
              </div>
            </div>

            {/* Target */}
            <div>
              <label className="block text-xs text-terminal-muted mb-1 uppercase tracking-wider">Target Variable</label>
              <select
                value={config.target}
                onChange={e => setConfig({ target: e.target.value })}
                className="bg-terminal-surface border border-terminal-border rounded px-3 py-1.5 text-xs text-terminal-text w-full focus:outline-none focus:border-terminal-accent"
              >
                {config.tickers.map(t => (
                  <option key={t} value={`${t}_ret`}>{t} daily return</option>
                ))}
              </select>
            </div>

            {/* Date range */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-terminal-muted mb-1 uppercase tracking-wider">Start Date</label>
                <input
                  type="date"
                  value={config.start_date}
                  onChange={e => setConfig({ start_date: e.target.value })}
                  className="bg-terminal-surface border border-terminal-border rounded px-3 py-1.5 text-xs text-terminal-text w-full focus:outline-none focus:border-terminal-accent"
                />
              </div>
              <div>
                <label className="block text-xs text-terminal-muted mb-1 uppercase tracking-wider">End Date</label>
                <input
                  type="date"
                  value={config.end_date}
                  onChange={e => setConfig({ end_date: e.target.value })}
                  className="bg-terminal-surface border border-terminal-border rounded px-3 py-1.5 text-xs text-terminal-text w-full focus:outline-none focus:border-terminal-accent"
                />
              </div>
            </div>

            {/* Advanced settings toggle */}
            <button
              onClick={() => setAdvanced(!advanced)}
              className="text-xs text-terminal-accent hover:text-terminal-text text-left flex items-center gap-1"
            >
              <span>{advanced ? '▼' : '▶'}</span>
              Advanced Settings
            </button>

            {advanced && (
              <div className="grid grid-cols-2 gap-4 p-4 bg-terminal-surface rounded border border-terminal-border">
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Max Lag: {config.max_lag}</label>
                  <input type="range" min="1" max="10" value={config.max_lag}
                    onChange={e => setConfig({ max_lag: Number(e.target.value) })}
                    className="w-full accent-terminal-accent"
                  />
                </div>
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Significance Level</label>
                  <select value={config.alpha}
                    onChange={e => setConfig({ alpha: Number(e.target.value) })}
                    className="bg-terminal-panel border border-terminal-border rounded px-2 py-1 text-xs w-full focus:outline-none focus:border-terminal-accent"
                  >
                    <option value={0.01}>0.01 (strict)</option>
                    <option value={0.05}>0.05 (standard)</option>
                    <option value={0.10}>0.10 (liberal)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Independence Test</label>
                  <select value={config.indep_test}
                    onChange={e => setConfig({ indep_test: e.target.value })}
                    className="bg-terminal-panel border border-terminal-border rounded px-2 py-1 text-xs w-full focus:outline-none focus:border-terminal-accent"
                  >
                    <option value="rcot">RCoT (recommended)</option>
                    <option value="kci">KCI (accurate, slow)</option>
                    <option value="fisherz">Fisher Z (fast)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Regimes: {config.n_regimes}</label>
                  <input type="range" min="2" max="5" value={config.n_regimes}
                    onChange={e => setConfig({ n_regimes: Number(e.target.value) })}
                    className="w-full accent-terminal-accent"
                  />
                </div>
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Forecast Horizon: {config.horizon} days</label>
                  <input type="range" min="1" max="20" value={config.horizon}
                    onChange={e => setConfig({ horizon: Number(e.target.value) })}
                    className="w-full accent-terminal-accent"
                  />
                </div>
                <div>
                  <label className="block text-xs text-terminal-muted mb-1">Initial Capital</label>
                  <input type="number" value={config.initial_capital}
                    onChange={e => setConfig({ initial_capital: Number(e.target.value) })}
                    className="bg-terminal-panel border border-terminal-border rounded px-2 py-1 text-xs w-full focus:outline-none focus:border-terminal-accent"
                  />
                </div>
                <div className="col-span-2 flex gap-4">
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input type="checkbox" checked={config.include_macro}
                      onChange={e => setConfig({ include_macro: e.target.checked })}
                      className="accent-terminal-accent"
                    />
                    Include Macro (VIX, DXY, TNX)
                  </label>
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input type="checkbox" checked={config.include_factors}
                      onChange={e => setConfig({ include_factors: e.target.checked })}
                      className="accent-terminal-accent"
                    />
                    Include Fama-French Factors
                  </label>
                </div>
              </div>
            )}

            {error && (
              <div className="p-3 bg-red-900/20 border border-terminal-red rounded text-xs text-terminal-red">
                {error}
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={loading || config.tickers.length === 0}
              className={`w-full py-3 rounded font-bold text-sm tracking-wider uppercase transition-all
                ${loading ? 'bg-terminal-muted text-terminal-bg cursor-not-allowed'
                  : 'bg-terminal-accent text-terminal-bg hover:brightness-110 active:brightness-90'}`}
            >
              {loading ? '◌ Starting...' : '▶ Run Pipeline'}
            </button>
          </div>
        </div>
      </div>

      {/* Previous runs */}
      <div className="w-80 flex flex-col gap-2">
        <div className="panel flex-1">
          <div className="panel-header">Previous Runs</div>
          <div className="p-2 overflow-auto max-h-[600px]">
            {runs.length === 0 ? (
              <div className="text-terminal-muted text-xs p-4 text-center">No runs yet</div>
            ) : (
              runs.map(run => (
                <div key={run.id}
                  className="p-2 mb-1 border border-terminal-border rounded hover:border-terminal-accent cursor-pointer"
                  onClick={() => { setActiveRunId(run.id); setActiveView('status') }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-xs text-terminal-accent">{run.id}</span>
                    <Badge status={run.status} />
                  </div>
                  <div className="text-xs text-terminal-muted">
                    Target: <span className="text-terminal-text">{run.target}</span>
                  </div>
                  <div className="text-xs text-terminal-muted truncate">
                    {run.tickers?.join(', ')}
                  </div>
                  {run.total_duration_sec != null && (
                    <div className="text-xs text-terminal-accent">{fmtDuration(run.total_duration_sec)}</div>
                  )}
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-xs text-terminal-muted">{fmtDateTime(run.created_at)}</span>
                    <div className="flex gap-1.5">
                      {run.status === 'completed' && (
                        <button
                          onClick={e => handleExport(e, run.id)}
                          disabled={exporting === run.id}
                          className="text-xs text-terminal-accent hover:text-terminal-text"
                          title="Export JSON"
                        >↓</button>
                      )}
                      <button
                        onClick={e => { e.stopPropagation(); handleDeleteRun(run.id) }}
                        className="text-xs text-terminal-muted hover:text-terminal-red"
                      >✕</button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
