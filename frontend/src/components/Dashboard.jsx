import React, { useEffect, useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { useApi } from '../hooks/useApi'
import PipelineControl from './PipelineControl'
import PipelineStatus from './PipelineStatus'
import CausalGraph from './CausalGraph'
import ForecastChart from './ForecastChart'
import BacktestResults from './BacktestResults'
import RegimeTimeline from './RegimeTimeline'
import RefutationTable from './RefutationTable'
import Guide from './Guide'
import RunsPage from './RunsPage'

const VIEWS = [
  { id: 'control', label: 'Control', icon: '⚙' },
  { id: 'status', label: 'Status', icon: '◉' },
  { id: 'graph', label: 'Causal Graph', icon: '⬡' },
  { id: 'forecast', label: 'Forecasts', icon: '◬' },
  { id: 'backtest', label: 'Backtest', icon: '◈' },
  { id: 'regime', label: 'Regimes', icon: '◧' },
  { id: 'validation', label: 'Validation', icon: '◎' },
  { id: 'runs', label: 'Runs', icon: '☰' },
  { id: 'guide', label: 'Guide', icon: '?' },
]

export default function Dashboard() {
  const { activeView, setActiveView, setRuns, activeRunId } = usePipelineStore()
  const { listRuns, health } = useApi()
  const [serverOk, setServerOk] = useState(null)

  // Bootstrap: load runs and check health
  useEffect(() => {
    const init = async () => {
      try {
        await health()
        setServerOk(true)
        const runs = await listRuns()
        setRuns(runs)
      } catch (e) {
        setServerOk(false)
      }
    }
    init()
    const interval = setInterval(async () => {
      try {
        const runs = await listRuns()
        setRuns(runs)
        if (!serverOk) setServerOk(true)
      } catch (e) {
        setServerOk(false)
      }
    }, 10000)
    return () => clearInterval(interval)
  }, [])

  const ViewComponent = {
    control: PipelineControl,
    status: PipelineStatus,
    graph: CausalGraph,
    forecast: ForecastChart,
    backtest: BacktestResults,
    regime: RegimeTimeline,
    validation: RefutationTable,
    runs: RunsPage,
    guide: Guide,
  }[activeView] || PipelineControl

  return (
    <div className="flex h-screen overflow-hidden bg-terminal-bg text-terminal-text font-mono">
      {/* Sidebar */}
      <aside className="w-48 flex-none bg-terminal-surface border-r border-terminal-border flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-terminal-border">
          <div className="text-terminal-accent font-bold text-sm tracking-widest uppercase">Causal</div>
          <div className="text-terminal-gold font-bold text-sm tracking-widest uppercase">Oracle</div>
          <div className="text-terminal-muted text-xs mt-1">v1.0.0</div>
        </div>

        {/* Server status */}
        <div className="px-4 py-2 border-b border-terminal-border flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${serverOk === true ? 'bg-terminal-green' : serverOk === false ? 'bg-terminal-red' : 'bg-terminal-muted'}`} />
          <span className="text-xs text-terminal-muted">
            {serverOk === true ? 'API Connected' : serverOk === false ? 'API Offline' : 'Connecting...'}
          </span>
        </div>

        {/* Active run indicator */}
        {activeRunId && (
          <div className="px-4 py-2 border-b border-terminal-border">
            <div className="text-xs text-terminal-muted">Active Run</div>
            <div className="text-xs text-terminal-accent font-mono">{activeRunId}</div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 py-2">
          {VIEWS.map(view => (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 text-xs transition-colors ${
                activeView === view.id
                  ? 'bg-terminal-accent/10 text-terminal-accent border-r-2 border-terminal-accent'
                  : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-panel'
              }`}
            >
              <span className="text-base leading-none">{view.icon}</span>
              <span className="font-medium">{view.label}</span>
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-terminal-border text-xs text-terminal-muted">
          <div>Causal Inference</div>
          <div>Stock Prediction</div>
          <div className="mt-1 text-terminal-border">CD-NOTS + TimesFM</div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex-none h-9 bg-terminal-surface border-b border-terminal-border flex items-center px-4 gap-4">
          <span className="text-xs text-terminal-muted uppercase tracking-widest">
            {VIEWS.find(v => v.id === activeView)?.label}
          </span>
          <div className="flex-1" />
          <div className="text-xs text-terminal-muted">
            {new Date().toISOString().slice(0, 19).replace('T', ' ')} UTC
          </div>
        </header>

        {/* View content */}
        <div className="flex-1 overflow-hidden">
          <ViewComponent />
        </div>
      </main>
    </div>
  )
}
