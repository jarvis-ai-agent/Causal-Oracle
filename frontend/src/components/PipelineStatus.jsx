import React, { useEffect, useRef } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { useApi } from '../hooks/useApi'

const STAGES = [
  { num: 1, name: 'Data Ingestion' },
  { num: 2, name: 'Features' },
  { num: 3, name: 'Lag Matrix' },
  { num: 4, name: 'Causal Discovery' },
  { num: 5, name: 'Validation' },
  { num: 6, name: 'Regime Detection' },
  { num: 7, name: 'Forecasting' },
  { num: 8, name: 'Backtesting' },
]

function StageIcon({ status, running }) {
  if (running) return <span className="inline-block w-4 h-4 rounded-full border-2 border-terminal-accent border-t-transparent animate-spin" />
  if (status === 'completed') return <span className="text-terminal-green text-sm">✓</span>
  if (status === 'failed') return <span className="text-terminal-red text-sm">✗</span>
  return <span className="inline-block w-3 h-3 rounded-full bg-terminal-border" />
}

export default function PipelineStatus() {
  const {
    activeRunId, currentRun, setCurrentRun,
    wsEvents, appendWsEvent,
    stageProgress, setStageProgress,
    setGraphData, setForecastData, setBacktestData, setRegimeData, setValidationData,
    setActiveView,
  } = usePipelineStore()
  const { getRun, getGraph, getForecast, getBacktest, getRegimes, getValidation } = useApi()
  const logRef = useRef(null)

  // Poll run state
  useEffect(() => {
    if (!activeRunId) return
    const fetch = async () => {
      try {
        const run = await getRun(activeRunId)
        setCurrentRun(run)
        // Load results when available
        if (run.status === 'completed') {
          try { setGraphData(await getGraph(activeRunId)) } catch (_) {}
          try { setForecastData(await getForecast(activeRunId)) } catch (_) {}
          try { setBacktestData(await getBacktest(activeRunId)) } catch (_) {}
          try { setRegimeData(await getRegimes(activeRunId)) } catch (_) {}
          try { setValidationData(await getValidation(activeRunId)) } catch (_) {}
        }
      } catch (e) {
        console.error('Poll error:', e)
      }
    }
    fetch()
    const interval = setInterval(fetch, 3000)
    return () => clearInterval(interval)
  }, [activeRunId])

  // WebSocket
  useWebSocket(activeRunId, (evt) => {
    appendWsEvent(evt)
    if (evt.stage && typeof evt.progress === 'number') {
      setStageProgress(evt.stage, evt.progress)
    }
  })

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [wsEvents])

  if (!activeRunId) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        No active pipeline run. Configure and start a run from the Control tab.
      </div>
    )
  }

  const run = currentRun
  const stageStatuses = run?.stage_statuses || {}

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-auto">
      {/* Run header */}
      <div className="panel">
        <div className="panel-header flex items-center justify-between">
          <span>Pipeline Run: <span className="text-terminal-text">{activeRunId}</span></span>
          <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${
            run?.status === 'completed' ? 'bg-terminal-green text-terminal-bg' :
            run?.status === 'failed' ? 'bg-terminal-red text-terminal-bg' :
            run?.status === 'running' ? 'bg-terminal-accent text-terminal-bg pulse-accent' :
            'bg-terminal-muted text-terminal-bg'
          }`}>{run?.status || 'loading'}</span>
        </div>
        {run?.config && (
          <div className="p-3 flex gap-6 text-xs text-terminal-muted">
            <span>Target: <span className="text-terminal-accent">{run.config.target}</span></span>
            <span>Tickers: <span className="text-terminal-text">{run.config.tickers?.join(', ')}</span></span>
            <span>Dates: <span className="text-terminal-text">{run.config.start_date} → {run.config.end_date}</span></span>
          </div>
        )}
      </div>

      {/* Stage pipeline bar */}
      <div className="panel">
        <div className="panel-header">Stage Progress</div>
        <div className="p-4">
          <div className="flex items-center gap-1 mb-4">
            {STAGES.map((s, idx) => {
              const st = stageStatuses[s.num] || {}
              const isRunning = run?.current_stage === s.num && run?.status === 'running'
              const isDone = st.status === 'completed'
              const isFailed = st.status === 'failed'
              return (
                <React.Fragment key={s.num}>
                  <div className={`flex flex-col items-center gap-1 flex-1 ${
                    isDone ? 'opacity-100' : isRunning ? 'opacity-100' : 'opacity-40'
                  }`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 ${
                      isDone ? 'border-terminal-green bg-terminal-green/10' :
                      isFailed ? 'border-terminal-red bg-terminal-red/10' :
                      isRunning ? 'border-terminal-accent bg-terminal-accent/10' :
                      'border-terminal-border bg-transparent'
                    }`}>
                      <StageIcon status={st.status} running={isRunning} />
                    </div>
                    <span className="text-xs text-terminal-muted text-center leading-tight" style={{ fontSize: '10px' }}>
                      {s.name}
                    </span>
                  </div>
                  {idx < STAGES.length - 1 && (
                    <div className={`h-px flex-none w-4 ${isDone ? 'bg-terminal-green' : 'bg-terminal-border'}`} />
                  )}
                </React.Fragment>
              )
            })}
          </div>

          {/* Current stage detail */}
          {run?.current_stage > 0 && (
            <div className="mt-2">
              {STAGES.filter(s => s.num === run.current_stage).map(s => {
                const st = stageStatuses[s.num] || {}
                const pct = (stageProgress[s.num] || 0) * 100
                return (
                  <div key={s.num} className="bg-terminal-surface rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-terminal-accent">{s.name}</span>
                      <span className="text-xs text-terminal-muted">{pct.toFixed(0)}%</span>
                    </div>
                    <div className="stage-bar">
                      <div className="stage-bar-fill" style={{ width: `${pct}%` }} />
                    </div>
                    {st.summary && (
                      <div className="mt-2 text-xs text-terminal-green">{st.summary}</div>
                    )}
                    {st.error && (
                      <div className="mt-2 text-xs text-terminal-red">{st.error}</div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Stage summaries */}
      <div className="grid grid-cols-4 gap-2">
        {STAGES.map(s => {
          const st = stageStatuses[s.num] || {}
          if (!st.summary && !st.error) return null
          return (
            <div key={s.num} className={`panel p-2 text-xs ${st.status === 'failed' ? 'border-terminal-red/50' : ''}`}>
              <div className="text-terminal-muted mb-1">{s.name}</div>
              {st.summary && <div className="text-terminal-text">{st.summary}</div>}
              {st.error && <div className="text-terminal-red">{st.error}</div>}
              {st.duration_sec && <div className="text-terminal-muted mt-1">{st.duration_sec}s</div>}
            </div>
          )
        })}
      </div>

      {/* Error banner — show full error when pipeline fails */}
      {run?.status === 'failed' && (() => {
        const errorLog = run?.logs?.find(l => l.level === 'ERROR')
        return errorLog ? (
          <div className="panel border-terminal-red/60">
            <div className="panel-header text-terminal-red flex items-center gap-2">
              <span>✗</span> Pipeline Error
            </div>
            <pre className="p-4 text-xs text-terminal-red font-mono leading-relaxed overflow-auto max-h-48 whitespace-pre-wrap">
              {errorLog.message}
            </pre>
          </div>
        ) : null
      })()}

      {/* Log stream */}
      <div className="panel flex-1 flex flex-col min-h-0">
        <div className="panel-header">Live Log Stream</div>
        <div ref={logRef} className="flex-1 overflow-auto p-3 font-mono text-xs leading-relaxed">
          {wsEvents.filter(e => e.event !== 'ping' && e.event !== 'connected').map((evt, i) => (
            <div key={i} className={`flex gap-2 mb-0.5 ${
              evt.event === 'stage_error' || evt.event === 'pipeline_error' ? 'text-terminal-red' :
              evt.event === 'stage_complete' || evt.event === 'pipeline_complete' ? 'text-terminal-green' :
              'text-terminal-muted'
            }`}>
              <span className="text-terminal-border flex-none">{evt.timestamp?.slice(11, 19) || ''}</span>
              <span className={`flex-none w-4 text-center ${
                evt.stage ? 'text-terminal-accent' : 'text-terminal-muted'
              }`}>{evt.stage || '—'}</span>
              <span className="text-terminal-text whitespace-pre-wrap">{evt.message}</span>
            </div>
          ))}
          {/* Fallback: show logs from DB if no WS events captured */}
          {wsEvents.filter(e => e.event !== 'ping' && e.event !== 'connected').length === 0 && run?.logs?.length > 0 && (
            run.logs.map((log, i) => (
              <div key={i} className={`flex gap-2 mb-0.5 ${log.level === 'ERROR' ? 'text-terminal-red' : 'text-terminal-muted'}`}>
                <span className="text-terminal-border flex-none">{log.timestamp?.slice(11, 19) || ''}</span>
                <span className="flex-none w-4 text-center text-terminal-muted">{log.stage || '—'}</span>
                <span className="text-terminal-text whitespace-pre-wrap">{log.message}</span>
              </div>
            ))
          )}
          {wsEvents.length === 0 && (!run?.logs || run.logs.length === 0) && (
            <div className="text-terminal-muted">Waiting for pipeline events...</div>
          )}
        </div>
      </div>

      {/* Navigate to results */}
      {run?.status === 'completed' && (
        <div className="flex gap-2">
          {[
            { view: 'graph', label: 'Causal Graph' },
            { view: 'forecast', label: 'Forecasts' },
            { view: 'backtest', label: 'Backtest' },
            { view: 'regime', label: 'Regimes' },
          ].map(({ view, label }) => (
            <button key={view} onClick={() => setActiveView(view)}
              className="flex-1 py-2 bg-terminal-surface border border-terminal-border rounded text-xs hover:border-terminal-accent hover:text-terminal-accent transition-colors"
            >
              {label} →
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
