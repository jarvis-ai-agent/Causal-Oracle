import React, { useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { useApi } from '../hooks/useApi'
import { fmtDate, fmtDateTime, fmtDuration } from '../utils/formatting'

const STAGE_NAMES = {
  1: 'Ingest', 2: 'Features', 3: 'Lag Matrix', 4: 'Causal',
  5: 'Validation', 6: 'Regimes', 7: 'Forecast', 8: 'Backtest',
}

function StatusBadge({ status }) {
  const styles = {
    completed: 'bg-terminal-green/20 text-terminal-green border-terminal-green/40',
    failed: 'bg-terminal-red/20 text-terminal-red border-terminal-red/40',
    running: 'bg-terminal-accent/20 text-terminal-accent border-terminal-accent/40',
    pending: 'bg-terminal-muted/20 text-terminal-muted border-terminal-muted/40',
  }
  return (
    <span className={`px-2 py-0.5 rounded border text-xs font-bold uppercase ${styles[status] || styles.pending}`}>
      {status}
    </span>
  )
}

function StagePips({ stageStatuses }) {
  return (
    <div className="flex gap-0.5">
      {[1,2,3,4,5,6,7,8].map(n => {
        const st = stageStatuses?.[n] || stageStatuses?.[String(n)] || {}
        const color =
          st.status === 'completed' ? 'bg-terminal-green' :
          st.status === 'failed'    ? 'bg-terminal-red' :
          st.status === 'running'   ? 'bg-terminal-accent animate-pulse' :
                                      'bg-terminal-border'
        return (
          <div key={n} title={`${STAGE_NAMES[n]}: ${st.status || 'pending'}`}
            className={`w-4 h-1.5 rounded-sm ${color}`} />
        )
      })}
    </div>
  )
}

export default function RunsPage() {
  const { runs, setRuns, setActiveRunId, setActiveView } = usePipelineStore()
  const { deleteRun, listRuns, exportRun } = useApi()
  const [exporting, setExporting] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const [filter, setFilter] = useState('all') // all | completed | failed | running

  const filtered = runs.filter(r => filter === 'all' || r.status === filter)

  const handleLoad = (run) => {
    setActiveRunId(run.id)
    setActiveView('status')
  }

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    if (!confirm(`Delete run ${id}? This cannot be undone.`)) return
    setDeleting(id)
    try {
      await deleteRun(id)
      const updated = await listRuns()
      setRuns(updated)
    } finally {
      setDeleting(null)
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
    } catch (e) {
      alert('Export failed: ' + e.message)
    } finally {
      setExporting(null)
    }
  }

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-terminal-accent text-lg font-bold tracking-wider uppercase">Pipeline Runs</h1>
          <p className="text-terminal-muted text-xs mt-0.5">{runs.length} total runs stored</p>
        </div>
        <div className="flex gap-2">
          {['all', 'completed', 'failed', 'running'].map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded text-xs uppercase tracking-wide border transition-colors ${
                filter === f
                  ? 'bg-terminal-accent text-terminal-bg border-terminal-accent'
                  : 'bg-terminal-surface text-terminal-muted border-terminal-border hover:border-terminal-accent'
              }`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-terminal-muted text-sm gap-2">
          <span className="text-3xl opacity-30">◉</span>
          <span>{filter === 'all' ? 'No runs yet. Start one from the Control page.' : `No ${filter} runs.`}</span>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {filtered.map(run => {
            const stageSummaries = Object.entries(run.stage_statuses || {})
              .filter(([, v]) => v.summary)

            return (
              <div key={run.id}
                onClick={() => handleLoad(run)}
                className="panel p-4 cursor-pointer hover:border-terminal-accent/60 transition-colors group"
              >
                {/* Top row */}
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-terminal-accent font-bold text-sm">{run.id}</span>
                    <StatusBadge status={run.status} />
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => handleExport(e, run.id)}
                      disabled={exporting === run.id || run.status !== 'completed'}
                      title={run.status === 'completed' ? 'Export full JSON' : 'Run not complete'}
                      className={`px-2.5 py-1 rounded text-xs border transition-colors ${
                        run.status === 'completed'
                          ? 'border-terminal-accent text-terminal-accent hover:bg-terminal-accent/10'
                          : 'border-terminal-border text-terminal-muted cursor-not-allowed opacity-40'
                      }`}
                    >
                      {exporting === run.id ? '⏳' : '↓ Export'}
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, run.id)}
                      disabled={deleting === run.id}
                      className="px-2.5 py-1 rounded text-xs border border-terminal-border text-terminal-muted hover:border-terminal-red hover:text-terminal-red transition-colors"
                    >
                      {deleting === run.id ? '⏳' : '✕ Delete'}
                    </button>
                  </div>
                </div>

                {/* Config summary */}
                <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-terminal-muted mb-3">
                  <span>Target: <span className="text-terminal-text">{run.target}</span></span>
                  <span>Tickers: <span className="text-terminal-text">{run.tickers?.join(', ')}</span></span>
                  <span>Range: <span className="text-terminal-text">{fmtDate(run.start_date)} → {fmtDate(run.end_date)}</span></span>
                </div>

                {/* Stage pips */}
                <div className="mb-3">
                  <StagePips stageStatuses={run.stage_statuses} />
                </div>

                {/* Stage summaries (completed stages only) */}
                {stageSummaries.length > 0 && (
                  <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 mb-3">
                    {stageSummaries.map(([num, st]) => (
                      <div key={num} className="text-xs text-terminal-muted">
                        <span className="text-terminal-border">{STAGE_NAMES[num] || `Stage ${num}`}:</span>
                        {' '}<span className="text-terminal-text">{st.summary}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Bottom row — timing */}
                <div className="flex items-center gap-6 text-xs text-terminal-muted border-t border-terminal-border/50 pt-2 mt-1">
                  <span>Started: <span className="text-terminal-text">{fmtDateTime(run.created_at)}</span></span>
                  {run.completed_at && (
                    <span>Completed: <span className="text-terminal-text">{fmtDateTime(run.completed_at)}</span></span>
                  )}
                  {run.total_duration_sec != null && (
                    <span>Duration: <span className="text-terminal-accent font-bold">{fmtDuration(run.total_duration_sec)}</span></span>
                  )}
                  {run.status === 'running' && (
                    <span className="text-terminal-accent animate-pulse">● Running — Stage {run.current_stage}/8</span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
