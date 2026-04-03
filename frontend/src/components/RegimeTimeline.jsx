import React, { useMemo } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { regimeColor } from '../utils/formatting'

function TransitionMatrix({ matrix, regimeMap }) {
  if (!matrix || !matrix.length) return null
  const n = matrix.length
  const names = Object.values(regimeMap || {})

  return (
    <div>
      <div className="text-xs text-terminal-muted mb-2 uppercase tracking-wider">Transition Matrix</div>
      <div className="inline-block">
        <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${n + 1}, minmax(60px, 1fr))` }}>
          {/* Header row */}
          <div className="text-xs text-terminal-muted p-1" />
          {names.slice(0, n).map((name, i) => (
            <div key={i} className="text-xs text-terminal-muted p-1 text-center font-bold truncate">
              {name || `R${i}`}
            </div>
          ))}
          {/* Data rows */}
          {matrix.map((row, i) => (
            <React.Fragment key={i}>
              <div className="text-xs text-terminal-muted p-1 font-bold truncate">{names[i] || `R${i}`}</div>
              {row.map((val, j) => {
                const intensity = Math.min(1, val || 0)
                return (
                  <div key={j} className="p-1 text-center text-xs font-mono"
                    style={{
                      background: `rgba(0, 212, 255, ${intensity * 0.5})`,
                      color: intensity > 0.5 ? '#ffffff' : '#c9d1e0',
                    }}
                  >
                    {(val || 0).toFixed(2)}
                  </div>
                )
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function RegimeTimeline() {
  const { regimeData, activeRunId } = usePipelineStore()

  if (!regimeData) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Regime data not yet available.' : 'No active run.'}
      </div>
    )
  }

  const { regime_map, transition_matrix, regime_distribution, current_regime, regime_labels } = regimeData

  // Build timeline bars
  const timelineData = useMemo(() => {
    if (!regime_labels) return []
    const entries = Object.entries(regime_labels)
    return entries.map(([date, label]) => ({
      date: date.slice(0, 10),
      label: Number(label),
      name: regime_map?.[label] || `regime_${label}`,
    }))
  }, [regime_labels, regime_map])

  // Compress consecutive same-regime spans for the timeline bar
  const spans = useMemo(() => {
    if (timelineData.length === 0) return []
    const result = []
    let cur = { name: timelineData[0].name, count: 1, start: timelineData[0].date }
    for (let i = 1; i < timelineData.length; i++) {
      if (timelineData[i].name === cur.name) {
        cur.count++
      } else {
        result.push({ ...cur, end: timelineData[i].date })
        cur = { name: timelineData[i].name, count: 1, start: timelineData[i].date }
      }
    }
    result.push({ ...cur, end: timelineData[timelineData.length - 1].date })
    return result
  }, [timelineData])

  const total = timelineData.length || 1

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-auto">
      {/* Current regime badge */}
      <div className="panel p-4 flex items-center gap-6">
        <div>
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">Current Regime</div>
          <div className="text-2xl font-bold" style={{ color: regimeColor(current_regime) }}>
            {current_regime || '—'}
          </div>
        </div>
        <div className="flex gap-4">
          {Object.entries(regime_distribution || {}).map(([name, pct]) => (
            <div key={name} className="text-center">
              <div className="text-xs text-terminal-muted mb-1">{name}</div>
              <div className="text-sm font-bold" style={{ color: regimeColor(name) }}>
                {(pct * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline bar */}
      <div className="panel">
        <div className="panel-header">Regime Timeline</div>
        <div className="p-4">
          <div className="flex h-8 rounded overflow-hidden border border-terminal-border">
            {spans.map((span, i) => (
              <div
                key={i}
                title={`${span.name}: ${span.start} → ${span.end} (${span.count} days)`}
                style={{
                  width: `${(span.count / total) * 100}%`,
                  background: regimeColor(span.name),
                  opacity: 0.8,
                  minWidth: span.count > 5 ? '2px' : undefined,
                }}
              />
            ))}
          </div>
          <div className="flex justify-between mt-1 text-xs text-terminal-muted">
            <span>{timelineData[0]?.date || ''}</span>
            <span>{timelineData[Math.floor(timelineData.length / 2)]?.date || ''}</span>
            <span>{timelineData[timelineData.length - 1]?.date || ''}</span>
          </div>
          <div className="flex gap-4 mt-3">
            {Object.values(regime_map || {}).map(name => (
              <div key={name} className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded" style={{ background: regimeColor(name) }} />
                <span className="text-terminal-text capitalize">{name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        {/* Distribution pie (as bar chart) */}
        <div className="panel flex-1">
          <div className="panel-header">Regime Distribution</div>
          <div className="p-4">
            {Object.entries(regime_distribution || {}).map(([name, pct]) => (
              <div key={name} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-terminal-text capitalize">{name}</span>
                  <span className="text-terminal-muted">{(pct * 100).toFixed(1)}%</span>
                </div>
                <div className="h-3 bg-terminal-surface rounded overflow-hidden">
                  <div
                    className="h-full rounded transition-all"
                    style={{ width: `${pct * 100}%`, background: regimeColor(name) }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Transition matrix */}
        <div className="panel flex-1">
          <div className="panel-header">Transition Probabilities</div>
          <div className="p-4">
            <TransitionMatrix matrix={transition_matrix} regimeMap={regime_map} />
          </div>
        </div>
      </div>
    </div>
  )
}
