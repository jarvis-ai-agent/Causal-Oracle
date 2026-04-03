import React, { useMemo } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { usePipelineStore } from '../store/pipelineStore'
import { fmtNum, fmtDate } from '../utils/formatting'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null
  return (
    <div className="panel p-2 text-xs border border-terminal-border">
      <div className="text-terminal-muted mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }} className="flex gap-2">
          <span>{p.name}:</span>
          <span className="font-bold">{fmtNum(p.value, 4)}</span>
        </div>
      ))}
    </div>
  )
}

export default function ForecastChart() {
  const { forecastData, regimeData, activeRunId } = usePipelineStore()

  if (!forecastData) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Forecast data not yet available.' : 'No active run.'}
      </div>
    )
  }

  const assets = Object.keys(forecastData).filter(k => k !== 'dates')
  const dates = forecastData.dates || []

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-auto">
      {assets.map(asset => {
        const fc = forecastData[asset]
        if (!fc || !fc.point) return null

        const point = fc.point
        const quantiles = fc.quantiles || []
        const arima = fc.arima_point || []
        const horizon = fc.horizon || point.length

        // Build chart data: last N historical points + forecast
        const histLen = Math.min(dates.length, 100)
        const chartData = []

        // Placeholder for historical (we'd need actual price series here)
        // Use quantile midpoint as proxy for recent history
        for (let i = histLen; i > 0; i--) {
          const date = dates[dates.length - i] || `T-${i}`
          chartData.push({ date, type: 'historical' })
        }

        // Forecast horizon
        for (let h = 0; h < horizon; h++) {
          const q = quantiles[h] || []
          chartData.push({
            date: `+${h + 1}d`,
            type: 'forecast',
            point: point[h],
            arima: arima[h],
            q10: q[0],
            q25: q[2],
            q75: q[7],
            q90: q[9],
          })
        }

        const signal = point[0] > 0.001 ? 'LONG' : point[0] < -0.001 ? 'SHORT' : 'FLAT'
        const signalColor = signal === 'LONG' ? '#22c55e' : signal === 'SHORT' ? '#ef4444' : '#f59e0b'

        return (
          <div key={asset} className="panel flex-1">
            <div className="panel-header flex items-center justify-between">
              <span>{asset} — Forecast (Horizon: {horizon} days)</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-terminal-muted">Regime: <span className="text-terminal-accent">{fc.regime_at_forecast || '—'}</span></span>
                <span className="text-xs text-terminal-muted">
                  Ensemble: <span className={fc.ensemble_agreement ? 'text-terminal-green' : 'text-terminal-red'}>
                    {fc.ensemble_agreement ? '✓ AGREE' : '✗ DISAGREE'}
                  </span>
                </span>
                <span className="px-2 py-0.5 rounded text-xs font-bold" style={{ backgroundColor: signalColor + '33', color: signalColor, border: `1px solid ${signalColor}` }}>
                  {signal}
                </span>
              </div>
            </div>
            <div className="p-4">
              <div className="grid grid-cols-5 gap-3 mb-4">
                {point.slice(0, 5).map((p, i) => (
                  <div key={i} className="bg-terminal-surface rounded p-2 text-center">
                    <div className="text-xs text-terminal-muted mb-1">+{i + 1}d</div>
                    <div className={`text-sm font-bold ${p > 0 ? 'text-terminal-green' : p < 0 ? 'text-terminal-red' : 'text-terminal-text'}`}>
                      {p > 0 ? '+' : ''}{fmtNum(p * 100, 2)}%
                    </div>
                    {arima[i] != null && (
                      <div className="text-xs text-terminal-muted">ARIMA: {arima[i] > 0 ? '+' : ''}{fmtNum(arima[i] * 100, 2)}%</div>
                    )}
                  </div>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <ComposedChart data={chartData.filter(d => d.type === 'forecast')}>
                  <CartesianGrid strokeDasharray="2 4" stroke="#1e2d45" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#5a6a7e' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#5a6a7e' }} tickFormatter={v => `${(v * 100).toFixed(1)}%`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#5a6a7e' }} />

                  {/* Quantile bands */}
                  <Area dataKey="q90" name="90th pct" stackId="1" stroke="none" fill="#00d4ff11" />
                  <Area dataKey="q75" name="75th pct" stackId="2" stroke="none" fill="#00d4ff22" />
                  <Area dataKey="q25" name="25th pct" stackId="3" stroke="none" fill="#00d4ff11" />
                  <Area dataKey="q10" name="10th pct" stackId="4" stroke="none" fill="#00d4ff05" />

                  <ReferenceLine y={0} stroke="#1e2d45" strokeDasharray="4 4" />
                  <Line dataKey="point" name="TimesFM" stroke="#00d4ff" strokeWidth={2} dot={{ r: 4, fill: '#00d4ff' }} />
                  <Line dataKey="arima" name="ARIMA" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>

              <div className="mt-3 flex gap-4 text-xs text-terminal-muted">
                <span>Confidence (lower=better): <span className="text-terminal-text">{fmtNum(fc.confidence, 4)}</span></span>
                <span>Forecast date: <span className="text-terminal-text">{fc.forecast_date || '—'}</span></span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
