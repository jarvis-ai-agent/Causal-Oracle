import React, { useState } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ReferenceLine
} from 'recharts'
import { usePipelineStore } from '../store/pipelineStore'
import { fmtNum, fmtPct, fmtMoney, fmtDate, signClass } from '../utils/formatting'

const MetricCard = ({ label, value, format = 'num', decimals = 2 }) => {
  let display = value
  if (format === 'pct') display = fmtPct(value)
  else if (format === 'money') display = fmtMoney(value)
  else display = fmtNum(value, decimals)
  const cls = signClass(value)
  return (
    <div className="panel p-3 text-center">
      <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-xl font-bold font-mono ${cls}`}>{display}</div>
    </div>
  )
}

export default function BacktestResults() {
  const { backtestData, activeRunId } = usePipelineStore()
  const [tab, setTab] = useState('overview')

  if (!backtestData) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Backtest results not yet available.' : 'No active run.'}
      </div>
    )
  }

  const { equity_curve, trades, metrics, signal_decay, regime_performance, comparison } = backtestData

  // Build equity curve chart data
  const equityData = Object.entries(equity_curve || {}).map(([date, val]) => ({
    date: date.slice(0, 10),
    strategy: val,
  }))

  // Build signal decay chart data
  const decayData = Object.entries(signal_decay || {}).map(([date, val]) => ({
    date: date.slice(0, 10),
    sharpe: val || 0,
  }))

  const tabs = ['overview', 'trades', 'regime']

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-auto">
      {/* Metrics grid */}
      <div className="grid grid-cols-6 gap-2">
        <MetricCard label="Sharpe Ratio" value={metrics?.sharpe} decimals={2} />
        <MetricCard label="Max Drawdown" value={metrics?.max_drawdown} format="pct" />
        <MetricCard label="Win Rate" value={metrics?.win_rate} format="pct" />
        <MetricCard label="Profit Factor" value={metrics?.profit_factor} decimals={2} />
        <MetricCard label="Total Trades" value={metrics?.total_trades} decimals={0} />
        <MetricCard label="Total Return" value={metrics?.total_return} format="pct" />
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1">
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-xs rounded uppercase tracking-wider font-bold transition-colors ${
              tab === t ? 'bg-terminal-accent text-terminal-bg' : 'bg-terminal-surface border border-terminal-border text-terminal-muted hover:text-terminal-text'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* Equity curve */}
          <div className="panel flex-1">
            <div className="panel-header flex justify-between items-center">
              <span>Equity Curve vs Buy & Hold</span>
              {comparison && (
                <div className="flex gap-4 text-xs text-terminal-muted">
                  <span>B&H Sharpe: <span className="text-terminal-text">{fmtNum(comparison.sharpe)}</span></span>
                  <span>B&H Return: <span className={signClass(comparison.total_return)}>{fmtPct(comparison.total_return)}</span></span>
                </div>
              )}
            </div>
            <div className="p-4">
              {equityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={equityData}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#1e2d45" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#5a6a7e' }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10, fill: '#5a6a7e' }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
                    <Tooltip formatter={(v) => fmtMoney(v)} contentStyle={{ background: '#0f1622', border: '1px solid #1e2d45', fontSize: 11 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="strategy" name="Strategy" stroke="#00d4ff" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center text-terminal-muted py-16 text-xs">No equity data</div>
              )}
            </div>
          </div>

          {/* Signal decay */}
          {decayData.length > 0 && (
            <div className="panel">
              <div className="panel-header">Signal Decay (30-day Rolling Sharpe)</div>
              <div className="p-4">
                <ResponsiveContainer width="100%" height={140}>
                  <ComposedChart data={decayData}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#1e2d45" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#5a6a7e' }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10, fill: '#5a6a7e' }} />
                    <Tooltip contentStyle={{ background: '#0f1622', border: '1px solid #1e2d45', fontSize: 11 }} />
                    <ReferenceLine y={0} stroke="#1e2d45" />
                    <Area dataKey="sharpe" name="Rolling Sharpe" stroke="#f59e0b" fill="#f59e0b22" strokeWidth={1.5} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'trades' && (
        <div className="panel flex-1">
          <div className="panel-header">Trade Log — {trades?.length || 0} trades</div>
          <div className="overflow-auto flex-1">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Asset</th>
                  <th>Side</th>
                  <th>Size</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>P&L</th>
                  <th>Regime</th>
                  <th>Hold (d)</th>
                </tr>
              </thead>
              <tbody>
                {(trades || []).slice(0, 500).map((t, i) => (
                  <tr key={i}>
                    <td className="text-terminal-muted">{fmtDate(t.date)}</td>
                    <td className="text-terminal-accent font-bold">{t.asset}</td>
                    <td className={t.side === 'LONG' ? 'text-terminal-green font-bold' : t.side === 'SHORT' ? 'text-terminal-red font-bold' : 'text-terminal-muted'}>
                      {t.side}
                    </td>
                    <td>{fmtMoney(t.size)}</td>
                    <td className="font-mono">{fmtNum(t.entry, 4)}</td>
                    <td className="font-mono">{fmtNum(t.exit, 4)}</td>
                    <td className={`font-bold ${signClass(t.pnl)}`}>{fmtMoney(t.pnl)}</td>
                    <td className="text-terminal-muted">{t.regime}</td>
                    <td className="text-terminal-muted">{t.horizon}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {(trades || []).length === 0 && (
              <div className="text-center text-terminal-muted py-8 text-xs">No trades</div>
            )}
          </div>
        </div>
      )}

      {tab === 'regime' && (
        <div className="panel flex-1">
          <div className="panel-header">Performance by Regime</div>
          <div className="p-4">
            {Object.keys(regime_performance || {}).length === 0 ? (
              <div className="text-terminal-muted text-xs text-center py-8">No regime data available</div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th>Total Trades</th>
                    <th>Win Rate</th>
                    <th>Total P&L</th>
                    <th>Avg P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(regime_performance).map(([regime, stats]) => (
                    <tr key={regime}>
                      <td className="font-bold" style={{ color: regime.includes('trend') ? '#22c55e' : regime.includes('crisis') ? '#ef4444' : '#f59e0b' }}>
                        {regime}
                      </td>
                      <td>{stats.total_trades}</td>
                      <td className={signClass(stats.win_rate - 0.5)}>{fmtPct(stats.win_rate)}</td>
                      <td className={signClass(stats.total_pnl)}>{fmtMoney(stats.total_pnl)}</td>
                      <td className={signClass(stats.avg_pnl)}>{fmtMoney(stats.avg_pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

