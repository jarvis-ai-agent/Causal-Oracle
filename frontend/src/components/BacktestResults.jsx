import React, { useState } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { usePipelineStore } from '../store/pipelineStore'
import { fmtNum, fmtPct, fmtMoney, fmtDate, signClass } from '../utils/formatting'

// ── tiny helpers ──────────────────────────────────────────────────────────────
const pct   = v => v == null ? '—' : `${(v * 100).toFixed(2)}%`
const money = v => v == null ? '—' : `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const num   = (v, d = 2) => v == null ? '—' : Number(v).toFixed(d)
const days  = v => v == null ? '—' : `${Number(v).toFixed(1)} d`
const plain = v => v == null ? '—' : String(v)

function cls(v) {
  if (v == null || v === 0) return 'text-terminal-text'
  return v > 0 ? 'text-terminal-green' : 'text-terminal-red'
}

// ── Section / Row components ───────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <div className="mb-0">
      <div className="bg-terminal-surface/60 border-b border-terminal-border px-4 py-1.5 text-xs font-bold text-terminal-accent uppercase tracking-widest">
        {title}
      </div>
      <table className="w-full text-xs">
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

function Row({ label, value, valueClass = 'text-terminal-text', note = null }) {
  return (
    <tr className="border-b border-terminal-border/30 hover:bg-terminal-surface/40 transition-colors">
      <td className="px-4 py-1.5 text-terminal-muted w-1/2">{label}</td>
      <td className={`px-4 py-1.5 font-mono font-semibold text-right ${valueClass}`}>
        {value}
        {note && <span className="ml-2 text-terminal-muted font-normal text-xs">{note}</span>}
      </td>
    </tr>
  )
}

function DualRow({ label1, value1, cls1, label2, value2, cls2 }) {
  return (
    <tr className="border-b border-terminal-border/30 hover:bg-terminal-surface/40 transition-colors">
      <td className="px-4 py-1.5 text-terminal-muted w-1/4">{label1}</td>
      <td className={`px-4 py-1.5 font-mono font-semibold text-right ${cls1 || 'text-terminal-text'}`}>{value1}</td>
      <td className="px-4 py-1.5 text-terminal-muted w-1/4 border-l border-terminal-border/20">{label2}</td>
      <td className={`px-4 py-1.5 font-mono font-semibold text-right ${cls2 || 'text-terminal-text'}`}>{value2}</td>
    </tr>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function BacktestResults() {
  const { backtestData, activeRunId } = usePipelineStore()
  const [tab, setTab] = useState('metrics')

  if (!backtestData) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Backtest results not yet available.' : 'No active run.'}
      </div>
    )
  }

  const { equity_curve, trades, metrics: m, signal_decay, regime_performance, comparison } = backtestData

  const equityData = Object.entries(equity_curve || {}).map(([date, val]) => ({
    date: date.slice(0, 10),
    strategy: val,
  }))

  const decayData = Object.entries(signal_decay || {}).map(([date, val]) => ({
    date: date.slice(0, 10),
    sharpe: val || 0,
  }))

  const tabs = [
    { id: 'metrics',  label: 'Performance Report' },
    { id: 'equity',   label: 'Equity Curve' },
    { id: 'trades',   label: `Trades (${(trades || []).length})` },
    { id: 'regime',   label: 'By Regime' },
  ]

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-auto">

      {/* Tab bar */}
      <div className="flex gap-1 flex-shrink-0">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-1.5 text-xs rounded uppercase tracking-wider font-bold transition-colors ${
              tab === t.id
                ? 'bg-terminal-accent text-terminal-bg'
                : 'bg-terminal-surface border border-terminal-border text-terminal-muted hover:text-terminal-text'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── METRICS TAB ───────────────────────────────────────────────────── */}
      {tab === 'metrics' && (
        <div className="panel flex-1 overflow-auto">
          <div className="panel-header flex items-center justify-between">
            <span>Performance Report</span>
            <span className="text-xs text-terminal-muted font-normal">All values computed from closed trades · No commission model · No leverage</span>
          </div>

          {/* ── Section 1: Capital & P&L ───────────────────────────────── */}
          <Section title="Capital & P&L">
            <Row label="Initial Capital"    value={money(m?.initial_capital)}   />
            <Row label="Open P&L"           value={money(m?.open_pnl)}          note="closed backtest — always $0" />
            <Row label="Net P&L"            value={money(m?.net_pnl)}           valueClass={cls(m?.net_pnl)} />
            <Row label="Gross Profit"       value={money(m?.gross_profit)}      valueClass="text-terminal-green" />
            <Row label="Gross Loss"         value={money(m?.gross_loss != null ? -m.gross_loss : null)} valueClass="text-terminal-red" />
            <Row label="Profit Factor"      value={num(m?.profit_factor)}       valueClass={cls(m?.profit_factor - 1)} />
            <Row label="Commission Paid"    value={money(m?.commission_paid)}   note="not modelled" />
            <Row label="Expected Payoff"    value={money(m?.expected_payoff)}   valueClass={cls(m?.expected_payoff)} />
          </Section>

          {/* ── Section 2: Benchmark Comparison ───────────────────────── */}
          <Section title="Benchmark Comparison">
            <Row label="Buy & Hold Return"          value={pct(m?.bh_return)}              valueClass={cls(m?.bh_return)} />
            <Row label="Buy & Hold % Gain"          value={pct(m?.bh_pct_gain)}            valueClass={cls(m?.bh_pct_gain)} />
            <Row label="Strategy Outperformance"    value={pct(m?.strategy_outperformance)} valueClass={cls(m?.strategy_outperformance)} />
          </Section>

          {/* ── Section 3: Risk-Adjusted Returns ──────────────────────── */}
          <Section title="Risk-Adjusted Returns">
            <Row label="Sharpe Ratio"   value={num(m?.sharpe)}   valueClass={cls(m?.sharpe)} />
            <Row label="Sortino Ratio"  value={num(m?.sortino)}  valueClass={cls(m?.sortino)} />
          </Section>

          {/* ── Section 4: Trade Statistics ───────────────────────────── */}
          <Section title="Trade Statistics">
            <DualRow
              label1="Total Trades"       value1={plain(m?.total_trades)}
              label2="Total Open Trades"  value2={plain(m?.total_open_trades)}
            />
            <DualRow
              label1="Winning Trades"   value1={plain(m?.winning_trades)}  cls1="text-terminal-green"
              label2="Losing Trades"    value2={plain(m?.losing_trades)}   cls2="text-terminal-red"
            />
            <Row label="Percent Profitable" value={pct(m?.win_rate)} valueClass={cls(m?.win_rate - 0.5)} />
            <Row label="Avg P&L"            value={money(m?.avg_pnl)} valueClass={cls(m?.avg_pnl)} />
            <DualRow
              label1="Avg Winning Trade"  value1={money(m?.avg_winning_trade)}  cls1="text-terminal-green"
              label2="Avg Losing Trade"   value2={money(m?.avg_losing_trade)}   cls2="text-terminal-red"
            />
            <Row label="Ratio Avg Win / Avg Loss" value={num(m?.ratio_avg_win_loss)} valueClass={cls(m?.ratio_avg_win_loss - 1)} />
          </Section>

          {/* ── Section 5: Largest Trades ─────────────────────────────── */}
          <Section title="Largest Trades">
            <DualRow
              label1="Largest Winning Trade"      value1={money(m?.largest_winning_trade)}        cls1="text-terminal-green"
              label2="Largest Winning Trade %"    value2={`${num(m?.largest_winning_trade_pct)}%`}  cls2="text-terminal-green"
            />
            <Row
              label="Largest Winner as % of Gross Profit"
              value={`${num(m?.largest_winner_pct_of_gross_profit)}%`}
              valueClass="text-terminal-green"
            />
            <DualRow
              label1="Largest Losing Trade"       value1={money(m?.largest_losing_trade)}          cls1="text-terminal-red"
              label2="Largest Losing Trade %"     value2={`${num(m?.largest_losing_trade_pct)}%`}   cls2="text-terminal-red"
            />
            <Row
              label="Largest Loser as % of Gross Loss"
              value={`${num(m?.largest_loser_pct_of_gross_loss)}%`}
              valueClass="text-terminal-red"
            />
          </Section>

          {/* ── Section 6: Holding Periods ────────────────────────────── */}
          <Section title="Holding Periods">
            <Row label="Avg # Bars in Trades"         value={days(m?.avg_bars_in_trades)} />
            <DualRow
              label1="Avg # Bars in Winning Trades"  value1={days(m?.avg_bars_in_winning_trades)}
              label2="Avg # Bars in Losing Trades"   value2={days(m?.avg_bars_in_losing_trades)}
            />
          </Section>

          {/* ── Section 7: Returns & Sizing ───────────────────────────── */}
          <Section title="Returns & Capital Efficiency">
            <Row label="Annualized Return (CAGR)"         value={pct(m?.cagr)}                      valueClass={cls(m?.cagr)} />
            <Row label="Return on Initial Capital"        value={pct(m?.return_on_initial_capital)}   valueClass={cls(m?.return_on_initial_capital)} />
            <Row label="Account Size Required"            value={money(m?.account_size_required)} />
            <Row label="Return on Account Size Required"  value={pct(m?.return_on_account_size_required)} valueClass={cls(m?.return_on_account_size_required)} />
            <Row label="Net Profit as % of Largest Loss"  value={`${num(m?.net_profit_pct_of_largest_loss)}%`} valueClass={cls(m?.net_profit_pct_of_largest_loss)} />
          </Section>

          {/* ── Section 8: Margin ─────────────────────────────────────── */}
          <Section title="Margin">
            <DualRow
              label1="Avg Margin Used"  value1={money(m?.avg_margin_used)}
              label2="Max Margin Used"  value2={money(m?.max_margin_used)}
            />
            <DualRow
              label1="Margin Efficiency"  value1={num(m?.margin_efficiency)}  cls1={cls(m?.margin_efficiency)}
              label2="Margin Calls"       value2={plain(m?.margin_calls)}
            />
          </Section>

          {/* ── Section 9: Equity Run-up ──────────────────────────────── */}
          <Section title="Equity Run-Up (Close-to-Close)">
            <DualRow
              label1="Avg Equity Run-Up Duration"  value1={days(m?.avg_equity_runup_duration)}
              label2="Avg Equity Run-Up"            value2={pct(m?.avg_equity_runup)}         cls2="text-terminal-green"
            />
            <DualRow
              label1="Max Equity Run-Up (Close)"    value1={pct(m?.max_equity_runup_close)}    cls1="text-terminal-green"
              label2="Max Equity Run-Up (Intrabar)" value2={pct(m?.max_equity_runup_intrabar)} cls2="text-terminal-green"
            />
            <Row
              label="Max Equity Run-Up as % of Initial Capital (Intrabar)"
              value={`${num(m?.max_equity_runup_pct_initial)}%`}
              valueClass="text-terminal-green"
            />
          </Section>

          {/* ── Section 10: Equity Drawdown ───────────────────────────── */}
          <Section title="Equity Drawdown (Close-to-Close)">
            <DualRow
              label1="Avg Equity Drawdown Duration"  value1={days(m?.avg_equity_drawdown_duration)}
              label2="Avg Equity Drawdown"            value2={pct(m?.avg_equity_drawdown_close)}  cls2="text-terminal-red"
            />
            <DualRow
              label1="Max Equity Drawdown (Close)"    value1={pct(m?.max_equity_drawdown_close)}    cls1="text-terminal-red"
              label2="Max Equity Drawdown (Intrabar)" value2={pct(m?.max_equity_drawdown_intrabar)} cls2="text-terminal-red"
            />
            <Row
              label="Max Equity Drawdown as % of Initial Capital (Intrabar)"
              value={`${num(m?.max_equity_drawdown_pct_initial)}%`}
              valueClass="text-terminal-red"
            />
            <Row
              label="Return of Max Equity Drawdown"
              value={num(m?.return_of_max_drawdown)}
              valueClass={cls(m?.return_of_max_drawdown)}
            />
          </Section>
        </div>
      )}

      {/* ── EQUITY CURVE TAB ──────────────────────────────────────────────── */}
      {tab === 'equity' && (
        <>
          {/* Summary strip */}
          <div className="grid grid-cols-6 gap-2 flex-shrink-0">
            {[
              { label: 'Net P&L',      v: money(m?.net_pnl),           vc: cls(m?.net_pnl) },
              { label: 'CAGR',         v: pct(m?.cagr),                vc: cls(m?.cagr) },
              { label: 'Sharpe',       v: num(m?.sharpe),              vc: cls(m?.sharpe) },
              { label: 'Sortino',      v: num(m?.sortino),             vc: cls(m?.sortino) },
              { label: 'Max DD',       v: pct(m?.max_equity_drawdown_close), vc: 'text-terminal-red' },
              { label: 'Outperform',   v: pct(m?.strategy_outperformance), vc: cls(m?.strategy_outperformance) },
            ].map(({ label, v, vc }) => (
              <div key={label} className="panel p-3 text-center">
                <div className="text-xs text-terminal-muted uppercase tracking-wider mb-1">{label}</div>
                <div className={`text-lg font-bold font-mono ${vc}`}>{v}</div>
              </div>
            ))}
          </div>

          <div className="panel flex-1">
            <div className="panel-header flex justify-between items-center">
              <span>Equity Curve</span>
              {comparison && (
                <div className="flex gap-4 text-xs text-terminal-muted">
                  <span>B&H Sharpe: <span className="text-terminal-text">{num(comparison.sharpe)}</span></span>
                  <span>B&H Return: <span className={cls(comparison.total_return)}>{pct(comparison.total_return)}</span></span>
                </div>
              )}
            </div>
            <div className="p-4">
              {equityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={equityData}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#1e2d45" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#5a6a7e' }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10, fill: '#5a6a7e' }} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip formatter={v => money(v)} contentStyle={{ background: '#0f1622', border: '1px solid #1e2d45', fontSize: 11 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="strategy" name="Strategy" stroke="#00d4ff" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center text-terminal-muted py-16 text-xs">No equity data</div>
              )}
            </div>
          </div>

          {decayData.length > 0 && (
            <div className="panel flex-shrink-0">
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

      {/* ── TRADES TAB ────────────────────────────────────────────────────── */}
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
                    <td className={
                      t.side === 'LONG'  ? 'text-terminal-green font-bold' :
                      t.side === 'SHORT' ? 'text-terminal-red font-bold'   :
                      'text-terminal-muted'
                    }>{t.side}</td>
                    <td>{money(t.size)}</td>
                    <td className="font-mono">{fmtNum(t.entry, 4)}</td>
                    <td className="font-mono">{fmtNum(t.exit, 4)}</td>
                    <td className={`font-bold ${cls(t.pnl)}`}>{money(t.pnl)}</td>
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

      {/* ── REGIME TAB ────────────────────────────────────────────────────── */}
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
                      <td className={cls(stats.win_rate - 0.5)}>{pct(stats.win_rate)}</td>
                      <td className={cls(stats.total_pnl)}>{money(stats.total_pnl)}</td>
                      <td className={cls(stats.avg_pnl)}>{money(stats.avg_pnl)}</td>
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
