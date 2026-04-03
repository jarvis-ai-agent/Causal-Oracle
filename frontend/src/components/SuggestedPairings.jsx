import React, { useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

const PAIRINGS = [
  {
    id: 'aapl-spy-gld',
    label: 'AAPL · SPY · GLD',
    tickers: ['AAPL', 'SPY', 'GLD'],
    target: 'AAPL_ret',
    badge: '★ Reference',
    badgeColor: 'text-terminal-gold border-terminal-gold/40 bg-terminal-gold/5',
    stats: { wr: '60.6%', sharpe: '2.20', horizon: 3 },
    description: 'The validated reference pairing. AAPL as target, SPY captures broad market causal momentum, GLD adds a risk-off cross-asset signal invisible to SPY alone.',
    reasoning: [
      { ticker: 'AAPL', role: 'Target', color: 'text-terminal-accent' },
      { ticker: 'SPY', role: 'Market proxy — SPY_ret_t-1 is the strongest predictive parent', color: 'text-terminal-gold' },
      { ticker: 'GLD', role: 'Risk-off signal — capital flows from equities to gold precede weakness', color: 'text-terminal-green' },
    ],
    settings: { max_lag: 7, alpha: 0.10, indep_test: 'rcot', horizon: 3, include_macro: true },
  },
  {
    id: 'msft-spy-tlt',
    label: 'MSFT · SPY · TLT',
    tickers: ['MSFT', 'SPY', 'TLT'],
    target: 'MSFT_ret',
    badge: 'Rate-sensitive',
    badgeColor: 'text-terminal-accent border-terminal-accent/40 bg-terminal-accent/5',
    stats: { wr: null, sharpe: null, horizon: 3 },
    description: 'Tech stock with bond duration sensitivity. TLT (20yr Treasury ETF) captures interest rate moves that drive MSFT valuation — a causal channel SPY alone misses.',
    reasoning: [
      { ticker: 'MSFT', role: 'Target', color: 'text-terminal-accent' },
      { ticker: 'SPY', role: 'Broad market causal context', color: 'text-terminal-gold' },
      { ticker: 'TLT', role: 'Rate sensitivity — rising rates compress high-multiple tech', color: 'text-terminal-green' },
    ],
    settings: { max_lag: 7, alpha: 0.10, indep_test: 'rcot', horizon: 3, include_macro: true },
  },
  {
    id: 'spy-tlt-gld',
    label: 'SPY · TLT · GLD',
    tickers: ['SPY', 'TLT', 'GLD'],
    target: 'SPY_ret',
    badge: 'Macro basket',
    badgeColor: 'text-terminal-green border-terminal-green/40 bg-terminal-green/5',
    stats: { wr: null, sharpe: null, horizon: 5 },
    description: 'Classic risk-on / risk-off macro basket. Captures the rotation between equities, bonds and gold that characterises market regime transitions. Good for broad market timing.',
    reasoning: [
      { ticker: 'SPY', role: 'Target — broad US equity market', color: 'text-terminal-accent' },
      { ticker: 'TLT', role: 'Flight-to-safety signal — bonds rally when equities fall', color: 'text-terminal-gold' },
      { ticker: 'GLD', role: 'Inflation/crisis hedge — orthogonal to bonds in stagflation', color: 'text-terminal-green' },
    ],
    settings: { max_lag: 5, alpha: 0.10, indep_test: 'rcot', horizon: 5, include_macro: true },
  },
  {
    id: 'nvda-smh-spy',
    label: 'NVDA · SMH · SPY',
    tickers: ['NVDA', 'SMH', 'SPY'],
    target: 'NVDA_ret',
    badge: 'Sector lead-lag',
    badgeColor: 'text-terminal-muted border-terminal-border bg-terminal-surface',
    stats: { wr: null, sharpe: null, horizon: 3 },
    description: 'Semiconductor name with sector ETF as causal context. SMH often leads individual semis by 1–2 days — the sector moves first, then specific names follow. SPY provides macro floor.',
    reasoning: [
      { ticker: 'NVDA', role: 'Target — individual semiconductor name', color: 'text-terminal-accent' },
      { ticker: 'SMH', role: 'Sector lead — SMH_ret_t-1 often precedes NVDA moves', color: 'text-terminal-gold' },
      { ticker: 'SPY', role: 'Macro context — separates sector from broad market moves', color: 'text-terminal-green' },
    ],
    settings: { max_lag: 7, alpha: 0.10, indep_test: 'rcot', horizon: 3, include_macro: false },
  },
]

export default function SuggestedPairings() {
  const { setConfig } = usePipelineStore()
  const [expanded, setExpanded] = useState(null)
  const [applied, setApplied] = useState(null)

  const handleApply = (pairing) => {
    setConfig({
      tickers: pairing.tickers,
      target: pairing.target,
      ...pairing.settings,
    })
    setApplied(pairing.id)
    setTimeout(() => setApplied(null), 2000)
  }

  return (
    <div className="panel">
      <div className="panel-header flex items-center justify-between">
        <span>Suggested Pairings</span>
        <span className="text-xs text-terminal-muted font-normal">
          Diversity of asset class &gt; number of tickers
        </span>
      </div>
      <div className="p-3 grid grid-cols-2 gap-2">
        {PAIRINGS.map(p => (
          <div
            key={p.id}
            className={`border rounded p-3 cursor-pointer transition-colors ${
              expanded === p.id
                ? 'border-terminal-accent/60 bg-terminal-accent/5'
                : 'border-terminal-border hover:border-terminal-accent/40'
            }`}
            onClick={() => setExpanded(expanded === p.id ? null : p.id)}
          >
            {/* Header row */}
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <span className="font-mono text-terminal-accent text-xs font-bold">{p.label}</span>
              <span className={`text-xs px-1.5 py-0.5 rounded border whitespace-nowrap ${p.badgeColor}`}>
                {p.badge}
              </span>
            </div>

            {/* Stats if available */}
            {p.stats.wr && (
              <div className="flex gap-3 mb-1.5">
                <span className="text-xs text-terminal-green">WR {p.stats.wr}</span>
                <span className="text-xs text-terminal-gold">Sharpe {p.stats.sharpe}</span>
                <span className="text-xs text-terminal-muted">H={p.stats.horizon}d</span>
              </div>
            )}

            {/* Description */}
            <p className="text-xs text-terminal-muted leading-relaxed">{p.description}</p>

            {/* Expanded: reasoning + apply */}
            {expanded === p.id && (
              <div className="mt-3 pt-3 border-t border-terminal-border/50">
                <div className="text-xs text-terminal-muted font-bold uppercase tracking-wider mb-2">Why this pairing:</div>
                <div className="space-y-1.5 mb-3">
                  {p.reasoning.map(r => (
                    <div key={r.ticker} className="flex gap-2 text-xs">
                      <span className={`font-mono font-bold w-12 flex-none ${r.color}`}>{r.ticker}</span>
                      <span className="text-terminal-muted">{r.role}</span>
                    </div>
                  ))}
                </div>
                <div className="text-xs text-terminal-muted mb-2">
                  Settings: lag={p.settings.max_lag}, α={p.settings.alpha}, {p.settings.indep_test.toUpperCase()}, H={p.settings.horizon}d
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleApply(p) }}
                  className={`w-full py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-all ${
                    applied === p.id
                      ? 'bg-terminal-green text-terminal-bg'
                      : 'bg-terminal-accent text-terminal-bg hover:brightness-110'
                  }`}
                >
                  {applied === p.id ? '✓ Applied' : '▶ Apply to Config'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Principle footer */}
      <div className="px-3 pb-3">
        <div className="border border-terminal-border/50 rounded p-2.5 text-xs text-terminal-muted leading-relaxed">
          <span className="text-terminal-text font-bold">The rule:</span> each ticker should represent a distinct causal channel —
          equity target, market proxy, and a cross-asset signal (bonds, gold, sector ETF).
          Co-linear assets (e.g. AAPL + MSFT + GOOGL) share one channel and produce noisy contemporaneous edges, not predictive lags.
        </div>
      </div>
    </div>
  )
}
