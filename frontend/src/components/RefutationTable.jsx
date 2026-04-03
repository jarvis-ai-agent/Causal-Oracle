import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { fmtNum } from '../utils/formatting'

export default function RefutationTable() {
  const { validationData, activeRunId } = usePipelineStore()

  if (!validationData) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Validation results not yet available.' : 'No active run.'}
      </div>
    )
  }

  const { validated_parents, dropped_parents, refutation_report, validation_metadata } = validationData

  const allParents = [
    ...Object.entries(validated_parents || {}).flatMap(([target, parents]) =>
      parents.map(p => ({ ...p, target, verdict_type: 'passed' }))
    ),
    ...Object.entries(dropped_parents || {}).flatMap(([target, parents]) =>
      parents.map(p => ({ ...p, target, verdict_type: 'failed' }))
    ),
  ]

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-auto">
      {/* Summary */}
      <div className="panel p-4 flex gap-6">
        <div className="text-center">
          <div className="text-xs text-terminal-muted mb-1">Total Tested</div>
          <div className="text-2xl font-bold text-terminal-text">{validation_metadata?.total_tested || 0}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-terminal-muted mb-1">Passed</div>
          <div className="text-2xl font-bold text-terminal-green">{validation_metadata?.passed || 0}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-terminal-muted mb-1">Failed</div>
          <div className="text-2xl font-bold text-terminal-red">{validation_metadata?.failed || 0}</div>
        </div>
        <div className="ml-auto text-xs text-terminal-muted leading-relaxed max-w-xs">
          <p>Refutation tests validate each discovered causal parent using:</p>
          <p>1. <strong className="text-terminal-text">Placebo treatment</strong> — replace cause with random noise</p>
          <p>2. <strong className="text-terminal-text">Random common cause</strong> — add random confounder</p>
          <p>3. <strong className="text-terminal-text">Data subset</strong> — stability across random subsets</p>
          <p className="mt-1 text-terminal-red">Fail ≥2 of 3 → dropped from feature set</p>
        </div>
      </div>

      {/* Parents table */}
      <div className="panel flex-1">
        <div className="panel-header">Causal Parent Validation Results</div>
        <div className="overflow-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Parent</th>
                <th>Target</th>
                <th>Effect</th>
                <th>Strength</th>
                <th>Lag</th>
                <th>Placebo P</th>
                <th>Rand Cause P</th>
                <th>Subset Stable</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {allParents.map((p, i) => {
                const report = refutation_report?.find?.(r => r.parent === p.name) || {}
                const isPassed = p.verdict_type === 'passed'
                return (
                  <tr key={i}>
                    <td className="font-mono text-terminal-text">{p.name}</td>
                    <td className="text-terminal-muted">{p.target}</td>
                    <td className="font-mono">{fmtNum(report.effect, 6)}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-terminal-border rounded overflow-hidden">
                          <div className="h-full bg-terminal-accent rounded" style={{ width: `${(p.strength || 0) * 100}%` }} />
                        </div>
                        <span className="font-mono text-terminal-muted">{fmtNum(p.strength, 3)}</span>
                      </div>
                    </td>
                    <td className="text-terminal-muted">{p.lag || 0}</td>
                    <td className={report.placebo_p > 0.05 ? 'text-terminal-green' : 'text-terminal-red'}>
                      {fmtNum(report.placebo_p, 3)}
                    </td>
                    <td className={report.random_cause_p > 0.05 ? 'text-terminal-green' : 'text-terminal-red'}>
                      {fmtNum(report.random_cause_p, 3)}
                    </td>
                    <td className={report.subset_stable !== false ? 'text-terminal-green' : 'text-terminal-red'}>
                      {report.subset_stable !== false ? '✓' : '✗'}
                    </td>
                    <td>
                      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${isPassed ? 'bg-terminal-green/20 text-terminal-green border border-terminal-green/30' : 'bg-terminal-red/20 text-terminal-red border border-terminal-red/30'}`}>
                        {isPassed ? 'PASS' : 'FAIL'}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {allParents.length === 0 && (
            <div className="text-center text-terminal-muted py-8 text-xs">No validation data</div>
          )}
        </div>
      </div>
    </div>
  )
}
