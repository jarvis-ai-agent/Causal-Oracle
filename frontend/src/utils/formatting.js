export function fmtNum(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—'
  return Number(val).toFixed(decimals)
}

export function fmtPct(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—'
  return `${(Number(val) * 100).toFixed(decimals)}%`
}

export function fmtMoney(val) {
  if (val == null || isNaN(val)) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0
  }).format(val)
}

export function fmtDate(isoStr) {
  if (!isoStr) return '—'
  return isoStr.slice(0, 10)
}

export function signClass(val) {
  if (val == null || isNaN(val)) return 'num-neutral'
  if (val > 0) return 'num-pos'
  if (val < 0) return 'num-neg'
  return 'num-neutral'
}

export function regimeColor(regime) {
  if (!regime) return '#5a6a7e'
  const r = regime.toLowerCase()
  if (r.includes('trend')) return '#22c55e'
  if (r.includes('crisis')) return '#ef4444'
  if (r.includes('mean')) return '#f59e0b'
  return '#00d4ff'
}
