import React, { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value)
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(handler)
  }, [value, delay])
  return debouncedValue
}

export default function TickerSearch({ onSelect, placeholder = 'Search ticker or company...' }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [highlightedIdx, setHighlightedIdx] = useState(-1)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)
  const debouncedQuery = useDebounce(query, 250)

  // Fetch search results
  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.trim().length < 1) {
      setResults([])
      setOpen(false)
      return
    }
    let cancelled = false
    setLoading(true)
    fetch(`${API_BASE}/api/search?q=${encodeURIComponent(debouncedQuery.trim())}`)
      .then(r => r.json())
      .then(data => {
        if (!cancelled) {
          setResults(Array.isArray(data) ? data : [])
          setOpen(Array.isArray(data) && data.length > 0)
          setHighlightedIdx(-1)
        }
      })
      .catch(() => {
        if (!cancelled) setResults([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [debouncedQuery])

  const handleSelect = useCallback((symbol) => {
    if (!symbol) return
    onSelect(symbol.trim().toUpperCase())
    setQuery('')
    setResults([])
    setOpen(false)
    setHighlightedIdx(-1)
  }, [onSelect])

  const handleKeyDown = (e) => {
    if (!open || results.length === 0) {
      if (e.key === 'Enter' && query.trim()) {
        handleSelect(query.trim().toUpperCase())
      }
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightedIdx(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightedIdx(i => Math.max(i - 1, -1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (highlightedIdx >= 0) {
        handleSelect(results[highlightedIdx].symbol)
      } else if (query.trim()) {
        handleSelect(query.trim().toUpperCase())
      }
    } else if (e.key === 'Escape') {
      setOpen(false)
      setHighlightedIdx(-1)
    }
  }

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (
        dropdownRef.current && !dropdownRef.current.contains(e.target) &&
        inputRef.current && !inputRef.current.contains(e.target)
      ) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const quoteTypeLabel = (type) => {
    const map = {
      EQUITY: 'Stock',
      ETF: 'ETF',
      MUTUALFUND: 'Fund',
      INDEX: 'Index',
      FUTURE: 'Future',
      CURRENCY: 'FX',
      CRYPTOCURRENCY: 'Crypto',
    }
    return map[type] || type || 'Stock'
  }

  return (
    <div className="relative w-full">
      {/* Input */}
      <div className="flex items-center gap-1 bg-terminal-surface border border-terminal-border rounded px-3 py-1.5 focus-within:border-terminal-accent transition-colors">
        {/* Search icon */}
        <svg className="w-3.5 h-3.5 text-terminal-muted shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => {
            setQuery(e.target.value.toUpperCase())
            if (e.target.value.length > 0) setOpen(true)
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder={placeholder}
          className="flex-1 bg-transparent border-none outline-none text-terminal-text placeholder:text-terminal-muted text-xs font-mono min-w-0"
          autoComplete="off"
          spellCheck={false}
        />
        {loading && (
          <span className="text-terminal-muted text-xs shrink-0 animate-pulse">…</span>
        )}
      </div>

      {/* Dropdown */}
      {open && results.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute top-full left-0 right-0 z-50 mt-1 bg-terminal-surface border border-terminal-border rounded shadow-2xl overflow-hidden"
        >
          {results.map((r, i) => (
            <button
              key={r.symbol}
              onMouseDown={e => { e.preventDefault(); handleSelect(r.symbol) }}
              onMouseEnter={() => setHighlightedIdx(i)}
              className={`w-full flex items-center gap-3 px-3 py-2 text-left transition-colors cursor-pointer ${
                i === highlightedIdx
                  ? 'bg-terminal-accent/15 border-l-2 border-terminal-accent'
                  : 'hover:bg-terminal-panel border-l-2 border-transparent'
              }`}
            >
              <span className="font-mono text-xs font-bold text-terminal-accent w-16 shrink-0 truncate">
                {r.symbol}
              </span>
              <span className="text-xs text-terminal-text truncate flex-1">
                {r.name}
              </span>
              <span className="text-xs text-terminal-muted shrink-0 ml-auto">
                {quoteTypeLabel(r.type)}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* No results message */}
      {open && !loading && results.length === 0 && query.length >= 2 && (
        <div
          ref={dropdownRef}
          className="absolute top-full left-0 right-0 z-50 mt-1 bg-terminal-surface border border-terminal-border rounded shadow-2xl px-3 py-2"
        >
          <span className="text-xs text-terminal-muted">No results for "{query}" — press Enter to add anyway</span>
        </div>
      )}
    </div>
  )
}
