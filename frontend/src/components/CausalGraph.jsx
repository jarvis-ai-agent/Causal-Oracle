import React, { useEffect, useRef, useState, useCallback } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

function nodeColor(type) {
  if (type === 'target') return '#fbbf24'
  if (type === 'time_node') return '#f97316'
  return '#00d4ff'
}

function edgeColor(validated) {
  if (validated === true) return '#22c55e'
  if (validated === false) return '#ef4444'
  return '#5a6a7e'
}

export default function CausalGraph() {
  const { graphData, validationData, activeRunId } = usePipelineStore()
  const canvasRef = useRef(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [showNonTarget, setShowNonTarget] = useState(true)
  const [minStrength, setMinStrength] = useState(0)
  const [validatedOnly, setValidatedOnly] = useState(false)
  const nodesRef = useRef([])
  const animRef = useRef(null)
  const positionsRef = useRef({})
  const velocitiesRef = useRef({})

  const data = graphData
  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        {activeRunId ? 'Causal graph not yet available. Run stages 1-4 first.' : 'No active run.'}
      </div>
    )
  }

  const filteredEdges = (data.edges || []).filter(e => {
    if (e.strength < minStrength) return false
    if (validatedOnly && e.validated !== true) return false
    return true
  })

  const visibleNodeIds = new Set()
  if (!showNonTarget) {
    filteredEdges.forEach(e => { visibleNodeIds.add(e.source); visibleNodeIds.add(e.target) })
    const target = (data.nodes || []).find(n => n.type === 'target')
    if (target) visibleNodeIds.add(target.id)
  } else {
    (data.nodes || []).forEach(n => visibleNodeIds.add(n.id))
  }

  const visibleNodes = (data.nodes || []).filter(n => visibleNodeIds.has(n.id))

  // Get validated parent names
  const validatedNames = new Set()
  const droppedNames = new Set()
  if (validationData) {
    Object.values(validationData.validated_parents || {}).forEach(parents => {
      parents.forEach(p => validatedNames.add(p.name))
    })
    Object.values(validationData.dropped_parents || {}).forEach(parents => {
      parents.forEach(p => droppedNames.add(p.name))
    })
  }

  const parentNodes = visibleNodes.filter(n => validatedNames.has(n.id) || droppedNames.has(n.id))
  const targetNode = visibleNodes.find(n => n.type === 'target')

  return (
    <div className="flex h-full gap-4 p-4 overflow-hidden">
      {/* Main graph area */}
      <div className="flex-1 flex flex-col gap-3 min-w-0">
        {/* Controls */}
        <div className="panel p-3 flex items-center gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={showNonTarget} onChange={e => setShowNonTarget(e.target.checked)} className="accent-terminal-accent" />
            Show all edges
          </label>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={validatedOnly} onChange={e => setValidatedOnly(e.target.checked)} className="accent-terminal-accent" />
            Validated only
          </label>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-terminal-muted">Min strength: {minStrength.toFixed(2)}</span>
            <input type="range" min="0" max="1" step="0.05" value={minStrength}
              onChange={e => setMinStrength(Number(e.target.value))}
              className="w-24 accent-terminal-accent"
            />
          </div>
          <div className="flex gap-3 text-xs text-terminal-muted ml-auto">
            <span><span className="text-terminal-gold">●</span> Target</span>
            <span><span className="text-terminal-accent">●</span> Feature</span>
            <span><span className="text-terminal-orange">●</span> Time node</span>
            <span><span className="text-terminal-green">─</span> Validated</span>
            <span><span className="text-terminal-red">─</span> Failed</span>
            <span><span className="text-terminal-muted">─</span> Untested</span>
          </div>
        </div>

        {/* Graph visualization */}
        <div className="panel flex-1 overflow-hidden relative">
          <div className="panel-header">Causal Graph — {visibleNodes.length} nodes, {filteredEdges.length} edges</div>
          <SimpleGraph
            nodes={visibleNodes}
            edges={filteredEdges}
            validatedNames={validatedNames}
            droppedNames={droppedNames}
            onNodeClick={setSelectedNode}
            selectedNode={selectedNode}
          />
        </div>
      </div>

      {/* Sidebar */}
      <div className="w-72 flex flex-col gap-3 overflow-auto">
        {/* Target parents */}
        <div className="panel">
          <div className="panel-header">Causal Parents of {targetNode?.id || 'Target'}</div>
          <div className="p-2">
            {parentNodes.length === 0 ? (
              <div className="text-terminal-muted text-xs p-2">No parents found</div>
            ) : (
              parentNodes.sort((a, b) => {
                const ea = filteredEdges.find(e => e.source === a.id)
                const eb = filteredEdges.find(e => e.source === b.id)
                return (eb?.strength || 0) - (ea?.strength || 0)
              }).map(node => {
                const edge = filteredEdges.find(e => e.source === node.id)
                const isValidated = validatedNames.has(node.id)
                const isDropped = droppedNames.has(node.id)
                return (
                  <div key={node.id} className={`flex items-center justify-between p-2 mb-1 rounded border ${
                    isValidated ? 'border-terminal-green/30 bg-terminal-green/5' :
                    isDropped ? 'border-terminal-red/30 bg-terminal-red/5' :
                    'border-terminal-border'
                  } cursor-pointer hover:opacity-80`}
                    onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                  >
                    <div>
                      <div className="text-xs font-mono text-terminal-text">{node.id}</div>
                      {edge && (
                        <div className="text-xs text-terminal-muted">
                          strength: {edge.strength?.toFixed(3)} | p: {edge.p_value?.toFixed(3)}
                        </div>
                      )}
                    </div>
                    <span className={`text-xs font-bold ${isValidated ? 'text-terminal-green' : isDropped ? 'text-terminal-red' : 'text-terminal-muted'}`}>
                      {isValidated ? 'PASS' : isDropped ? 'FAIL' : '—'}
                    </span>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Nonstationary vars */}
        {data.nodes?.some(n => n.nonstationary) && (
          <div className="panel">
            <div className="panel-header">Nonstationary Variables</div>
            <div className="p-2">
              {data.nodes.filter(n => n.nonstationary).map(n => (
                <div key={n.id} className="text-xs text-terminal-orange py-0.5 px-2 font-mono">{n.id}</div>
              ))}
            </div>
          </div>
        )}

        {/* Selected node info */}
        {selectedNode && (
          <div className="panel">
            <div className="panel-header">Node: {selectedNode}</div>
            <div className="p-3">
              <div className="text-xs space-y-1">
                {filteredEdges
                  .filter(e => e.source === selectedNode || e.target === selectedNode)
                  .map((e, i) => (
                    <div key={i} className="flex gap-2 text-terminal-muted">
                      <span className="text-terminal-text font-mono">{e.source}</span>
                      <span>{e.directed ? '→' : '─'}</span>
                      <span className="text-terminal-text font-mono">{e.target}</span>
                      <span className="ml-auto">{e.strength?.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Simple force-directed graph using canvas
function SimpleGraph({ nodes, edges, validatedNames, droppedNames, onNodeClick, selectedNode }) {
  const canvasRef = useRef(null)
  const stateRef = useRef({ positions: {}, velocities: {}, dragging: null })
  const animRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || nodes.length === 0) return
    const canvas = canvasRef.current
    const state = stateRef.current

    // Initialize positions in a circle
    if (Object.keys(state.positions).length === 0 || Object.keys(state.positions).length !== nodes.length) {
      state.positions = {}
      state.velocities = {}
      const cx = canvas.width / 2
      const cy = canvas.height / 2
      nodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / nodes.length
        const radius = Math.min(canvas.width, canvas.height) * 0.35
        state.positions[n.id] = {
          x: cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 20,
          y: cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 20,
        }
        state.velocities[n.id] = { x: 0, y: 0 }
      })
    }

    const draw = () => {
      const W = canvas.width
      const H = canvas.height
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, W, H)

      // Background
      ctx.fillStyle = '#0f1622'
      ctx.fillRect(0, 0, W, H)

      // Force simulation step
      const k = 80
      const positions = state.positions

      for (const a of nodes) {
        for (const b of nodes) {
          if (a.id === b.id) continue
          const pa = positions[a.id], pb = positions[b.id]
          if (!pa || !pb) continue
          const dx = pa.x - pb.x, dy = pa.y - pb.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const force = (k * k) / dist
          state.velocities[a.id].x += (dx / dist) * force * 0.01
          state.velocities[a.id].y += (dy / dist) * force * 0.01
        }
      }

      for (const e of edges) {
        const pa = positions[e.source], pb = positions[e.target]
        if (!pa || !pb) continue
        const dx = pb.x - pa.x, dy = pb.y - pa.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const desired = 120
        const force = (dist - desired) * 0.008
        const fx = (dx / dist) * force, fy = (dy / dist) * force
        if (state.velocities[e.source]) {
          state.velocities[e.source].x += fx
          state.velocities[e.source].y += fy
        }
        if (state.velocities[e.target]) {
          state.velocities[e.target].x -= fx
          state.velocities[e.target].y -= fy
        }
      }

      for (const n of nodes) {
        if (!positions[n.id] || !state.velocities[n.id]) continue
        if (state.dragging === n.id) continue
        const v = state.velocities[n.id]
        v.x *= 0.85
        v.y *= 0.85
        positions[n.id].x += v.x
        positions[n.id].y += v.y
        positions[n.id].x = Math.max(20, Math.min(W - 20, positions[n.id].x))
        positions[n.id].y = Math.max(20, Math.min(H - 20, positions[n.id].y))
      }

      // Draw edges
      for (const e of edges) {
        const pa = positions[e.source], pb = positions[e.target]
        if (!pa || !pb) continue
        const isValidated = validatedNames.has(e.source)
        const isDropped = droppedNames.has(e.source)
        const color = isValidated ? '#22c55e' : isDropped ? '#ef4444' : '#5a6a7e'
        const alpha = Math.max(0.2, Math.min(1, e.strength || 0.3))
        ctx.beginPath()
        ctx.strokeStyle = color
        ctx.lineWidth = Math.max(0.5, (e.strength || 0.1) * 3)
        ctx.globalAlpha = alpha
        ctx.moveTo(pa.x, pa.y)
        ctx.lineTo(pb.x, pb.y)
        ctx.stroke()

        // Arrowhead
        if (e.directed) {
          ctx.globalAlpha = alpha
          const angle = Math.atan2(pb.y - pa.y, pb.x - pa.x)
          const arrowLen = 8
          const nodeR = 12
          const ax = pb.x - (nodeR + 2) * Math.cos(angle)
          const ay = pb.y - (nodeR + 2) * Math.sin(angle)
          ctx.beginPath()
          ctx.fillStyle = color
          ctx.moveTo(ax, ay)
          ctx.lineTo(ax - arrowLen * Math.cos(angle - 0.4), ay - arrowLen * Math.sin(angle - 0.4))
          ctx.lineTo(ax - arrowLen * Math.cos(angle + 0.4), ay - arrowLen * Math.sin(angle + 0.4))
          ctx.closePath()
          ctx.fill()
        }
        ctx.globalAlpha = 1
      }

      // Draw nodes
      for (const n of nodes) {
        const p = positions[n.id]
        if (!p) continue
        const r = n.type === 'target' ? 16 : n.type === 'time_node' ? 12 : 10
        const color = nodeColor(n.type)
        const isSelected = n.id === selectedNode

        ctx.beginPath()
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
        ctx.fillStyle = color + '22'
        ctx.fill()
        ctx.strokeStyle = isSelected ? '#ffffff' : color
        ctx.lineWidth = isSelected ? 2 : 1.5
        ctx.stroke()

        // Label
        const shortId = n.id.length > 18 ? n.id.slice(0, 16) + '..' : n.id
        ctx.fillStyle = isSelected ? '#ffffff' : '#c9d1e0'
        ctx.font = `${isSelected ? 'bold ' : ''}10px monospace`
        ctx.textAlign = 'center'
        ctx.fillText(shortId, p.x, p.y + r + 12)
      }
    }

    const loop = () => {
      draw()
      animRef.current = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [nodes, edges, validatedNames, droppedNames, selectedNode])

  // Resize
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
      stateRef.current.positions = {}
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [])

  // Click handler
  const handleClick = (e) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    for (const n of nodes) {
      const p = stateRef.current.positions[n.id]
      if (!p) continue
      const r = n.type === 'target' ? 16 : 12
      const dx = mx - p.x, dy = my - p.y
      if (dx * dx + dy * dy <= r * r) {
        onNodeClick(n.id === selectedNode ? null : n.id)
        return
      }
    }
    onNodeClick(null)
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="w-full h-full cursor-crosshair"
      style={{ display: 'block' }}
    />
  )
}
