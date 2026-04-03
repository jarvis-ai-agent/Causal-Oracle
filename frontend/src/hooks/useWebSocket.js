import { useEffect, useRef, useCallback } from 'react'

export function useWebSocket(runId, onMessage) {
  const wsRef = useRef(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (!runId) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    const wsUrl = `ws://${window.location.host}/ws/pipeline/${runId}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log(`WS connected for run ${runId}`)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessageRef.current(data)
      } catch (e) {
        console.error('WS parse error:', e)
      }
    }

    ws.onerror = (err) => {
      console.error('WS error:', err)
    }

    ws.onclose = () => {
      console.log(`WS closed for run ${runId}`)
    }
  }, [runId])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { connect }
}
