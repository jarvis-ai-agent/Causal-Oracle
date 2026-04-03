import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export function useApi() {
  const startRun = async (config) => {
    const res = await api.post('/pipeline/run', config)
    return res.data
  }

  const listRuns = async () => {
    const res = await api.get('/pipeline/runs')
    return res.data
  }

  const getRun = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}`)
    return res.data
  }

  const deleteRun = async (runId) => {
    const res = await api.delete(`/pipeline/runs/${runId}`)
    return res.data
  }

  const getGraph = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/graph`)
    return res.data
  }

  const getForecast = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/forecast`)
    return res.data
  }

  const getBacktest = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/backtest`)
    return res.data
  }

  const getRegimes = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/regimes`)
    return res.data
  }

  const getValidation = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/validation`)
    return res.data
  }

  const getLogs = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/logs`)
    return res.data
  }

  const exportRun = async (runId) => {
    const res = await api.get(`/pipeline/runs/${runId}/export`)
    return res.data
  }

  const health = async () => {
    const res = await api.get('/health')
    return res.data
  }

  return {
    startRun, listRuns, getRun, deleteRun,
    getGraph, getForecast, getBacktest, getRegimes, getValidation, getLogs, exportRun, health,
  }
}
