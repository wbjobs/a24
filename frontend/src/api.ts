import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000,
  headers: { 'Content-Type': 'application/json' },
})

export interface TrainingHistoryPoint {
  epoch: number
  loss: number
  loss_ic?: number
  loss_bc?: number
  loss_pde?: number
}

export interface PDEType {
  id: string
  name: string
  latex: string
  description: string
  default_params: Record<string, number>
}

export interface GridData {
  x: number[]
  t: number[]
  u: number[][]
  u_std?: number[][]
  u_uncertainty?: number[][]
  has_uncertainty?: boolean
}

export interface SolveRequest {
  pde_type: string
  equation_latex: string
  ic_expression: string
  bc_left_expression: string
  bc_right_expression: string
  domain: { x_min: number; x_max: number; t_min: number; t_max: number }
  params: Record<string, number>
  layers: number[]
  n_collocation: number
  n_initial: number
  n_boundary: number
  epochs: number
  learning_rate: number
  name: string
  use_fourier?: boolean
  use_adaptive_activation?: boolean
  use_hard_constraint?: boolean
  use_mc_dropout?: boolean
  fourier_bands?: number[]
  fourier_freqs?: number
  dropout_rate?: number
  mc_samples?: number
  log_interval?: number
}

export interface SolveStartResponse {
  task_id: string
  status: string
  message: string
}

export interface SolveCompleteResponse {
  id: number
  status: string
  final_loss: number
  training_history: TrainingHistoryPoint[]
  grid_data: GridData
  final_epochs: number
}

export interface AdvancedOptions {
  use_fourier: boolean
  use_adaptive_activation: boolean
  use_hard_constraint: boolean
  use_mc_dropout: boolean
  fourier_bands: number[]
  fourier_freqs: number
  dropout_rate: number
  mc_samples: number
}

export interface HistoryRecord {
  id: number
  name: string
  pde_type: string
  equation_latex: string
  ic_expression: string
  bc_left_expression: string
  bc_right_expression: string
  domain_x_min: number
  domain_x_max: number
  domain_t_min: number
  domain_t_max: number
  params: Record<string, number>
  layers: number[]
  n_collocation: number
  n_initial: number
  n_boundary: number
  epochs: number
  learning_rate: number
  final_loss: number | null
  training_history: TrainingHistoryPoint[]
  created_at: string | null
}

export const fetchPDETypes = () => api.get<{ types: PDEType[] }>('/pde-types')

export const fetchAdvancedOptions = () => api.get<AdvancedOptions>('/advanced-options')

export const solvePDE = (data: SolveRequest) =>
  api.post<SolveStartResponse>('/solve', data)

export const predictPDE = (
  id: number,
  nx = 100,
  nt = 100,
  mc_samples = 30,
  compute_uncertainty = true
) =>
  api.post<GridData>(`/predict/${id}`, {
    nx, nt, mc_samples, compute_uncertainty
  })

export const fetchHistory = () => api.get<HistoryRecord[]>('/history')

export const fetchRecord = (id: number) => api.get<HistoryRecord>(`/history/${id}`)

export const deleteRecord = (id: number) => api.delete(`/history/${id}`)

export const getExportUrl = (id: number, format: 'csv' | 'vtk' | 'vtk3d') =>
  `${API_BASE}/export/${id}/${format}`

export interface SSEProgressEvent {
  epoch: number
  loss: number
  loss_ic?: number
  loss_bc?: number
  loss_pde?: number
}

export type SSEHandler = {
  onProgress?: (e: SSEProgressEvent) => void
  onComplete?: (r: SolveCompleteResponse) => void
  onError?: (err: string) => void
  onConnected?: (clientId: string) => void
}

export function createSSEConnection(handler: SSEHandler): EventSource {
  const es = new EventSource(`${API_BASE}/stream`, { withCredentials: false })

  es.addEventListener('connected', (ev: any) => {
    try {
      const data = JSON.parse(ev.data)
      if (handler.onConnected) handler.onConnected(data.client_id)
    } catch {}
  })

  es.addEventListener('progress', (ev: any) => {
    try {
      const data = JSON.parse(ev.data) as SSEProgressEvent
      if (handler.onProgress) handler.onProgress(data)
    } catch {}
  })

  es.addEventListener('complete', (ev: any) => {
    try {
      const data = JSON.parse(ev.data) as SolveCompleteResponse
      if (handler.onComplete) handler.onComplete(data)
      es.close()
    } catch {}
  })

  es.addEventListener('error', (ev: any) => {
    try {
      const parsed = JSON.parse(ev.data)
      if (handler.onError) handler.onError(parsed.error || 'Unknown error')
    } catch {
      if (ev.readyState === EventSource.CLOSED) {
        es.close()
      }
    }
  })

  return es
}

export default api
