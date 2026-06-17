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
}

export interface GridData {
  x: number[]
  t: number[]
  u: number[][]
}

export interface SolveResponse {
  id: number
  status: string
  final_loss: number
  training_history: TrainingHistoryPoint[]
  grid_data: GridData
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

export const solvePDE = (data: SolveRequest) => api.post<SolveResponse>('/solve', data)

export const predictPDE = (id: number, nx = 100, nt = 100) =>
  api.post<GridData>(`/predict/${id}`, { nx, nt })

export const fetchHistory = () => api.get<HistoryRecord[]>('/history')

export const fetchRecord = (id: number) => api.get<HistoryRecord>(`/history/${id}`)

export const deleteRecord = (id: number) => api.delete(`/history/${id}`)

export const getExportUrl = (id: number, format: 'csv' | 'vtk' | 'vtk3d') =>
  `${API_BASE}/export/${id}/${format}`

export default api
