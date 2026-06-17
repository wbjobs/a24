import React, { useState, useEffect, useCallback, useRef } from 'react'
import FormulaEditor from './components/FormulaEditor'
import Visualization3D from './components/Visualization3D'
import HistoryPanel from './components/HistoryPanel'
import ExportPanel from './components/ExportPanel'
import LossChart from './components/LossChart'
import {
  fetchPDETypes,
  solvePDE,
  fetchHistory,
  predictPDE,
  PDEType,
  SolveRequest,
  GridData,
  HistoryRecord,
  SolveResponse,
  TrainingHistoryPoint,
} from './api'

const PDE_PRESETS: Record<string, { equation: string; ic: string; bcLeft: string; bcRight: string }> = {
  heat: {
    equation: '\\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2}',
    ic: 'sin(pi*x)',
    bcLeft: '0',
    bcRight: '0',
  },
  wave: {
    equation: '\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\frac{\\partial^2 u}{\\partial x^2}',
    ic: 'sin(pi*x)',
    bcLeft: '0',
    bcRight: '0',
  },
  elliptic: {
    equation: '\\alpha \\frac{\\partial^2 u}{\\partial x^2} = 0',
    ic: '0',
    bcLeft: '0',
    bcRight: '1',
  },
  general: {
    equation: '\\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2}',
    ic: 'sin(pi*x)',
    bcLeft: '0',
    bcRight: '0',
  },
}

function App() {
  const [pdeTypes, setPdeTypes] = useState<PDEType[]>([])
  const [selectedType, setSelectedType] = useState<string>('heat')
  const [equationLatex, setEquationLatex] = useState(PDE_PRESETS.heat.equation)
  const [icExpression, setIcExpression] = useState(PDE_PRESETS.heat.ic)
  const [bcLeftExpression, setBcLeftExpression] = useState(PDE_PRESETS.heat.bcLeft)
  const [bcRightExpression, setBcRightExpression] = useState(PDE_PRESETS.heat.bcRight)
  const [domain, setDomain] = useState({ x_min: 0, x_max: 1, t_min: 0, t_max: 1 })
  const [params, setParams] = useState<Record<string, number>>({ alpha: 0.01 })
  const [layers, setLayers] = useState<number[]>([2, 64, 64, 64, 1])
  const [nCollocation, setNCollocation] = useState(10000)
  const [nInitial, setNInitial] = useState(200)
  const [nBoundary, setNBoundary] = useState(200)
  const [epochs, setEpochs] = useState(5000)
  const [learningRate, setLearningRate] = useState(0.001)
  const [solveName, setSolveName] = useState('')
  const [isSolving, setIsSolving] = useState(false)
  const [solveStatus, setSolveStatus] = useState<string>('')
  const [gridData, setGridData] = useState<GridData | null>(null)
  const [history, setHistory] = useState<HistoryRecord[]>([])
  const [selectedRecordId, setSelectedRecordId] = useState<number | null>(null)
  const [currentTimeIndex, setCurrentTimeIndex] = useState(999)
  const [isAnimating, setIsAnimating] = useState(false)
  const [colorScheme, setColorScheme] = useState<string>('plasma')
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryPoint[]>([])
  const [currentRecordId, setCurrentRecordId] = useState<number | null>(null)
  const animRef = useRef<number | null>(null)

  useEffect(() => {
    fetchPDETypes()
      .then((res) => setPdeTypes(res.data.types))
      .catch(console.error)
    loadHistory()
  }, [])

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetchHistory()
      setHistory(res.data)
    } catch (err) {
      console.error('Failed to load history:', err)
    }
  }, [])

  useEffect(() => {
    if (!isAnimating || !gridData) return
    const nt = gridData.t.length
    let frame = currentTimeIndex
    const animate = () => {
      frame = (frame + 1) % nt
      setCurrentTimeIndex(frame)
      animRef.current = requestAnimationFrame(animate)
    }
    animRef.current = requestAnimationFrame(animate)
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [isAnimating, gridData])

  const handleTypeSelect = (type: string) => {
    setSelectedType(type)
    const preset = PDE_PRESETS[type]
    if (preset) {
      setEquationLatex(preset.equation)
      setIcExpression(preset.ic)
      setBcLeftExpression(preset.bcLeft)
      setBcRightExpression(preset.bcRight)
    }
    const pdeType = pdeTypes.find((t) => t.id === type)
    if (pdeType) {
      setParams(pdeType.default_params)
    }
  }

  const handleSolve = async () => {
    setIsSolving(true)
    setSolveStatus('solving')
    try {
      const request: SolveRequest = {
        pde_type: selectedType,
        equation_latex: equationLatex,
        ic_expression: icExpression,
        bc_left_expression: bcLeftExpression,
        bc_right_expression: bcRightExpression,
        domain,
        params,
        layers,
        n_collocation: nCollocation,
        n_initial: nInitial,
        n_boundary: nBoundary,
        epochs,
        learning_rate: learningRate,
        name: solveName || `${selectedType}_${new Date().toLocaleTimeString('zh-CN')}`,
      }
      const res = await solvePDE(request)
      const data = res.data
      setGridData(data.grid_data)
      setCurrentTimeIndex(data.grid_data.t.length - 1)
      setTrainingHistory(data.training_history)
      setCurrentRecordId(data.id)
      setSelectedRecordId(data.id)
      setSolveStatus('completed')
      await loadHistory()
    } catch (err: any) {
      console.error('Solve failed:', err)
      setSolveStatus('error')
      alert(`求解失败: ${err.response?.data?.error || err.message}`)
    } finally {
      setIsSolving(false)
    }
  }

  const handleSelectRecord = async (record: HistoryRecord) => {
    setSelectedRecordId(record.id)
    setCurrentRecordId(record.id)
    try {
      const res = await predictPDE(record.id, 100, 100)
      setGridData(res.data)
      setCurrentTimeIndex(res.data.t.length - 1)
      setTrainingHistory(record.training_history || [])
      setSelectedType(record.pde_type)
      setEquationLatex(record.equation_latex)
      setIcExpression(record.ic_expression)
      setBcLeftExpression(record.bc_left_expression)
      setBcRightExpression(record.bc_right_expression)
      setDomain({
        x_min: record.domain_x_min,
        x_max: record.domain_x_max,
        t_min: record.domain_t_min,
        t_max: record.domain_t_max,
      })
      setParams(record.params)
    } catch (err) {
      console.error('Failed to load record:', err)
    }
  }

  const handleDeleteRecord = (id: number) => {
    setHistory((prev) => prev.filter((r) => r.id !== id))
    if (selectedRecordId === id) {
      setSelectedRecordId(null)
    }
    if (currentRecordId === id) {
      setCurrentRecordId(null)
    }
  }

  const layersStr = layers.join(', ')

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>PINN Solver</h1>
          <p>物理信息神经网络 · 偏微分方程数值求解器</p>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
          <div className="section">
            <div className="section-title">方程类型</div>
            <div className="pde-type-selector">
              {[
                { id: 'heat', name: '热传导', latex: 'u_t = αu_{xx}' },
                { id: 'wave', name: '波动方程', latex: 'u_{tt} = c²u_{xx}' },
                { id: 'elliptic', name: '椭圆方程', latex: '∇²u = f' },
                { id: 'general', name: '通用PDE', latex: 'F(u)=0' },
              ].map((type) => (
                <div
                  key={type.id}
                  className={`pde-type-btn ${selectedType === type.id ? 'active' : ''}`}
                  onClick={() => handleTypeSelect(type.id)}
                >
                  <span className="type-name">{type.name}</span>
                  <span className="type-latex">{type.latex}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="section">
            <div className="section-title">PDE方程 (LaTeX)</div>
            <FormulaEditor value={equationLatex} onChange={setEquationLatex} />
          </div>

          <div className="section">
            <div className="section-title">初始条件</div>
            <div className="form-group">
              <label>u(x, 0) =</label>
              <input
                type="text"
                value={icExpression}
                onChange={(e) => setIcExpression(e.target.value)}
                placeholder="sin(pi*x)"
              />
            </div>
          </div>

          <div className="section">
            <div className="section-title">边界条件</div>
            <div className="form-group">
              <label>u(0, t) =</label>
              <input
                type="text"
                value={bcLeftExpression}
                onChange={(e) => setBcLeftExpression(e.target.value)}
                placeholder="0"
              />
            </div>
            <div className="form-group">
              <label>u(L, t) =</label>
              <input
                type="text"
                value={bcRightExpression}
                onChange={(e) => setBcRightExpression(e.target.value)}
                placeholder="0"
              />
            </div>
          </div>

          <div className="section">
            <div className="section-title">求解域</div>
            <div className="domain-grid">
              <div className="form-group">
                <label>x 最小值</label>
                <input
                  type="number"
                  value={domain.x_min}
                  onChange={(e) => setDomain({ ...domain, x_min: parseFloat(e.target.value) || 0 })}
                  step="0.1"
                />
              </div>
              <div className="form-group">
                <label>x 最大值</label>
                <input
                  type="number"
                  value={domain.x_max}
                  onChange={(e) => setDomain({ ...domain, x_max: parseFloat(e.target.value) || 1 })}
                  step="0.1"
                />
              </div>
              <div className="form-group">
                <label>t 最小值</label>
                <input
                  type="number"
                  value={domain.t_min}
                  onChange={(e) => setDomain({ ...domain, t_min: parseFloat(e.target.value) || 0 })}
                  step="0.1"
                />
              </div>
              <div className="form-group">
                <label>t 最大值</label>
                <input
                  type="number"
                  value={domain.t_max}
                  onChange={(e) => setDomain({ ...domain, t_max: parseFloat(e.target.value) || 1 })}
                  step="0.1"
                />
              </div>
            </div>
          </div>

          <div className="section">
            <div className="section-title">物理参数</div>
            {Object.entries(params).map(([key, value]) => (
              <div className="param-row" key={key}>
                <label>{key}</label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) || 0 })}
                  step="0.001"
                />
              </div>
            ))}
          </div>

          <div className="section">
            <div className="section-title">网络结构</div>
            <div className="form-group">
              <label>隐藏层节点</label>
              <input
                type="text"
                value={layersStr}
                onChange={(e) => {
                  const parsed = e.target.value.split(',').map((s) => parseInt(s.trim())).filter((n) => !isNaN(n))
                  if (parsed.length >= 2) setLayers(parsed)
                }}
              />
            </div>
          </div>

          <div className="section">
            <div className="section-title">训练参数</div>
            <div className="training-params">
              <div className="form-group">
                <label>配置点数</label>
                <input
                  type="number"
                  value={nCollocation}
                  onChange={(e) => setNCollocation(parseInt(e.target.value) || 10000)}
                />
              </div>
              <div className="form-group">
                <label>初始点数</label>
                <input
                  type="number"
                  value={nInitial}
                  onChange={(e) => setNInitial(parseInt(e.target.value) || 200)}
                />
              </div>
              <div className="form-group">
                <label>边界点数</label>
                <input
                  type="number"
                  value={nBoundary}
                  onChange={(e) => setNBoundary(parseInt(e.target.value) || 200)}
                />
              </div>
              <div className="form-group">
                <label>训练轮数</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 5000)}
                />
              </div>
              <div className="form-group">
                <label>学习率</label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                  step="0.0001"
                />
              </div>
              <div className="form-group">
                <label>求解名称</label>
                <input
                  type="text"
                  value={solveName}
                  onChange={(e) => setSolveName(e.target.value)}
                  placeholder="可选名称"
                />
              </div>
            </div>
          </div>

          <button
            className="btn btn-primary btn-block"
            onClick={handleSolve}
            disabled={isSolving}
            style={{ marginTop: '8px', padding: '12px' }}
          >
            {isSolving ? (
              <>
                <span className="loading-spinner" />
                正在求解...
              </>
            ) : (
              <>🚀 开始求解</>
            )}
          </button>

          {solveStatus && (
            <div style={{ marginTop: '8px' }}>
              <span className={`status-badge ${solveStatus}`}>
                {solveStatus === 'solving' && '求解中'}
                {solveStatus === 'completed' && '已完成'}
                {solveStatus === 'error' && '出错'}
              </span>
            </div>
          )}

          {trainingHistory.length > 0 && (
            <div className="training-progress">
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>训练损失曲线</div>
              <LossChart history={trainingHistory} />
            </div>
          )}

          <HistoryPanel
            records={history}
            selectedId={selectedRecordId}
            onSelect={handleSelectRecord}
            onDelete={handleDeleteRecord}
            onRefresh={loadHistory}
          />

          <ExportPanel recordId={currentRecordId} disabled={isSolving} />
        </div>
      </div>

      <div className="main-content">
        <div className="top-bar">
          <span className="top-bar-title">数值解可视化</span>
          <div className="toolbar">
            <select
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
              style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                color: 'var(--text-primary)',
                padding: '4px 8px',
                fontSize: '12px',
              }}
            >
              <option value="plasma">Plasma</option>
              <option value="viridis">Viridis</option>
              <option value="coolwarm">Coolwarm</option>
            </select>
            <div className="divider" />
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setIsAnimating(!isAnimating)}
            >
              {isAnimating ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => {
                if (gridData) setCurrentTimeIndex(gridData.t.length - 1)
              }}
            >
              ⏭ 末帧
            </button>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setCurrentTimeIndex(0)}
            >
              ⏮ 首帧
            </button>
          </div>
        </div>

        <div className="content-area">
          <Visualization3D
            gridData={gridData}
            isAnimating={isAnimating}
            currentTimeIndex={currentTimeIndex}
            colorScheme={colorScheme}
          />

          {gridData && (
            <div className="time-slider-container">
              <label>时间步</label>
              <input
                type="range"
                min={0}
                max={gridData.t.length - 1}
                value={currentTimeIndex}
                onChange={(e) => setCurrentTimeIndex(parseInt(e.target.value))}
              />
              <span className="time-value">t = {gridData.t[currentTimeIndex]?.toFixed(3)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
