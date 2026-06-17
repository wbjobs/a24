import React, { useState, useEffect, useCallback, useRef } from 'react'
import FormulaEditor from './components/FormulaEditor'
import Visualization3D, { VizMode } from './components/Visualization3D'
import HistoryPanel from './components/HistoryPanel'
import ExportPanel from './components/ExportPanel'
import LossChart from './components/LossChart'
import LibraryPanel from './components/LibraryPanel'
import HyperOptPanel from './components/HyperOptPanel'
import ComparisonPanel from './components/ComparisonPanel'
import {
  fetchPDETypes,
  fetchAdvancedOptions,
  solvePDE,
  fetchHistory,
  predictPDE,
  createSSEConnection,
  PDEType,
  SolveRequest,
  GridData,
  HistoryRecord,
  TrainingHistoryPoint,
  AdvancedOptions,
  SSEProgressEvent,
  SolveCompleteResponse,
  fetchLibraryCase,
  solveLibraryCase,
  LibraryCaseDetail,
  ComparisonResult,
  LibrarySolveCompleteResponse,
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
    ic: 'sin(pi*x) + 0.1*sin(10*pi*x)',
    bcLeft: '0',
    bcRight: '0',
  },
}

const DEFAULT_ADVANCED: AdvancedOptions = {
  use_fourier: true,
  use_adaptive_activation: true,
  use_hard_constraint: true,
  use_mc_dropout: true,
  fourier_bands: [0.01, 0.1, 1, 10],
  fourier_freqs: 32,
  dropout_rate: 0.1,
  mc_samples: 30,
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

  const [advanced, setAdvanced] = useState<AdvancedOptions>(DEFAULT_ADVANCED)
  const [showAdvanced, setShowAdvanced] = useState(true)
  const [vizMode, setVizMode] = useState<VizMode>('mean')
  const [showUncBand, setShowUncBand] = useState(true)
  const [logInterval, setLogInterval] = useState(20)
  const [progressEta, setProgressEta] = useState<{ epoch: number; progressPct: number; elapsed: number } | null>(null)

  const [isSolving, setIsSolving] = useState(false)
  const [solveStatus, setSolveStatus] = useState<string>('')
  const [solveErrorMsg, setSolveErrorMsg] = useState<string>('')

  const [gridData, setGridData] = useState<GridData | null>(null)
  const [history, setHistory] = useState<HistoryRecord[]>([])
  const [selectedRecordId, setSelectedRecordId] = useState<number | null>(null)
  const [currentTimeIndex, setCurrentTimeIndex] = useState(999)
  const [isAnimating, setIsAnimating] = useState(false)
  const [colorScheme, setColorScheme] = useState<string>('plasma')
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryPoint[]>([])
  const [currentRecordId, setCurrentRecordId] = useState<number | null>(null)
  const animRef = useRef<number | null>(null)
  const sseRef = useRef<EventSource | null>(null)
  const startTimeRef = useRef<number>(0)
  const clientIdRef = useRef<string>('')

  const [leftTab, setLeftTab] = useState<'custom' | 'library' | 'history'>('library')
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [selectedCaseDetail, setSelectedCaseDetail] = useState<LibraryCaseDetail | null>(null)
  const [comparison, setComparison] = useState<ComparisonResult | null>(null)
  const [mainBottomTab, setMainBottomTab] = useState<'viz' | 'compare'>('viz')
  const [runFdmOnSolve, setRunFdmOnSolve] = useState(true)
  const [runHpOnSolve, setRunHpOnSolve] = useState(false)

  useEffect(() => {
    fetchPDETypes()
      .then((res) => setPdeTypes(res.data.types))
      .catch(console.error)
    fetchAdvancedOptions()
      .then((res) => { if (res.data) setAdvanced({ ...DEFAULT_ADVANCED, ...res.data }) })
      .catch(() => setAdvanced(DEFAULT_ADVANCED))
    loadHistory()
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

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetchHistory()
      setHistory(res.data)
    } catch (err) {
      console.error('Failed to load history:', err)
    }
  }, [])

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

  const closeSSE = () => {
    if (sseRef.current) {
      try { sseRef.current.close() } catch {}
      sseRef.current = null
    }
  }

  const handleLibraryCaseSelect = async (caseId: string) => {
    setSelectedCaseId(caseId)
    try {
      const { data } = await fetchLibraryCase(caseId)
      setSelectedCaseDetail(data)
      setSelectedType(data.pde_type)
      setEquationLatex(data.equation_latex)
      setIcExpression(data.ic_expression)
      setBcLeftExpression(data.bc_left_expression)
      setBcRightExpression(data.bc_right_expression)
      const dm: any = { x_min: 0, x_max: 1, t_min: 0, t_max: 1 }
      if (data.domain) {
        if (data.domain.x) { dm.x_min = data.domain.x[0]; dm.x_max = data.domain.x[1] }
        if (data.domain.t) { dm.t_min = data.domain.t[0]; dm.t_max = data.domain.t[1] }
      }
      setDomain(dm)
      setParams({ ...data.params })
      if (data.default_layers) setLayers(data.default_layers)
      if (data.default_epochs) setEpochs(data.default_epochs)
      setSolveName(`${data.name}_${new Date().toLocaleTimeString('zh-CN')}`)
    } catch (err) {
      console.error('Failed to load case:', err)
    }
  }

  const handleSolveLibraryCase = async (caseId: string) => {
    setIsSolving(true)
    setSolveStatus('solving')
    setSolveErrorMsg('')
    setProgressEta(null)
    setTrainingHistory([])
    setGridData(null)
    setCurrentRecordId(null)
    setComparison(null)
    setVizMode('mean')
    setMainBottomTab('viz')
    startTimeRef.current = Date.now()

    try {
      await handleLibraryCaseSelect(caseId)
      closeSSE()

      const startRes = await solveLibraryCase(caseId, {
        epochs,
        layers,
        use_fourier: advanced.use_fourier,
        use_adaptive_activation: advanced.use_adaptive_activation,
        use_hard_constraint: advanced.use_hard_constraint,
        use_mc_dropout: advanced.use_mc_dropout,
        fourier_bands: advanced.fourier_bands,
        fourier_freqs: advanced.fourier_freqs,
        dropout_rate: advanced.dropout_rate,
        mc_samples: advanced.mc_samples,
        log_interval: logInterval,
        n_collocation: nCollocation,
        learning_rate: learningRate,
        run_fdm_compare: runFdmOnSolve,
        run_hyperopt: runHpOnSolve,
        hp_n_trials: 8,
        hp_quick_epochs: 150,
        name: solveName,
      })

      const taskId = startRes.data.task_id
      console.log('Library solve started:', taskId)

      const sse = createSSEConnection({
        onConnected: (cid) => { clientIdRef.current = cid },
        onProgress: (ev: SSEProgressEvent) => {
          setTrainingHistory((prev) => {
            const prev2 = prev.length >= 2000 ? prev.slice(-1000) : prev
            return [...prev2, ev]
          })
          const elapsed = (Date.now() - startTimeRef.current) / 1000
          const pct = Math.min(100, (ev.epoch / Math.max(1, epochs)) * 100)
          setProgressEta({ epoch: ev.epoch, progressPct: pct, elapsed })
        },
        onComplete: (raw: any) => {
          const data = raw as LibrarySolveCompleteResponse
          console.log('Library solve complete:', data.id, 'loss=', data.final_loss, 'comparison=', !!data.comparison)
          setGridData(data.grid_data)
          setCurrentTimeIndex(data.grid_data.t.length - 1)
          setTrainingHistory(data.training_history)
          setCurrentRecordId(data.id)
          setSelectedRecordId(data.id)
          if (data.comparison) {
            setComparison(data.comparison)
            setMainBottomTab('compare')
          }
          setSolveStatus('completed')
          setIsSolving(false)
          closeSSE()
          loadHistory()
        },
        onError: (err: string) => {
          setSolveStatus('error')
          setSolveErrorMsg(err)
          setIsSolving(false)
          closeSSE()
        },
      })
      sseRef.current = sse

    } catch (err: any) {
      setSolveStatus('error')
      setSolveErrorMsg(err?.response?.data?.error || err.message || 'Unknown error')
      setIsSolving(false)
      closeSSE()
    }
  }

  const applyHyperOptBest = (best: Record<string, any>) => {
    if (best.layers) setLayers(best.layers)
    if (best.learning_rate) setLearningRate(best.learning_rate)
    if (best.n_collocation) setNCollocation(best.n_collocation)
  }

  const handleSolve = async () => {
    setIsSolving(true)
    setSolveStatus('solving')
    setSolveErrorMsg('')
    setProgressEta(null)
    setTrainingHistory([])
    setGridData(null)
    setCurrentRecordId(null)
    setVizMode('mean')
    startTimeRef.current = Date.now()

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
        use_fourier: advanced.use_fourier,
        use_adaptive_activation: advanced.use_adaptive_activation,
        use_hard_constraint: advanced.use_hard_constraint,
        use_mc_dropout: advanced.use_mc_dropout,
        fourier_bands: advanced.fourier_bands,
        fourier_freqs: advanced.fourier_freqs,
        dropout_rate: advanced.dropout_rate,
        mc_samples: advanced.mc_samples,
        log_interval: logInterval,
      }

      closeSSE()

      const startRes = await solvePDE(request)
      const taskId = startRes.data.task_id
      console.log('Started task:', taskId)

      const sse = createSSEConnection({
        onConnected: (cid) => { clientIdRef.current = cid; console.log('SSE connected, client=', cid) },
        onProgress: (ev: SSEProgressEvent) => {
          setTrainingHistory((prev) => {
            const prev2 = prev.length >= 2000 ? prev.slice(-1000) : prev
            return [...prev2, ev]
          })
          const elapsed = (Date.now() - startTimeRef.current) / 1000
          const pct = Math.min(100, (ev.epoch / Math.max(1, epochs)) * 100)
          setProgressEta({ epoch: ev.epoch, progressPct: pct, elapsed })
        },
        onComplete: (data: SolveCompleteResponse) => {
          console.log('SSE complete:', data.id, 'final_loss=', data.final_loss)
          setGridData(data.grid_data)
          setCurrentTimeIndex(data.grid_data.t.length - 1)
          setTrainingHistory(data.training_history)
          setCurrentRecordId(data.id)
          setSelectedRecordId(data.id)
          setSolveStatus('completed')
          setIsSolving(false)
          closeSSE()
          loadHistory()
        },
        onError: (err: string) => {
          console.error('SSE error:', err)
          setSolveStatus('error')
          setSolveErrorMsg(err)
          setIsSolving(false)
          closeSSE()
        },
      })
      sseRef.current = sse

    } catch (err: any) {
      console.error('Solve start failed:', err)
      setSolveStatus('error')
      setSolveErrorMsg(err?.response?.data?.error || err.message || 'Unknown error')
      setIsSolving(false)
      closeSSE()
    }
  }

  const handleCancel = () => {
    setIsSolving(false)
    setSolveStatus('')
    closeSSE()
  }

  const handleSelectRecord = async (record: HistoryRecord) => {
    setSelectedRecordId(record.id)
    setCurrentRecordId(record.id)
    try {
      const res = await predictPDE(record.id, 100, 100, advanced.mc_samples, advanced.use_mc_dropout)
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
      setVizMode('mean')
    } catch (err) {
      console.error('Failed to load record:', err)
    }
  }

  const handleDeleteRecord = (id: number) => {
    setHistory((prev) => prev.filter((r) => r.id !== id))
    if (selectedRecordId === id) setSelectedRecordId(null)
    if (currentRecordId === id) setCurrentRecordId(null)
  }

  useEffect(() => {
    return () => closeSSE()
  }, [])

  const layersStr = layers.join(', ')

  const hasUncertainty = !!gridData?.has_uncertainty && (gridData?.u_uncertainty || gridData?.u_std)

  const updateFourierBands = (val: string) => {
    const parsed = val.split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n) && n > 0)
    if (parsed.length >= 1) setAdvanced({ ...advanced, fourier_bands: parsed })
  }

  const uncAll = gridData?.u_uncertainty ? gridData.u_uncertainty.flat() : []
  const uncMax = uncAll.length > 0 ? Math.max(...uncAll) : 0
  const uncMean = uncAll.length > 0 ? uncAll.reduce((a: number, b: number) => a + b, 0) / uncAll.length : 0

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>PINN Solver</h1>
          <p>物理信息神经网络 · 偏微分方程数值求解器</p>
        </div>

        <div className="left-tabs">
          <button
            className={`left-tab ${leftTab === 'library' ? 'active' : ''}`}
            onClick={() => setLeftTab('library')}
          >
            📚 方程库
          </button>
          <button
            className={`left-tab ${leftTab === 'custom' ? 'active' : ''}`}
            onClick={() => setLeftTab('custom')}
          >
            ⚙️ 自定义
          </button>
          <button
            className={`left-tab ${leftTab === 'history' ? 'active' : ''}`}
            onClick={() => setLeftTab('history')}
          >
            📜 历史
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px 16px' }}>

          {leftTab === 'library' && (
            <LibraryPanel
              selectedCaseId={selectedCaseId}
              onSelectCase={handleLibraryCaseSelect}
              onSolveCase={handleSolveLibraryCase}
              busy={isSolving}
            />
          )}

          {leftTab === 'custom' && (
            <>
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

          <div className="section">
            <div className="section-title" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setShowAdvanced(!showAdvanced)}>
              <span style={{ marginRight: 6 }}>{showAdvanced ? '▾' : '▸'}</span>
              高级训练选项
            </div>
            {showAdvanced && (
              <div style={{ padding: '4px 0', borderTop: '1px solid var(--border-color)', marginTop: 8 }}>
                <div className="checkbox-row">
                  <label className="check-label">
                    <input type="checkbox" checked={advanced.use_hard_constraint} onChange={(e) => setAdvanced({ ...advanced, use_hard_constraint: e.target.checked })} />
                    <span>硬约束网络 (距离函数边界编码)</span>
                  </label>
                </div>
                <div className="checkbox-row">
                  <label className="check-label">
                    <input type="checkbox" checked={advanced.use_adaptive_activation} onChange={(e) => setAdvanced({ ...advanced, use_adaptive_activation: e.target.checked })} />
                    <span>自适应激活函数 (可学习斜率)</span>
                  </label>
                </div>
                <div className="checkbox-row">
                  <label className="check-label">
                    <input type="checkbox" checked={advanced.use_fourier} onChange={(e) => setAdvanced({ ...advanced, use_fourier: e.target.checked })} />
                    <span>多频段傅里叶特征映射</span>
                  </label>
                </div>
                <div className="checkbox-row">
                  <label className="check-label">
                    <input type="checkbox" checked={advanced.use_mc_dropout} onChange={(e) => setAdvanced({ ...advanced, use_mc_dropout: e.target.checked })} />
                    <span>蒙特卡洛Dropout (不确定性估计)</span>
                  </label>
                </div>
                {advanced.use_fourier && (
                  <div className="form-group" style={{ marginTop: 6 }}>
                    <label>傅里叶频段 (逗号分隔)</label>
                    <input
                      type="text"
                      value={advanced.fourier_bands.join(', ')}
                      onChange={(e) => updateFourierBands(e.target.value)}
                    />
                  </div>
                )}
                <div className="form-group" style={{ marginTop: 4 }}>
                  <label>每频段频率数</label>
                  <input
                    type="number"
                    min={4}
                    max={256}
                    value={advanced.fourier_freqs}
                    onChange={(e) => setAdvanced({ ...advanced, fourier_freqs: Math.max(4, Math.min(256, parseInt(e.target.value) || 32)) })}
                  />
                </div>
                {advanced.use_mc_dropout && (
                  <>
                    <div className="form-group" style={{ marginTop: 4 }}>
                      <label>Dropout概率</label>
                      <input
                        type="number"
                        min={0}
                        max={0.9}
                        step={0.01}
                        value={advanced.dropout_rate}
                        onChange={(e) => setAdvanced({ ...advanced, dropout_rate: Math.max(0, Math.min(0.9, parseFloat(e.target.value) || 0.1)) })}
                      />
                    </div>
                    <div className="form-group" style={{ marginTop: 4 }}>
                      <label>MC采样数</label>
                      <input
                        type="number"
                        min={2}
                        max={200}
                        value={advanced.mc_samples}
                        onChange={(e) => setAdvanced({ ...advanced, mc_samples: Math.max(2, Math.min(200, parseInt(e.target.value) || 30)) })}
                      />
                    </div>
                  </>
                )}
                <div className="form-group" style={{ marginTop: 4 }}>
                  <label>日志间隔 (epoch)</label>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    value={logInterval}
                    onChange={(e) => setLogInterval(Math.max(1, Math.min(500, parseInt(e.target.value) || 20)))}
                  />
                </div>
                <div style={{ marginTop: 10, borderTop: '1px solid var(--border-color)', paddingTop: 10 }}>
                  <div className="checkbox-row">
                    <label className="check-label">
                      <input type="checkbox" checked={runFdmOnSolve} onChange={(e) => setRunFdmOnSolve(e.target.checked)} />
                      <span style={{ color: '#5eead4' }}>📊 求解后运行FDM并对比</span>
                    </label>
                  </div>
                  <div className="checkbox-row">
                    <label className="check-label">
                      <input type="checkbox" checked={runHpOnSolve} onChange={(e) => setRunHpOnSolve(e.target.checked)} />
                      <span style={{ color: '#c4b5fd' }}>🔍 先运行超参搜索再训练</span>
                    </label>
                  </div>
                </div>
                <div style={{ marginTop: 10 }}>
                  <HyperOptPanel
                    defaultPdeType={selectedType}
                    disabled={isSolving}
                    onApplyBest={applyHyperOptBest}
                  />
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn btn-primary btn-block"
              onClick={handleSolve}
              disabled={isSolving}
              style={{ marginTop:8, padding: '12px', flex: 1 }}
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
            {isSolving && (
              <button
                className="btn btn-secondary"
                onClick={handleCancel}
                style={{ marginTop: 8, padding: '12px 16px' }}
                title="取消训练"
              >
                ✕
              </button>
            )}
          </div>

          {solveStatus && (
            <div style={{ marginTop: 8 }}>
              <span className={`status-badge ${solveStatus}`}>
                {solveStatus === 'solving' && progressEta && `求解中 · epoch ${progressEta.epoch}/${epochs} (${progressEta.progressPct.toFixed(1)}%)`}
                {solveStatus === 'solving' && !progressEta && '求解中 (初始化中...)'}
                {solveStatus === 'completed' && '已完成'}
                {solveStatus === 'error' && '出错'}
              </span>
              {solveErrorMsg && (
                <div style={{ color: '#ff6b6b', fontSize: 12, marginTop: 6, lineHeight: 1.5, wordBreak: 'break-word' }}>
                  {solveErrorMsg}
                </div>
              )}
              {progressEta && solveStatus === 'solving' && (
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                  已用时 {progressEta.elapsed.toFixed(1)}s
                </div>
              )}
            </div>
          )}

          {trainingHistory.length > 0 && (
            <div className="training-progress">
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>训练损失曲线</div>
              <LossChart history={trainingHistory} />
            </div>
          )}
          </>
          )}

          {leftTab === 'history' && (
            <>
              {trainingHistory.length > 0 && (
                <div className="training-progress" style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>最近训练损失曲线</div>
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
            </>
          )}

          {leftTab !== 'history' && (
            <>
              {trainingHistory.length > 0 && leftTab === 'library' && (
                <div className="training-progress" style={{ marginTop: 12 }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>训练损失曲线</div>
                  <LossChart history={trainingHistory} />
                </div>
              )}
              {solveStatus && leftTab === 'library' && (
                <div style={{ marginTop: 8 }}>
                  <span className={`status-badge ${solveStatus}`}>
                    {solveStatus === 'solving' && progressEta && `求解中 · epoch ${progressEta.epoch}/${epochs} (${progressEta.progressPct.toFixed(1)}%)`}
                    {solveStatus === 'solving' && !progressEta && '求解中 (初始化中...)'}
                    {solveStatus === 'completed' && '已完成'}
                    {solveStatus === 'error' && '出错'}
                  </span>
                  {solveErrorMsg && (
                    <div style={{ color: '#ff6b6b', fontSize: 12, marginTop: 6, lineHeight: 1.5, wordBreak: 'break-word' }}>
                      {solveErrorMsg}
                    </div>
                  )}
                </div>
              )}
              <ExportPanel recordId={currentRecordId} disabled={isSolving} />
            </>
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="top-bar">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: 2, background: 'var(--bg-tertiary)', borderRadius: 6, padding: 3 }}>
              <button
                onClick={() => setMainBottomTab('viz')}
                className={`main-tab ${mainBottomTab === 'viz' ? 'active' : ''}`}
              >
                📈 可视化
              </button>
              <button
                onClick={() => setMainBottomTab('compare')}
                disabled={!comparison}
                className={`main-tab ${mainBottomTab === 'compare' ? 'active' : ''} ${!comparison ? 'disabled' : ''}`}
              >
                {comparison ? '📊 方法对比' : '📊 对比 (无数据)'}
              </button>
            </div>
            {selectedCaseDetail && (
              <span style={{
                fontSize: 11,
                padding: '3px 10px',
                borderRadius: 12,
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                color: '#93c5fd',
              }}>
                📚 {selectedCaseDetail.name}
              </span>
            )}
          </div>
          <span className="top-bar-title" style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
            数值解
          </span>
          <div className="toolbar">
            {hasUncertainty && (
              <>
                <div style={{ display: 'flex', gap: 2, background: 'var(--bg-tertiary)', borderRadius: 4, padding: 2 }}>
                  <button
                    onClick={() => setVizMode('mean')}
                    style={{
                      background: vizMode === 'mean' ? 'var(--accent-primary)' : 'transparent',
                      border: 'none',
                      color: vizMode === 'mean' ? '#fff' : 'var(--text-secondary)',
                      padding: '4px 10px',
                      borderRadius: 3,
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    均值 u(x,t)
                  </button>
                  <button
                    onClick={() => setVizMode('uncertainty')}
                    style={{
                      background: vizMode === 'uncertainty' ? '#c13b55' : 'transparent',
                      border: 'none',
                      color: vizMode === 'uncertainty' ? '#fff' : 'var(--text-secondary)',
                      padding: '4px 10px',
                      borderRadius: 3,
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    不确定性 2σ
                  </button>
                </div>
                <div className="divider" />
                <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showUncBand}
                    onChange={(e) => setShowUncBand(e.target.checked)}
                    style={{ marginTop: 0 }}
                  />
                  置信带
                </label>
                <div className="divider" />
              </>
            )}
            <select
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
              style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: 4,
                color: 'var(--text-primary)',
                padding: '4px 8px',
                fontSize: 12,
              }}
            >
              <option value="plasma">Plasma</option>
              <option value="viridis">Viridis</option>
              <option value="coolwarm">Coolwarm</option>
            </select>
            <div className="divider" />
            <button className="btn btn-sm btn-secondary" onClick={() => setIsAnimating(!isAnimating)}>
              {isAnimating ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => { if (gridData) setCurrentTimeIndex(gridData.t.length - 1) }}
            >
              ⏭ 末帧
            </button>
            <button className="btn btn-sm btn-secondary" onClick={() => setCurrentTimeIndex(0)}>
              ⏮ 首帧
            </button>
          </div>
        </div>

        <div className="content-area">
          {mainBottomTab === 'viz' && (
            <>
          <Visualization3D
            gridData={gridData}
            isAnimating={isAnimating}
            currentTimeIndex={currentTimeIndex}
            colorScheme={colorScheme}
            vizMode={vizMode}
            showUncertaintyBand={showUncBand}
          />

          {gridData && (
            <>
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
              {hasUncertainty && (
                <div style={{
                  padding: '8px 16px',
                  background: 'rgba(193, 59, 85, 0.08)',
                  border: '1px solid rgba(193, 59, 85, 0.3)',
                  borderRadius: 6,
                  margin: '0 16px 8px',
                  display: 'flex',
                  gap: 20,
                  fontSize: 12,
                  flexWrap: 'wrap',
                }}>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>最大不确定性 (95%CI):</span>
                    <strong style={{ color: '#ff8a8a', marginLeft: 4 }}>
                      {uncMax.toExponential(3)}
                    </strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>平均不确定性:</span>
                    <strong style={{ color: '#ff8a8a', marginLeft: 4 }}>
                      {uncMean.toExponential(3)}
                    </strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>MC采样数:</span>
                    <strong style={{ color: '#ff8a8a', marginLeft: 4 }}>{advanced.mc_samples}</strong>
                  </div>
                </div>
              )}
            </>
          )}
            </>
          )}

          {mainBottomTab === 'compare' && (
            <div style={{ padding: 20, overflowY: 'auto' }}>
              {comparison ? (
                <ComparisonPanel
                  result={comparison}
                  caseName={selectedCaseDetail?.name}
                />
              ) : (
                <div style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  gap: 12,
                  color: 'var(--text-secondary)',
                }}>
                  <div style={{ fontSize: 48 }}>📊</div>
                  <div style={{ fontSize: 16 }}>暂无对比数据</div>
                  <div style={{ fontSize: 12, textAlign: 'center', maxWidth: 400 }}>
                    请在左侧高级选项中勾选"📊 求解后运行FDM并对比"，或从方程库选择支持FDM对比的案例（标有"FDM对比"标签），运行求解后即可显示PINN与传统方法的误差、速度对比。
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
