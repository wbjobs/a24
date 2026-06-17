import React, { useEffect, useRef, useState } from 'react'
import { getHyperOptStatus, startHyperOpt, HyperOptStartRequest, HyperOptStatusResponse } from '../api'

interface Props {
  defaultPdeType?: string
  disabled?: boolean
  onApplyBest?: (params: Record<string, any>) => void
}

const HyperOptPanel: React.FC<Props> = ({ defaultPdeType = 'heat', disabled, onApplyBest }) => {
  const [config, setConfig] = useState<HyperOptStartRequest>({
    pde_type: defaultPdeType,
    n_trials: 8,
    quick_epochs: 150,
  })
  const [status, setStatus] = useState<HyperOptStatusResponse | null>(null)
  const [starting, setStarting] = useState(false)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    setConfig(c => ({ ...c, pde_type: defaultPdeType }))
  }, [defaultPdeType])

  useEffect(() => {
    (async () => {
      try {
        const { data } = await getHyperOptStatus()
        setStatus(data)
      } catch {}
    })()
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [])

  const startPolling = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = window.setInterval(async () => {
      try {
        const { data } = await getHyperOptStatus()
        setStatus(data)
        if (!data.running) {
          if (timerRef.current) window.clearInterval(timerRef.current)
        }
      } catch {}
    }, 3000)
  }

  const handleStart = async () => {
    if (disabled || status?.running || starting) return
    setStarting(true)
    try {
      await startHyperOpt(config)
      startPolling()
    } finally {
      setStarting(false)
    }
  }

  const pct = status?.running
    ? Math.min(99, Math.round((status.progress_count / Math.max(1, config.n_trials || 1)) * 100))
    : status?.result ? 100 : 0

  return (
    <div className="bg-[#1e293b]/60 rounded-xl p-4 border border-slate-700">
      <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
        🔍 贝叶斯超参数搜索 <span className="text-[10px] font-normal text-slate-400">（Optuna）</span>
      </h3>

      <div className="grid grid-cols-3 gap-2 mb-3">
        <div>
          <label className="text-[10px] text-slate-400 block mb-1">尝试次数</label>
          <input
            type="number"
            min={3} max={100}
            value={config.n_trials}
            disabled={status?.running || disabled}
            onChange={e => setConfig(c => ({ ...c, n_trials: Number(e.target.value) }))}
            className="w-full px-2 py-1 rounded bg-slate-900 text-white text-xs border border-slate-700 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-400 block mb-1">快速试训epochs</label>
          <input
            type="number"
            min={20} max={2000}
            value={config.quick_epochs}
            disabled={status?.running || disabled}
            onChange={e => setConfig(c => ({ ...c, quick_epochs: Number(e.target.value) }))}
            className="w-full px-2 py-1 rounded bg-slate-900 text-white text-xs border border-slate-700 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-400 block mb-1">目标方程</label>
          <input
            type="text"
            value={config.pde_type}
            disabled
            className="w-full px-2 py-1 rounded bg-slate-900 text-white text-xs border border-slate-700 opacity-60"
          />
        </div>
      </div>

      <button
        onClick={handleStart}
        disabled={status?.running || disabled || starting}
        className={`w-full text-xs py-1.5 rounded font-medium transition mb-3 ${
          status?.running || disabled || starting
            ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
            : 'bg-indigo-600 hover:bg-indigo-500 text-white'
        }`}
      >
        {starting ? '启动中...' : status?.running ? '⏳ 搜索进行中...' : '🚀 开始超参搜索'}
      </button>

      {/* 进度条 */}
      {(status?.running || status?.result) && (
        <>
          <div className="mb-2">
            <div className="flex justify-between text-[10px] text-slate-400 mb-1">
              <span>
                {status?.running
                  ? `第 ${status.progress_count || 0} / ${config.n_trials} 次尝试`
                  : status?.result ? `完成 ${status.result.all_trials?.length || config.n_trials} 次` : ''}
              </span>
              <span className="font-mono">{pct}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  status.running ? 'bg-indigo-500 animate-pulse' : 'bg-green-500'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {/* 当前最佳 */}
          {(status?.result || status?.progress?.length) && (
            <div className="bg-slate-800/60 rounded-lg p-2.5 border border-slate-700">
              <div className="text-[10px] text-slate-400 mb-1">🏆 当前最优</div>
              <div className="text-xs text-white font-mono space-y-0.5">
                {(() => {
                  const best = status?.result
                    ? status.result.best_params
                    : status?.progress?.[status.progress.length - 1]?.best_params
                  const bestLoss = status?.result
                    ? status.result.best_loss
                    : status?.progress?.[status.progress.length - 1]?.best_loss
                  return (
                    <>
                      <div className="flex justify-between">
                        <span className="text-slate-400">最优损失</span>
                        <span className="text-green-400 font-bold">{bestLoss?.toExponential(3) || '-'}</span>
                      </div>
                      {best && Object.entries(best).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-slate-400">{k}</span>
                          <span>{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                        </div>
                      ))}
                      {status?.result && onApplyBest && best && (
                        <button
                          onClick={() => onApplyBest(best)}
                          className="mt-2 w-full bg-green-600/70 hover:bg-green-600 text-white text-[11px] py-1 rounded transition"
                        >
                          ✅ 应用这些超参数到主界面
                        </button>
                      )}
                    </>
                  )
                })()}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default HyperOptPanel
