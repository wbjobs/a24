import React from 'react'
import { ComparisonResult } from '../api'

interface Props {
  result: ComparisonResult
  caseName?: string
}

const formatNum = (n: number, p = 2) => {
  if (n === undefined || n === null || isNaN(n)) return '-'
  if (n === 0) return '0'
  const abs = Math.abs(n)
  if (abs >= 1000) return n.toFixed(0)
  if (abs >= 1) return n.toFixed(p)
  if (abs >= 0.001) return n.toFixed(p + 1)
  return n.toExponential(p)
}

const ComparisonPanel: React.FC<Props> = ({ result, caseName }) => {
  const l2Color = result.l2_error < 1e-2 ? 'text-green-400' : result.l2_error < 1e-1 ? 'text-yellow-400' : 'text-red-400'
  const speedupColor = result.speedup_ratio >= 1 ? 'text-green-400' : 'text-red-400'
  const speedupArrow = result.speedup_ratio >= 1 ? '🚀' : '🐢'

  const timePinn = result.pinn_time_sec ?? 0
  const timeFdm = result.fdm_time_sec ?? (result.fdm_info?.elapsed_sec || 0)

  return (
    <div className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] rounded-xl p-5 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          📊 PINN vs 传统数值方法对比
          {caseName && <span className="text-sm font-normal text-slate-400">— {caseName}</span>}
        </h3>
        {result.fdm_info?.method && (
          <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-xs border border-cyan-500/40">
            FDM: {result.fdm_info.method}
          </span>
        )}
      </div>

      {/* 核心指标卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
          <div className="text-[11px] text-slate-400 mb-1">L2 误差</div>
          <div className={`text-xl font-bold font-mono ${l2Color}`}>{formatNum(result.l2_error, 3)}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">PINN − 参考解</div>
        </div>
        <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
          <div className="text-[11px] text-slate-400 mb-1">L∞ 误差</div>
          <div className={`text-xl font-bold font-mono ${l2Color}`}>{formatNum(result.linf_error, 3)}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">最大逐点误差</div>
        </div>
        <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
          <div className="text-[11px] text-slate-400 mb-1">PINN 用时</div>
          <div className="text-xl font-bold font-mono text-blue-400">{formatNum(timePinn)} s</div>
          <div className="text-[10px] text-slate-500 mt-0.5">含网络构建</div>
        </div>
        <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
          <div className="text-[11px] text-slate-400 mb-1">{speedupArrow} 加速比</div>
          <div className={`text-xl font-bold font-mono ${speedupColor}`}>
            {formatNum(result.speedup_ratio || (timeFdm / (timePinn || 1)), 2)}×
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">FDM用时 / PINN用时</div>
        </div>
      </div>

      {/* 详细参数 */}
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">🧠 PINN 神经网络</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">训练用时</span>
              <span className="text-white font-mono">{formatNum(timePinn, 2)} s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">相对误差</span>
              <span className={`font-mono ${l2Color}`}>{formatNum(result.relative_error * 100, 3)} %</span>
            </div>
          </div>
        </div>
        <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2">💻 FDM 有限差分</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">方法</span>
              <span className="text-white font-mono">{result.fdm_method || result.fdm_info?.method || '-'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">网格</span>
              <span className="text-white font-mono">
                {result.fdm_info ? `${result.fdm_info.nx || '?'} × ${result.fdm_info.nt || '?'}` : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">用时</span>
              <span className="text-white font-mono">{formatNum(timeFdm, 3)} s</span>
            </div>
          </div>
        </div>
      </div>

      {/* 结论条 */}
      <div className={`rounded-lg p-3 border text-sm ${
        result.speedup_ratio >= 1 && result.l2_error < 0.05
          ? 'bg-green-500/10 border-green-500/40 text-green-300'
          : result.l2_error < 0.05
            ? 'bg-yellow-500/10 border-yellow-500/40 text-yellow-300'
            : 'bg-red-500/10 border-red-500/40 text-red-300'
      }`}>
        {
          result.speedup_ratio >= 1 && result.l2_error < 0.05
            ? `✅ 结论：PINN 在获得高精度 (L2=${formatNum(result.l2_error, 2)}) 的同时，比 ${result.fdm_method || 'FDM'} 快 ${formatNum(result.speedup_ratio, 1)}×，速度和精度双优！`
            : result.l2_error < 0.05
              ? `📝 结论：PINN 精度良好 (L2=${formatNum(result.l2_error, 2)})，与传统方法速度接近。可通过增加训练配置点数提升精度。`
              : `⚠️ 结论：当前 PINN 精度仍需优化 (L2=${formatNum(result.l2_error, 2)})，建议增加 epoch 或配置点数，或启用超参搜索。`
        }
      </div>

      {/* 逐时间步误差条 */}
      {result.per_time_step_errors && Object.keys(result.per_time_step_errors).length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold text-slate-300 mb-2">📈 逐时间步 L2 误差 (采样)</h4>
          <div className="space-y-1">
            {Object.entries(result.per_time_step_errors).slice(0, 8).map(([t, e]) => (
              <div key={t} className="flex items-center gap-2 text-[10px]">
                <span className="text-slate-400 w-14 shrink-0">t={Number(t).toFixed(2)}</span>
                <div className="flex-1 bg-slate-800 rounded h-4 relative overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                    style={{
                      width: `${Math.min(100, Math.max(2, (Math.log10((e || 1e-12) + 1) * 100)))}%`,
                    }}
                  />
                </div>
                <span className="font-mono text-slate-300 w-20 text-right">{formatNum(e as number, 2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ComparisonPanel
