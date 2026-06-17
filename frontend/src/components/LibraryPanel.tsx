import React, { useEffect, useState } from 'react'
import { fetchLibraryIndex, LibraryCase } from '../api'

interface Props {
  onSelectCase: (caseId: string) => void
  selectedCaseId?: string | null
  onSolveCase: (caseId: string) => void
  busy: boolean
}

const LibraryPanel: React.FC<Props> = ({ onSelectCase, selectedCaseId, onSolveCase, busy }) => {
  const [cases, setCases] = useState<LibraryCase[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [pdeTypes, setPdeTypes] = useState<string[]>([])
  const [catFilter, setCatFilter] = useState<string>('')
  const [diffFilter, setDiffFilter] = useState<string>('')
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [searchText, setSearchText] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      try {
        const { data } = await fetchLibraryIndex()
        setCases(data.cases)
        setCategories(data.categories)
        setPdeTypes(data.pde_types)
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const filtered = cases.filter(c => {
    if (catFilter && c.category !== catFilter) return false
    if (typeFilter && c.pde_type !== typeFilter) return false
    const dstr = String(c.difficulty)
    if (diffFilter && !dstr.startsWith(diffFilter)) return false
    if (searchText) {
      const s = searchText.toLowerCase()
      if (!c.name.toLowerCase().includes(s) && !c.description.toLowerCase().includes(s)
        && !c.pde_type.toLowerCase().includes(s)) return false
    }
    return true
  })

  const diffColor = (d: string | number) => {
    const s = String(d)
    if (s.startsWith('1') || s.startsWith('简单')) return 'bg-green-500/20 text-green-300 border-green-500/40'
    if (s.startsWith('2') || s.startsWith('中')) return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40'
    return 'bg-red-500/20 text-red-300 border-red-500/40'
  }

  return (
    <div className="flex flex-col h-full">
      <div className="mb-3 space-y-2">
        <input
          type="text"
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          placeholder="🔍 搜索方程名或描述..."
          className="w-full px-3 py-1.5 rounded-lg bg-[#1e293b] text-white text-sm border border-slate-700 focus:outline-none focus:border-blue-500"
        />
        <div className="flex gap-2">
          <select
            value={catFilter}
            onChange={e => setCatFilter(e.target.value)}
            className="flex-1 px-2 py-1 rounded bg-[#1e293b] text-white text-xs border border-slate-700"
          >
            <option value="">全部分类</option>
            {categories.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <select
            value={typeFilter}
            onChange={e => setTypeFilter(e.target.value)}
            className="flex-1 px-2 py-1 rounded bg-[#1e293b] text-white text-xs border border-slate-700"
          >
            <option value="">全部类型</option>
            {pdeTypes.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="flex gap-2">
          {['', '简单', '中等', '困难'].map(d => (
            <button
              key={d || 'all'}
              onClick={() => setDiffFilter(d)}
              className={`flex-1 text-xs py-1 rounded border transition ${
                diffFilter === d
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-[#1e293b] border-slate-700 text-slate-300 hover:border-blue-500'
              }`}
            >
              {d || '全部'}
            </button>
          ))}
        </div>
      </div>

      <div className="text-xs text-slate-400 mb-2">
        {loading ? '加载中...' : `共 ${cases.length} 个案例，筛选后 ${filtered.length} 个`}
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
        {loading && (
          <div className="text-center text-slate-400 py-8">📚 加载方程库...</div>
        )}
        {filtered.map(c => (
          <div
            key={c.id}
            onClick={() => onSelectCase(c.id)}
            className={`p-3 rounded-lg border cursor-pointer transition group ${
              selectedCaseId === c.id
                ? 'bg-blue-600/20 border-blue-500'
                : 'bg-[#1e293b]/50 border-slate-700 hover:border-blue-400 hover:bg-[#1e293b]'
            }`}
          >
            <div className="flex items-start justify-between mb-1.5">
              <h4 className="font-semibold text-sm text-white flex-1">{c.name}</h4>
              <div className={`text-[10px] px-1.5 py-0.5 rounded border ${diffColor(c.difficulty)} ml-2 shrink-0`}>
                {c.difficulty}
              </div>
            </div>
            <div className="flex flex-wrap gap-1 mb-2">
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {c.pde_type}
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {c.category}
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                {c.n_dims === 1 ? '1D+t' : c.n_dims === 2 ? '2D+t' : c.n_dims === 3 ? '3D+t' : `${c.n_dims}D`}
              </span>
              {c.has_fdm_reference && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  FDM对比
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 mb-2 line-clamp-2">{c.description}</p>
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-500">
                推荐: {c.default_epochs} epochs × [{(c.default_layers || []).join('×') || 'default'}]
              </span>
              <button
                onClick={e => { e.stopPropagation(); onSolveCase(c.id) }}
                disabled={busy}
                className={`text-xs px-3 py-1 rounded font-medium transition ${
                  busy
                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-500 text-white'
                }`}
              >
                ▶ 运行
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default LibraryPanel
