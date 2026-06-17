import React from 'react'
import { HistoryRecord, deleteRecord } from '../api'

interface HistoryPanelProps {
  records: HistoryRecord[]
  selectedId: number | null
  onSelect: (record: HistoryRecord) => void
  onDelete: (id: number) => void
  onRefresh: () => void
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({
  records,
  selectedId,
  onSelect,
  onDelete,
  onRefresh,
}) => {
  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation()
    try {
      await deleteRecord(id)
      onDelete(id)
    } catch (err) {
      console.error('Failed to delete record:', err)
    }
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return ''
    const d = new Date(dateStr)
    return d.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const pdeTypeLabels: Record<string, string> = {
    heat: '热传导',
    wave: '波动方程',
    elliptic: '椭圆方程',
    general: '通用PDE',
  }

  return (
    <div className="section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="section-title">求解历史</div>
        <button className="btn btn-sm btn-secondary" onClick={onRefresh}>
          ↻ 刷新
        </button>
      </div>
      {records.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '12px', textAlign: 'center' }}>
          暂无历史记录
        </div>
      ) : (
        <ul className="history-list">
          {records.map((record) => (
            <li
              key={record.id}
              className={`history-item ${selectedId === record.id ? 'selected' : ''}`}
              onClick={() => onSelect(record)}
            >
              <div className="history-item-name">
                {record.name || `求解 #${record.id}`}
              </div>
              <div className="history-item-meta">
                {pdeTypeLabels[record.pde_type] || record.pde_type}
                {record.created_at && ` · ${formatDate(record.created_at)}`}
              </div>
              {record.final_loss !== null && (
                <div className="history-item-loss">
                  Loss: {record.final_loss.toExponential(4)}
                </div>
              )}
              <button
                className="btn btn-sm btn-danger"
                style={{ marginTop: '4px', padding: '2px 6px', fontSize: '10px' }}
                onClick={(e) => handleDelete(e, record.id)}
              >
                删除
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default HistoryPanel
