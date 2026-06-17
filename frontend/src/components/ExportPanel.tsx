import React from 'react'
import { getExportUrl } from '../api'

interface ExportPanelProps {
  recordId: number | null
  disabled: boolean
}

const ExportPanel: React.FC<ExportPanelProps> = ({ recordId, disabled }) => {
  const handleExport = (format: 'csv' | 'vtk' | 'vtk3d') => {
    if (!recordId) return
    const url = getExportUrl(recordId, format)
    window.open(url, '_blank')
  }

  return (
    <div className="export-panel">
      <div className="section-title" style={{ marginBottom: '8px' }}>导出结果</div>
      <div className="export-buttons">
        <button
          className="btn btn-sm btn-secondary"
          disabled={disabled || !recordId}
          onClick={() => handleExport('csv')}
          title="导出为CSV格式"
        >
          📄 CSV
        </button>
        <button
          className="btn btn-sm btn-secondary"
          disabled={disabled || !recordId}
          onClick={() => handleExport('vtk')}
          title="导出为VTK格式 (2D平面)"
        >
          📦 VTK 2D
        </button>
        <button
          className="btn btn-sm btn-secondary"
          disabled={disabled || !recordId}
          onClick={() => handleExport('vtk3d')}
          title="导出为VTK格式 (3D曲面)"
        >
          📦 VTK 3D
        </button>
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
        VTK格式可直接导入ParaView进行后处理
      </div>
    </div>
  )
}

export default ExportPanel
