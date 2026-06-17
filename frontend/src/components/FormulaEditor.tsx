import React, { useEffect, useRef, useState } from 'react'
import katex from 'katex'

interface FormulaEditorProps {
  value: string
  onChange: (value: string) => void
  label?: string
  placeholder?: string
}

const QUICK_INSERTS = [
  { label: '\\frac{\\partial}{\\partial t}', value: '\\frac{\\partial }{\\partial t}' },
  { label: '\\frac{\\partial^2}{\\partial x^2}', value: '\\frac{\\partial^2 }{\\partial x^2}' },
  { label: '\\alpha', value: '\\alpha' },
  { label: '\\nabla^2', value: '\\nabla^2' },
  { label: 'u(x,t)', value: 'u(x,t)' },
  { label: '\\sin', value: '\\sin' },
  { label: '\\cos', value: '\\cos' },
  { label: '\\pi', value: '\\pi' },
]

const FormulaEditor: React.FC<FormulaEditorProps> = ({
  value,
  onChange,
  label,
  placeholder = '输入LaTeX公式...',
}) => {
  const previewRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!previewRef.current) return
    if (!value.trim()) {
      previewRef.current.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">预览区域</span>'
      setError(null)
      return
    }
    try {
      katex.render(value, previewRef.current, {
        displayMode: true,
        throwOnError: true,
        strict: false,
      })
      setError(null)
    } catch (e: any) {
      try {
        katex.render(value, previewRef.current, {
          displayMode: true,
          throwOnError: false,
          strict: false,
        })
      } catch {}
      setError(e.message || 'LaTeX语法错误')
    }
  }, [value])

  const handleQuickInsert = (insertValue: string) => {
    const textarea = textareaRef.current
    if (!textarea) return
    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    const before = value.substring(0, start)
    const after = value.substring(end)
    const newValue = before + insertValue + after
    onChange(newValue)
    setTimeout(() => {
      textarea.focus()
      const newPos = start + insertValue.length
      textarea.setSelectionRange(newPos, newPos)
    }, 0)
  }

  return (
    <div className="formula-editor">
      {label && <label className="form-group-label">{label}</label>}
      <div className="quick-inserts">
        {QUICK_INSERTS.map((item) => (
          <button
            key={item.value}
            className="btn btn-sm btn-secondary"
            onClick={() => handleQuickInsert(item.value)}
            title={item.value}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="form-group" style={{ marginTop: '8px' }}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={2}
          style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '13px' }}
        />
      </div>
      <div className="latex-preview" ref={previewRef}>
        <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>预览区域</span>
      </div>
      {error && (
        <div style={{ color: 'var(--accent-red)', fontSize: '11px', marginTop: '4px' }}>
          {error}
        </div>
      )}
    </div>
  )
}

export default FormulaEditor
