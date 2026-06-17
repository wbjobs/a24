import React, { useRef, useEffect } from 'react'

import { TrainingHistoryPoint } from '../api'

interface LossChartProps {
  history: TrainingHistoryPoint[]
}

const LossChart: React.FC<LossChartProps> = ({ history }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || history.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const w = rect.width
    const h = rect.height
    const pad = { top: 10, right: 10, bottom: 20, left: 50 }
    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom

    ctx.fillStyle = 'var(--bg-tertiary)'
    ctx.fillRect(0, 0, w, h)

    const losses = history.map((h) => Math.log10(Math.max(h.loss, 1e-15)))
    const lossIc = history.map((h) => Math.log10(Math.max(h.loss_ic ?? h.loss, 1e-15)))
    const lossBc = history.map((h) => Math.log10(Math.max(h.loss_bc ?? h.loss, 1e-15)))
    const lossPde = history.map((h) => Math.log10(Math.max(h.loss_pde ?? h.loss, 1e-15)))

    const allVals = [...losses, ...lossIc, ...lossBc, ...lossPde]
    const minVal = Math.min(...allVals)
    const maxVal = Math.max(...allVals)
    const valRange = maxVal - minVal || 1

    const toX = (i: number) => pad.left + (i / (history.length - 1 || 1)) * plotW
    const toY = (v: number) => pad.top + (1 - (v - minVal) / valRange) * plotH

    const drawLine = (data: number[], color: string) => {
      ctx.beginPath()
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      for (let i = 0; i < data.length; i++) {
        const x = toX(i)
        const y = toY(data[i])
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    ctx.strokeStyle = '#2d2d4a'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(pad.left + plotW, y)
      ctx.stroke()

      const val = maxVal - (i / 4) * valRange
      ctx.fillStyle = '#6a6a82'
      ctx.font = '10px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(`1e${val.toFixed(1)}`, pad.left - 4, y + 3)
    }

    drawLine(losses, '#4a9eff')
    drawLine(lossIc, '#22c55e')
    drawLine(lossBc, '#f59e0b')
    drawLine(lossPde, '#ef4444')

    const labels = [
      { color: '#4a9eff', text: 'Total' },
      { color: '#22c55e', text: 'IC' },
      { color: '#f59e0b', text: 'BC' },
      { color: '#ef4444', text: 'PDE' },
    ]
    ctx.font = '10px sans-serif'
    labels.forEach((l, i) => {
      ctx.fillStyle = l.color
      ctx.fillRect(pad.left + i * 55, h - 12, 8, 8)
      ctx.fillStyle = '#a0a0b8'
      ctx.textAlign = 'left'
      ctx.fillText(l.text, pad.left + i * 55 + 10, h - 4)
    })
  }, [history])

  return (
    <div className="loss-chart">
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}

export default LossChart
