import React, { useState } from 'react'
import {
  SplitSquareVertical,
  Sparkles,
  Save,
  FileText,
  Eye,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { useI18n } from '@/i18n/I18nContext'

interface ReviewWorkbenchProps {
  jobId: string | null
}

export function ReviewWorkbench({ }: ReviewWorkbenchProps) {
  const { t } = useI18n()
  const [viewMode, setViewMode] = useState<'segments' | 'visual_witness'>('segments')

  const [segments, setSegments] = useState([
    {
      id: 'seg-101',
      page: 12,
      source: 'In deep learning, backpropagation computes the gradient of the loss function with respect to the weights by the chain rule.',
      target: '在深度学习中，反向传播利用链式法则计算损失函数关于权重的梯度。',
      status: 'verified',
      qeScore: 0.99,
      hasFormula: true,
      hasTermViolation: false,
    },
    {
      id: 'seg-102',
      page: 12,
      source: 'Let $L(\\theta)$ denote the objective function, where $\\theta \\in \\mathbb{R}^d$ represents the parameter vector.',
      target: '设 $L(\\theta)$ 为目标函数，其中 $\\theta \\in \\mathbb{R}^d$ 代表参数矢量。',
      status: 'warning',
      qeScore: 0.88,
      hasFormula: true,
      hasTermViolation: true,
      termIssue: '推荐规范用词: "参数向量" (而非 "参数矢量")',
    },
    {
      id: 'seg-103',
      page: 13,
      source: 'Stochastic gradient descent updates the parameter vector iteratively: $\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$.',
      target: '随机梯度下降迭代更新参数向量：$\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$。',
      status: 'verified',
      qeScore: 0.98,
      hasFormula: true,
      hasTermViolation: false,
    },
    {
      id: 'seg-104',
      page: 13,
      source: 'When the learning rate $\\eta$ is set excessively large, the optimization trajectory may diverge uncontrollably.',
      target: '当学习率 $\\eta$ 设置过大时，优化轨迹可能会不可控地发散。',
      status: 'verified',
      qeScore: 0.97,
      hasFormula: true,
      hasTermViolation: false,
    },
  ])

  const handleEditTarget = (id: string, newText: string) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === id ? { ...s, target: newText, status: 'verified', hasTermViolation: false } : s))
    )
  }

  const handleApplyTermFix = (id: string) => {
    setSegments((prev) =>
      prev.map((s) => {
        if (s.id === id) {
          return {
            ...s,
            target: s.target.replace('参数矢量', '参数向量'),
            hasTermViolation: false,
            status: 'verified',
          }
        }
        return s
      })
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden bg-[var(--paper-bg)] text-[var(--ink-primary)]">
      {/* Precision Ribbon */}
      <div className="h-11 px-6 border-b border-[var(--paper-border)] bg-[var(--paper-surface)] flex items-center justify-between shrink-0 shadow-2xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--ink-primary)]">
            <SplitSquareVertical className="h-4 w-4 text-[#15803d]" />
            <span>{t.review.title}</span>
          </div>
          <div className="h-3 w-px bg-[var(--paper-border)]" />
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="text-[#b45309] flex items-center gap-1 font-medium">
              <span className="h-1.5 w-1.5 rounded-full bg-[#b45309]" />
              1 {t.review.faultCount}
            </span>
            <span className="text-[#15803d] flex items-center gap-1 font-medium">
              <span className="h-1.5 w-1.5 rounded-full bg-[#15803d]" />
              3 {t.review.verifiedCount}
            </span>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-[4px] border border-[var(--paper-border)]">
          <button
            onClick={() => setViewMode('segments')}
            className={`px-3 py-1 text-xs rounded-[3px] transition-colors font-medium ${
              viewMode === 'segments' ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-2xs' : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
            }`}
          >
            {t.review.modeSegments}
          </button>
          <button
            onClick={() => setViewMode('visual_witness')}
            className={`px-3 py-1 text-xs rounded-[3px] transition-colors font-medium ${
              viewMode === 'visual_witness' ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-2xs' : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
            }`}
          >
            {t.review.modeVisual}
          </button>
        </div>
      </div>

      {/* Main Workbench Viewport */}
      {viewMode === 'segments' ? (
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-4 max-w-5xl mx-auto w-full">
          {segments.map((seg) => (
            <div
              key={seg.id}
              className={`rounded-lg border transition-colors shadow-2xs bg-[var(--paper-surface)] ${
                seg.hasTermViolation
                  ? 'border-[#b45309]/50'
                  : 'border-[var(--paper-border)]'
              }`}
            >
              {/* Segment Header */}
              <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono text-[var(--ink-muted)]">
                <div className="flex items-center gap-2">
                  <span className="text-[var(--ink-primary)] font-bold">{seg.id}</span>
                  <span>Page {seg.page}</span>
                  {seg.hasFormula && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--paper-border)] text-[var(--ink-secondary)] font-semibold">
                      LaTeX
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-semibold text-[var(--ink-primary)]">QE: {(seg.qeScore * 100).toFixed(0)}%</span>
                  <Badge variant={seg.hasTermViolation ? 'warning' : 'success'} dot>
                    {seg.hasTermViolation ? 'REVIEW' : 'PASS'}
                  </Badge>
                </div>
              </div>

              {/* Bilingual Parallel Split */}
              <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-[var(--paper-border)] text-xs">
                {/* Source Column */}
                <div className="p-4 text-[var(--ink-secondary)] font-mono leading-relaxed select-text bg-[var(--paper-subsurface)]/20">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1 font-semibold">
                    {t.review.sourceSegment}
                  </div>
                  {seg.source}
                </div>

                {/* Target Column (Inline Editor) */}
                <div className="p-4 flex flex-col justify-between bg-[var(--paper-surface)]">
                  <div>
                    <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1.5 flex items-center justify-between font-semibold">
                      <span>{t.review.targetSegment}</span>
                      {seg.hasTermViolation && (
                        <span className="text-[#b45309] font-normal font-sans">
                          {seg.termIssue}
                        </span>
                      )}
                    </div>
                    <textarea
                      value={seg.target}
                      onChange={(e) => handleEditTarget(seg.id, e.target.value)}
                      rows={2}
                      className="w-full bg-transparent text-[var(--ink-primary)] text-xs resize-none focus:outline-none font-mono leading-relaxed placeholder:text-[var(--ink-muted)]"
                    />
                  </div>

                  <div className="flex items-center justify-end gap-2 pt-2.5 border-t border-[var(--paper-border)] mt-3">
                    {seg.hasTermViolation && (
                      <Button
                        onClick={() => handleApplyTermFix(seg.id)}
                        variant="primary"
                        size="sm"
                        className="text-[11px] h-6 px-2.5 font-bold"
                      >
                        <Sparkles className="h-3 w-3 mr-1" />
                        {t.review.applyNorm}
                      </Button>
                    )}
                    <Button variant="secondary" size="sm" className="text-[11px] h-6 px-2.5 font-medium">
                      <Save className="h-3 w-3 mr-1 text-[#15803d]" />
                      {t.review.saveFeedback}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Visual Witness PDF Mode */
        <div className="flex-1 flex overflow-hidden p-6 gap-6">
          <div className="flex-1 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col shadow-2xs">
            <div className="h-8 px-3 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono text-[var(--ink-muted)] flex items-center justify-between font-semibold">
              <span>{t.review.sourceCanvas}</span>
              <span>100% SCALE</span>
            </div>
            <div className="flex-1 flex items-center justify-center text-xs text-[var(--ink-muted)] text-center p-6">
              <div>
                <FileText className="h-8 w-8 mx-auto text-[var(--paper-border-hover)] mb-2" />
                <span className="text-[var(--ink-secondary)] font-semibold">Original PDF Canvas</span>
                <p className="text-[11px] text-[var(--ink-muted)] mt-0.5">Rendered via native pypdfium2 / PDF.js</p>
              </div>
            </div>
          </div>

          <div className="flex-1 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col shadow-2xs">
            <div className="h-8 px-3 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono text-[var(--ink-muted)] flex items-center justify-between font-semibold">
              <span>{t.review.targetCanvas}</span>
              <span className="text-[#15803d] font-bold">{t.review.pixelDiff}: 0.12%</span>
            </div>
            <div className="flex-1 flex items-center justify-center text-xs text-[var(--ink-muted)] text-center p-6">
              <div>
                <Eye className="h-8 w-8 mx-auto text-[#15803d]/40 mb-2" />
                <span className="text-[var(--ink-primary)] font-bold">Typst Dual Spread</span>
                <p className="text-[11px] text-[var(--ink-muted)] mt-0.5">Layout gates and collision checks validated</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
