import React, { useCallback, useEffect, useState } from 'react'
import { Search, Plus, Trash2, Loader2, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getGlossary,
  upsertGlossaryTerm,
  deleteGlossaryTerm,
  listTm,
  evictTm,
  type GlossaryTerm,
  type TmEntry,
} from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

type Tab = 'glossary' | 'tm' | 'bible'

export function LanguageAssetsView() {
  const { t } = useI18n()
  const [activeTab, setActiveTab] = useState<Tab>('glossary')
  const [searchQuery, setSearchQuery] = useState('')

  // Glossary
  const [terms, setTerms] = useState<GlossaryTerm[]>([])
  const [glossaryError, setGlossaryError] = useState<string | null>(null)
  const [glossaryLoading, setGlossaryLoading] = useState(false)
  const [newSource, setNewSource] = useState('')
  const [newTarget, setNewTarget] = useState('')
  const [savingTerm, setSavingTerm] = useState(false)

  // Translation memory
  const [tmEntries, setTmEntries] = useState<TmEntry[]>([])
  const [tmTotal, setTmTotal] = useState(0)
  const [tmLoading, setTmLoading] = useState(false)
  const [tmError, setTmError] = useState<string | null>(null)

  const loadGlossary = useCallback(async () => {
    setGlossaryLoading(true)
    setGlossaryError(null)
    try {
      setTerms(await getGlossary())
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : 'Failed to load glossary')
    } finally {
      setGlossaryLoading(false)
    }
  }, [])

  const loadTm = useCallback(async () => {
    setTmLoading(true)
    setTmError(null)
    try {
      const data = await listTm({ limit: 200 })
      setTmEntries(data.entries)
      setTmTotal(data.total)
    } catch (err) {
      setTmError(err instanceof Error ? err.message : 'Failed to load translation memory')
    } finally {
      setTmLoading(false)
    }
  }, [])

  useEffect(() => {
    if (activeTab === 'glossary') void loadGlossary()
    if (activeTab === 'tm') void loadTm()
  }, [activeTab, loadGlossary, loadTm])

  const handleAddTerm = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newSource.trim() || !newTarget.trim() || savingTerm) return
    setSavingTerm(true)
    try {
      setTerms(await upsertGlossaryTerm(newSource.trim(), newTarget.trim()))
      setNewSource('')
      setNewTarget('')
      setGlossaryError(null)
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : 'Failed to save term')
    } finally {
      setSavingTerm(false)
    }
  }

  const handleDeleteTerm = async (source: string) => {
    try {
      setTerms(await deleteGlossaryTerm(source))
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : 'Failed to delete term')
    }
  }

  const handleEvict = async (id: number) => {
    try {
      await evictTm([id])
      setTmEntries((prev) => prev.filter((entry) => entry.id !== id))
      setTmTotal((prev) => Math.max(0, prev - 1))
    } catch (err) {
      setTmError(err instanceof Error ? err.message : 'Failed to evict entry')
    }
  }

  const filteredTerms = terms.filter(
    (term) =>
      term.source.toLowerCase().includes(searchQuery.toLowerCase()) ||
      term.target.includes(searchQuery)
  )

  const tabs: { id: Tab; label: string }[] = [
    { id: 'glossary', label: t.assets.tabGlossary },
    { id: 'tm', label: t.assets.tabTm },
    { id: 'bible', label: t.assets.tabBible },
  ]

  return (
    <div className="flex-1 overflow-y-auto px-8 py-7 space-y-7 max-w-5xl mx-auto w-full">
      {/* View Header */}
      <div className="flex items-center justify-between border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.assets.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">{t.assets.subtitle}</p>
        </div>

        <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-[5px] border border-[var(--paper-border)]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1 rounded-[4px] text-xs font-semibold transition-colors ${
                activeTab === tab.id
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] shadow-2xs'
                  : 'text-[var(--ink-secondary)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'glossary' && (
        <div className="space-y-4">
          <div className="p-3.5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] shadow-2xs">
            <form onSubmit={handleAddTerm} className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              <input
                type="text"
                placeholder={t.assets.sourceTerm}
                value={newSource}
                onChange={(e) => setNewSource(e.target.value)}
                className="h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-3 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none font-mono placeholder:text-[var(--ink-muted)]"
              />
              <input
                type="text"
                placeholder={t.assets.targetTerm}
                value={newTarget}
                onChange={(e) => setNewTarget(e.target.value)}
                className="h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-3 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none placeholder:text-[var(--ink-muted)]"
              />
              <Button type="submit" variant="primary" size="sm" disabled={savingTerm}>
                {savingTerm ? (
                  <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                ) : (
                  <Plus className="h-3.5 w-3.5 mr-1" />
                )}
                {t.assets.submitTerm}
              </Button>
            </form>
          </div>

          {glossaryError && (
            <div className="p-3 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-secondary)] flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0" />
              {glossaryError}
            </div>
          )}

          <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
            <div className="h-10 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3">
              <div className="relative flex-1 max-w-xs">
                <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-[var(--ink-muted)]" />
                <input
                  type="text"
                  placeholder={t.common.search}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full h-7 pl-8 pr-3 rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none placeholder:text-[var(--ink-muted)]"
                />
              </div>
              <Badge variant="outline">
                {filteredTerms.length} {t.assets.activeTerms}
              </Badge>
            </div>

            <table className="w-full text-xs text-left">
              <thead className="bg-[var(--paper-subsurface)] text-[var(--ink-muted)] font-mono border-b border-[var(--paper-border)]">
                <tr>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider">
                    SOURCE
                  </th>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider">
                    TARGET
                  </th>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider text-right">
                    ACTION
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--paper-border)]">
                {glossaryLoading ? (
                  <tr>
                    <td colSpan={3} className="py-6 text-center text-[var(--ink-muted)]">
                      <Loader2 className="h-4 w-4 animate-spin mx-auto" />
                    </td>
                  </tr>
                ) : filteredTerms.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="py-6 text-center text-[var(--ink-muted)]">
                      No glossary terms.
                    </td>
                  </tr>
                ) : (
                  filteredTerms.map((term) => (
                    <tr
                      key={term.source}
                      className="hover:bg-[var(--paper-subsurface)]/60 transition-colors"
                    >
                      <td className="py-2 px-4 font-mono font-bold text-[var(--ink-primary)]">
                        {term.source}
                      </td>
                      <td className="py-2 px-4 text-[#15803d] font-bold">{term.target}</td>
                      <td className="py-2 px-4 text-right">
                        <button
                          onClick={() => handleDeleteTerm(term.source)}
                          className="text-[var(--ink-muted)] hover:text-[#b91c1c] transition-colors p-1"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'tm' && (
        <div className="space-y-4">
          {tmError && (
            <div className="p-3 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-secondary)] flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0" />
              {tmError}
            </div>
          )}
          <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
            <div className="h-10 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {t.assets.tmTitle}
              </span>
              <Badge variant="outline">{tmTotal} entries</Badge>
            </div>
            <table className="w-full text-xs text-left">
              <thead className="bg-[var(--paper-subsurface)] text-[var(--ink-muted)] font-mono border-b border-[var(--paper-border)]">
                <tr>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider">
                    SOURCE
                  </th>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider">
                    TARGET
                  </th>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider">
                    PROVENANCE
                  </th>
                  <th className="py-2.5 px-4 font-semibold uppercase text-[10px] tracking-wider text-right">
                    ACTION
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--paper-border)]">
                {tmLoading ? (
                  <tr>
                    <td colSpan={4} className="py-6 text-center text-[var(--ink-muted)]">
                      <Loader2 className="h-4 w-4 animate-spin mx-auto" />
                    </td>
                  </tr>
                ) : tmEntries.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-6 text-center text-[var(--ink-muted)]">
                      {t.assets.tmDesc}
                    </td>
                  </tr>
                ) : (
                  tmEntries.map((entry) => (
                    <tr
                      key={entry.id}
                      className="hover:bg-[var(--paper-subsurface)]/60 transition-colors"
                    >
                      <td className="py-2 px-4 font-mono text-[var(--ink-primary)] max-w-xs truncate">
                        {entry.source_text}
                      </td>
                      <td className="py-2 px-4 text-[#15803d] max-w-xs truncate">
                        {entry.target_text}
                      </td>
                      <td className="py-2 px-4 font-mono text-[11px] text-[var(--ink-muted)]">
                        {entry.provenance}
                      </td>
                      <td className="py-2 px-4 text-right">
                        <button
                          onClick={() => handleEvict(entry.id)}
                          className="text-[var(--ink-muted)] hover:text-[#b91c1c] transition-colors p-1"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'bible' && (
        <div className="p-12 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-center text-[var(--ink-muted)] space-y-2 shadow-2xs">
          <h2 className="text-xs font-bold text-[var(--ink-primary)]">{t.assets.bibleTitle}</h2>
          <p className="text-xs text-[var(--ink-secondary)] max-w-md mx-auto leading-relaxed">
            {t.assets.bibleDesc}
          </p>
        </div>
      )}
    </div>
  )
}
