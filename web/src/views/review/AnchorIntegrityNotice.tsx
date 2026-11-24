import { AlertTriangle, Plus } from 'lucide-react'
import { useI18n } from '@/i18n/useI18n'
import { checkInlineAnchors } from '@/lib/inlineAnchors'

export function AnchorIntegrityNotice({
  source,
  draft,
  onInsertTag,
}: {
  source: string
  draft: string
  onInsertTag: (tag: string) => void
}) {
  const { t } = useI18n()
  const { missingRigid, unpairedTags } = checkInlineAnchors(source, draft)
  if (missingRigid.length === 0 && unpairedTags.length === 0) return null

  return (
    <div className="space-y-1.5 rounded-md border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200">
      {missingRigid.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <AlertTriangle className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
          <span>{t.review.anchorMissingWarning}</span>
          {missingRigid.map((tag) => (
            <button
              key={tag}
              type="button"
              onClick={() => onInsertTag(tag)}
              className="inline-flex items-center gap-0.5 rounded border border-amber-600/35 bg-background/80 px-1.5 py-0.5 font-mono text-[11px] font-medium text-amber-900 transition-colors hover:bg-background dark:text-amber-100"
              title={t.review.anchorInsertTag.replace('{tag}', tag)}
            >
              <Plus className="size-3" />
              {tag}
            </button>
          ))}
        </div>
      ) : null}
      {unpairedTags.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <AlertTriangle className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
          <span>{t.review.anchorUnpairedWarning}</span>
          <span className="font-mono font-medium">{unpairedTags.join(' ')}</span>
        </div>
      ) : null}
    </div>
  )
}
