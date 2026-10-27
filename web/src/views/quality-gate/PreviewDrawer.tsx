import { useState } from 'react'
import { AlertTriangle, Download, LoaderCircle, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  deliverableDownloadUrl,
  triggerBrowserDownload,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { deliverableName, fmtSize, isPreviewableProduct } from './artifactMeta'

/**
 * The finished product, in place.
 *
 * This drawer deliberately previews only reader-facing PDF/HTML. Reports,
 * XLIFF and metrics are technical attachments: they are listed once in the
 * job's Deliverables tab, not re-rendered as a second document manager here.
 */
export function PreviewDrawer({
  jobId,
  item,
  onClose,
}: {
  jobId: string
  item: DeliverableItem
  onClose: () => void
}) {
  const { t } = useI18n()
  const [failed, setFailed] = useState(false)
  const name = deliverableName(item.key, item.label, t)
  const download = () =>
    triggerBrowserDownload(
      deliverableDownloadUrl(jobId, item.key),
      item.filename
    )

  if (!isPreviewableProduct(item)) return null

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        showCloseButton={false}
        className="flex h-[min(90vh,960px)] max-w-6xl flex-col gap-0 overflow-hidden p-0"
      >
        <DialogHeader className="flex-row items-center justify-between gap-4 space-y-0 border-b border-border/70 px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <DialogTitle className="truncate text-sm">{name}</DialogTitle>
            <DialogDescription className="truncate font-mono text-xs">
              {item.filename} · {fmtSize(item.size_bytes)}
            </DialogDescription>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <Button variant="ghost" size="sm" onClick={download}>
              <Download />
              {t.quality.downloadAction}
            </Button>
            <DialogClose asChild>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                aria-label={t.quality.previewClose}
              >
                <X />
              </Button>
            </DialogClose>
          </div>
        </DialogHeader>

        <div className="relative min-h-0 flex-1 bg-white">
          {failed ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
              <AlertTriangle className="size-6 text-amber-600" />
              <p className="max-w-sm text-sm text-muted-foreground">
                {t.quality.previewDownloadInstead}
              </p>
              <Button variant="secondary" size="sm" onClick={download}>
                <Download />
                {t.quality.downloadAction}
              </Button>
            </div>
          ) : (
            <>
              <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <LoaderCircle className="size-4 animate-spin" />
                {t.quality.previewLoading}
              </div>
              <iframe
                src={deliverableDownloadUrl(jobId, item.key, { inline: true })}
                title={name}
                className="size-full border-0"
                onError={() => setFailed(true)}
              />
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
