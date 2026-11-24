import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkBreaks from 'remark-breaks'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

import { useI18n } from '@/i18n/useI18n'
import {
  parseAnchorBadgeToken,
  prepareInlineAnchorsForMarkdown,
} from '@/lib/inlineAnchors'
import { hasStructuredMarkup } from '@/lib/markup'
import { cn } from '@/lib/utils'

interface RichTextProps {
  text: string
  source?: boolean
  className?: string
  /** Use compact prose inside table cells and dialogs. */
  density?: 'default' | 'compact'
}

const RASTER_DATA_URL = /^data:image\/(?:png|jpe?g|gif|webp|avif)[;,]/i
const ABSOLUTE_URL = /^[a-z][a-z0-9+.-]*:/i

/**
 * Whether an <img> source may be fetched without leaving this origin.
 *
 * Translated text is attacker-controlled: it comes from a book someone else
 * wrote. Rendering its images directly turned every review session into a
 * beacon (`![x](https://evil.example/pixel.png)` proves the reviewer is reading
 * that book) and gave relative URLs a same-origin GET that carries the session
 * cookie. Artifact images are served from this origin, so they keep working.
 */
function imageIsSameOrigin(src: string | undefined): boolean {
  const value = (src ?? '').trim()
  if (!value) return true
  if (RASTER_DATA_URL.test(value)) return true
  if (value.startsWith('blob:')) return true
  // Protocol-relative (`//host/x.png`) and any non-http scheme is off-origin.
  if (value.startsWith('//')) return false
  if (ABSOLUTE_URL.test(value) && !/^https?:/i.test(value)) return false
  if (/^https?:/i.test(value)) {
    try {
      return new URL(value, window.location.origin).origin === window.location.origin
    } catch {
      return false
    }
  }
  // A relative path resolves against this origin.
  return true
}

function imageHost(src: string | undefined): string {
  try {
    return new URL((src ?? '').trim(), window.location.origin).host
  } catch {
    return 'remote'
  }
}

/** One renderer for review, translation memory and any future text surface. */
export function RichText({
  text,
  source = false,
  className,
  density = 'default',
}: RichTextProps) {
  const { t } = useI18n()
  if (!text.trim()) {
    return <span className="text-muted-foreground">—</span>
  }
  if (source && !hasStructuredMarkup(text)) {
    return (
      <p className={cn('break-words whitespace-pre-wrap', className)}>
        {text}
      </p>
    )
  }
  const renderedText = prepareInlineAnchorsForMarkdown(text)

  return (
    <div
      className={cn(
        'rich-text break-words',
        density === 'compact' && 'rich-text-compact',
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          code: ({ className: codeClassName, children, ...rest }) => {
            const rawText = typeof children === 'string'
              ? children
              : Array.isArray(children) && children.every((c) => typeof c === 'string')
                ? children.join('')
                : null
            const anchor = !codeClassName && rawText ? parseAnchorBadgeToken(rawText) : null
            if (anchor) {
              const kindLabel =
                anchor.kind === 'f'
                  ? t.review.anchorBadgeFormula
                  : anchor.kind === 'n'
                    ? t.review.anchorBadgeNote
                    : t.review.anchorBadgeObject
              return (
                <span
                  title={anchor.rawTag}
                  className={cn(
                    'mx-0.5 inline-flex items-center gap-1 rounded border px-1.5 py-0 font-mono text-[11px] font-medium align-baseline',
                    anchor.kind === 'f'
                      ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300'
                      : anchor.kind === 'n'
                        ? 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300'
                        : 'border-border bg-muted/70 text-muted-foreground',
                  )}
                >
                  <span>{kindLabel}</span>
                  <span className="opacity-75">{anchor.id}</span>
                </span>
              )
            }
            return (
              <code className={codeClassName} {...rest}>
                {children}
              </code>
            )
          },
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
          img: ({ src, alt }) => {
            if (!imageIsSameOrigin(typeof src === 'string' ? src : undefined)) {
              const host = imageHost(typeof src === 'string' ? src : undefined)
              return (
                <span className="inline-flex items-center rounded border border-dashed border-muted-foreground/40 px-1.5 py-0.5 text-xs text-muted-foreground">
                  {t.common.remoteImageBlocked.replace('{host}', host)}
                  {alt ? ` · ${alt}` : ''}
                </span>
              )
            }
            return (
              <img
                src={src}
                alt={alt ?? ''}
                loading="lazy"
                className="max-w-full rounded"
              />
            )
          },
        }}
      >
        {renderedText}
      </ReactMarkdown>
    </div>
  )
}
