import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkBreaks from 'remark-breaks'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

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
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
          img: ({ src, alt }) => {
            if (!imageIsSameOrigin(typeof src === 'string' ? src : undefined)) {
              return (
                <span className="inline-flex items-center rounded border border-dashed border-muted-foreground/40 px-1.5 py-0.5 text-xs text-muted-foreground">
                  远程图片（{imageHost(typeof src === 'string' ? src : undefined)}）未加载
                  {alt ? `：${alt}` : ''}
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
        {text}
      </ReactMarkdown>
    </div>
  )
}
