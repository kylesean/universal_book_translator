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
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
