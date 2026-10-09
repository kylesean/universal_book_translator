import React from 'react'
import { clsx } from 'clsx'

export type BadgeVariant = 'default' | 'success' | 'destructive' | 'warning' | 'info' | 'outline'

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: BadgeVariant
  dot?: boolean
}

export function Badge({ className, variant = 'default', dot = false, children, ...props }: BadgeProps) {
  // Semantic tints stay as literals: Tailwind v3 cannot apply an opacity
  // modifier to a `var()` colour — `bg-[var(--ink-amber)]/10` emits *no rule at
  // all*, so the badge would render transparent. Only the solid dot goes
  // through a token (no modifier needed).
  const variantStyles = {
    default: 'bg-[var(--paper-subsurface)] text-[var(--ink-secondary)] border-[var(--paper-border)]',
    success: 'bg-[#15803d]/10 text-[#15803d] border-[#15803d]/25',
    destructive: 'bg-[#b91c1c]/10 text-[#b91c1c] border-[#b91c1c]/25',
    warning: 'bg-[#b45309]/10 text-[#b45309] border-[#b45309]/25',
    info: 'bg-[#0369a1]/10 text-[#0369a1] border-[#0369a1]/25',
    outline: 'text-[var(--ink-muted)] border-[var(--paper-border)] bg-transparent',
  }

  const dotColors = {
    default: 'bg-[var(--ink-muted)]',
    success: 'bg-[var(--ink-highlight)]',
    destructive: 'bg-[var(--ink-rose)]',
    warning: 'bg-[var(--ink-amber)]',
    info: 'bg-[var(--ink-info)]',
    outline: 'bg-[var(--ink-muted)]',
  }

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-[4px] border px-2 py-0.5 text-xs font-mono font-medium transition-colors select-none',
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {dot && <span className={clsx('h-1.5 w-1.5 rounded-full shrink-0', dotColors[variant])} />}
      {children}
    </div>
  )
}
