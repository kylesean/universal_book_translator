import React from 'react'
import { clsx } from 'clsx'

export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'danger' | 'ghost'
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  shortcut?: string
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = 'secondary', size = 'md', shortcut, disabled, children, ...props },
  ref
) {
  const base =
    'inline-flex items-center justify-center font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ink-primary)] disabled:pointer-events-none disabled:opacity-40 select-none rounded-[5px] tracking-tight gap-2'

  const variants = {
    // Solid Carbon Ink button with crisp paper-white reverse text
    primary:
      'bg-[var(--btn-bg)] text-[var(--btn-fg)] font-semibold hover:bg-[var(--btn-hover)] shadow-[0_1px_2px_rgba(0,0,0,0.15)] active:translate-y-[0.5px]',
    // Paper subsurface with hairline border
    secondary:
      'bg-[var(--paper-subsurface)] text-[var(--ink-primary)] hover:bg-[var(--paper-surface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)]',
    outline:
      'border border-[var(--paper-border)] bg-transparent text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] hover:border-[var(--paper-border-hover)]',
    danger:
      'bg-[#b91c1c]/10 text-[#b91c1c] hover:bg-[#b91c1c]/20 border border-[#b91c1c]/25',
    ghost:
      'text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)]',
  }

  const sizes = {
    sm: 'h-7 px-2.5 text-xs',
    md: 'h-8 px-3 text-xs',
    lg: 'h-9 px-4 text-sm',
    icon: 'h-8 w-8 p-0',
  }

  return (
    <button
      ref={ref}
      className={clsx(base, variants[variant], sizes[size], className)}
      disabled={disabled}
      {...props}
    >
      {children}
      {shortcut && (
        <span className="text-xs font-mono px-1 py-0.5 rounded bg-[var(--paper-border)] text-[var(--ink-primary)] ml-1 opacity-80">
          {shortcut}
        </span>
      )}
    </button>
  )
})
