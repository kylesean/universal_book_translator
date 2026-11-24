import React from 'react'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'

export interface ChoiceSelectProps {
  id: string
  value: string
  onValueChange: (value: string) => void
  options: ReadonlyArray<{ value: string; label: string }>
  placeholder?: string
  disabled?: boolean
}

/** A thin composition of the official shadcn Select primitives. */
export function ChoiceSelect({
  id,
  value,
  onValueChange,
  options,
  placeholder,
  disabled,
}: ChoiceSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger id={id}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function FieldLabel({
  htmlFor,
  children,
  className,
}: {
  htmlFor?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <Label
      htmlFor={htmlFor}
      className={cn('text-sm font-medium leading-none text-foreground', className)}
    >
      {children}
    </Label>
  )
}

export function WizardSection({
  step,
  title,
  meta,
  children,
  className,
}: {
  step: number
  title: string
  meta?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={cn(
        'border-t border-border/70 pt-6 first:border-t-0 first:pt-0',
        className,
      )}
    >
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-3">
          <span className="font-mono text-xs font-medium text-muted-foreground tabular-nums">
            {String(step).padStart(2, '0')}
          </span>
          <h2 className="text-base font-semibold tracking-tight text-foreground">{title}</h2>
        </div>
        {meta ? <div className="max-w-full text-xs text-muted-foreground">{meta}</div> : null}
      </div>
      {children}
    </section>
  )
}
