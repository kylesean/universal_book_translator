export function numericFieldInvalid(raw: string, min: number): boolean {
  if (!raw.trim()) return false // empty = follow the operator's default
  const value = Number(raw)
  return !Number.isFinite(value) || value < min
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
