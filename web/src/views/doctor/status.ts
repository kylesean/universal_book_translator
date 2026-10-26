export type DoctorTone = 'ok' | 'warn' | 'fail' | 'skip'

export function toneForStatus(status: string): DoctorTone {
  if (status === 'FAIL') return 'fail'
  if (status === 'WARN') return 'warn'
  if (status === 'SKIP') return 'skip'
  return 'ok'
}
