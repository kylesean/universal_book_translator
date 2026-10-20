import type { DoctorCheck } from '@/api/client'
import type { DoctorCodeTables } from './translations/doctorCodes'

/**
 * Render one doctor check in the console's language.
 *
 * The engine ships English copy plus stable codes and the values interpolated into
 * that copy (see `_Check` in `ubt/cli/commands/doctor.py`). This resolves each
 * piece against the active catalogue:
 *
 * 1. `group_code` / `code` / `detail_code` / `fix_code` → localized template
 * 2. `{name}` placeholders → `check.params[name]`
 * 3. no template → the engine's English text
 *
 * The fallback is deliberate and is the whole reason the codes exist: a console
 * that lags the engine shows one untranslated line rather than a raw `disk_space.low`
 * key, and adding a check to the engine cannot break an older console.
 */

/** Replace `{key}` with `params[key]`; leave the placeholder when absent. */
function fill(template: string, params: Record<string, unknown>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    const value = params[key]
    if (value === undefined || value === null) return match
    // Lists (dropped font names, missing binaries) read better comma-joined than
    // as JS's default array stringification.
    if (Array.isArray(value)) return value.join(', ')
    return String(value)
  })
}

/**
 * A variant template is `"<detail_code>.<suffix>"`, chosen only when the param it
 * needs is non-empty.
 *
 * The QE check reads `judge=gemma4-e4b` or `judge=workbuddy/... via local-judge`
 * depending on whether a judge provider block is configured, and the two read
 * badly as one template ("via " with nothing after it). Keeping both in the
 * catalogue and picking by param beats interpolating `{provider}` into a
 * sentence that grammatically needs it.
 */
const PARAM_VARIANTS: ReadonlyArray<{ code: string; suffix: string; param: string }> = [
  { code: 'qe.tiered', suffix: 'via', param: 'provider' },
]

/** The most specific template for `detailCode`, or undefined. */
function pickTemplate(
  details: Record<string, string>,
  detailCode: string,
  params: Record<string, unknown>,
): string | undefined {
  for (const { code, suffix, param } of PARAM_VARIANTS) {
    if (detailCode !== code) continue
    const value = params[param]
    const variant = details[`${code}.${suffix}`]
    if (variant && value !== undefined && value !== null && String(value) !== '') {
      return variant
    }
  }
  return details[detailCode]
}

export interface LocalizedCheck {
  group: string
  name: string
  detail: string
  fix: string | null
}

export function localizeDoctorCheck(
  check: DoctorCheck,
  codes: DoctorCodeTables,
): LocalizedCheck {
  const params = check.params ?? {}
  const group = codes.groups[check.group_code] ?? check.group
  const name = codes.names[check.code] ?? check.name

  const detailTemplate = pickTemplate(codes.details, check.detail_code, params)
  const detail = detailTemplate ? fill(detailTemplate, params) : check.detail

  let fix: string | null = check.fix
  if (check.fix) {
    const fixTemplate = codes.fixes[check.fix_code]
    fix = fixTemplate ? fill(fixTemplate, params) : check.fix
  }

  return { group, name, detail, fix }
}
