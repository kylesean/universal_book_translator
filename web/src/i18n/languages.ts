import type { Language, TranslationDictionary } from './types'
import { en } from './translations/en'
import { zh } from './translations/zh'

export interface LanguageOption {
  code: Language
  /** The language's own name, used in selectors regardless of UI locale. */
  label: string
  /** Compact form for constrained chrome. */
  shortLabel: string
  /** BCP-47 locale for dates, numbers and the document language attribute. */
  locale: string
  /** Browser/OS language prefixes that select this dictionary. */
  matchPrefixes: readonly string[]
  direction: 'ltr' | 'rtl'
  listSeparator: string
}

/**
 * The single registry for languages the console can display.
 *
 * Adding a language means adding one entry here, one typed dictionary, and one
 * translator lookup. Selectors, browser detection and document direction all
 * read this registry, so no UI component needs a new hard-coded language pair.
 */
export const LANGUAGE_OPTIONS: readonly LanguageOption[] = [
  {
    code: 'zh',
    label: '简体中文',
    shortLabel: '中',
    locale: 'zh-CN',
    matchPrefixes: ['zh'],
    direction: 'ltr',
    listSeparator: '、',
  },
  {
    code: 'en',
    label: 'English',
    shortLabel: 'EN',
    locale: 'en',
    matchPrefixes: ['en'],
    direction: 'ltr',
    listSeparator: ', ',
  },
] as const

export const SUPPORTED_LANGUAGES: readonly Language[] = LANGUAGE_OPTIONS.map(
  (option) => option.code
)
export const DEFAULT_LANGUAGE: Language = 'en'

const TRANSLATIONS: Record<Language, TranslationDictionary> = {
  en,
  zh,
}

export function languageOption(lang: Language): LanguageOption {
  return (
    LANGUAGE_OPTIONS.find((option) => option.code === lang) ??
    LANGUAGE_OPTIONS.find((option) => option.code === DEFAULT_LANGUAGE)!
  )
}

export function matchLanguageTag(raw: string): Language | null {
  const tag = raw.trim().toLowerCase()
  if (!tag) return null
  return (
    LANGUAGE_OPTIONS.find((option) =>
      option.matchPrefixes.some((prefix) => tag.startsWith(prefix))
    )?.code ?? null
  )
}

export function resolveTranslation(lang: Language): TranslationDictionary {
  return TRANSLATIONS[lang] ?? TRANSLATIONS[DEFAULT_LANGUAGE]
}
