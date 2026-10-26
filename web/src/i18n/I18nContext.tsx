import React, { useEffect, useState } from 'react'
import type { Language } from './types'
import { I18nContext } from './useI18n'
import {
  DEFAULT_LANGUAGE,
  SUPPORTED_LANGUAGES,
  languageOption,
  matchLanguageTag,
  resolveTranslation,
} from './languages'

function detectDefaultLanguage(): Language {
  if (typeof window === 'undefined') return DEFAULT_LANGUAGE

  // The write side already tolerates refused storage (private mode); the read
  // side must too — a SecurityError here throws during render, above every
  // error boundary, and blanks the whole console for users with site data
  // disabled.
  let saved: string | null = null
  try {
    saved = localStorage.getItem('ubt_lang')
  } catch {
    saved = null
  }
  if (saved && (SUPPORTED_LANGUAGES as readonly string[]).includes(saved)) {
    return saved as Language
  }

  // Check browser / OS languages with full hierarchy
  const candidateLangs: string[] = []
  if (Array.isArray(navigator.languages)) {
    candidateLangs.push(...navigator.languages)
  }
  if (navigator.language) {
    candidateLangs.push(navigator.language)
  }
  if ((navigator as any).userLanguage) {
    candidateLangs.push((navigator as any).userLanguage)
  }

  for (const raw of candidateLangs) {
    const matched = matchLanguageTag(raw ?? '')
    if (matched) return matched
  }

  // Canonical international fallback is DEFAULT_LANGUAGE ('en').
  // Never assume "if not English, it must be Chinese" or vice versa.
  return DEFAULT_LANGUAGE
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>(detectDefaultLanguage)

  useEffect(() => {
    const option = languageOption(language)
    document.documentElement.lang = option.locale
    document.documentElement.dir = option.direction
  }, [language])

  const setLanguage = (lang: Language) => {
    setLanguageState(lang)
    try {
      localStorage.setItem('ubt_lang', lang)
    } catch {
      // Ignore storage errors in private modes
    }
  }

  const t = resolveTranslation(language)

  return (
    <I18nContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </I18nContext.Provider>
  )
}
