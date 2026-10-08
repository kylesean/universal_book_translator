import React, { useState } from 'react'
import type { Language } from './types'
import { en } from './translations/en'
import { zh } from './translations/zh'
import { I18nContext } from './useI18n'

function detectDefaultLanguage(): Language {
  if (typeof window === 'undefined') return 'en'
  const saved = localStorage.getItem('ubt_lang') as Language | null
  if (saved === 'en' || saved === 'zh') return saved

  // Check browser / OS language
  const browserLang = navigator.language || (navigator as any).userLanguage || ''
  return browserLang.toLowerCase().startsWith('zh') ? 'zh' : 'en'
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>(detectDefaultLanguage)

  const setLanguage = (lang: Language) => {
    setLanguageState(lang)
    try {
      localStorage.setItem('ubt_lang', lang)
    } catch {
      // Ignore storage errors in private modes
    }
  }

  const t = language === 'zh' ? zh : en

  return (
    <I18nContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </I18nContext.Provider>
  )
}
