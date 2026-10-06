import React, { createContext, useContext, useEffect, useState } from 'react'
import type { Language, TranslationDictionary } from './types'
import { en } from './translations/en'
import { zh } from './translations/zh'

interface I18nContextType {
  language: Language
  setLanguage: (lang: Language) => void
  t: TranslationDictionary
}

const I18nContext = createContext<I18nContextType | null>(null)

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

export function useI18n(): I18nContextType {
  const ctx = useContext(I18nContext)
  if (!ctx) {
    throw new Error('useI18n must be used within an I18nProvider')
  }
  return ctx
}
