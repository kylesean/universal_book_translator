import { createContext, useContext } from 'react'
import type { Language, TranslationDictionary } from './types'

export interface I18nContextType {
  language: Language
  setLanguage: (lang: Language) => void
  t: TranslationDictionary
}

//: The context and its reader live apart from `I18nContext.tsx` on purpose:
//: a module that exports both a component and a hook cannot be fast-refreshed
//: (React discards the module's state on every edit), which is what
//: `react/only-export-components` reports. The provider imports the context
//: from here; consumers import only the hook.
export const I18nContext = createContext<I18nContextType | null>(null)

export function useI18n(): I18nContextType {
  const ctx = useContext(I18nContext)
  if (!ctx) {
    throw new Error('useI18n must be used within an I18nProvider')
  }
  return ctx
}
