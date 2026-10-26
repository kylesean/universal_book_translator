export type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'ubt_theme'

/** Stored choice first, then the operating system preference. */
export function detectTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // Private browsing can refuse storage; the system preference still works.
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function applyTheme(theme: Theme): void {
  const dark = theme === 'dark'
  document.documentElement.classList.toggle('dark', dark)
  document.documentElement.style.colorScheme = theme
}

export function rememberTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // The current page still reflects the choice when persistence is denied.
  }
}
