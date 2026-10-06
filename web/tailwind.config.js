/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          bg: 'var(--paper-bg)',
          surface: 'var(--paper-surface)',
          subsurface: 'var(--paper-subsurface)',
          border: 'var(--paper-border)',
          'border-hover': 'var(--paper-border-hover)',
        },
        ink: {
          primary: 'var(--ink-primary)',
          secondary: 'var(--ink-secondary)',
          muted: 'var(--ink-muted)',
          highlight: 'var(--ink-highlight)',
          amber: 'var(--ink-amber)',
          rose: 'var(--ink-rose)',
        },
        ubt: {
          passed: '#15803d',    // Forest / Archival Emerald Green
          blocked: '#b91c1c',   // Cinnabar Red
          warning: '#b45309',   // Amber Wax Seal
          running: '#0369a1',   // Deep Indigo / Sky
          primary: '#18181b',   // Solid Carbon Ink
        }
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Inter', 'Geist', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'ui-monospace', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        serif: ['Source Han Serif SC', 'Noto Serif SC', 'Songti SC', 'Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}
