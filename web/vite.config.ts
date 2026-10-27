import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `/jobs`, `/assets` and `/system` are both SPA routes and API prefixes. A
// browser navigation sends `Accept: text/html`; fetches from the console send
// `*/*`. Let the SPA handle navigations and proxy only data requests, otherwise
// opening a route directly in dev serves the backend's stale embedded bundle.
function proxyBypass(req: {
  headers: { accept?: string; 'sec-fetch-dest'?: string }
}) {
  // An iframe is a document navigation that should receive the actual
  // deliverable, not the console shell. Without this the preview pane embeds
  // the SPA, which looks exactly like the app rendering itself.
  return req.headers.accept?.includes('text/html') &&
    req.headers['sec-fetch-dest'] === 'document'
    ? '/index.html'
    : undefined
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // The shell + i18n dictionaries are shared by every route. Keep the
    // warning useful for route chunks instead of firing on the vendor shell.
    chunkSizeWarningLimit: 650,
  },
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/jobs': { target: 'http://127.0.0.1:8000', bypass: proxyBypass },
      '/health': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/assets': { target: 'http://127.0.0.1:8000', bypass: proxyBypass },
      '/system': { target: 'http://127.0.0.1:8000', bypass: proxyBypass },
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
