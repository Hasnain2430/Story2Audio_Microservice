import { fileURLToPath, URL } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    // The gateway owns /v1 and the WebSocket upgrade for job events. Proxying in dev
    // keeps the browser on one origin, so cookies and CORS behave the same as in prod.
    proxy: {
      '/v1': {
        target: process.env.VITE_API_ORIGIN ?? 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
