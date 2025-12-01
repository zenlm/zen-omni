import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@luxfi/ui': path.resolve('/Users/z/work/lux/web/pkg/ui'),
      '@luxfi/ui/commerce': path.resolve('/Users/z/work/lux/web/pkg/ui/commerce/ui/context.tsx'),
      '@luxfi/ui/style': path.resolve('/Users/z/work/lux/web/pkg/ui/style'),
      '@hanzo/ui': path.resolve('/Users/z/work/hanzo/ui/pkg/ui'),
      '@hanzo/ui/primitives': path.resolve('/Users/z/work/hanzo/ui/pkg/ui/primitives'),
    },
  },
  optimizeDeps: {
    include: ['lucide-react'],
    exclude: ['@luxfi/ui', '@hanzo/ui'],
  },
  server: {
    port: 3000,
    open: true,
    fs: {
      // Allow serving files from lux web directory
      allow: [
        '.', 
        '/Users/z/work/lux/web',
        '/Users/z/work/hanzo/ui'
      ]
    }
  },
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  },
})