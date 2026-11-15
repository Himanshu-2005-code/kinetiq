import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// Relative base so the build works on GitHub Pages, Vercel, Netlify or any static host.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    rollupOptions: { input: { main: resolve(__dirname, 'index.html'), app: resolve(__dirname, 'app.html') } },
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
