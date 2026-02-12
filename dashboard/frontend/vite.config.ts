import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// TLS enabled by default (set DASHBOARD_TLS=0 to disable)
const tlsEnabled = process.env.DASHBOARD_TLS !== '0' && process.env.DASHBOARD_TLS !== 'false';

// Load certificates if TLS is enabled
const httpsConfig = tlsEnabled ? {
  key: readFileSync(resolve(__dirname, '../certs/key.pem')),
  cert: readFileSync(resolve(__dirname, '../certs/cert.pem')),
} : undefined;

// Backend URL scheme based on TLS setting
const backendScheme = tlsEnabled ? 'https' : 'http';
const wsScheme = tlsEnabled ? 'wss' : 'ws';
const backendPort = process.env.DASHBOARD_PORT || '3000';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    host: '0.0.0.0',  // Listen on all interfaces for macstudio.local access
    allowedHosts: ['macstudio.local', 'localhost'],
    https: httpsConfig,
    proxy: {
      '/api': {
        target: `${backendScheme}://localhost:${backendPort}`,
        secure: false,  // Accept self-signed certificates
      },
      '/ws': {
        target: `${wsScheme}://localhost:${backendPort}`,
        ws: true,
        secure: false,  // Accept self-signed certificates
      }
    }
  }
});
