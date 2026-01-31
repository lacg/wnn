import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

// Check if TLS is enabled via environment variable
const tlsEnabled = process.env.DASHBOARD_TLS === '1' || process.env.DASHBOARD_TLS === 'true';

// Load certificates if TLS is enabled
const httpsConfig = tlsEnabled ? {
  key: fs.readFileSync(path.resolve(__dirname, '../certs/key.pem')),
  cert: fs.readFileSync(path.resolve(__dirname, '../certs/cert.pem')),
} : undefined;

// Backend URL scheme based on TLS setting
const backendScheme = tlsEnabled ? 'https' : 'http';
const wsScheme = tlsEnabled ? 'wss' : 'ws';
const backendPort = process.env.DASHBOARD_PORT || '3000';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
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
