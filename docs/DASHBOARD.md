# WNN Dashboard - Startup Guide

The WNN Dashboard consists of three components:
1. **Backend** - Rust server (Axum) providing the API
2. **Frontend** - SvelteKit UI for managing experiments
3. **Worker** - Python process that executes queued flows

## Quick Start

### Terminal 1 - Backend

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard
cargo run
```

Runs at: http://localhost:3000

### Terminal 2 - Frontend

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard/frontend
npm run dev
```

Runs at: http://localhost:5173

### Terminal 3 - Worker

**Recommended: Use the wrapper script**

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
./start-worker.sh
```

The script:
- Prevents duplicate workers (checks if one is already running)
- Runs in background with `nohup`
- Logs to `worker.out`

**Manual startup:**

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
source wnn/bin/activate
python -u -m wnn.ram.experiments.worker --url http://localhost:3000
```

**Important:** Do NOT add `src/wnn` to PYTHONPATH - it causes import conflicts with HuggingFace's `tokenizers` package.

## Starting with TLS (HTTPS)

### 1. Generate Certificates (first time only)

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard
bash scripts/gen-certs.sh
```

This creates:
- `certs/cert.pem` - Self-signed certificate (valid 10 years)
- `certs/key.pem` - Private key

### 2. Start Backend with TLS

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard
DASHBOARD_TLS=1 cargo run
```

Runs at: https://localhost:3000

### 3. Start Frontend with TLS

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard/frontend
DASHBOARD_TLS=1 npm run dev
```

### 4. Start Worker with TLS

**Recommended: Use the wrapper script**

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
./start-worker.sh --tls
```

**Manual:**

```bash
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
source wnn/bin/activate
python -u -m wnn.ram.experiments.worker --url https://localhost:3000 --insecure
```

The `--insecure` flag is needed for self-signed certificates.

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_PORT` | `3000` | Server port |
| `DATABASE_URL` | `sqlite:dashboard.db?mode=rwc` | SQLite database path |
| `RUST_LOG` | `info` | Logging level |
| `DASHBOARD_TLS` | (unset) | Set to `1` or `true` to enable TLS |
| `DASHBOARD_CERT` | `certs/cert.pem` | TLS certificate path |
| `DASHBOARD_KEY` | `certs/key.pem` | TLS private key path |

### Worker

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:3000` | Dashboard URL |
| `--poll-interval` | `10` | Seconds between polls |
| `--insecure` | (off) | Skip TLS verification |
| `--checkpoint-dir` | `checkpoints` | Save/resume directory |

## Background Execution

For long-running sessions:

```bash
# Backend
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard
nohup cargo run > ../logs/backend.log 2>&1 &

# Frontend
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard/frontend
nohup npm run dev > ../../logs/frontend.log 2>&1 &

# Worker (uses start-worker.sh which already runs in background)
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
./start-worker.sh          # HTTP mode
./start-worker.sh --tls    # TLS mode
# Logs: tail -f worker.out
```

## Checking Status

```bash
# Check if backend is responding
curl http://localhost:3000/api/health

# Check running processes
ps aux | grep -E "wnn-dashboard|vite|worker" | grep -v grep

# View logs
tail -f logs/backend.log
tail -f logs/frontend.log
tail -f logs/worker.log
```

## Stopping Components

```bash
# Find PIDs
ps aux | grep -E "wnn-dashboard|vite|worker" | grep -v grep

# Kill by PID
kill <pid>

# Or kill all at once
pkill -f "wnn-dashboard|vite dev|wnn.ram.experiments.worker"
```

## Architecture

```
Frontend (SvelteKit)        Backend (Axum)           Worker (Python)
   :5173                       :3000
     │                           │                         │
     │     HTTP/WebSocket        │                         │
     └──────────────────────────►│◄────── HTTP polling ────┘
                                 │
                            SQLite DB
                          (dashboard.db)
```

- **Frontend** provides the UI for creating/monitoring experiments
- **Backend** stores flows in SQLite, serves the API
- **Worker** polls for queued flows, executes them, updates status
