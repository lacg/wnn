# WNN Architecture Search Dashboard

Real-time monitoring dashboard for phased architecture search experiments.

## Architecture

- **Backend**: Rust (axum + tokio + sqlx)
- **Frontend**: SvelteKit
- **Database**: SQLite
- **Real-time**: WebSocket

## Features

- Live CE/accuracy metrics during experiment runs
- Phase timeline visualization
- Comparison table across all phases
- Real-time iteration updates via WebSocket
- Historical experiment browser

## Development

### Backend

```bash
cd dashboard
cargo run
```

### Frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

The frontend dev server proxies API requests to the Rust backend.

## Project Structure

```
dashboard/
├── src/
│   ├── main.rs          # Entry point
│   ├── api/             # HTTP routes
│   ├── db/              # Database operations
│   ├── models/          # Data structures
│   └── parser/          # Log file parser
└── frontend/
    └── src/
        ├── lib/         # Stores, types, utilities
        └── routes/      # SvelteKit pages
```

## TODO

- [ ] File watcher for live log parsing
- [ ] Chart.js integration for CE over time
- [ ] Multiple experiment comparison
- [ ] Export to CSV/JSON
