# Experiments Manager

The Experiments Manager transforms the monolithic phased search into a composable experiment system with first-class support for flows, experiments, and checkpoints.

## Overview

```
┌─────────────────────────────────────────────────────┐
│                    CLI (wnn-exp)                     │
│  flow create/list/show | checkpoint list/delete     │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│              Rust Backend (Axum)                     │
│  /api/flows | /api/checkpoints | /api/experiments   │
│  SQLite: metadata + checkpoint paths                │
└─────────────────────┬───────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼───────────────────────────────┐
│            Svelte Frontend                           │
│  Flow Builder | Checkpoint Browser | History View   │
└─────────────────────────────────────────────────────┘
```

## Concepts

### Flows

A **Flow** is a sequence of experiments, like the current 6-phase pass:

1. Phase 1a: GA Neurons
2. Phase 1b: TS Neurons
3. Phase 2a: GA Bits
4. Phase 2b: TS Bits
5. Phase 3a: GA Connections
6. Phase 3b: TS Connections

Flows can be:
- Created from templates (e.g., `standard-6-phase`)
- Seeded from existing checkpoints
- Monitored in real-time via WebSocket

### Experiments

An **Experiment** is a single GA or TS optimization run. Each experiment:
- Optimizes one dimension (neurons, bits, or connections)
- Produces checkpoints during and after optimization
- Reports progress via WebSocket

### Checkpoints

A **Checkpoint** is a saved genome state with metadata:
- File path to `.json.gz` file
- Fitness (CE loss) and accuracy
- Genome statistics (clusters, neurons, connections)
- Reference count for safe deletion

## CLI Usage

### Installation

The CLI is installed automatically with the package:

```bash
pip install -e .
wnn-exp --help
```

### Flow Commands

```bash
# List all flows
wnn-exp flow list

# List running flows only
wnn-exp flow list --status running

# Show flow details
wnn-exp flow show 1

# Create a new flow
wnn-exp flow create --name "Pass 2" --patience 15

# Create flow with custom settings
wnn-exp flow create \
  --name "High Patience Run" \
  --ga-gens 500 \
  --ts-iters 500 \
  --patience 20 \
  --phase-order bits_first

# Delete a flow
wnn-exp flow delete 1
```

### Checkpoint Commands

```bash
# List all checkpoints
wnn-exp checkpoint list

# List final checkpoints only
wnn-exp checkpoint list --final-only

# Show checkpoint details
wnn-exp checkpoint show 5

# Create flow seeded from checkpoint
wnn-exp checkpoint seed 5 --flow-name "Pass 3 (seeded)"

# Delete checkpoint (with confirmation)
wnn-exp checkpoint delete 5

# Force delete (even if referenced)
wnn-exp checkpoint delete 5 --force
```

### Status Command

```bash
# Check dashboard connection
wnn-exp status
```

## Python API

### Using Flow and Experiment

```python
from pathlib import Path
from wnn.ram.experiments import (
    Flow,
    FlowConfig,
    DashboardClient,
)

# Create flow configuration
config = FlowConfig.standard_6_phase(
    name="Pass 1",
    patience=10,
    ga_generations=250,
    ts_iterations=250,
    tier_config=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
    optimize_tier0_only=True,
)

# Optional: Connect to dashboard for tracking
client = DashboardClient()

# Create and run flow
flow = Flow(
    config=config,
    evaluator=cached_evaluator,  # Your CachedEvaluator instance
    logger=print,
    checkpoint_dir=Path("checkpoints/pass1"),
    dashboard_client=client,
)

result = flow.run()
print(f"Final CE: {result.final_fitness:.4f}")
print(f"Final Accuracy: {result.final_accuracy:.2%}")
```

### Using DashboardClient Directly

```python
from wnn.ram.experiments import DashboardClient, DashboardClientConfig

# Create client
config = DashboardClientConfig(base_url="http://localhost:3000")
client = DashboardClient(config)

# Check connection
if client.ping():
    print("Dashboard is running")

# List flows
flows = client.list_flows(status="completed")
for flow in flows:
    print(f"{flow['id']}: {flow['name']} - {flow['status']}")

# List checkpoints
checkpoints = client.list_checkpoints(is_final=True)
for ckpt in checkpoints:
    print(f"{ckpt['id']}: {ckpt['name']} (CE: {ckpt['final_fitness']:.4f})")
```

## REST API

### Flows

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/flows` | List flows |
| POST | `/api/flows` | Create flow |
| GET | `/api/flows/:id` | Get flow |
| PATCH | `/api/flows/:id` | Update flow |
| DELETE | `/api/flows/:id` | Delete flow |
| GET | `/api/flows/:id/experiments` | List flow experiments |

### Checkpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/checkpoints` | List checkpoints |
| POST | `/api/checkpoints` | Create checkpoint |
| GET | `/api/checkpoints/:id` | Get checkpoint |
| DELETE | `/api/checkpoints/:id` | Delete checkpoint |
| GET | `/api/checkpoints/:id/download` | Download checkpoint file |

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/experiments` | List experiments |
| POST | `/api/experiments` | Create experiment |
| GET | `/api/experiments/:id` | Get experiment |

## WebSocket Events

Connect to `/ws` to receive real-time updates:

```typescript
// Message types
type WsMessage =
  | { type: "Snapshot", data: DashboardSnapshot }
  | { type: "IterationUpdate", data: Iteration }
  | { type: "PhaseStarted", data: Phase }
  | { type: "PhaseCompleted", data: { phase: Phase, result: PhaseResult } }
  | { type: "FlowStarted", data: Flow }
  | { type: "FlowCompleted", data: Flow }
  | { type: "FlowFailed", data: { flow: Flow, error: string } }
  | { type: "CheckpointCreated", data: Checkpoint }
  | { type: "CheckpointDeleted", data: { id: number } }
```

## Database Schema

```sql
-- Flows (sequences of experiments)
CREATE TABLE flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    seed_checkpoint_id INTEGER REFERENCES checkpoints(id)
);

-- Checkpoints (first-class, with file paths)
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    created_at TEXT NOT NULL,
    final_fitness REAL,
    final_accuracy REAL,
    iterations_run INTEGER,
    genome_stats_json TEXT,
    is_final BOOLEAN DEFAULT FALSE,
    reference_count INTEGER DEFAULT 0
);
```

## Migration

To migrate existing checkpoint files to the database:

```bash
# Dry run to see what would be migrated
python scripts/migrate_checkpoints.py --dry-run

# Actually migrate
python scripts/migrate_checkpoints.py

# Custom checkpoint directory
python scripts/migrate_checkpoints.py --checkpoint-dir /path/to/checkpoints
```

## Frontend Pages

| Route | Description |
|-------|-------------|
| `/flows` | Flow list with status badges |
| `/flows/new` | Flow builder with configuration options |
| `/flows/[id]` | Flow detail with experiment list |
| `/checkpoints` | Checkpoint browser with download/delete |
| `/experiments` | Historical experiment browser |

## Architecture

### Files

| File | Purpose |
|------|---------|
| `dashboard/src/db/mod.rs` | Database schema and queries |
| `dashboard/src/models/mod.rs` | Rust data models |
| `dashboard/src/api/mod.rs` | REST API handlers |
| `src/wnn/ram/experiments/dashboard_client.py` | Python HTTP client |
| `src/wnn/ram/experiments/experiment.py` | Single experiment runner |
| `src/wnn/ram/experiments/flow.py` | Flow orchestration |
| `src/wnn/ram/experiments/cli.py` | CLI tool (wnn-exp) |
| `scripts/migrate_checkpoints.py` | Migration script |

### Data Flow

```
Flow.run()
  │
  ├─▶ dashboard.create_flow() → flow_id
  ├─▶ dashboard.flow_started(flow_id)
  │
  └─▶ for each experiment:
        ├─▶ dashboard.create_experiment() → experiment_id
        ├─▶ Experiment.run()
        │     └─▶ dashboard.checkpoint_created(experiment_id, ...)
        │
  └─▶ dashboard.flow_completed(flow_id)
```

## Backwards Compatibility

The existing `run_coarse_fine_search.py` continues to work unchanged. The new system is opt-in:
- Use `Flow` class for dashboard integration
- Use `PhasedSearchRunner` for standalone operation

Both produce the same checkpoint files and can be used interchangeably.
