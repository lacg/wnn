//! Log file watcher using notify crate
//!
//! Watches log files for changes and broadcasts parsed events via WebSocket.
//! Supports dynamic switching to a new log file.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher, EventKind};
use tokio::sync::{broadcast, mpsc, oneshot, RwLock};
use tracing::{info, warn, error};

use crate::api::DashboardState;
use crate::models::*;
use crate::parser::{parse_line, LogEvent};

/// Handle to stop a running watcher
pub struct WatcherHandle {
    stop_tx: oneshot::Sender<()>,
}

/// Watches a log file and broadcasts parsed events
pub struct LogWatcher {
    log_path: String,
    ws_tx: broadcast::Sender<WsMessage>,
    dashboard: Arc<RwLock<DashboardState>>,
}

/// State for tracking parsed events
struct ParserState {
    current_phase_id: i64,
    iteration_id: i64,
    current_phase_name: String,
    current_phase_type: PhaseType,
    /// Collecting final summary rows
    in_final_results: bool,
    summary_rows: Vec<PhaseSummaryRow>,
    last_phase_name: String,
    /// Track genome progress within a generation
    current_generation: Option<i32>,
    gen_best_ce: f64,
    gen_best_acc: f64,
    gen_genome_count: i32,
    gen_start_time: std::time::Instant,
    /// Last generation we emitted an iteration for (to avoid duplicates)
    last_emitted_generation: Option<i32>,
}

impl ParserState {
    fn new() -> Self {
        Self {
            current_phase_id: 0,
            iteration_id: 0,
            current_phase_name: String::new(),
            current_phase_type: PhaseType::GaNeurons,
            in_final_results: false,
            summary_rows: Vec::new(),
            last_phase_name: String::new(),
            current_generation: None,
            gen_best_ce: f64::INFINITY,
            gen_best_acc: 0.0,
            gen_genome_count: 0,
            gen_start_time: std::time::Instant::now(),
            last_emitted_generation: None,
        }
    }

    /// Reset generation tracking for a new generation
    fn start_new_generation(&mut self, generation: i32) {
        self.current_generation = Some(generation);
        self.gen_best_ce = f64::INFINITY;
        self.gen_best_acc = 0.0;
        self.gen_genome_count = 0;
        self.gen_start_time = std::time::Instant::now();
    }

    /// Update with a genome's metrics
    fn update_genome(&mut self, ce: f64, accuracy: f64) {
        self.gen_genome_count += 1;
        if ce < self.gen_best_ce {
            self.gen_best_ce = ce;
        }
        if accuracy > self.gen_best_acc {
            self.gen_best_acc = accuracy;
        }
    }
}

impl WatcherHandle {
    /// Stop the watcher
    pub fn stop(self) {
        let _ = self.stop_tx.send(());
    }
}

impl LogWatcher {
    pub fn new(
        log_path: String,
        ws_tx: broadcast::Sender<WsMessage>,
        dashboard: Arc<RwLock<DashboardState>>,
    ) -> Self {
        Self { log_path, ws_tx, dashboard }
    }

    /// Start watching the log file, returns a handle to stop the watcher
    pub async fn start(self, start_from_end: bool) -> Result<WatcherHandle> {
        let (stop_tx, stop_rx) = oneshot::channel::<()>();
        let path = self.log_path.clone();
        let ws_tx = self.ws_tx.clone();
        let dashboard = self.dashboard.clone();

        // Channel for notify events -> async task
        let (notify_tx, mut notify_rx) = mpsc::channel::<()>(10);

        // Set up file watcher
        let notify_tx_clone = notify_tx.clone();
        let path_clone = path.clone();

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<notify::Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        if matches!(event.kind, EventKind::Modify(_)) {
                            let _ = notify_tx_clone.blocking_send(());
                        }
                    }
                    Err(e) => warn!("Watch error: {:?}", e),
                }
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        )?;

        // Watch the parent directory
        let parent = Path::new(&path_clone).parent().unwrap_or(Path::new("."));
        watcher.watch(parent, RecursiveMode::NonRecursive)?;

        info!("Watching log file: {}", path);

        // Spawn the reader task with stop signal support
        tokio::spawn(async move {
            let mut stop_rx = stop_rx;

            let mut file = match File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    error!("Failed to open log file: {}", e);
                    return;
                }
            };

            if start_from_end {
                if let Err(e) = file.seek(SeekFrom::End(0)) {
                    warn!("Failed to seek to end: {}", e);
                }
            }

            let mut reader = BufReader::new(file);
            let mut line_buf = String::new();
            let mut state = ParserState::new();

            // === PHASE 1: Parse existing content ===
            if !start_from_end {
                info!("Parsing existing log content...");
                let mut lines_parsed = 0;
                let mut events_found = 0;

                loop {
                    line_buf.clear();
                    match reader.read_line(&mut line_buf) {
                        Ok(0) => break,
                        Ok(_) => {
                            lines_parsed += 1;
                            let line = line_buf.trim_end();
                            let messages = process_line(line, &mut state);
                            for msg in messages {
                                events_found += 1;
                                // Update shared dashboard state
                                {
                                    let mut dash = dashboard.write().await;
                                    dash.update_from_message(&msg);
                                }
                                // Broadcast (may have no subscribers yet, that's ok)
                                let _ = ws_tx.send(msg);
                            }
                        }
                        Err(e) => {
                            warn!("Error reading line: {}", e);
                            break;
                        }
                    }
                }

                info!(
                    "Historical parsing complete: {} lines, {} events, {} phases",
                    lines_parsed, events_found, state.current_phase_id
                );
            }

            // === PHASE 2: Watch for new content ===
            info!("Now watching for new log entries...");

            loop {
                tokio::select! {
                    // Stop signal received
                    _ = &mut stop_rx => {
                        info!("Watcher stopped by request");
                        break;
                    }
                    // File change notification
                    result = notify_rx.recv() => {
                        if result.is_none() {
                            break;
                        }

                        loop {
                            line_buf.clear();
                            match reader.read_line(&mut line_buf) {
                                Ok(0) => break,
                                Ok(_) => {
                                    let line = line_buf.trim_end();
                                    let messages = process_line(line, &mut state);
                                    for msg in messages {
                                        // Update shared dashboard state
                                        {
                                            let mut dash = dashboard.write().await;
                                            dash.update_from_message(&msg);
                                        }
                                        let _ = ws_tx.send(msg);
                                    }
                                }
                                Err(e) => {
                                    warn!("Error reading line: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            drop(watcher);
        });

        Ok(WatcherHandle { stop_tx })
    }
}

/// Process a single log line and return WebSocket messages if relevant
fn process_line(line: &str, state: &mut ParserState) -> Vec<WsMessage> {
    let Some((_time, event)) = parse_line(line) else {
        return vec![];
    };

    match event {
        LogEvent::PhaseStart { name, phase_type } => {
            let mut messages = Vec::new();

            // Complete the previous phase if there was one
            if state.current_phase_id > 0 {
                let completed_phase = Phase {
                    id: state.current_phase_id,
                    experiment_id: 1,
                    name: state.current_phase_name.clone(),
                    phase_type: state.current_phase_type.clone(),
                    started_at: chrono::Utc::now(), // approximate
                    ended_at: Some(chrono::Utc::now()),
                    status: PhaseStatus::Completed,
                };
                // Create a placeholder result (actual metrics not available from log parsing)
                let result = PhaseResult {
                    id: state.current_phase_id,
                    phase_id: state.current_phase_id,
                    metric_type: MetricType::TopKMean,
                    ce: 0.0,
                    accuracy: 0.0,
                    memory_bytes: 0,
                    improvement_pct: 0.0,
                };
                messages.push(WsMessage::PhaseCompleted { phase: completed_phase, result });
            }

            state.current_phase_id += 1;
            state.current_phase_name = name.clone();
            state.current_phase_type = phase_type.clone();
            state.iteration_id = 0;

            let phase = Phase {
                id: state.current_phase_id,
                experiment_id: 1,
                name,
                phase_type,
                started_at: chrono::Utc::now(),
                ended_at: None,
                status: PhaseStatus::Running,
            };
            messages.push(WsMessage::PhaseStarted(phase));
            messages
        }
        LogEvent::GaIteration { generation, best, avg, elapsed, .. } => {
            // Skip if we already emitted an iteration from GenomeProgress for this generation
            // (GenomeProgress includes accuracy, GaIteration does not)
            if state.last_emitted_generation == Some(generation) {
                // Already handled by GenomeProgress, skip duplicate
                return vec![];
            }
            state.iteration_id += 1;
            state.last_emitted_generation = Some(generation);
            let iteration = Iteration {
                id: state.iteration_id,
                phase_id: state.current_phase_id,
                iteration_num: generation,
                best_ce: best,
                avg_ce: Some(avg),
                best_accuracy: None,
                elapsed_secs: elapsed,
                timestamp: chrono::Utc::now(),
            };
            vec![WsMessage::IterationUpdate(iteration)]
        }
        LogEvent::TsIteration { iter, best_harmonic_ce, best_harmonic_acc, best_ce, elapsed, .. } => {
            state.iteration_id += 1;
            let iteration = Iteration {
                id: state.iteration_id,
                phase_id: state.current_phase_id,
                iteration_num: iter,
                best_ce: best_harmonic_ce,
                avg_ce: Some(best_ce),
                best_accuracy: Some(best_harmonic_acc),
                elapsed_secs: elapsed,
                timestamp: chrono::Utc::now(),
            };
            vec![WsMessage::IterationUpdate(iteration)]
        }
        LogEvent::GenomeProgress { generation, total_gens: _, genome_idx, total_genomes, is_elite: _, ce, accuracy } => {
            let mut messages = Vec::new();

            // Check if this is a new generation
            if state.current_generation != Some(generation) {
                // Emit iteration update for previous generation if we have data
                if state.current_generation.is_some() && state.gen_genome_count > 0 {
                    state.iteration_id += 1;
                    state.last_emitted_generation = state.current_generation;
                    let elapsed = state.gen_start_time.elapsed().as_secs_f64();
                    let iteration = Iteration {
                        id: state.iteration_id,
                        phase_id: state.current_phase_id,
                        iteration_num: state.current_generation.unwrap(),
                        best_ce: state.gen_best_ce,
                        avg_ce: None, // No average in genome-by-genome format
                        best_accuracy: Some(state.gen_best_acc),
                        elapsed_secs: elapsed,
                        timestamp: chrono::Utc::now(),
                    };
                    messages.push(WsMessage::IterationUpdate(iteration));
                }
                // Start tracking the new generation
                state.start_new_generation(generation);
            }

            // Update generation tracking with this genome's metrics
            state.update_genome(ce, accuracy);

            // If this is the last genome in the generation, emit the iteration update now
            if genome_idx == total_genomes {
                state.iteration_id += 1;
                state.last_emitted_generation = Some(generation);
                let elapsed = state.gen_start_time.elapsed().as_secs_f64();
                let iteration = Iteration {
                    id: state.iteration_id,
                    phase_id: state.current_phase_id,
                    iteration_num: generation,
                    best_ce: state.gen_best_ce,
                    avg_ce: None,
                    best_accuracy: Some(state.gen_best_acc),
                    elapsed_secs: elapsed,
                    timestamp: chrono::Utc::now(),
                };
                messages.push(WsMessage::IterationUpdate(iteration));
                // Reset for next generation
                state.current_generation = None;
            }

            messages
        }
        LogEvent::HealthCheck { k, ce, accuracy } => {
            let health = HealthCheck {
                id: 0,
                phase_id: state.current_phase_id,
                top_k_ce: ce,
                top_k_accuracy: accuracy,
                best_ce: 0.0,
                best_ce_accuracy: 0.0,
                best_acc_ce: 0.0,
                best_acc_accuracy: 0.0,
                k,
                timestamp: chrono::Utc::now(),
            };
            vec![WsMessage::HealthCheck(health)]
        }
        LogEvent::FinalResults => {
            // Start collecting summary rows
            state.in_final_results = true;
            state.summary_rows.clear();
            state.last_phase_name.clear();
            vec![]
        }
        LogEvent::PhaseSummaryRow { phase_name, metric_type, ce, ppl, accuracy } => {
            if state.in_final_results {
                // Track the phase name for continuation rows
                let actual_phase_name = if phase_name.is_empty() {
                    state.last_phase_name.clone()
                } else {
                    state.last_phase_name = phase_name.clone();
                    phase_name
                };

                state.summary_rows.push(PhaseSummaryRow {
                    phase_name: actual_phase_name,
                    metric_type,
                    ce,
                    ppl,
                    accuracy,
                });

                // Check if we have all rows (Baseline + 6 phases × 3 metrics = 21 rows)
                // Send summary when we get enough rows
                if state.summary_rows.len() >= 21 {
                    let summary = PhaseSummary {
                        rows: state.summary_rows.clone(),
                        timestamp: chrono::Utc::now(),
                    };
                    state.in_final_results = false;
                    state.summary_rows.clear();
                    return vec![WsMessage::PhaseSummary(summary)];
                }
            }
            vec![]
        }
        _ => vec![],
    }
}
