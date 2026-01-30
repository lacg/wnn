//! WNN Architecture Search Dashboard
//!
//! Real-time monitoring for phased architecture search experiments.

mod api;
mod db;
mod models;
mod parser;
mod watcher;

use std::sync::Arc;
use std::path::PathBuf;
use anyhow::Result;
use tokio::sync::{broadcast, Mutex, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use api::{AppState, DashboardState};
use models::WsMessage;
use watcher::{LogWatcher, WatcherHandle};

/// Find the most recent log file in the logs directory
fn find_most_recent_log() -> Option<PathBuf> {
    // Try both current dir and parent dir (for when running from dashboard/)
    let possible_dirs = vec![
        PathBuf::from("logs"),
        PathBuf::from("../logs"),
    ];

    let logs_dir = possible_dirs.into_iter().find(|d| d.exists())?;

    let mut newest: Option<(PathBuf, std::time::SystemTime)> = None;

    // Walk through logs directory recursively
    fn walk_dir(dir: &PathBuf, newest: &mut Option<(PathBuf, std::time::SystemTime)>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk_dir(&path, newest);
                } else if path.extension().map_or(false, |e| e == "log") {
                    if let Ok(metadata) = path.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if newest.as_ref().map_or(true, |(_, t)| modified > *t) {
                                *newest = Some((path, modified));
                            }
                        }
                    }
                }
            }
        }
    }

    walk_dir(&logs_dir, &mut newest);
    newest.map(|(path, _)| path)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Initialize database
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:dashboard.db?mode=rwc".into());
    let db = db::init_db(&db_url).await?;

    // WebSocket broadcast channel
    let (ws_tx, _) = broadcast::channel::<WsMessage>(100);

    // Shared dashboard state
    let dashboard = Arc::new(RwLock::new(DashboardState::new()));

    // Watcher handle for dynamic log switching
    let watcher_handle: Arc<Mutex<Option<WatcherHandle>>> = Arc::new(Mutex::new(None));

    // Start log watcher: use LOG_PATH env var, or auto-detect most recent log file
    let log_path = std::env::var("LOG_PATH").ok().or_else(|| {
        find_most_recent_log().map(|p| p.to_string_lossy().into_owned())
    });

    if let Some(log_path) = log_path {
        let watcher = LogWatcher::new(log_path.clone(), ws_tx.clone(), dashboard.clone());
        // Start from end of file (don't replay old events) unless it's very recent
        let start_from_end = true;
        let handle = watcher.start(start_from_end).await?;
        *watcher_handle.lock().await = Some(handle);
        tracing::info!("Watching log file: {}", log_path);
    } else {
        tracing::info!("No log file found - waiting for /api/watch call");
    }

    let state = Arc::new(AppState {
        db,
        ws_tx,
        dashboard,
        watcher_handle,
    });

    // Build router
    let app = api::routes(state)
        .nest_service("/", ServeDir::new("frontend/dist"))
        .layer(CorsLayer::permissive());

    let addr = "0.0.0.0:3000";
    tracing::info!("Starting dashboard at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
