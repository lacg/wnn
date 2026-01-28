//! WNN Architecture Search Dashboard
//!
//! Real-time monitoring for phased architecture search experiments.

mod api;
mod db;
mod models;
mod parser;
mod watcher;

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use api::{AppState, DashboardState};
use models::WsMessage;
use watcher::LogWatcher;

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

    // Start log watcher if LOG_PATH is set
    if let Ok(log_path) = std::env::var("LOG_PATH") {
        let watcher = LogWatcher::new(log_path.clone(), ws_tx.clone(), dashboard.clone());
        watcher.start(false).await?;
        tracing::info!("Watching log file: {}", log_path);
    }

    let state = Arc::new(AppState { db, ws_tx, dashboard });

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
