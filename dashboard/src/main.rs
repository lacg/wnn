//! WNN Architecture Search Dashboard
//!
//! Real-time monitoring for phased architecture search experiments.
//! Uses database polling for real-time updates (no log parsing).

mod api;
mod db;
mod models;

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use api::AppState;
use models::WsMessage;

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

    // WebSocket broadcast channel (for flow status changes)
    let (ws_tx, _) = broadcast::channel::<WsMessage>(100);

    let state = Arc::new(AppState {
        db,
        ws_tx,
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
