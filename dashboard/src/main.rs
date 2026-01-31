//! WNN Architecture Search Dashboard
//!
//! Real-time monitoring for phased architecture search experiments.
//! Uses database polling for real-time updates (no log parsing).
//!
//! TLS Support:
//! - Set DASHBOARD_TLS=1 to enable HTTPS
//! - Certificates default to certs/cert.pem and certs/key.pem
//! - Override with DASHBOARD_CERT and DASHBOARD_KEY env vars

mod api;
mod db;
mod models;

use std::sync::Arc;
use std::net::SocketAddr;
use std::path::PathBuf;
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

    // Get port from environment or use default
    let port: u16 = std::env::var("DASHBOARD_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    // Check if TLS is enabled
    let tls_enabled = std::env::var("DASHBOARD_TLS")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if tls_enabled {
        // Load TLS certificates
        let cert_path = PathBuf::from(
            std::env::var("DASHBOARD_CERT").unwrap_or_else(|_| "certs/cert.pem".into())
        );
        let key_path = PathBuf::from(
            std::env::var("DASHBOARD_KEY").unwrap_or_else(|_| "certs/key.pem".into())
        );

        tracing::info!("Loading TLS certificates from {:?} and {:?}", cert_path, key_path);

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            &cert_path,
            &key_path,
        )
        .await
        .expect("Failed to load TLS certificates");

        tracing::info!("Starting dashboard with TLS at https://{}", addr);

        axum_server::bind_rustls(addr, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        tracing::info!("Starting dashboard at http://{}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
