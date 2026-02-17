//! WNN Architecture Search Dashboard
//!
//! Real-time monitoring for phased architecture search experiments.
//! Uses database polling for real-time updates (no log parsing).
//!
//! TLS Support (enabled by default):
//! - Set DASHBOARD_TLS=0 to disable HTTPS (not recommended)
//! - Certificates default to certs/cert.pem and certs/key.pem
//! - Override with DASHBOARD_CERT and DASHBOARD_KEY env vars
//! - HTTP requests are automatically redirected to HTTPS

mod api;
mod db;
mod models;

use std::sync::Arc;
use std::net::SocketAddr;
use std::path::PathBuf;
use anyhow::Result;
use axum::{
    response::Redirect,
    routing::any,
    Router,
    extract::Host,
    http::Uri,
};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
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

    // Initialize database (standard location: db/wnn.db relative to project root)
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:../db/wnn.db?mode=rwc".into());
    let db = db::init_db(&db_url).await?;

    // WebSocket broadcast channel (for flow status changes)
    let (ws_tx, _) = broadcast::channel::<WsMessage>(100);

    let state = Arc::new(AppState {
        db,
        ws_tx,
        current_log_path: tokio::sync::RwLock::new(None),
    });

    // Build router
    let app = api::routes(state)
        .nest_service("/", ServeDir::new("frontend/dist").fallback(ServeFile::new("frontend/dist/index.html")))
        .layer(CorsLayer::permissive());

    // Get port from environment or use default
    let https_port: u16 = std::env::var("DASHBOARD_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let http_port: u16 = std::env::var("DASHBOARD_HTTP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3080);
    let https_addr = SocketAddr::from(([0, 0, 0, 0], https_port));
    let http_addr = SocketAddr::from(([0, 0, 0, 0], http_port));

    // Check if TLS is enabled (default: true)
    let tls_enabled = std::env::var("DASHBOARD_TLS")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);

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

        // Start HTTP redirect server in background
        let redirect_https_port = https_port;
        tokio::spawn(async move {
            let redirect_app = Router::new()
                .fallback(any(move |Host(host): Host, uri: Uri| async move {
                    // Extract hostname without port
                    let host_without_port = host.split(':').next().unwrap_or(&host);
                    let https_uri = if redirect_https_port == 443 {
                        format!("https://{}{}", host_without_port, uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/"))
                    } else {
                        format!("https://{}:{}{}", host_without_port, redirect_https_port, uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/"))
                    };
                    Redirect::permanent(&https_uri)
                }));

            tracing::info!("Starting HTTP redirect server at http://{} -> https://", http_addr);

            if let Ok(listener) = tokio::net::TcpListener::bind(http_addr).await {
                let _ = axum::serve(listener, redirect_app).await;
            } else {
                tracing::warn!("Failed to bind HTTP redirect server on port {}", http_port);
            }
        });

        tracing::info!("Starting dashboard with TLS at https://{}", https_addr);

        axum_server::bind_rustls(https_addr, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        tracing::info!("Starting dashboard at http://{} (TLS disabled)", https_addr);

        let listener = tokio::net::TcpListener::bind(https_addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
