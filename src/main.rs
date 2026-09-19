mod classify;
mod config;
mod handlers;
mod jev;
mod openai;
mod transcript;

use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::handlers::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cfg = Arc::new(Config::from_env());
    let port = cfg.port;

    // rustls with bundled WebPKI roots: TLS works in scratch/distroless images.
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("http client");

    // Gate concurrent Jev calls: MinusPod fires all windows in parallel and
    // free-tier backends burst-limit aggressively.
    let sem = Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent));
    tracing::info!(
        "jev backends: primary {} ({}), secondary {} ({}), max_concurrent={}",
        cfg.primary.base_url,
        cfg.primary.model,
        cfg.secondary.base_url,
        cfg.secondary.model,
        cfg.max_concurrent,
    );

    let state = AppState { cfg, client, sem };
    let app = Router::new()
        .route("/v1/models", get(handlers::models))
        .route("/v1/chat/completions", post(handlers::completions))
        .route("/health", get(handlers::health))
        // Bare paths for base URLs configured without /v1.
        .route("/models", get(handlers::models))
        .route("/chat/completions", post(handlers::completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("minuspod-jev-proxy listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}
