use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use std::time::Duration;

use minuspod_jev_proxy::config::{self, Config};
use minuspod_jev_proxy::handlers::{self, AppState};
use minuspod_jev_proxy::health::{self, BackendHealth};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Warn about config this build ignores, before anything is read, so a
    // threshold tuned against an older build cannot look applied.
    config::warn_unknown_env_vars();

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
    // The third slot is optional, so say so explicitly: an empty base_url here
    // is normal, not a misconfiguration.
    if cfg.tertiary.base_url.trim().is_empty() {
        tracing::info!("tertiary backend: not configured (skipped)");
    } else {
        tracing::info!(
            "tertiary backend: {} ({})",
            cfg.tertiary.base_url,
            cfg.tertiary.model,
        );
    }
    tracing::info!(
        "thresholds: recall={} attach={} edge={} (piece={}s context={}s) review={}",
        cfg.recall_threshold,
        cfg.attach_threshold,
        cfg.edge_threshold,
        cfg.edge_piece_secs,
        cfg.edge_context_secs,
        cfg.review_threshold,
    );
    tracing::info!(
        "segments: target={}s max={}s gap={}s per_call={}",
        cfg.segment_target_secs,
        cfg.segment_max_secs,
        cfg.segment_gap_secs,
        cfg.max_segments_per_call,
    );

    let health = BackendHealth::with_docker(cfg.clone());
    health.adopt_if_paused().await;
    tokio::spawn(health::run_scheduler(health.clone()));
    tracing::info!(
        container = %cfg.minuspod_container,
        socket = %cfg.docker_socket,
        billing_dead_secs = cfg.billing_dead_secs,
        client_timeout_secs = cfg.client_timeout_secs,
        "minuspod pause control: all-dead backends pause the container until the next dead window ends",
    );

    let state = AppState {
        cfg,
        client,
        sem,
        health,
    };
    let app = Router::new()
        .route("/v1/models", get(handlers::models))
        .route("/v1/chat/completions", post(handlers::completions))
        .route("/health", get(handlers::health))
        .route("/status", get(handlers::status))
        // Bare paths for base URLs configured without /v1.
        .route("/models", get(handlers::models))
        .route("/chat/completions", post(handlers::completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("minuspod-jev-proxy listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}
