//! Library surface for the ad-detection pipeline.
//!
//! The binary in `main.rs` is a thin HTTP wrapper around this. It exists as
//! a separate target so the detection pipeline can be exercised directly from
//! integration tests, without a running server and without an API key.

pub mod classify;
pub mod config;
mod docker;
pub mod handlers;
pub mod health;
pub mod jev;
pub mod openai;
pub mod transcript;