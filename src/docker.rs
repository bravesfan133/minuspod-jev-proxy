//! Minimal Docker Engine client over the unix socket.
//!
//! The proxy only pauses, unpauses, and inspects one named container. It does
//! not create containers or pass through arbitrary API calls.

use serde_json::Value;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const API: &str = "/v1.41";

#[derive(Clone)]
pub enum PauseCtl {
    /// Talk to the engine on `socket` about `container` only.
    Docker { socket: String, container: String },
    /// Tests: pause and unpause succeed without a daemon.
    Record,
}

impl PauseCtl {
    pub fn docker(socket: impl Into<String>, container: impl Into<String>) -> Self {
        Self::Docker {
            socket: socket.into(),
            container: container.into(),
        }
    }

    pub async fn pause(&self) -> Result<(), String> {
        self.action("pause").await
    }

    pub async fn unpause(&self) -> Result<(), String> {
        self.action("unpause").await
    }

    /// `Ok(true)` when the container is frozen. `Err` when this controller
    /// cannot see the daemon; the caller then uses the last command it sent.
    pub async fn inspect_paused(&self) -> Result<bool, String> {
        let Self::Docker { socket, container } = self else {
            return Err("pause controller is not connected to Docker".into());
        };
        let path = format!("{API}/containers/{}/json", encode(container));
        let (status, body) = docker_req(socket, "GET", &path).await?;
        if status == 404 {
            return Err(format!("container {container} not found"));
        }
        if !(200..300).contains(&status) {
            return Err(format!("docker inspect HTTP {status}"));
        }
        paused_from_body(&body).ok_or_else(|| "docker inspect had no State.Paused".into())
    }

    async fn action(&self, action: &str) -> Result<(), String> {
        let Self::Docker { socket, container } = self else {
            return Ok(());
        };
        if container.trim().is_empty() {
            return Err("JEV_MINUSPOD_CONTAINER is empty".into());
        }
        let path = format!("{API}/containers/{}/{action}", encode(container));
        let (status, body) = docker_req(socket, "POST", &path).await?;
        // 204: transitioned. 409: already in the requested state.
        if status == 204 || status == 409 || (200..300).contains(&status) {
            return Ok(());
        }
        let head: String = body.chars().take(200).collect();
        Err(format!("docker {action} {container} HTTP {status}: {head}"))
    }
}

fn encode(name: &str) -> String {
    let mut out = String::new();
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

async fn docker_req(socket: &str, method: &str, path: &str) -> Result<(u16, String), String> {
    if socket.trim().is_empty() {
        return Err("JEV_DOCKER_SOCKET is empty".into());
    }
    let request = async {
        let mut stream = UnixStream::connect(socket)
            .await
            .map_err(|e| format!("connect {socket}: {e}"))?;
        let req = format!("{method} {path} HTTP/1.0\r\nHost: docker\r\n\r\n");
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| format!("write docker socket: {e}"))?;
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| format!("read docker socket: {e}"))?;
        Ok(buf)
    };
    let buf = tokio::time::timeout(Duration::from_secs(3), request)
        .await
        .map_err(|_| "docker socket timed out".to_string())?
        .map_err(|e: String| e)?;
    let text = String::from_utf8_lossy(&buf).to_string();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad docker response: {}", text.chars().take(80).collect::<String>()))?;
    let body = text
        .split_once("\r\n\r\n")
        .or_else(|| text.split_once("\n\n"))
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok((status, body))
}

fn paused_from_body(body: &str) -> Option<bool> {
    let parse = |raw: &str| -> Option<bool> {
        let v: Value = serde_json::from_str(raw).ok()?;
        v.pointer("/State/Paused")?.as_bool()
    };
    if let Some(v) = parse(body) {
        return Some(v);
    }
    let start = body.find('{')?;
    parse(&body[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    async fn serve_once(status: u16, body: &str) -> (String, tokio::task::JoinHandle<String>) {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("jev-docker-{}-{n}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let path_s = path.to_string_lossy().to_string();
        let body = body.to_string();
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let n = sock.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let resp = format!("HTTP/1.0 {status} X\r\nContent-Length: {}\r\n\r\n{body}", body.len());
            sock.write_all(resp.as_bytes()).await.unwrap();
            let _ = path;
            req
        });
        (path_s, handle)
    }

    #[tokio::test]
    async fn pause_treats_204_and_409_as_success() {
        let (path, srv) = serve_once(204, "").await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        ctl.pause().await.expect("204");
        let req = srv.await.unwrap();
        assert!(req.contains("POST /v1.41/containers/minuspod/pause"));
        let _ = std::fs::remove_file(&path);

        let (path, srv) = serve_once(409, "already paused").await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        ctl.pause().await.expect("409 is already paused");
        let _ = srv.await;
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn inspect_reads_paused_flag() {
        let (path, srv) = serve_once(200, r#"{"State":{"Paused":true,"Status":"paused"}}"#).await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        assert!(ctl.inspect_paused().await.unwrap());
        let _ = srv.await;
        let _ = std::fs::remove_file(&path);
    }
}
