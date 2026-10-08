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
        // Connection: close plus a write shutdown. Without both, the engine
        // keeps the socket open and a read-to-end waits until the timeout.
        let req = format!(
            "{method} {path} HTTP/1.0\r\nHost: docker\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
        );
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| format!("write docker socket: {e}"))?;
        stream
            .shutdown()
            .await
            .map_err(|e| format!("shutdown docker socket: {e}"))?;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            let n = stream
                .read(&mut tmp)
                .await
                .map_err(|e| format!("read docker socket: {e}"))?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if response_complete(&buf) {
                break;
            }
        }
        Ok(buf)
    };
    let buf = tokio::time::timeout(Duration::from_secs(10), request)
        .await
        .map_err(|_| "docker socket timed out".to_string())?
        .map_err(|e: String| e)?;
    let status = parse_status(&buf)?;
    let body = decode_body(&buf)?;
    Ok((status, body))
}

/// True once headers and a full body are in hand.
///
/// The engine answers with chunked HTTP/1.1 even when the request is HTTP/1.0.
/// Stopping at the header block leaves `State.Paused` unread.
fn response_complete(buf: &[u8]) -> bool {
    let Some((head, body)) = split_http(buf) else {
        return false;
    };
    if header_is(head, "transfer-encoding", "chunked") {
        return decode_chunked(body).is_ok();
    }
    if let Some(n) = content_length(head) {
        return body.len() >= n;
    }
    false
}

fn split_http(buf: &[u8]) -> Option<(&[u8], &[u8])> {
    let sep = buf.windows(4).position(|w| w == b"\r\n\r\n")?;
    Some((&buf[..sep], &buf[sep + 4..]))
}

fn header_is(head: &[u8], name: &str, needle: &str) -> bool {
    let Ok(text) = std::str::from_utf8(head) else {
        return false;
    };
    text.lines().any(|line| {
        let Some((n, v)) = line.split_once(':') else {
            return false;
        };
        n.eq_ignore_ascii_case(name) && v.to_ascii_lowercase().contains(needle)
    })
}

fn content_length(head: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(head).ok()?;
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

fn parse_status(buf: &[u8]) -> Result<u16, String> {
    let line_end = buf.iter().position(|b| *b == b'\n').unwrap_or(buf.len());
    let line = std::str::from_utf8(&buf[..line_end]).unwrap_or("");
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            format!(
                "bad docker response: {}",
                line.chars().take(80).collect::<String>()
            )
        })
}

fn decode_body(buf: &[u8]) -> Result<String, String> {
    let Some((head, body)) = split_http(buf) else {
        return Err("docker response had no header terminator".into());
    };
    let bytes = if header_is(head, "transfer-encoding", "chunked") {
        decode_chunked(body)?
    } else if let Some(n) = content_length(head) {
        body.get(..n).unwrap_or(body).to_vec()
    } else {
        body.to_vec()
    };
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = body.windows(2).position(|w| w == b"\r\n") else {
            return Err("truncated chunk size".into());
        };
        let line =
            std::str::from_utf8(&body[..nl]).map_err(|_| "chunk size is not utf-8".to_string())?;
        let size_hex = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| format!("bad chunk size {size_hex}"))?;
        body = body.get(nl + 2..).ok_or("truncated chunk")?;
        if size == 0 {
            return Ok(out);
        }
        if body.len() < size + 2 {
            return Err("truncated chunk data".into());
        }
        out.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }
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

    /// Answers once and then holds the socket open. The client must finish
    /// from Content-Length; waiting for EOF is the hang this guards against.
    async fn serve_once(
        status: u16,
        body: &str,
    ) -> (
        String,
        tokio::sync::oneshot::Receiver<String>,
        tokio::task::JoinHandle<()>,
    ) {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("jev-docker-{}-{n}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let path_s = path.to_string_lossy().to_string();
        let body = body.to_string();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 2048];
            let n = sock.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let _ = tx.send(req);
            let resp = format!(
                "HTTP/1.0 {status} X\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            std::future::pending::<()>().await;
        });
        (path_s, rx, handle)
    }

    #[tokio::test]
    async fn pause_treats_204_and_409_as_success() {
        let (path, rx, srv) = serve_once(204, "").await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        ctl.pause().await.expect("204");
        let req = rx.await.unwrap();
        assert!(req.contains("POST /v1.41/containers/minuspod/pause"));
        assert!(req.contains("Connection: close"));
        srv.abort();
        let _ = std::fs::remove_file(&path);

        let (path, rx, srv) = serve_once(409, "already paused").await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        ctl.pause().await.expect("409 is already paused");
        let _ = rx.await;
        srv.abort();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chunked_body_decodes_to_the_paused_flag() {
        let payload = r#"{"State":{"Paused":true,"Status":"paused"}}"#;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{payload}\r\n0\r\n\r\n",
            payload.len()
        );
        assert!(response_complete(raw.as_bytes()));
        let body = decode_body(raw.as_bytes()).unwrap();
        assert_eq!(paused_from_body(&body), Some(true));
        let partial = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello";
        assert!(!response_complete(partial));
    }

    #[tokio::test]
    async fn inspect_reads_paused_flag() {
        let (path, rx, srv) =
            serve_once(200, r#"{"State":{"Paused":true,"Status":"paused"}}"#).await;
        let ctl = PauseCtl::docker(&path, "minuspod");
        assert!(ctl.inspect_paused().await.unwrap());
        let _ = rx.await;
        srv.abort();
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn inspect_reads_a_chunked_body_without_waiting_for_eof() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("jev-docker-chunk-{n}.sock"));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let path_s = path.to_string_lossy().to_string();
        let srv = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 2048];
            let _ = sock.read(&mut buf).await.unwrap();
            let payload = r#"{"State":{"Paused":false}}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{payload}\r\n0\r\n\r\n",
                payload.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            std::future::pending::<()>().await;
        });
        let ctl = PauseCtl::docker(&path_s, "minuspod");
        assert!(!ctl.inspect_paused().await.unwrap());
        srv.abort();
        let _ = std::fs::remove_file(&path);
    }
}
