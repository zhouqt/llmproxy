use std::time::Duration;

use reqwest::Client;

use crate::config::ProxyConfig;
use crate::error::{ProxyError, Result};

/// Build a reqwest client with optional SOCKS/HTTP proxy.
///
/// The scheme of the proxy URL controls the proxy type:
/// - `socks5://...` and `socks5h://...` → SOCKS5
/// - `http://...` and `https://...` → HTTP CONNECT
pub fn build_client(cfg: &ProxyConfig) -> Result<Client> {
    let mut builder = Client::builder()
        .user_agent(concat!("llmproxy/", env!("CARGO_PKG_VERSION")))
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(cfg.timeout_secs.unwrap_or(600)));

    if let Some(url) = &cfg.url {
        let proxy = reqwest::Proxy::all(url).map_err(|e| {
            ProxyError::Config(format!("invalid proxy url '{url}': {e}"))
        })?;
        builder = builder.proxy(proxy);
    }

    builder.build().map_err(ProxyError::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_client_without_proxy() {
        let client = build_client(&ProxyConfig::default()).unwrap();
        let _ = client;
    }

    #[test]
    fn builds_with_http_proxy() {
        let cfg = ProxyConfig {
            url: Some("http://127.0.0.1:8080".into()),
            timeout_secs: Some(120),
        };
        let client = build_client(&cfg).unwrap();
        let _ = client;
    }

    #[test]
    fn builds_with_socks5_proxy() {
        let cfg = ProxyConfig {
            url: Some("socks5://user:pass@127.0.0.1:1080".into()),
            timeout_secs: None,
        };
        let client = build_client(&cfg).unwrap();
        let _ = client;
    }

    #[test]
    fn invalid_proxy_url_errors() {
        let cfg = ProxyConfig {
            url: Some("http://[::1".into()),
            timeout_secs: None,
        };
        let err = build_client(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid proxy url"), "got: {msg}");
    }
}
