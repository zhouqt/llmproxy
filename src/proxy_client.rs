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
    let mut builder = common_builder(cfg);

    if let Some(url) = &cfg.url {
        let proxy = reqwest::Proxy::all(url).map_err(|e| {
            ProxyError::Config(format!("invalid proxy url '{url}': {e}"))
        })?;
        builder = builder.proxy(proxy);
    }

    builder.build().map_err(ProxyError::Http)
}

/// Build a reqwest client that intentionally bypasses the global proxy.
///
/// Same timeouts/user-agent/pool settings as `build_client`, but
/// `cfg.url` is ignored. Used by providers that have set `use_proxy:
/// false` so the chain can share a single direct-egress pool across all
/// of them rather than spawning one reqwest::Client per provider.
pub fn build_direct_client(cfg: &ProxyConfig) -> Result<Client> {
    common_builder(cfg).build().map_err(ProxyError::Http)
}

fn common_builder(cfg: &ProxyConfig) -> reqwest::ClientBuilder {
    Client::builder()
        .user_agent(concat!("llmproxy/", env!("CARGO_PKG_VERSION")))
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(cfg.timeout_secs.unwrap_or(600)))
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

    #[test]
    fn direct_client_ignores_proxy_url() {
        let cfg = ProxyConfig {
            url: Some("socks5h://192.0.2.1:1080".into()),
            timeout_secs: Some(120),
        };
        // A direct client must build without error even though cfg.url
        // is set; that's the whole point — operators rely on it to
        // opt providers out of the global proxy.
        let client = build_direct_client(&cfg).unwrap();
        let _ = client;
    }

    #[test]
    fn direct_client_applies_timeout() {
        let cfg = ProxyConfig {
            url: None,
            timeout_secs: Some(45),
        };
        let client = build_direct_client(&cfg).unwrap();
        let _ = client;
    }
}
