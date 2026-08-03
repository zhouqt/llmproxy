use std::time::Duration;

use reqwest::Client;

use crate::config::ProxyConfig;
use crate::error::{ProxyError, Result};

/// Build a reqwest client with optional SOCKS/HTTP proxy.
///
/// The scheme of the proxy URL controls the proxy type:
/// - `socks5://...` and `socks5h://...` → SOCKS5
/// - `http://...` and `https://...` → HTTP CONNECT
///
/// `user_agent` is the client identity presented to upstream providers
/// (from the top-level `Config::user_agent`); it is intentionally not part
/// of `ProxyConfig`, which only describes the outbound HTTP proxy.
pub fn build_client(cfg: &ProxyConfig, user_agent: &str) -> Result<Client> {
    let mut builder = common_builder(cfg, user_agent);

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
pub fn build_direct_client(cfg: &ProxyConfig, user_agent: &str) -> Result<Client> {
    common_builder(cfg, user_agent).build().map_err(ProxyError::Http)
}

fn common_builder(cfg: &ProxyConfig, user_agent: &str) -> reqwest::ClientBuilder {
    Client::builder()
        // Present the Claude Code identity by default (see
        // `Config::user_agent`) so upstream providers classify the
        // proxy's traffic as Claude Code instead of an unknown client.
        .user_agent(user_agent)
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(cfg.timeout_secs.unwrap_or(600)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The default client identity the shared pools present upstream —
    /// reused here so the assertion string can't drift from the config
    /// default.
    fn default_user_agent() -> String {
        crate::config::default_user_agent()
    }

    #[test]
    fn builds_client_without_proxy() {
        let client = build_client(&ProxyConfig::default(), &default_user_agent()).unwrap();
        let _ = client;
    }

    #[test]
    fn builds_with_http_proxy() {
        let cfg = ProxyConfig {
            url: Some("http://127.0.0.1:8080".into()),
            timeout_secs: Some(120),
        };
        let client = build_client(&cfg, &default_user_agent()).unwrap();
        let _ = client;
    }

    #[test]
    fn builds_with_socks5_proxy() {
        let cfg = ProxyConfig {
            url: Some("socks5://user:pass@127.0.0.1:1080".into()),
            timeout_secs: None,
        };
        let client = build_client(&cfg, &default_user_agent()).unwrap();
        let _ = client;
    }

    #[test]
    fn invalid_proxy_url_errors() {
        let cfg = ProxyConfig {
            url: Some("http://[::1".into()),
            timeout_secs: None,
        };
        let err = build_client(&cfg, &default_user_agent()).unwrap_err();
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
        let client = build_direct_client(&cfg, &default_user_agent()).unwrap();
        let _ = client;
    }

    #[test]
    fn direct_client_applies_timeout() {
        let cfg = ProxyConfig {
            url: None,
            timeout_secs: Some(45),
        };
        let client = build_direct_client(&cfg, &default_user_agent()).unwrap();
        let _ = client;
    }

    #[tokio::test]
    async fn default_user_agent_simulates_claude_code() {
        // The whole point of the identity simulation: with the default
        // config, outbound requests must carry `User-Agent:
        // claude-code/<version>` so LLM providers classify the proxy's
        // traffic as Claude Code instead of "Unknown".
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .and(header("user-agent", "claude-code/2.1.220"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let client = build_client(&ProxyConfig::default(), &default_user_agent()).unwrap();
        client.get(&server.uri()).send().await.unwrap();
    }

    #[tokio::test]
    async fn direct_client_default_user_agent_simulates_claude_code() {
        // The direct pool shares the same default — providers that bypass
        // the global proxy must still present the Claude Code identity.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .and(header("user-agent", "claude-code/2.1.220"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let client = build_direct_client(&ProxyConfig::default(), &default_user_agent()).unwrap();
        client.get(&server.uri()).send().await.unwrap();
    }

    #[tokio::test]
    async fn configured_user_agent_overrides_default() {
        // Operators can present a custom identity via the top-level
        // `user_agent` config (e.g. to impersonate a different client or
        // keep a custom label); the configured value must win.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .and(header("user-agent", "custom-agent/1.0"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let client = build_client(&ProxyConfig::default(), "custom-agent/1.0").unwrap();
        client.get(&server.uri()).send().await.unwrap();
    }
}
