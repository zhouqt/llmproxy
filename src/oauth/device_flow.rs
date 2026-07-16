//! GitHub OAuth Device Flow (RFC 8628).
//!
//! Reference: copilot-api-py/src/services/github/{get_device_code,poll_access_token}.py

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

use crate::error::{ProxyError, Result};
use crate::oauth::{GITHUB_BASE_URL, GITHUB_CLIENT_ID, GITHUB_SCOPES};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    pub interval: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct AccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

pub async fn request_device_code(http: &reqwest::Client) -> Result<DeviceCodeResponse> {
    request_device_code_at(http, github_base_url()).await
}

fn github_base_url() -> &'static str {
    #[cfg(test)]
    {
        if let Ok(value) = std::env::var("LLMPROXY_TEST_GITHUB_BASE_URL") {
            return Box::leak(value.into_boxed_str());
        }
    }
    GITHUB_BASE_URL
}

async fn request_device_code_at(
    http: &reqwest::Client,
    base_url: &str,
) -> Result<DeviceCodeResponse> {
    let url = format!("{}/login/device/code", base_url.trim_end_matches('/'));
    let resp = http
        .post(&url)
        .header("accept", "application/json")
        .header("content-type", "application/json")
        .json(&serde_json::json!({
            "client_id": GITHUB_CLIENT_ID,
            "scope": GITHUB_SCOPES,
        }))
        .send()
        .await?;

    let status = resp.status();
    let body: Value = resp.json().await?;
    if !status.is_success() {
        return Err(ProxyError::Other(anyhow::anyhow!(
            "device code request failed: {status} {body}"
        )));
    }
    serde_json::from_value(body).map_err(ProxyError::Json)
}

pub async fn poll_access_token(
    http: &reqwest::Client,
    device_code: &str,
) -> Result<Option<String>> {
    poll_access_token_at(http, github_base_url(), device_code).await
}

async fn poll_access_token_at(
    http: &reqwest::Client,
    base_url: &str,
    device_code: &str,
) -> Result<Option<String>> {
    let url = format!(
        "{}/login/oauth/access_token",
        base_url.trim_end_matches('/')
    );
    let resp = http
        .post(&url)
        .header("accept", "application/json")
        .header("content-type", "application/json")
        .json(&serde_json::json!({
            "client_id": GITHUB_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }))
        .send()
        .await?;
    let status = resp.status();
    let value: Value = resp.json().await?;
    if !status.is_success() {
        return Err(ProxyError::Other(anyhow::anyhow!(
            "access token request failed: {status} {value}"
        )));
    }
    let body: AccessTokenResponse = serde_json::from_value(value)?;
    match body.error.as_deref() {
        Some("authorization_pending") => Ok(None),
        Some("slow_down") => Ok(None),
        Some("expired_token") => Err(ProxyError::Other(anyhow::anyhow!("device code expired"))),
        Some(other) => Err(ProxyError::Other(anyhow::anyhow!(
            "oauth error: {other}: {}",
            body.error_description.unwrap_or_default()
        ))),
        None => match body.access_token {
            Some(t) => Ok(Some(t)),
            None => Err(ProxyError::Other(anyhow::anyhow!(
                "missing access_token in response"
            ))),
        },
    }
}

/// Block until the user authorizes the device, printing the user code to
/// stdout periodically. Returns the github access token on success.
pub async fn device_flow_blocking(http: &reqwest::Client) -> Result<String> {
    let dc = request_device_code(http).await?;
    println!();
    println!("GitHub Copilot authentication required.");
    println!("Open: {}", dc.verification_uri);
    println!("Enter code: {}", dc.user_code);
    println!("(waiting up to {} seconds)\n", dc.expires_in);
    poll_loop(http, github_base_url(), &dc).await
}

async fn poll_loop(
    http: &reqwest::Client,
    base_url: &str,
    dc: &DeviceCodeResponse,
) -> Result<String> {
    let interval = Duration::from_secs(dc.interval.max(5) + 1);
    let deadline = std::time::Instant::now() + Duration::from_secs(dc.expires_in);

    loop {
        if std::time::Instant::now() >= deadline {
            return Err(ProxyError::Other(anyhow::anyhow!(
                "device flow timed out after {}s",
                dc.expires_in
            )));
        }
        tokio::time::sleep(interval).await;
        match poll_access_token_at(http, base_url, &dc.device_code).await? {
            Some(token) => return Ok(token),
            None => continue,
        }
    }
}

#[cfg(test)]
pub(crate) async fn device_flow_blocking_at(
    http: &reqwest::Client,
    base_url: &str,
) -> Result<String> {
    let dc = request_device_code_at(http, base_url).await?;
    poll_loop(http, base_url, &dc).await
}

/// Process-wide lock guarding tests that mutate the
    /// `LLMPROXY_TEST_GITHUB_BASE_URL` env var (or rely on the `Box::leak`
    /// cache inside `github_base_url`). Cargo runs tests in parallel by
    /// default; without this lock, two env-mutating tests can interleave
    /// their setup and one will hit the wrong mock server. `copilot.rs`
    /// tests use the same env var via `device_flow_blocking`, so they share
    /// this lock.
    #[cfg(test)]
    pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    pub(crate) use super::ENV_LOCK;

    async fn poll_response(status: u16, body: Value) -> (MockServer, reqwest::Client) {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .and(header("accept", "application/json"))
            .and(body_json(json!({
                "client_id": GITHUB_CLIENT_ID,
                "device_code": "device-1",
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            })))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        (server, reqwest::Client::new())
    }

    #[tokio::test]
    async fn request_device_code_sends_expected_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .and(header("accept", "application/json"))
            .and(body_json(json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": GITHUB_SCOPES,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "device-1",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://example.test/device",
                "expires_in": 900,
                "interval": 5,
            })))
            .expect(1)
            .mount(&server)
            .await;

        let result = request_device_code_at(&reqwest::Client::new(), &server.uri())
            .await
            .unwrap();

        assert_eq!(result.device_code, "device-1");
        assert_eq!(result.user_code, "ABCD-EFGH");
        assert_eq!(result.verification_uri, "https://example.test/device");
        assert_eq!(result.expires_in, 900);
        assert_eq!(result.interval, 5);
    }

    #[tokio::test]
    async fn request_device_code_rejects_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(
                ResponseTemplate::new(403).set_body_json(json!({"message": "forbidden"})),
            )
            .mount(&server)
            .await;

        let error = request_device_code_at(&reqwest::Client::new(), &server.uri())
            .await
            .unwrap_err();

        assert!(error.to_string().contains("device code request failed"));
        assert!(error.to_string().contains("403"));
    }

    #[tokio::test]
    async fn poll_access_token_returns_token() {
        let (server, client) = poll_response(200, json!({"access_token": "gh-token"})).await;

        let token = poll_access_token_at(&client, &server.uri(), "device-1")
            .await
            .unwrap();

        assert_eq!(token.as_deref(), Some("gh-token"));
    }

    #[tokio::test]
    async fn poll_access_token_waits_for_pending_and_slow_down() {
        for oauth_error in ["authorization_pending", "slow_down"] {
            let (server, client) = poll_response(200, json!({"error": oauth_error})).await;

            let token = poll_access_token_at(&client, &server.uri(), "device-1")
                .await
                .unwrap();

            assert!(token.is_none());
        }
    }

    #[tokio::test]
    async fn poll_access_token_rejects_expired_code() {
        let (server, client) = poll_response(200, json!({"error": "expired_token"})).await;

        let error = poll_access_token_at(&client, &server.uri(), "device-1")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("device code expired"));
    }

    #[tokio::test]
    async fn poll_access_token_reports_oauth_error_description() {
        let (server, client) = poll_response(
            200,
            json!({"error": "access_denied", "error_description": "user denied access"}),
        )
        .await;

        let error = poll_access_token_at(&client, &server.uri(), "device-1")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("access_denied"));
        assert!(error.to_string().contains("user denied access"));
    }

    #[tokio::test]
    async fn poll_access_token_rejects_missing_token_and_http_error() {
        let (server, client) = poll_response(200, json!({})).await;
        let missing = poll_access_token_at(&client, &server.uri(), "device-1")
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("missing access_token"));

        let (server, client) = poll_response(500, json!({"message": "unavailable"})).await;
        let upstream = poll_access_token_at(&client, &server.uri(), "device-1")
            .await
            .unwrap_err();
        assert!(upstream.to_string().contains("access token request failed"));
        assert!(upstream.to_string().contains("500"));
    }

    #[tokio::test(start_paused = true)]
    async fn public_wrappers_and_blocking_flow_share_env_override() {
        // Combined test: covers the env-var-override path of all three
        // public wrappers (request_device_code, poll_access_token,
        // device_flow_blocking) within a single MockServer. They share the
        // process-wide LLMPROXY_TEST_GITHUB_BASE_URL env var (and the
        // `Box::leak` cache inside `github_base_url`), so we hold
        // ENV_LOCK to serialize against other env-mutating tests in the
        // crate (notably the copilot refresh_token tests).
        let _env_guard = ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .and(body_json(json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": GITHUB_SCOPES,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "public-device",
                "user_code": "PUBLIC-1234",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .and(body_json(json!({
                "client_id": GITHUB_CLIENT_ID,
                "device_code": "public-device",
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "public-token"
            })))
            .expect(2)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());

        let result = request_device_code(&reqwest::Client::new()).await.unwrap();
        assert_eq!(result.device_code, "public-device");
        assert_eq!(result.user_code, "PUBLIC-1234");
        assert_eq!(result.expires_in, 600);

        let token = poll_access_token(&reqwest::Client::new(), "public-device")
            .await
            .unwrap();
        assert_eq!(token.as_deref(), Some("public-token"));

        // Now exercise the production-blocking entry point: the prints and
        // the first request_device_code call live only here, so this run
        // covers that code path under the same env override.
        let _stop = spawn_auto_advance();
        let token = device_flow_blocking(&reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(token, "public-token");

        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");
    }

    #[test]
    fn github_base_url_default_and_override_lifecycle() {
        // Combined test: covers the default fallback branch of
        // `github_base_url()` plus exercises the production wrappers under
        // the env-var override. Both halves mutate the process-wide env var
        // and `Box::leak`-cached pointer, so we hold ENV_LOCK to serialize
        // against other env-mutating tests in the crate.
        let _env_guard = ENV_LOCK.lock().unwrap();
        let saved = std::env::var_os("LLMPROXY_TEST_GITHUB_BASE_URL");
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        let default_url = github_base_url();
        assert_eq!(default_url, "https://github.com");

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", "https://override.test");
        let override_url = github_base_url();
        assert_eq!(override_url, "https://override.test");

        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        if let Some(value) = saved {
            std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", value);
        } else {
            std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");
        }
    }

    async fn mount_device_code(server: &MockServer, expires_in: u64, interval: u64) {
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .and(header("accept", "application/json"))
            .and(body_json(json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": GITHUB_SCOPES,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "device-1",
                "user_code": "CODE-1234",
                "verification_uri": "https://example.test/device",
                "expires_in": expires_in,
                "interval": interval,
            })))
            .expect(1)
            .mount(server)
            .await;
    }

    /// Spawn a task that advances paused tokio time in 7-second jumps
    /// (enough to wake each `tokio::time::sleep(interval+1)` block).
    /// Returns a oneshot that the test awaits to stop the auto-advance task
    /// once the device flow under test has resolved.
    fn spawn_auto_advance() -> tokio::sync::oneshot::Sender<()> {
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    _ = tokio::time::sleep(Duration::from_secs(7)) => {
                        tokio::time::advance(Duration::from_secs(7)).await;
                    }
                }
            }
        });
        stop_tx
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_flow_returns_token_on_first_poll() {
        let server = MockServer::start().await;
        mount_device_code(&server, 900, 5).await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "first-poll-token"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let _stop = spawn_auto_advance();
        let client = reqwest::Client::new();
        let token = device_flow_blocking_at(&client, &server.uri())
            .await
            .unwrap();
        assert_eq!(token, "first-poll-token");
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_flow_returns_token_after_pending_polls() {
        let server = MockServer::start().await;
        mount_device_code(&server, 900, 5).await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "error": "authorization_pending"
            })))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "later-token"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let _stop = spawn_auto_advance();
        let client = reqwest::Client::new();
        let token = device_flow_blocking_at(&client, &server.uri())
            .await
            .unwrap();
        assert_eq!(token, "later-token");
    }

    #[tokio::test]
    async fn blocking_flow_times_out_at_zero_deadline() {
        // Without paused time: expires_in=0 makes the deadline check fire on
        // the very first loop iteration, before any sleep is awaited.
        let server = MockServer::start().await;
        mount_device_code(&server, 0, 0).await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "error": "authorization_pending"
            })))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let error = device_flow_blocking_at(&client, &server.uri())
            .await
            .unwrap_err();

        assert!(error.to_string().contains("device flow timed out"));
        assert!(error.to_string().contains("0s"));
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_flow_surfaces_poll_errors_as_oauth_failure() {
        let server = MockServer::start().await;
        mount_device_code(&server, 900, 5).await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "error": "access_denied",
                "error_description": "user said no"
            })))
            .mount(&server)
            .await;

        let _stop = spawn_auto_advance();
        let client = reqwest::Client::new();
        let error = device_flow_blocking_at(&client, &server.uri())
            .await
            .unwrap_err();

        assert!(error.to_string().contains("access_denied"));
        assert!(error.to_string().contains("user said no"));
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_flow_handles_slow_down_polls() {
        let server = MockServer::start().await;
        mount_device_code(&server, 900, 5).await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "error": "slow_down"
            })))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "after-slowdown"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let _stop = spawn_auto_advance();
        let client = reqwest::Client::new();
        let token = device_flow_blocking_at(&client, &server.uri())
            .await
            .unwrap();
        assert_eq!(token, "after-slowdown");
    }
}
