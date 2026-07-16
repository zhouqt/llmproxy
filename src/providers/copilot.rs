//! GitHub Copilot provider.
//!
//! Reference: copilot-api-py/src/lib/token.py, src/services/copilot/*
//!
//! Flow:
//! 1. GitHub device flow → github_access_token
//! 2. Exchange github token at api.github.com/copilot_internal/v2/token → copilot_token
//! 3. Use copilot_token with required Copilot headers against api.githubcopilot.com

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};

use crate::anthropic::MessagesRequest;
use crate::config::ApiFormat;
use crate::error::{ProxyError, Result};
use crate::oauth::device_flow::device_flow_blocking;
use crate::oauth::token_store::{StoredTokens, TokenStore};
use crate::providers::openai_compat::OpenAiSseToAnthropic;
use crate::providers::{Provider, ProviderOutput};

const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const GITHUB_API_VERSION: &str = "2025-04-01";
const COPILOT_INTERNAL_TOKEN_URL: &str =
    "https://api.github.com/copilot_internal/v2/token";

pub struct CopilotProvider {
    name: String,
    vscode_version: String,
    account_type: String,
    http: reqwest::Client,
    state: Arc<CopilotState>,
    #[cfg(test)]
    api_base_override: Option<String>,
    #[cfg(test)]
    copilot_token_url: String,
}

struct CopilotState {
    tokens: RwLock<Option<StoredTokens>>,
    store: TokenStore,
    refresh_lock: Mutex<()>,
}

/// Distinguishes credential failures (must clear store + re-authenticate)
/// from transient failures (network blip, upstream 5xx, malformed body —
/// keep store, surface error to caller, do NOT trigger device flow).
#[derive(Debug)]
enum CopilotFetchError {
    /// GitHub / Copilot actively rejected the token: 401, 403, or 404 on
    /// the token-exchange endpoint. The stored github token is no longer
    /// usable; the operator must re-authenticate.
    AuthRejected { status: u16, body: String },
    /// Network error, upstream 5xx, malformed JSON, missing token field,
    /// etc. The stored credentials may still be valid; surface the error
    /// and try again later.
    Transient(String),
}

impl std::fmt::Display for CopilotFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopilotFetchError::AuthRejected { status, body } => {
                write!(f, "auth rejected ({status}): {body}")
            }
            CopilotFetchError::Transient(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for CopilotFetchError {}

impl CopilotProvider {
    pub fn new(
        name: String,
        vscode_version: String,
        account_type: String,
        http: reqwest::Client,
    ) -> Result<Self> {
        let store = TokenStore::new()?;
        let initial = store.load().unwrap_or_else(|e| {
            tracing::warn!(
                provider = "copilot",
                error = %e,
                path = %store.path().display(),
                "failed to read token store; treating as empty"
            );
            None
        });
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(initial),
            store,
            refresh_lock: Mutex::new(()),
        });
        Ok(Self {
            name,
            vscode_version,
            account_type,
            http,
            state,
            #[cfg(test)]
            api_base_override: None,
            #[cfg(test)]
            copilot_token_url: COPILOT_INTERNAL_TOKEN_URL.to_string(),
        })
    }

    fn base_url(&self) -> String {
        #[cfg(test)]
        if let Some(api_base) = &self.api_base_override {
            return api_base.trim_end_matches('/').to_string();
        }
        match self.account_type.as_str() {
            "individual" => "https://api.githubcopilot.com".to_string(),
            other => format!("https://api.{other}.githubcopilot.com"),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url())
    }

    fn token_url(&self) -> &str {
        #[cfg(test)]
        {
            &self.copilot_token_url
        }
        #[cfg(not(test))]
        {
            COPILOT_INTERNAL_TOKEN_URL
        }
    }

    fn headers(&self, token: &str) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert("authorization", format!("Bearer {token}").parse().unwrap());
        h.insert("copilot-integration-id", "vscode-chat".parse().unwrap());
        h.insert(
            "editor-version",
            format!("vscode/{}", self.vscode_version).parse().unwrap(),
        );
        h.insert(
            "editor-plugin-version",
            EDITOR_PLUGIN_VERSION.parse().unwrap(),
        );
        h.insert("user-agent", USER_AGENT.parse().unwrap());
        h.insert("openai-intent", "conversation-panel".parse().unwrap());
        h.insert("x-github-api-version", GITHUB_API_VERSION.parse().unwrap());
        h.insert(
            "x-request-id",
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
        );
        h.insert(
            "x-vscode-user-agent-library-version",
            "electron-fetch".parse().unwrap(),
        );
        h
    }

    async fn ensure_token(&self) -> Result<String> {
        let need_refresh = {
            let guard = self.state.tokens.read().await;
            match guard.as_ref() {
                Some(t) => {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64;
                    t.copilot_expires_at - now < 60
                }
                None => true,
            }
        };

        if need_refresh {
            self.refresh_token().await?;
        }

        let guard = self.state.tokens.read().await;
        Ok(guard
            .as_ref()
            .ok_or_else(|| ProxyError::Other(anyhow::anyhow!("no copilot token after refresh")))?
            .copilot_token
            .clone())
    }

    pub async fn refresh_token(&self) -> Result<()> {
        let _guard = self.state.refresh_lock.lock().await;

        let existing = self.state.store.load().unwrap_or_else(|e| {
            tracing::warn!(
                provider = "copilot",
                error = %e,
                path = %self.state.store.path().display(),
                "failed to read token store; treating as empty"
            );
            None
        });
        let github_token = match existing.as_ref() {
            Some(t) => t.github_access_token.clone(),
            None => device_flow_blocking(&self.http).await?,
        };

        match self.fetch_copilot_token(&github_token).await {
            Ok(new_tokens) => {
                self.state.store.save(&new_tokens)?;
                *self.state.tokens.write().await = Some(new_tokens);
                Ok(())
            }
            Err(CopilotFetchError::AuthRejected { status, body }) => {
                tracing::warn!(
                    provider = "copilot",
                    status,
                    body = %body,
                    "copilot rejected stored credentials; re-running device flow"
                );
                self.state.store.clear().ok();
                let gh = device_flow_blocking(&self.http).await?;
                let new_tokens = self
                    .fetch_copilot_token(&gh)
                    .await
                    .map_err(|e| ProxyError::Other(anyhow::anyhow!("copilot token fetch after device flow failed: {e}")))?;
                self.state.store.save(&new_tokens)?;
                *self.state.tokens.write().await = Some(new_tokens);
                Ok(())
            }
            Err(CopilotFetchError::Transient(reason)) => {
                // Network blip / 5xx / parse error: keep the stored token
                // so the next attempt can use it, surface the failure to
                // the caller, and do NOT trigger a blocking device flow.
                tracing::warn!(
                    provider = "copilot",
                    reason = %reason,
                    "copilot token refresh failed (transient); keeping stored credentials"
                );
                Err(ProxyError::Other(anyhow::anyhow!(reason)))
            }
        }
    }

    async fn fetch_copilot_token(&self, github_token: &str) -> std::result::Result<StoredTokens, CopilotFetchError> {
        let resp = match self
            .http
            .get(self.token_url())
            .header("authorization", format!("token {github_token}"))
            .header("user-agent", USER_AGENT)
            .header("accept", "application/json")
            .header("x-github-api-version", GITHUB_API_VERSION)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(CopilotFetchError::Transient(format!(
                    "network error contacting copilot token endpoint: {e}"
                )));
            }
        };
        let status = resp.status();
        if !status.is_success() {
            // Read the body as text — many transient failures (5xx HTML
            // error pages, 429 plain text, etc.) are not valid JSON and
            // we don't want to fail at the parse step before we get a
            // chance to classify the status.
            let text = resp.text().await.unwrap_or_default();
            // 401 / 403 / 404 mean the stored github token is invalid or
            // lost access — the operator must re-authenticate. Anything
            // else (5xx, 408, 429) is treated as transient so we don't
            // wipe the store on a flaky upstream.
            if matches!(status.as_u16(), 401 | 403 | 404) {
                return Err(CopilotFetchError::AuthRejected {
                    status: status.as_u16(),
                    body: text,
                });
            }
            return Err(CopilotFetchError::Transient(format!(
                "copilot token fetch failed: {status} {text}"
            )));
        }
        let body: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                return Err(CopilotFetchError::Transient(format!(
                    "copilot token response not valid JSON: {e}"
                )));
            }
        };
        let token = match body.get("token").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => {
                return Err(CopilotFetchError::Transient(
                    "missing token field in copilot response".to_string(),
                ));
            }
        };
        let expires_at = body.get("expires_at").and_then(|v| v.as_i64()).unwrap_or_else(|| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            now + 1500
        });
        let refresh_in = body.get("refresh_in").and_then(|v| v.as_i64()).unwrap_or(1500);
        Ok(StoredTokens {
            github_access_token: github_token.to_string(),
            copilot_token: token,
            copilot_expires_at: expires_at,
            refresh_in,
        })
    }

    /// Spawn a background refresh loop. Returns a join handle for shutdown.
    pub fn spawn_refresh_loop(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let sleep_secs = {
                    let guard = self.state.tokens.read().await;
                    match guard.as_ref() {
                        Some(t) => (t.refresh_in - 60).max(60) as u64,
                        None => 60,
                    }
                };
                tokio::time::sleep(Duration::from_secs(sleep_secs)).await;
                if let Err(e) = self.refresh_token().await {
                    tracing::error!("background copilot refresh failed: {e}");
                }
            }
        })
    }

    async fn send_with_token(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response> {
        let token = self.ensure_token().await?;
        let resp = self
            .http
            .post(self.chat_url())
            .headers(self.headers(&token))
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await?;
        if resp.status().as_u16() == 401 {
            self.refresh_token().await?;
            let token = self.ensure_token().await?;
            let resp2 = self
                .http
                .post(self.chat_url())
                .headers(self.headers(&token))
                .header("content-type", "application/json")
                .json(body)
                .send()
                .await?;
            return Ok(resp2);
        }
        Ok(resp)
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn api_format(&self) -> ApiFormat {
        ApiFormat::Openai
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, model_rewrite);
        openai_req.stream = false;
        openai_req.stream_options = None;
        let body = serde_json::to_value(openai_req)?;

        let resp = self.send_with_token(&body).await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        // GitHub Copilot (like DeepSeek) returns HTTP 200 with an OpenAI
        // error envelope when the model name isn't recognized, instead of
        // a 4xx. Detect the envelope before deserializing as ChatResponse
        // so the client sees the real upstream message rather than a
        // generic 500 "missing field `object`" — see fix-R8 in
        // docs/TEST_ISSUES.md.
        let parsed: serde_json::Value = serde_json::from_str(&text)
            .map_err(ProxyError::Json)?;
        if crate::openai::looks_like_error_envelope(&parsed) {
            return Err(ProxyError::Upstream { status: 400, body: text });
        }
        let chat: crate::openai::ChatResponse = serde_json::from_value(parsed)?;
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let anthropic =
            crate::conversion::openai_to_anthropic_response(&chat, &req.model, &msg_id)?;
        Ok(ProviderOutput::Json(serde_json::to_value(anthropic)?))
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, model_rewrite);
        openai_req.stream = true;
        openai_req.stream_options = Some(crate::openai::StreamOptions {
            include_usage: true,
        });
        let body = serde_json::to_value(openai_req)?;

        let resp = self.send_with_token(&body).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let stream = resp.bytes_stream();
        let sse = OpenAiSseToAnthropic::new(stream, &req.model);
        Ok(ProviderOutput::Stream(Box::new(sse)))
    }

    fn spawn_background(self: Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        Some(self.spawn_refresh_loop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;
    use futures_util::StreamExt;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn now() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    fn stored_tokens(github: &str, copilot: &str, expires_in: i64) -> StoredTokens {
        StoredTokens {
            github_access_token: github.to_string(),
            copilot_token: copilot.to_string(),
            copilot_expires_at: now() + expires_in,
            refresh_in: 1500,
        }
    }

    fn test_provider(
        server: Option<&MockServer>,
        initial: Option<StoredTokens>,
    ) -> (tempfile::TempDir, CopilotProvider) {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        if let Some(tokens) = &initial {
            store.save(tokens).unwrap();
        }
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(initial),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: server.map(MockServer::uri),
            copilot_token_url: server
                .map(|server| format!("{}/copilot_internal/v2/token", server.uri()))
                .unwrap_or_else(|| COPILOT_INTERNAL_TOKEN_URL.to_string()),
        };
        (dir, provider)
    }

    fn request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "system": "system prompt",
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    fn completion_response(content: &str) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "copilot-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6
            }
        })
    }

    #[test]
    fn base_urls_and_headers_match_copilot_contract() {
        let (_dir, mut provider) = test_provider(None, None);
        assert_eq!(provider.base_url(), "https://api.githubcopilot.com");
        provider.account_type = "business".to_string();
        assert_eq!(
            provider.base_url(),
            "https://api.business.githubcopilot.com"
        );
        assert_eq!(
            provider.chat_url(),
            "https://api.business.githubcopilot.com/chat/completions"
        );

        let headers = provider.headers("token-1");
        assert_eq!(headers["authorization"], "Bearer token-1");
        assert_eq!(headers["copilot-integration-id"], "vscode-chat");
        assert_eq!(headers["editor-version"], "vscode/1.95.0");
        assert_eq!(headers["editor-plugin-version"], EDITOR_PLUGIN_VERSION);
        assert_eq!(headers["user-agent"], USER_AGENT);
        assert_eq!(headers["openai-intent"], "conversation-panel");
        assert_eq!(headers["x-github-api-version"], GITHUB_API_VERSION);
        assert_eq!(
            headers["x-vscode-user-agent-library-version"],
            "electron-fetch"
        );
        uuid::Uuid::parse_str(headers["x-request-id"].to_str().unwrap()).unwrap();
    }

    #[tokio::test]
    async fn ensure_token_reuses_unexpired_memory_token() {
        let (_dir, provider) = test_provider(
            None,
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );

        assert_eq!(provider.ensure_token().await.unwrap(), "copilot-token");
    }

    #[tokio::test]
    async fn fetch_copilot_token_sends_headers_and_parses_values() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .and(header("user-agent", USER_AGENT))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "new-copilot-token",
                "expires_at": 1234567890,
                "refresh_in": 1200
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);

        let tokens = provider.fetch_copilot_token("github-token").await.unwrap();

        assert_eq!(tokens.github_access_token, "github-token");
        assert_eq!(tokens.copilot_token, "new-copilot-token");
        assert_eq!(tokens.copilot_expires_at, 1234567890);
        assert_eq!(tokens.refresh_in, 1200);
    }

    #[tokio::test]
    async fn fetch_copilot_token_applies_defaults_and_rejects_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"token": "token"})))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);

        let before = now();
        let tokens = provider.fetch_copilot_token("github").await.unwrap();
        assert_eq!(tokens.refresh_in, 1500);
        assert!(tokens.copilot_expires_at >= before + 1500);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(403).set_body_json(json!({"message": "denied"})))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);
        let error = provider
            .fetch_copilot_token("github")
            .await
            .unwrap_err();
        // 403 is now classified as AuthRejected (not Transient), so the
        // error message reflects that — the caller uses this to decide
        // whether to clear the store.
        assert!(error.to_string().contains("auth rejected"));
        assert!(error.to_string().contains("403"));

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);
        let error = provider
            .fetch_copilot_token("github")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("missing token"));
    }

    #[tokio::test]
    async fn ensure_token_refreshes_when_memory_is_empty() {
        // The store has a valid GitHub token but the in-memory cache was
        // cleared (e.g. process restart). ensure_token should re-fetch the
        // Copilot token instead of failing.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "fresh-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        store
            .save(&stored_tokens("github-token", "stale", 600))
            .unwrap();
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };

        let token = provider.ensure_token().await.unwrap();
        assert_eq!(token, "fresh-token");
    }

    #[tokio::test]
    async fn refresh_token_uses_stored_github_token_and_persists_result() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "refreshed-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "expired-token", -1)),
        );

        provider.refresh_token().await.unwrap();

        let memory = provider.state.tokens.read().await;
        assert_eq!(memory.as_ref().unwrap().copilot_token, "refreshed-token");
        drop(memory);
        let disk = provider.state.store.load().unwrap().unwrap();
        assert_eq!(disk.copilot_token, "refreshed-token");
        assert_eq!(disk.refresh_in, 800);
    }

    #[tokio::test]
    async fn complete_converts_request_and_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer copilot-token"))
            .and(header("editor-version", "vscode/1.95.0"))
            .and(body_partial_json(json!({
                "model": "copilot-model",
                "stream": false,
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "hello"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response("world")))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "copilot-model".to_string());

        let output = provider.complete(&request(false), &rewrite).await.unwrap();

        assert_eq!(provider.name(), "copilot");
        assert_eq!(provider.api_format(), ApiFormat::Openai);
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "world");
            assert_eq!(body["usage"]["input_tokens"], 4);
        });
    }

    #[tokio::test]
    async fn unauthorized_chat_refreshes_and_retries_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer old-token"))
            .respond_with(ResponseTemplate::new(401).set_body_string("expired"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "new-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer new-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response("retried")))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "old-token", 600)),
        );

        let output = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();

        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "retried");
        });
    }

    #[tokio::test]
    async fn stream_converts_sse_and_background_task_can_be_aborted() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"streamed\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );

        let output = provider
            .stream(&request(true), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut encoded = String::new();
            while let Some(item) = output.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(encoded.contains("\"text\":\"streamed\""));
            assert!(encoded.contains("event: message_stop"));
        });

        let handle = Arc::new(provider)
            .spawn_background()
            .expect("copilot should have a background refresh task");
        handle.abort();
    }

    #[tokio::test]
    async fn complete_and_stream_preserve_upstream_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .expect(2)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );

        let complete = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("complete should fail");
        let stream = provider
            .stream(&request(true), &HashMap::new())
            .await
            .err()
            .expect("stream should fail");

        assert!(matches!(
            complete,
            ProxyError::Upstream { status: 503, ref body } if body == "unavailable"
        ));
        assert!(matches!(
            stream,
            ProxyError::Upstream { status: 503, ref body } if body == "unavailable"
        ));
    }

    #[tokio::test]
    async fn complete_surfaces_error_envelope_on_http_200() {
        // GitHub Copilot returns HTTP 200 with an OpenAI error envelope
        // when the requested model isn't supported. Without the envelope
        // check (mirroring OpenAiCompatProvider's fix-F), ChatResponse
        // deserialization fails with "missing field `object`" and the
        // client sees a generic 500. See fix-R8 in docs/TEST_ISSUES.md.
        let server = MockServer::start().await;
        let envelope = json!({
            "error": {
                "message": "Model not supported",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&envelope))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("error envelope should surface as Err");

        expect_variant!(error, ProxyError::Upstream { status, body } => {
            assert_eq!(status, 400);
            assert!(body.contains("Model not supported"), "body was: {body}");
            assert!(body.contains("model_not_found"), "body was: {body}");
        });
    }

    #[tokio::test(start_paused = true)]
    async fn spawn_refresh_loop_runs_one_iteration_and_aborts() {
        // The background refresh loop in spawn_refresh_loop normally runs
        // forever. With paused time + auto-advance we let the initial sleep
        // elapse, then let the inner refresh_token complete one cycle before
        // aborting the handle. This exercises the loop body (sleep_secs
        // computation + the refresh_token call inside the loop) without
        // waiting hours of real wall-clock time.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Mock the copilot token endpoint so refresh_token completes
        // successfully without falling back to device flow.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "loop-copilot-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1..)
            .mount(&server)
            .await;

        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let provider = Arc::new(provider);
        let handle = provider.clone().spawn_refresh_loop();

        // Advance paused time enough to wake the loop out of its first
        // sleep (refresh_in=1500 → sleep for 1440s) and run refresh_token.
        for _ in 0..300 {
            tokio::time::advance(std::time::Duration::from_secs(60)).await;
            // Yield so the spawned task can be scheduled.
            tokio::task::yield_now().await;
            // Give the runtime a chance to actually poll the spawned task.
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            // If refresh_token already ran, the assertion would have
            // succeeded; bail early to keep the test fast.
            if provider.state.tokens.read().await.as_ref().unwrap().copilot_token
                == "loop-copilot-token"
            {
                break;
            }
        }

        handle.abort();
        let _ = handle.await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // After the loop ran, the memory cache should reflect the new
        // token fetched by refresh_token.
        let memory = provider.state.tokens.read().await;
        assert_eq!(
            memory.as_ref().unwrap().copilot_token,
            "loop-copilot-token"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn spawn_refresh_loop_uses_short_sleep_when_no_tokens() {
        // When the memory cache is empty (no prior token), the loop body
        // falls into the `None => 60` branch and sleeps only 60 seconds
        // before its first refresh attempt. Verify by starting with an
        // empty in-memory cache and a populated on-disk store; after one
        // refresh iteration, memory should reflect the new token.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "empty-loop-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1..)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        store
            .save(&stored_tokens("github-token", "stale", 600))
            .unwrap();
        // Memory cache starts empty (the `None => 60` branch reads).
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = Arc::new(CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        });
        let handle = provider.clone().spawn_refresh_loop();

        for _ in 0..300 {
            tokio::time::advance(std::time::Duration::from_secs(60)).await;
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if provider.state.tokens.read().await.is_some() {
                break;
            }
        }

        handle.abort();
        let _ = handle.await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        let memory = provider.state.tokens.read().await;
        assert_eq!(memory.as_ref().unwrap().copilot_token, "empty-loop-token");
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_runs_device_flow_when_store_is_empty() {
        // When no github token is on disk, refresh_token must run the full
        // device flow to obtain one, then exchange it for a Copilot token.
        // Exercises the `None => device_flow_blocking(...)` arm in
        // refresh_token (production code path that previously had no
        // coverage).
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // GitHub device flow mocks (routed via LLMPROXY_TEST_GITHUB_BASE_URL).
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "df-device",
                "user_code": "DF-CODE",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "device-flow-github"
            })))
            .expect(1)
            .mount(&server)
            .await;
        // Copilot token exchange (after device flow yields github token).
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token device-flow-github"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "fresh-copilot-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        // Note: store has NO saved tokens — refresh_token must run device flow.
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        provider.refresh_token().await.unwrap();
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        let memory = provider.state.tokens.read().await;
        assert_eq!(
            memory.as_ref().unwrap().copilot_token,
            "fresh-copilot-token"
        );
        drop(memory);
        let disk = provider.state.store.load().unwrap().unwrap();
        assert_eq!(disk.github_access_token, "device-flow-github");
        assert_eq!(disk.copilot_token, "fresh-copilot-token");
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_proceeds_when_store_load_fails() {
        // When the token store exists but is unreadable (here: contains
        // corrupted JSON), refresh_token must NOT silently treat the
        // failure as "no token" and proceed to device flow — it must log
        // the error so the operator can see why their credentials didn't
        // load. The end-state is still Ok (device flow runs), but the
        // log path is exercised.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "df-device",
                "user_code": "DF-CODE",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "device-flow-github"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token device-flow-github"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "fresh-copilot-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("github_token.json");
        // Write corrupted JSON so load() returns Err(Json).
        std::fs::write(&store_path, b"{not valid json").unwrap();
        let store = TokenStore::from_path(store_path);
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        // The load error is logged (warn-level) but does not abort the
        // refresh path. Device flow still runs because existing=None.
        provider.refresh_token().await.unwrap();
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // The corrupted file is replaced by the device-flow result.
        let disk = provider.state.store.load().unwrap().unwrap();
        assert_eq!(disk.github_access_token, "device-flow-github");
        assert_eq!(disk.copilot_token, "fresh-copilot-token");
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_reruns_device_flow_when_copilot_endpoint_rejects() {
        // When the stored github token is rejected by the Copilot token
        // endpoint, refresh_token should clear the store, re-run the device
        // flow to get a new github token, and retry the exchange.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Device flow mocks (re-used for both initial and recovery runs).
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "rf-device",
                "user_code": "RF-CODE",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "recovered-github-token"
            })))
            .expect(1)
            .mount(&server)
            .await;
        // Copilot token endpoint: first call (with stale github token)
        // fails; second call (with new github token from device flow)
        // succeeds.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token stale-github"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"message": "expired"})))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token recovered-github-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "recovered-copilot-token",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        // Start with a stale github token so refresh_token will try the
        // endpoint and fail before re-running the device flow.
        store
            .save(&stored_tokens("stale-github", "old-copilot", -10))
            .unwrap();
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(Some(stored_tokens(
                "stale-github",
                "old-copilot",
                -10,
            ))),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        provider.refresh_token().await.unwrap();
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        let memory = provider.state.tokens.read().await;
        assert_eq!(
            memory.as_ref().unwrap().copilot_token,
            "recovered-copilot-token"
        );
        drop(memory);
        let disk = provider.state.store.load().unwrap().unwrap();
        assert_eq!(disk.github_access_token, "recovered-github-token");
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_keeps_store_on_transient_5xx() {
        // When the Copilot token endpoint returns a transient failure
        // (5xx), refresh_token must NOT clear the store or trigger the
        // device flow. It returns Err so the caller sees the failure,
        // but the stored credentials remain intact for the next attempt.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Copilot token endpoint always 503 (server error, transient).
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
            .expect(1) // exactly one call — device flow must NOT run
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let pre_existing = stored_tokens("still-valid-github", "still-valid-copilot", 900);
        store.save(&pre_existing).unwrap();
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(Some(pre_existing.clone())),
            store: store.clone(),
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };

        let result = provider.refresh_token().await;
        assert!(result.is_err(), "transient 5xx must surface as Err");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("503"),
            "transient error should preserve upstream status: {err}"
        );

        // Store must NOT have been cleared — the credentials are still
        // valid, only the network/upstream blipped.
        let disk = store.load().unwrap().expect("store must still exist");
        assert_eq!(disk.github_access_token, "still-valid-github");
        assert_eq!(disk.copilot_token, "still-valid-copilot");
    }

    /// Spawn a background task that advances paused tokio time every 7
    /// seconds so device-flow polls wake up. Returns a guard that stops
    /// the task when dropped.
    fn spawn_test_time_advance() -> impl Drop {
        struct Stopper(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for Stopper {
            fn drop(&mut self) {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
            }
        }
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(7)) => {
                        tokio::time::advance(std::time::Duration::from_secs(7)).await;
                    }
                }
            }
        });
        Stopper(Some(tx))
    }
}
