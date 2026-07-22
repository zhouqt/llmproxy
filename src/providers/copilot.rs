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
use crate::error::{ProxyError, Result};
use crate::oauth::device_flow::{request_device_code, DeviceCodeResponse};
use crate::oauth::token_store::{StoredTokens, TokenStore};
use crate::providers::openai_compat::OpenAiSseToAnthropic;
use crate::providers::{Provider, ProviderOutput};

const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const GITHUB_API_VERSION: &str = "2025-04-01";
const COPILOT_INTERNAL_TOKEN_URL: &str =
    "https://api.github.com/copilot_internal/v2/token";

/// Copilot rejects `/chat/completions` for GPT-5.x with
/// `unsupported_api_for_model` — those models must hit `/responses`
/// instead. Reference: copilot-api-py `endpoint_router.py`. Pure
/// function so we can call it from both `complete` and `stream`
/// without sharing provider state.
fn endpoint_for_model(model: &str) -> &'static str {
    if model.starts_with("gpt-5") {
        "responses"
    } else {
        "chat_completions"
    }
}

pub struct CopilotProvider {
    name: String,
    vscode_version: String,
    account_type: String,
    model_rewrite: HashMap<String, String>,
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
        model_rewrite: HashMap<String, String>,
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
            model_rewrite,
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

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url())
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
            None => {
                // No stored credentials. Don't run the device flow here
                // — that would block the request for up to 10 minutes
                // while waiting for the operator to authorize. Instead,
                // fast-fail with 401 so the fallback chain skips Copilot
                // immediately. Bootstrap is owned by `start_bootstrap`
                // (called from the background refresh loop and from
                // POST /admin/copilot/auth). See fix-R2.
                return Err(ProxyError::Upstream {
                    status: 401,
                    body: "github_copilot not authenticated".to_string(),
                });
            }
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
                    "copilot rejected stored credentials; clearing store and signalling bootstrap needed"
                );
                self.state.store.clear().ok();
                // Don't run the device flow inline — same reason as the
                // empty-store branch above. The background loop will
                // notice the cleared store on its next iteration and
                // trigger bootstrap; operators can also call
                // POST /admin/copilot/auth to start it immediately.
                Err(ProxyError::Upstream {
                    status: 401,
                    body: format!(
                        "github_copilot credentials rejected (was {status}); trigger bootstrap via /admin/copilot/auth"
                    ),
                })
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

    /// Request a fresh device code, spawn a background task to complete
    /// the OAuth device flow + Copilot-token exchange, and return the
    /// user-facing device-code info immediately. Used by both the
    /// background refresh loop and the admin endpoint so concurrent
    /// triggers fail fast with "already in progress" instead of
    /// duplicating the device flow. See fix-R2.
    pub async fn start_bootstrap(self: Arc<Self>) -> Result<DeviceCodeResponse> {
        // Best-effort check for a concurrent bootstrap. The actual
        // bootstrap lock is re-acquired by the spawned task (see
        // below) — this fast-path check just lets the caller fail
        // immediately with a clear message instead of printing the
        // banner twice. The race window is small: GitHub will reject
        // the second device-code request anyway.
        if self.state.refresh_lock.try_lock().is_err() {
            return Err(ProxyError::Other(anyhow::anyhow!(
                "copilot bootstrap already in progress"
            )));
        }

        let dc = request_device_code(&self.http).await?;

        // Print the user code so operators see it in the proxy logs
        // even when bootstrap was triggered by the background loop
        // (no admin endpoint was called).
        println!();
        println!("GitHub Copilot authentication required.");
        println!("Open: {}", dc.verification_uri);
        println!("Enter code: {}", dc.user_code);
        println!("(waiting up to {} seconds)\n", dc.expires_in);

        let provider_for_task = self.clone();
        let name = self.name.clone();
        let dc_for_task = dc.clone();
        tokio::spawn(async move {
            // Hold the refresh lock for the lifetime of the bootstrap
            // so refresh_token and other concurrent start_bootstrap
            // calls block until we're done.
            let _g = provider_for_task.state.refresh_lock.lock().await;
            let result = provider_for_task.complete_bootstrap(dc_for_task).await;
            match result {
                Ok(()) => tracing::info!(provider = %name, "copilot bootstrap completed"),
                Err(e) => tracing::error!(provider = %name, error = %e, "copilot bootstrap failed"),
            }
        });

        Ok(dc)
    }

    /// Background half of `start_bootstrap`: poll for the GitHub
    /// access token, exchange it for a Copilot token, persist to the
    /// store, and update in-memory state. The caller (spawned task)
    /// must hold `state.refresh_lock` for the entire duration.
    async fn complete_bootstrap(&self, dc: DeviceCodeResponse) -> Result<()> {
        let gh = crate::oauth::device_flow::poll_device_token(&self.http, &dc).await?;
        let new_tokens = self.fetch_copilot_token(&gh).await.map_err(|e| {
            ProxyError::Other(anyhow::anyhow!(
                "copilot token fetch after device flow failed: {e}"
            ))
        })?;
        self.state.store.save(&new_tokens)?;
        *self.state.tokens.write().await = Some(new_tokens);
        Ok(())
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
                // Decide based on whether the on-disk store actually
                // holds credentials. The memory cache may be empty
                // even when the store has a token (e.g. just after
                // startup); in that case refresh_token will load it.
                // refresh_token clears the store on AuthRejected, so
                // the next iteration falls through to start_bootstrap.
                let has_credentials = self
                    .state
                    .store
                    .load()
                    .ok()
                    .flatten()
                    .is_some();
                if has_credentials {
                    if let Err(e) = self.refresh_token().await {
                        tracing::error!("background copilot refresh failed: {e}");
                    }
                } else if let Err(e) = self.clone().start_bootstrap().await {
                    // Common: "bootstrap already in progress" — quiet.
                    // Other errors (network blip, GitHub 5xx) are worth
                    // logging so the operator sees why auth isn't
                    // progressing.
                    let msg = e.to_string();
                    if !msg.contains("already in progress") {
                        tracing::warn!("background copilot bootstrap failed: {e}");
                    }
                }
            }
        })
    }

    async fn send_with_token(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<reqwest::Response> {
        let token = self.ensure_token().await?;
        tracing::debug!(
            provider = "copilot",
            url = url,
            model = body.get("model").and_then(|v| v.as_str()).unwrap_or(""),
            stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
            "sending copilot request"
        );
        let resp = self
            .http
            .post(url)
            .headers(self.headers(&token))
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await?;
        tracing::debug!(
            provider = "copilot",
            url = url,
            status = resp.status().as_u16(),
            "copilot response status"
        );
        if resp.status().as_u16() == 401 {
            self.refresh_token().await?;
            let token = self.ensure_token().await?;
            let resp2 = self
                .http
                .post(url)
                .headers(self.headers(&token))
                .header("content-type", "application/json")
                .json(body)
                .send()
                .await?;
            return Ok(resp2);
        }
        Ok(resp)
    }

    /// Complete path for GPT-5.x models — Copilot rejects
    /// `/chat/completions` for those (`unsupported_api_for_model`), so
    /// we route them to `/responses` instead. See
    /// `endpoint_for_model`.
    async fn complete_responses(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = merge_rewrites(&self.model_rewrite, model_rewrite);
        let mut responses_req =
            crate::conversion::anthropic_to_responses_request(req, &merged);
        responses_req.stream = false;
        let body = serde_json::to_value(responses_req)?;

        let resp = self.send_with_token(&self.responses_url(), &body).await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let parsed: crate::responses::ResponsesResponse = serde_json::from_str(&text)?;
        let msg_id = crate::conversion::make_message_id();
        let anthropic = crate::conversion::responses_to_anthropic_response(
            &parsed,
            &req.model,
            &msg_id,
        )?;
        Ok(ProviderOutput::Json(serde_json::to_value(anthropic)?))
    }

    /// Streaming twin of `complete_responses`.
    async fn stream_responses(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = merge_rewrites(&self.model_rewrite, model_rewrite);
        let mut responses_req =
            crate::conversion::anthropic_to_responses_request(req, &merged);
        responses_req.stream = true;
        let body = serde_json::to_value(responses_req)?;

        let resp = self.send_with_token(&self.responses_url(), &body).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let stream = resp.bytes_stream();
        let sse = crate::providers::openai_responses::ResponsesSseToAnthropic::new(
            stream,
            &req.model,
        );
        Ok(ProviderOutput::Stream(Box::new(sse)))
    }
}

/// Combine the configured provider-level rewrite table with the
/// runtime per-call map. Runtime entries override configured ones
/// when keys collide (mirrors `OpenAiCompatProvider`).
fn merge_rewrites(
    configured: &HashMap<String, String>,
    runtime: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = configured.clone();
    merged.extend(runtime.iter().map(|(k, v)| (k.clone(), v.clone())));
    merged
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_serve_model(&self, model: &str) -> bool {
        // Mirrors OpenAiCompatProvider: empty rewrite table accepts any
        // model verbatim (Copilot exposes its own catalog); a non-empty
        // table is an explicit allow-list — see fix-R11.
        self.model_rewrite.is_empty() || self.model_rewrite.contains_key(model)
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = merge_rewrites(&self.model_rewrite, model_rewrite);
        let upstream_model = merged
            .get(&req.model)
            .map(String::as_str)
            .unwrap_or(&req.model);
        let endpoint = endpoint_for_model(upstream_model);
        if endpoint == "responses" {
            return self.complete_responses(req, &merged).await;
        }

        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, &merged);
        openai_req.stream = false;
        openai_req.stream_options = None;
        let body = serde_json::to_value(openai_req)?;

        let resp = self.send_with_token(&self.chat_url(), &body).await?;
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
        let merged = merge_rewrites(&self.model_rewrite, model_rewrite);
        let upstream_model = merged
            .get(&req.model)
            .map(String::as_str)
            .unwrap_or(&req.model);
        let endpoint = endpoint_for_model(upstream_model);
        if endpoint == "responses" {
            return self.stream_responses(req, &merged).await;
        }

        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, &merged);
        openai_req.stream = true;
        openai_req.stream_options = Some(crate::openai::StreamOptions {
            include_usage: true,
        });
        let body = serde_json::to_value(openai_req)?;

        let resp = self.send_with_token(&self.chat_url(), &body).await?;
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

    fn as_any_copilot(self: Arc<Self>) -> Option<Arc<CopilotProvider>> {
        Some(self)
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
            model_rewrite: HashMap::new(),
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

    fn responses_response_json(content: &str) -> Value {
        json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.5",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
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
    async fn fetch_copilot_token_classifies_5xx_as_transient() {
        // A 5xx from the copilot token endpoint must be Transient (not
        // AuthRejected) so the caller keeps the stored github token
        // instead of wiping it on a flaky upstream (lines 379-381).
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);
        let error = provider.fetch_copilot_token("github").await.unwrap_err();
        assert!(
            matches!(error, CopilotFetchError::Transient(ref m) if m.contains("503") && m.contains("upstream down")),
            "5xx must be Transient, got: {error:?}"
        );
    }

    #[tokio::test]
    async fn fetch_copilot_token_classifies_invalid_json_body_as_transient() {
        // A 200 response whose body is not valid JSON must be Transient
        // (lines 385-388) — a truncated/garbled success payload is a
        // flaky-upstream symptom, not an auth failure.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string("this is not json"),
            )
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);
        let error = provider.fetch_copilot_token("github").await.unwrap_err();
        assert!(
            matches!(error, CopilotFetchError::Transient(ref m) if m.contains("not valid JSON")),
            "invalid JSON body must be Transient, got: {error:?}"
        );
    }

    #[tokio::test]
    async fn fetch_copilot_token_classifies_network_error_as_transient() {
        // When the token endpoint is unreachable (connection refused),
        // the send() call itself errors and must map to Transient
        // (lines 356-359). We point the provider at a port nobody is
        // listening on to force a connection error deterministically.
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state: Arc::new(CopilotState {
                tokens: RwLock::new(None),
                store,
                refresh_lock: Mutex::new(()),
            }),
            api_base_override: None,
            // 127.0.0.1:1 is in the reserved low-port range; nothing
            // listens there, so the TCP connect fails immediately.
            copilot_token_url: "http://127.0.0.1:1/copilot_internal/v2/token".to_string(),
        };
        let error = provider.fetch_copilot_token("github").await.unwrap_err();
        assert!(
            matches!(error, CopilotFetchError::Transient(ref m) if m.contains("network error")),
            "connection failure must be Transient, got: {error:?}"
        );
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
            model_rewrite: HashMap::new(),
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
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "world");
            assert_eq!(body["usage"]["input_tokens"], 4);
        });
    }

    #[tokio::test]
    async fn complete_routes_gpt5_to_responses_endpoint() {
        // GPT-5.x models must hit /responses, not /chat/completions —
        // Copilot rejects the latter with unsupported_api_for_model.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("authorization", "Bearer copilot-token"))
            .and(body_partial_json(json!({
                "model": "gpt-5",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello-from-responses"}]
                }],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(false);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider.complete(&gpt5_req, &HashMap::new()).await.unwrap();

        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "hello-from-responses");
            assert_eq!(body["stop_reason"], "end_turn");
            assert_eq!(body["usage"]["input_tokens"], 5);
        });
    }

    #[tokio::test]
    async fn stream_routes_gpt5_to_responses_endpoint() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"streamed\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(true);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider.stream(&gpt5_req, &HashMap::new()).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut stream) => {
            let mut encoded = String::new();
            while let Some(item) = stream.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(encoded.contains("event: message_start"));
            assert!(encoded.contains("\"text\":\"streamed\""));
            assert!(encoded.contains("event: message_stop"));
        });
    }

    #[tokio::test]
    async fn complete_responses_preserves_upstream_error() {
        // Copilot's /responses endpoint can also 5xx; the error body
        // must surface to the caller unchanged so the router can
        // decide whether to fall back. Mirrors the chat-completions
        // path's error preservation.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(502).set_body_string("bad gateway"))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(false);
        gpt5_req.model = "gpt-5".to_string();

        let error = provider
            .complete(&gpt5_req, &HashMap::new())
            .await
            .err()
            .expect("upstream 502 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 502, ref body } if body == "bad gateway"
        ));
    }

    #[tokio::test]
    async fn stream_responses_preserves_upstream_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(true);
        gpt5_req.model = "gpt-5".to_string();

        let error = provider
            .stream(&gpt5_req, &HashMap::new())
            .await
            .err()
            .expect("upstream 429 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 429, ref body } if body == "rate limited"
        ));
    }

    #[tokio::test]
    async fn stream_responses_converts_sse_to_anthropic_for_gpt5() {
        // Success path for the GPT-5 /responses streaming surface: a
        // Responses-API SSE sequence (response.created → output_item
        // .added → output_text.delta → completed) must translate into
        // valid Anthropic SSE frames (message_start … content_block_delta
        // … message_stop). This exercises complete lines 543-548
        // (bytes_stream → ResponsesSseToAnthropic) which the error-only
        // test above never reaches.
        let server = MockServer::start().await;
        let sse = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_s\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5\",\"status\":\"in_progress\",\"output\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"output_index\":0,\"content_index\":0,\"delta\":\"streamed-via-responses\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_s\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(true);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider
            .stream(&gpt5_req, &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut encoded = String::new();
            while let Some(item) = output.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(encoded.contains("event: message_start"), "missing message_start: {encoded}");
            assert!(
                encoded.contains("\"text\":\"streamed-via-responses\""),
                "missing text delta: {encoded}"
            );
            assert!(encoded.contains("event: message_stop"), "missing message_stop: {encoded}");
        });
    }

    #[tokio::test]
    async fn gpt5_request_never_touches_chat_completions_endpoint() {
        // Regression guard: ensure GPT-5 dispatch is exclusive to
        // /responses. We assert `expect(1)` on /responses AND `expect(0)`
        // on /chat/completions via a mock that 404s if hit, so any
        // accidental chat-side request would fail.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_x",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut gpt5_req = request(false);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider
            .complete(&gpt5_req, &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "ok");
        });
    }

    #[tokio::test]
    async fn non_gpt5_request_never_touches_responses_endpoint() {
        // A non-GPT-5 source model with no rewrite resolving to a GPT-5
        // upstream name must NOT be routed to /responses. The mock on
        // /responses asserts it is never hit; the chat-completions mock
        // is what serves the request.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({"model": "claude-sonnet-4.6"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response("via-chat")))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut req = request(false);
        req.model = "claude-sonnet-4.6".to_string();
        // No rewrite: upstream_model == req.model, classified as
        // chat_completions.
        let rewrite = HashMap::new();

        let output = provider
            .complete(&req, &rewrite)
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "via-chat");
        });
    }

    #[tokio::test]
    async fn rewritten_to_gpt5_routes_to_responses_endpoint() {
        // The user's `work-high → gpt-5.5` mapping must dispatch to
        // /responses, not /chat/completions. Previously dispatch keyed
        // off the original `req.model` and missed the rewrite.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({"model": "gpt-5.5"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                responses_response_json("via-responses"),
            ))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut req = request(false);
        req.model = "work-high".to_string();
        let mut rewrite = HashMap::new();
        rewrite.insert("work-high".to_string(), "gpt-5.5".to_string());

        let output = provider
            .complete(&req, &rewrite)
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "via-responses");
        });
    }

    #[tokio::test]
    async fn streaming_rewritten_to_gpt5_routes_to_responses_endpoint() {
        // Same routing must apply on the streaming path: a rewrite
        // `work-high → gpt-5.5` must dispatch to /responses and the
        // response.stream event text must reach the client.
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5.5\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"output_index\":0,\"content_index\":0,\"delta\":\"via-stream\"}\n\n",
            "data: {\"type\":\"response.output_text.done\",\"item_id\":\"m1\",\"output_index\":0,\"content_index\":0,\"text\":\"via-stream\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-5.5\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({"model": "gpt-5.5"})))
            .respond_with(ResponseTemplate::new(200).set_body_string(sse))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut req = request(true);
        req.model = "work-high".to_string();
        let mut rewrite = HashMap::new();
        rewrite.insert("work-high".to_string(), "gpt-5.5".to_string());

        let output = provider.stream(&req, &rewrite).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut encoded = String::new();
            while let Some(item) = output.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(
                encoded.contains("\"text\":\"via-stream\""),
                "expected text delta in stream, got: {encoded}"
            );
        });
    }

    #[test]
    fn endpoint_for_model_classifies_by_prefix() {
        assert_eq!(endpoint_for_model("gpt-5"), "responses");
        assert_eq!(endpoint_for_model("gpt-5-mini"), "responses");
        assert_eq!(endpoint_for_model("gpt-5.5"), "responses");
        assert_eq!(endpoint_for_model("gpt-4"), "chat_completions");
        assert_eq!(endpoint_for_model("claude-sonnet-4.6"), "chat_completions");
        assert_eq!(endpoint_for_model(""), "chat_completions");
    }

    #[test]
    fn endpoint_for_model_is_case_sensitive() {
        // Routing is case-sensitive: only the exact lowercase prefix
        // "gpt-5" hits /responses. Mixed-case model names fall through
        // to /chat/completions, which is the safer default — we'd
        // rather retry on the wrong endpoint than silently mis-route
        // an unknown model.
        assert_eq!(endpoint_for_model("GPT-5"), "chat_completions");
        assert_eq!(endpoint_for_model("Gpt-5-mini"), "chat_completions");
        assert_eq!(endpoint_for_model("GPT5"), "chat_completions");
        // Real-world GPT-5 variants: all lowercase prefix matches.
        assert_eq!(endpoint_for_model("gpt-5.5-mini"), "responses");
        assert_eq!(endpoint_for_model("gpt-5-2025-08-07"), "responses");
    }

    #[test]
    fn can_serve_model_accepts_any_model_when_rewrite_is_empty() {
        // Mirrors OpenAiCompatProvider: empty rewrite table accepts
        // every model verbatim (Copilot exposes its own catalog).
        // Without this, the router would skip Copilot for every
        // request unless the operator explicitly enumerated every
        // model name.
        let (_dir, provider) = test_provider(None, None);
        assert!(provider.can_serve_model("claude-opus-4"));
        assert!(provider.can_serve_model("gpt-5"));
        assert!(provider.can_serve_model("any-random-name"));
    }

    #[test]
    fn can_serve_model_filters_by_rewrite_keys_when_set() {
        // When the operator explicitly configures a rewrite table, it
        // becomes an allow-list — same semantics as OpenAiCompat.
        // The router relies on this to skip Copilot for unsupported
        // models without making a doomed HTTP call.
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-sonnet-4.6".to_string(), "copilot-claude".to_string());
        let (_dir, mut provider) = test_provider(None, None);
        provider.model_rewrite = rewrite;
        assert!(provider.can_serve_model("claude-sonnet-4.6"));
        assert!(!provider.can_serve_model("claude-opus-4"));
        assert!(!provider.can_serve_model("gpt-5"));
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
            model_rewrite: HashMap::new(),
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
    async fn spawn_refresh_loop_logs_error_when_refresh_fails() {
        // When the background refresh loop's refresh_token call returns
        // Err (here: 5xx from the Copilot token endpoint), the loop
        // body must surface the failure via tracing::error and keep
        // running (not abort). The next iteration will try again. We
        // hit line 442 (tracing::error arm) on the first failed cycle.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Copilot token endpoint returns 5xx on every call. The loop's
        // refresh_token path classifies this as transient (store stays
        // populated, returns Err), so the loop hits the
        // `tracing::error!` arm at line 442.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream-down"))
            .mount(&server)
            .await;

        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("loop-fail-github", "loop-fail-copilot", 600)),
        );
        let provider = Arc::new(provider);
        let handle = provider.clone().spawn_refresh_loop();

        // Advance enough paused time to wake the loop past its first
        // sleep (refresh_in=1500 → 1440s) and execute the refresh call.
        for _ in 0..30 {
            tokio::time::advance(std::time::Duration::from_secs(60)).await;
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        handle.abort();
        let _ = handle.await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // The store must still hold the original credentials — the
        // refresh failure was transient, so the loop must not have
        // cleared it.
        let disk = provider.state.store.load().unwrap().expect("store intact");
        assert_eq!(disk.github_access_token, "loop-fail-github");
    }

    #[tokio::test(start_paused = true)]
    async fn spawn_refresh_loop_logs_warn_when_bootstrap_fails() {
        // When the background loop finds no credentials on disk, it
        // calls start_bootstrap. If the device-flow endpoint returns
        // 5xx, start_bootstrap returns Err with a non-"already in
        // progress" message, and the loop hits the tracing::warn arm
        // (lines 444-453). We don't assert on the log output — the
        // observable contract is just "loop must keep running" and
        // "no credentials got persisted". Hitting the warn path is
        // enough.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Device code endpoint always 5xx so start_bootstrap fails.
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(503).set_body_string("device-flow-down"))
            .mount(&server)
            .await;
        // Make sure no unexpected calls reach the copilot endpoint.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        // Empty on-disk store → loop hits the bootstrap branch.
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store: store.clone(),
            refresh_lock: Mutex::new(()),
        });
        let provider = Arc::new(CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        });
        let handle = provider.clone().spawn_refresh_loop();

        // The empty-memory branch sleeps only 60s; advance enough
        // paused time for at least one full iteration.
        for _ in 0..5 {
            tokio::time::advance(std::time::Duration::from_secs(60)).await;
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        handle.abort();
        let _ = handle.await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // No tokens got persisted — bootstrap failed and the loop just
        // logs and retries.
        assert!(
            store.load().unwrap().is_none(),
            "store must remain empty after a failed bootstrap attempt"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_returns_401_when_store_is_empty() {
        // When no github token is on disk, refresh_token must NOT run the
        // device flow inline — that would block the request for up to 10
        // minutes. It must fast-fail with a clear 401 so the fallback
        // chain skips Copilot immediately. Bootstrap is owned by
        // `start_bootstrap`. See fix-R2.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // The device-code endpoint must NOT be hit — refresh_token
        // should bail out before even requesting a device code.
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "should-not-be-used",
                "user_code": "SHOULD-NOT-BE-USED",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(0)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        let err = provider.refresh_token().await.err().expect("must fast-fail");
        match err {
            ProxyError::Upstream { status, body } => {
                assert_eq!(status, 401);
                assert!(body.contains("not authenticated"), "body: {body}");
            }
            other => panic!("expected Upstream 401, got: {other:?}"),
        }
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // Memory cache stays empty; bootstrap hasn't run.
        let memory = provider.state.tokens.read().await;
        assert!(memory.is_none(), "refresh_token must not populate cache on fast-fail");
        // Store stays empty (no file written).
        assert!(provider.state.store.load().unwrap().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_warns_and_returns_401_when_store_load_fails() {
        // When the token store exists but is unreadable (here: contains
        // corrupted JSON), refresh_token must NOT silently treat the
        // failure as "no token" and proceed to device flow — it must log
        // the warn and fast-fail with 401 so the operator can see why
        // credentials didn't load and the fallback chain skips Copilot.
        // Device flow is owned by `start_bootstrap` now. See fix-R2.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // The device-code endpoint must NOT be hit.
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "should-not-be-used",
                "user_code": "SHOULD-NOT-BE-USED",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(0)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("github_token.json");
        // Write corrupted JSON so load() returns Err(Json).
        std::fs::write(&store_path, b"{not valid json").unwrap();
        let store = TokenStore::from_path(store_path.clone());
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        let err = provider.refresh_token().await.err().expect("must fast-fail");
        match err {
            ProxyError::Upstream { status, body } => {
                assert_eq!(status, 401);
                assert!(body.contains("not authenticated"), "body: {body}");
            }
            other => panic!("expected Upstream 401, got: {other:?}"),
        }
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // Corrupted file is preserved — refresh_token doesn't rewrite
        // it. Bootstrap (or a manual fix) is what should resolve this.
        let raw = std::fs::read(&store_path).unwrap();
        assert_eq!(raw, b"{not valid json");
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_token_clears_store_and_returns_401_when_copilot_rejects() {
        // When the stored github token is rejected by the Copilot token
        // endpoint (401), refresh_token must clear the store so the
        // background loop / admin endpoint can re-bootstrap, but it
        // must NOT inline a device flow — that would block the request
        // for up to 10 minutes. Return Err 401 instead. See fix-R2.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // Copilot token endpoint rejects the stored token.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token stale-github"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"message": "expired"})))
            .expect(1)
            .mount(&server)
            .await;
        // Device flow must NOT be triggered inline.
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "should-not-be-used",
                "user_code": "SHOULD-NOT-BE-USED",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(0)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        // Start with a stale github token so refresh_token will try the
        // endpoint and fail.
        store
            .save(&stored_tokens("stale-github", "old-copilot", -10))
            .unwrap();
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(Some(stored_tokens(
                "stale-github",
                "old-copilot",
                -10,
            ))),
            store: store.clone(),
            refresh_lock: Mutex::new(()),
        });
        let provider = CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        };
        let _auto_advance = spawn_test_time_advance();

        let err = provider.refresh_token().await.err().expect("must fast-fail");
        match err {
            ProxyError::Upstream { status, body } => {
                assert_eq!(status, 401);
                assert!(
                    body.contains("rejected") || body.contains("trigger bootstrap"),
                    "body should explain the recovery path, got: {body}"
                );
            }
            other => panic!("expected Upstream 401, got: {other:?}"),
        }
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        // Store is cleared so the background loop sees no-token and
        // triggers bootstrap on its next iteration.
        assert!(
            store.load().unwrap().is_none(),
            "store must be cleared after AuthRejected"
        );
        // In-memory state stays stale (not overwritten with Err).
        let memory = provider.state.tokens.read().await;
        assert_eq!(
            memory.as_ref().unwrap().github_access_token,
            "stale-github",
            "memory cache is not modified on the rejection path"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn start_bootstrap_runs_device_flow_and_persists_tokens() {
        // Exercises start_bootstrap's spawned task: request device
        // code, return DeviceCodeResponse immediately, then complete
        // the device flow + Copilot token exchange in the background
        // and persist the result. After the loop completes, the store
        // and in-memory cache must both hold the new tokens. See fix-R2.
        // Uses real (non-paused) tokio time because the spawned task's
        // poll loop sleeps on tokio::time::sleep — pausing time would
        // require manually advancing the clock from the test body, but
        // the spawned task needs CPU time too.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "sb-device",
                "user_code": "SB-CODE",
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
                "access_token": "bootstrap-github"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token bootstrap-github"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "bootstrap-copilot",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store: store.clone(),
            refresh_lock: Mutex::new(()),
        });
        let provider = Arc::new(CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        });

        // start_bootstrap returns immediately with the device code.
        let dc = provider.clone().start_bootstrap().await.unwrap();
        assert_eq!(dc.user_code, "SB-CODE");
        assert_eq!(dc.device_code, "sb-device");
        // Keep the env var set: the spawned task reads github_base_url()
        // each time it polls, so we have to keep the override alive
        // until bootstrap finishes (not just until start_bootstrap
        // returns the device code).

        // Poll the memory cache until the spawned bootstrap task
        // completes (or we time out). With real time + interval=5s
        // +1=6s poll, this should complete within ~6s.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            async {
                loop {
                    if provider.state.tokens.read().await.is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            },
        )
        .await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");
        assert!(result.is_ok(), "bootstrap task did not complete in 30s");

        // Both store and memory cache must hold the new tokens.
        let memory = provider.state.tokens.read().await;
        assert_eq!(
            memory.as_ref().unwrap().copilot_token,
            "bootstrap-copilot",
            "background bootstrap must populate memory cache"
        );
        assert_eq!(
            memory.as_ref().unwrap().github_access_token,
            "bootstrap-github"
        );
        drop(memory);
        let disk = store.load().unwrap().unwrap();
        assert_eq!(disk.copilot_token, "bootstrap-copilot");
        assert_eq!(disk.github_access_token, "bootstrap-github");
    }

    #[tokio::test(start_paused = true)]
    async fn start_bootstrap_returns_already_in_progress_when_concurrent() {
        // When start_bootstrap is called while another bootstrap is
        // already running, the second call must fail fast with a clear
        // "already in progress" error instead of kicking off a second
        // device flow. See fix-R2.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        // The device-code endpoint must NOT be hit: the second call's
        // try_lock fails before it ever reaches request_device_code.
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "should-not-be-used",
                "user_code": "SHOULD-NOT-BE-USED",
                "verification_uri": "https://example.test/device",
                "expires_in": 600,
                "interval": 5,
            })))
            .expect(0)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(None),
            store,
            refresh_lock: Mutex::new(()),
        });
        let provider = Arc::new(CopilotProvider {
            name: "copilot".to_string(),
            vscode_version: "1.95.0".to_string(),
            account_type: "individual".to_string(),
            model_rewrite: HashMap::new(),
            http: reqwest::Client::new(),
            state,
            api_base_override: Some(server.uri()),
            copilot_token_url: format!("{}/copilot_internal/v2/token", server.uri()),
        });

        // Hold the lock manually to simulate an in-flight bootstrap.
        let _lock_held = provider.state.refresh_lock.lock().await;

        let err = provider
            .clone()
            .start_bootstrap()
            .await
            .err()
            .expect("second start_bootstrap must fail when lock held");
        match err {
            ProxyError::Other(msg) => {
                assert!(
                    msg.to_string().contains("already in progress"),
                    "expected 'already in progress', got: {msg}"
                );
            }
            other => panic!("expected Other with 'already in progress', got: {other:?}"),
        }
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");
        // Drop the lock guard before the mock verification runs so the
        // task holding the lock (none here) doesn't race with mock
        // teardown.
        drop(_lock_held);
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
            model_rewrite: HashMap::new(),
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
