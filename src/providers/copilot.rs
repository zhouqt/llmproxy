//! GitHub Copilot provider.
//!
//! Reference: copilot-api-py/src/lib/token.py, src/services/copilot/*
//!
//! Flow:
//! 1. GitHub device flow → github_access_token
//! 2. Exchange github token at api.github.com/copilot_internal/v2/token → copilot_token
//! 3. Use copilot_token with required Copilot headers against api.githubcopilot.com

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};

use crate::anthropic::MessagesRequest;
use crate::error::{ProxyError, Result};
use crate::oauth::device_flow::{request_device_code, DeviceCodeResponse};
use crate::oauth::token_store::{StoredTokens, TokenStore};
use crate::oauth::USER_AGENT;
use crate::providers::openai_compat::OpenAiSseToAnthropic;
use crate::providers::{Provider, ProviderOutput};

const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const GITHUB_API_VERSION: &str = "2025-04-01";
const COPILOT_INTERNAL_TOKEN_URL: &str =
    "https://api.github.com/copilot_internal/v2/token";

/// A Copilot-discovered model stripped to the fields `/v1/models` needs.
/// Policy state and capabilities from the upstream response are discarded
/// — the cache only keeps entries whose `policy.state == "enabled"`, so
/// disabled/preview models are never advertised.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CopilotModel {
    pub id: String,
    pub name: String,
    pub vendor: String,
    /// Endpoints Copilot advertises for this model, e.g.
    /// `["/chat/completions", "/responses"]`. Drives endpoint selection in
    /// [`CopilotProvider::endpoint_for_model`]: routing is responses-default,
    /// chat chosen only when the model advertises chat alone, or advertises
    /// both while matching the EnterWorktree whitelist (`gpt-5*` — the
    /// Responses path risks malformed `name` values that fail Claude Code's
    /// `Kor(name)` validator). Missing/empty on older Copilot payloads is
    /// schema drift, treated like a cold cache: responses default.
    #[serde(default)]
    pub supported_endpoints: Vec<String>,
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
    cached_models: RwLock<Option<Vec<CopilotModel>>>,
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
    /// Copilot /models endpoint returned 401 or 403: the token is
    /// rejected. The caller should clear the model cache so stale
    /// entries are not advertised.
    Auth(String),
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
            CopilotFetchError::Auth(msg) => write!(f, "{msg}"),
            CopilotFetchError::Transient(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for CopilotFetchError {}

/// Whether `/responses` triggers malformed `EnterWorktree` tool arguments
/// for this model — empty `name`s, absolute paths, or both mutex keys
/// non-empty, all of which fail Claude Code's `Kor(name)` validator. When
/// such a model advertises chat in the `/models` cache we route it to
/// `/chat/completions` as a whitelist fallback.
///
/// The whitelist only takes effect when the cache is hot AND the entry
/// advertises both endpoints. On a cold cache even these models go to
/// `/responses` — the same as the historical `gpt-5` prefix heuristic, so
/// this is not a regression. Kept until a real-device smoke test (Phase 4)
/// proves the Responses path is safe, then narrowed.
fn prefers_chat_via_enterworktree(model: &str) -> bool {
    model.starts_with("gpt-5")
}

/// Location of the persisted `/models` snapshot. Co-located with the token
/// store (`github_token.json`) so tempdir-scoped tests stay isolated.
fn models_cache_path_for(store: &TokenStore) -> PathBuf {
    store
        .path()
        .parent()
        .unwrap_or(Path::new("."))
        .join("copilot_models.json")
}

/// Reads a previously persisted `/models` snapshot into memory at cold
/// start. Missing file → `None` (true cold start); unreadable or corrupt
/// → warn + `None` (a bad cache must never block startup, same contract as
/// the token store).
fn load_models_from_disk(path: &Path) -> Option<Vec<CopilotModel>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "copilot: failed to read models cache; starting cold"
            );
            return None;
        }
    };
    match serde_json::from_str::<Vec<CopilotModel>>(&raw) {
        Ok(models) => Some(models),
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "copilot: failed to parse models cache; starting cold"
            );
            None
        }
    }
}

impl CopilotProvider {
    /// Pick the upstream endpoint for a model.
    ///
    /// Defaults to Copilot's `/responses` endpoint - the ecosystem default
    /// (Codex dropped `/chat/completions` in Feb 2026; Assistants API shuts
    /// down 2026-08-26) and the only endpoint for responses-only models such
    /// as grok-4.5, which used to 400 on a cold cache. Three
    /// cache-advertised exceptions:
    ///
    /// - chat-only entry -> `/chat/completions`;
    /// - dual-endpoint entry for an EnterWorktree-whitelisted model (`gpt-5*`)
    ///   -> `/chat/completions` (the Responses path makes these emit
    ///   malformed `EnterWorktree` tool args - see
    ///   [`prefers_chat_via_enterworktree`]);
    /// - everything else -> `/responses`.
    ///
    /// `Some(entry)` with an empty `supported_endpoints` is schema drift on
    /// an older Copilot payload and is treated as a cold cache ->
    /// `/responses`. This silently changes behavior for non-gpt-5 models that
    /// used to fall back to `/chat/completions`; that is the intended
    /// direction of travel.
    async fn endpoint_for_model(&self, model: &str) -> &'static str {
        let entry = self
            .cached_models()
            .await
            .and_then(|ms| ms.into_iter().find(|m| m.id == model));
        match entry {
            Some(e) => {
                let chat = e
                    .supported_endpoints
                    .iter()
                    .any(|s| s == "/chat/completions" || s == "chat_completions");
                let responses = e
                    .supported_endpoints
                    .iter()
                    .any(|s| s == "/responses" || s == "responses");
                if chat && !responses {
                    "chat_completions"
                } else if chat && responses && prefers_chat_via_enterworktree(model) {
                    "chat_completions"
                } else {
                    "responses"
                }
            }
            None => "responses", // cold cache / unlisted -> responses default
        }
    }

    pub fn new(
        name: String,
        vscode_version: String,
        account_type: String,
        model_rewrite: HashMap<String, String>,
        http: reqwest::Client,
    ) -> Result<Self> {
        let store = TokenStore::new()?;
        Ok(Self::from_store(
            store,
            name,
            vscode_version,
            account_type,
            model_rewrite,
            http,
        ))
    }

    /// Test-only: like [`Self::new`] but with a caller-provided token
    /// store, so the cold-start disk-cache load can be exercised against
    /// a tempdir instead of the real `XDG_DATA_HOME`.
    #[cfg(test)]
    pub(crate) fn new_with_store(store: TokenStore) -> Self {
        Self::from_store(
            store,
            "copilot".to_string(),
            "1.95.0".to_string(),
            "individual".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
    }

    fn from_store(
        store: TokenStore,
        name: String,
        vscode_version: String,
        account_type: String,
        model_rewrite: HashMap<String, String>,
        http: reqwest::Client,
    ) -> Self {
        let initial = store.load().unwrap_or_else(|e| {
            tracing::warn!(
                provider = "copilot",
                error = %e,
                path = %store.path().display(),
                "failed to read token store; treating as empty"
            );
            None
        });
        let cached_models = load_models_from_disk(&models_cache_path_for(&store));
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(initial),
            store,
            refresh_lock: Mutex::new(()),
            cached_models: RwLock::new(cached_models),
        });
        Self {
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
        }
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

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url())
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

    /// Returns true when the copilot token has expired.
    ///
    /// The 60-second buffer that the old `expires_at - now < 60`
    /// provided was a refresh heuristic, not an auth-freshness check;
    /// that heuristic lives in the refresh loop's `refresh_in - 60`
    /// interval logic instead.
    fn token_expired(tokens: &StoredTokens, now_unix: i64) -> bool {
        tokens.copilot_expires_at <= now_unix
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
                    Self::token_expired(t, now)
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
            // fetch_copilot_token never returns Auth, but Rust needs the
            // match to be exhaustive. Treat it as transient if it appears.
            Err(CopilotFetchError::Auth(msg)) => {
                tracing::warn!(
                    provider = "copilot",
                    reason = %msg,
                    "copilot token refresh failed (unexpected auth); keeping stored credentials"
                );
                Err(ProxyError::Other(anyhow::anyhow!(msg)))
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
                Ok(()) => tracing::debug!(provider = %name, "copilot bootstrap completed"),
                Err(e) => tracing::debug!(provider = %name, error = %e, "copilot bootstrap failed"),
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
        let copilot_token = new_tokens.copilot_token.clone();
        self.state.store.save(&new_tokens)?;
        *self.state.tokens.write().await = Some(new_tokens);

        // A fresh OAuth cycle may belong to a different account with a
        // different model set. Drop the on-disk cache now so a cold start
        // between this point and the fetch below cannot route on the
        // previous account's endpoint metadata.
        let _ = std::fs::remove_file(models_cache_path_for(&self.state.store));

        // Token is fresh — populate the model list immediately without
        // calling ensure_token (which would re-enter refresh_lock).
        self.cache_models_with_token(&copilot_token).await;

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
            // Sanitize the body so a 5xx HTML error page (e.g. the
            // GitHub 502 with its inline stylesheet and image refs)
            // doesn't pollute the logs. Keep only the first short,
            // tag-free line as a hint; URLs / images / links / scripts
            // are stripped entirely.
            let hint = crate::util::summarize_for_log(&text, "<empty body>");
            // 401 / 403 / 404 mean the stored github token is invalid or
            // lost access — the operator must re-authenticate. Anything
            // else (5xx, 408, 429) is treated as transient so we don't
            // wipe the store on a flaky upstream.
            if matches!(status.as_u16(), 401 | 403 | 404) {
                return Err(CopilotFetchError::AuthRejected {
                    status: status.as_u16(),
                    body: hint,
                });
            }
            return Err(CopilotFetchError::Transient(format!(
                "copilot token fetch failed: {status} {hint}"
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
            // At the top of the first iteration, BEFORE the first sleep:
            // If the on-disk store has credentials, kick a best-effort
            // cache_models(). No lock is held here, so cache_models() can
            // call ensure_token() safely.
            let has_credentials = self
                .state
                .store
                .load()
                .ok()
                .flatten()
                .is_some();
            if has_credentials {
                self.cache_models().await;
            }

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
                    } else {
                        // Refresh the model list in the background so
                        // /v1/models stays current.
                        self.cache_models().await;
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

    /// Return a clone of the cached Copilot model list, if available.
    pub async fn cached_models(&self) -> Option<Vec<CopilotModel>> {
        self.state.cached_models.read().await.clone()
    }

    /// Fetch models from the Copilot API and update the cache.
    ///
    /// Best-effort: obtains a token via `ensure_token`, then delegates to
    /// `cache_models_with_token`. Logs a warning on failure but does not
    /// propagate the error — a stale cache (or no cache) is better than
    /// blocking the caller on a transient `/models` failure.
    pub async fn cache_models(&self) {
        match self.ensure_token().await {
            Ok(t) => self.cache_models_with_token(&t).await,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to get copilot token for model refresh; keeping stale cache"
                );
            }
        }
    }

    /// Fetch models using the given token and store them directly.
    ///
    /// Does not call `ensure_token` — the caller is responsible for
    /// providing a valid token. This avoids re-entering `refresh_lock`
    /// when the caller already holds a fresh token (e.g. after bootstrap
    /// or at startup in the refresh loop).
    pub async fn cache_models_with_token(&self, token: &str) {
        match self.fetch_models(token).await {
            Ok(models) => {
                // Background cache maintenance fires on every refresh-loop
                // iteration, so this is TRACE — the count is useful only
                // when debugging model-cache issues, and at INFO/DEBUG it
                // would just pollute the log.
                tracing::trace!(
                    model_count = models.len(),
                    "copilot models cached"
                );
                // Persist FIRST, then move into memory: `save_models_to_disk`
                // borrows, `Some(models)` moves.
                self.save_models_to_disk(&models);
                *self.state.cached_models.write().await = Some(models);
            }
            Err(CopilotFetchError::Auth(msg)) => {
                tracing::warn!("{msg}");
                // Deliberately keep the on-disk cache: it is endpoint-routing
                // metadata, independent of authentication state. Only memory is
                // cleared so `/v1/models` stops advertising models we currently
                // cannot reach. The on-disk cache is dropped at the next fresh
                // OAuth cycle (see `complete_bootstrap`), which is when the
                // account can actually have changed.
                *self.state.cached_models.write().await = None;
            }
            Err(e) => {
                tracing::warn!(error = %e, "copilot /models fetch failed; keeping stale cache");
            }
        }
    }

    /// Persists the `/models` snapshot next to the token store so a cold
    /// restart can route without a warm-up fetch. Best effort: a failed
    /// write only logs; the in-memory cache still serves. Entries are
    /// sorted by `id` so refresh cycles do not churn the file.
    ///
    /// The file holds no secrets (model ids/names/vendors only), so unlike
    /// the 0o600 token file it is explicitly widened to 0o644 — a
    /// low-privilege process must be able to read a cache written by
    /// another user, or it silently falls back to a cold start.
    fn save_models_to_disk(&self, models: &[CopilotModel]) {
        let path = models_cache_path_for(&self.state.store);
        let mut sorted: Vec<&CopilotModel> = models.iter().collect();
        sorted.sort_by(|a, b| a.id.cmp(&b.id));
        let raw = match serde_json::to_vec_pretty(&sorted) {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!(error = %e, "copilot: failed to serialize models cache");
                return;
            }
        };
        if let Err(e) = crate::oauth::token_store::write_atomic(&path, &raw) {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "copilot: failed to persist models cache"
            );
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644));
        }
    }

    async fn fetch_models(
        &self,
        token: &str,
    ) -> std::result::Result<Vec<CopilotModel>, CopilotFetchError> {
        let url = self.models_url();
        let resp = match self
            .http
            .get(&url)
            .headers(self.headers(token))
            .header("accept", "application/json")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(CopilotFetchError::Transient(format!(
                    "network error fetching copilot models: {e}"
                )));
            }
        };
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let code = status.as_u16();
            return Err(match code {
                401 | 403 => {
                    tracing::warn!(
                        status = code,
                        body = %text,
                        "copilot token rejected by /models; clearing cache"
                    );
                    CopilotFetchError::Auth(format!("copilot /models returned {status}"))
                }
                s if s >= 500 => {
                    CopilotFetchError::Transient(format!("copilot /models returned {s}"))
                }
                s => CopilotFetchError::Transient(format!(
                    "copilot /models returned {s}: {text}"
                )),
            });
        }
        let body: Value = resp.json().await.map_err(|e| {
            CopilotFetchError::Transient(format!(
                "copilot models response not valid JSON: {e}"
            ))
        })?;
        let data = body
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| {
                CopilotFetchError::Transient(
                    "copilot models response missing 'data' array".to_string(),
                )
            })?;
        let enabled: Vec<CopilotModel> = data
            .iter()
            .filter(|entry| {
                entry
                    .get("policy")
                    .and_then(|p| p.get("state"))
                    .and_then(|s| s.as_str())
                    == Some("enabled")
            })
            .filter_map(|entry| {
                match serde_json::from_value::<CopilotModel>(entry.clone()) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        let id = entry
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?");
                        tracing::warn!(
                            entry_id = id,
                            error = %e,
                            "dropping copilot model entry: schema drift"
                        );
                        None
                    }
                }
            })
            .collect();
        Ok(enabled)
    }

    async fn send_with_token(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<reqwest::Response> {
        let token = self.ensure_token().await?;
        let resp = self
            .http
            .post(url)
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

    /// Complete path via the Responses API. Reached when
    /// [`endpoint_for_model`](Self::endpoint_for_model) selects `/responses`
    /// — either because the `/models` cache advertises only `/responses` for
    /// the model, or as the cold-cache fallback for `gpt-5*`.
    async fn complete_responses(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = self.merged_rewrite(model_rewrite);
        let mut responses_req =
            crate::conversion::anthropic_to_responses_request(req, &merged)?;
        responses_req.stream = false;
        // PR-9: Copilot strips high-risk fields before serialization.
        // Copilot's request-side tolerance is unverified (closed source,
        // strict validators, unknown models return 200 error envelopes);
        // sending these would risk 400 errors. service_tier is the most
        // likely to 400; prediction/logit_bias/logprobs/metadata/
        // safety_identifier/verbosity are scrubbed defensively.
        strip_high_risk_fields_responses(&mut responses_req);
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
        let merged = self.merged_rewrite(model_rewrite);
        let mut responses_req =
            crate::conversion::anthropic_to_responses_request(req, &merged)?;
        responses_req.stream = true;
        // PR-9: strip high-risk fields before serialization (see
        // complete_responses for rationale).
        strip_high_risk_fields_responses(&mut responses_req);
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
/// PR-9: strip high-risk fields from a Chat request before sending to
/// Copilot. Copilot's request-side tolerance is unverified (closed
/// source, strict validators, unknown models return 200 error
/// envelopes); these fields have the highest 400 risk on Copilot.
///
/// Stripped (all PR-9 P2 high-risk):
/// - `service_tier` — OpenAI tier hint; would 400 on Copilot
/// - `prediction` — speculative content
/// - `logit_bias` — token-bias map
/// - `logprobs` / `top_logprobs` — log probability outputs
/// - `metadata` — request tags (Copilot doesn't accept)
/// - `safety_identifier` — user identity, Copilot may differ
/// - `verbosity` — text verbosity
fn strip_high_risk_fields_chat(req: &mut crate::openai::ChatRequest) {
    req.service_tier = None;
    req.prediction = None;
    req.logit_bias = None;
    req.logprobs = None;
    req.top_logprobs = None;
    req.metadata = None;
    req.safety_identifier = None;
    req.verbosity = None;
}

/// PR-9: strip high-risk fields from a Responses request before sending
/// to Copilot. The Responses path puts some fields under `extra.text`
/// (verbosity), so we have to clear both the typed field and the
/// nested text object.
fn strip_high_risk_fields_responses(req: &mut crate::responses::ResponsesRequest) {
    req.service_tier = None;
    // verbosity lives under `extra.text.verbosity` on Responses; clear
    // it there too. `extra.text.format` is preserved (Copilot accepts
    // json_schema shaping).
    if let Some(text_obj) = req.extra.get_mut("text").and_then(|v| v.as_object_mut()) {
        text_obj.remove("verbosity");
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn merged_rewrite<'a>(
        &'a self,
        runtime: &'a HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(runtime.iter().map(|(k, v)| (k.clone(), v.clone())));
        merged
    }

    fn can_serve_model(&self, model: &str) -> bool {
        // Mirrors OpenAiCompatProvider: empty rewrite table accepts any
        // model verbatim (Copilot exposes its own catalog); a non-empty
        // table is an explicit allow-list — see fix-R11.
        self.model_rewrite.is_empty() || self.model_rewrite.contains_key(model)
    }

    async fn list_models(&self) -> Option<Vec<serde_json::Value>> {
        let cached = self.cached_models().await?;
        Some(
            cached
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "id": m.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": m.vendor,
                        "display_name": m.name,
                    })
                })
                .collect(),
        )
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = self.merged_rewrite(model_rewrite);
        let upstream_model = merged
            .get(&req.model)
            .map(String::as_str)
            .unwrap_or(&req.model);
        let endpoint = self.endpoint_for_model(upstream_model).await;
        if endpoint == "responses" {
            return self.complete_responses(req, &merged).await;
        }

        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, &merged, false)?;
        openai_req.stream = false;
        openai_req.stream_options = None;
        // PR-9: strip high-risk fields before serialization (see
        // complete_responses for rationale).
        strip_high_risk_fields_chat(&mut openai_req);
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
        let msg_id = crate::conversion::make_message_id();
        let anthropic =
            crate::conversion::openai_to_anthropic_response(&chat, &req.model, &msg_id)?;
        Ok(ProviderOutput::Json(serde_json::to_value(anthropic)?))
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = self.merged_rewrite(model_rewrite);
        let upstream_model = merged
            .get(&req.model)
            .map(String::as_str)
            .unwrap_or(&req.model);
        let endpoint = self.endpoint_for_model(upstream_model).await;
        if endpoint == "responses" {
            return self.stream_responses(req, &merged).await;
        }

        let mut openai_req =
            crate::conversion::anthropic_to_openai_request(req, &merged, false)?;
        openai_req.stream = true;
        openai_req.stream_options = Some(crate::openai::StreamOptions {
            include_usage: true,
            // PR-9: include_obfuscation has no Anthropic source; keep
            // absent on the wire (upstream default false).
            include_obfuscation: None,
        });
        // PR-9: strip high-risk fields before serialization (see
        // complete_responses for rationale).
        strip_high_risk_fields_chat(&mut openai_req);
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
    use crate::test_support::JsonFieldAbsent;
    use futures_util::StreamExt;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── PR-9 · high-risk field stripping ────────────────────────────────

    /// PR-9: `strip_high_risk_fields_chat` clears every P2 high-risk
    /// field Copilot's strict validators are most likely to 400 on.
    /// The Anthropic→OpenAI translator may set some of these fields
    /// (service_tier, verbosity, safety_identifier); the Copilot path
    /// must drop them before serialization.
    #[test]
    fn strip_high_risk_fields_chat_clears_all_listed_fields() {
        let mut req = crate::openai::ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: false,
            stream_options: None,
            tools: None,
            tool_choice: None,
            user: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            service_tier: Some("auto".into()),
            parallel_tool_calls: Some(false),
            safety_identifier: Some("user-42".into()),
            verbosity: Some("low".into()),
            n: Some(2),
            logit_bias: Some(std::collections::HashMap::from([("x".into(), 1)])),
            logprobs: Some(true),
            top_logprobs: Some(5),
            prediction: Some(json!({"type": "content", "content": "x"})),
            metadata: Some(json!({"trace": "abc"})),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(-0.5),
            seed: Some(12345),
            extra: json!({}),
        };
        strip_high_risk_fields_chat(&mut req);
        assert!(req.service_tier.is_none());
        assert!(req.prediction.is_none());
        assert!(req.logit_bias.is_none());
        assert!(req.logprobs.is_none());
        assert!(req.top_logprobs.is_none());
        assert!(req.metadata.is_none());
        assert!(req.safety_identifier.is_none());
        assert!(req.verbosity.is_none());
        // Non-stripped fields are preserved (Copilot accepts these).
        assert_eq!(req.parallel_tool_calls, Some(false));
        assert_eq!(req.presence_penalty, Some(0.5));
        assert_eq!(req.frequency_penalty, Some(-0.5));
        assert_eq!(req.seed, Some(12345));
    }

    /// PR-9: `strip_high_risk_fields_responses` clears service_tier and
    /// the nested `extra.text.verbosity`. `extra.text.format` is
    /// preserved (Copilot accepts json_schema shaping).
    #[test]
    fn strip_high_risk_fields_responses_clears_service_tier_and_text_verbosity() {
        use serde_json::Value;
        let mut req = crate::responses::ResponsesRequest {
            model: "gpt-5".into(),
            input: vec![],
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: Some(false),
            user: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            store: None,
            service_tier: Some("auto".into()),
            extra: Value::Object({
                let mut m = serde_json::Map::new();
                m.insert(
                    "text".into(),
                    json!({"verbosity": "high", "format": {"type": "json_schema"}}),
                );
                m
            }),
        };
        strip_high_risk_fields_responses(&mut req);
        assert!(req.service_tier.is_none());
        // format is preserved.
        assert_eq!(
            req.extra["text"]["format"]["type"],
            "json_schema",
            "format must survive the strip; got {}",
            req.extra
        );
        // verbosity is gone.
        assert!(
            req.extra["text"].get("verbosity").is_none(),
            "verbosity must be cleared from text; got {}",
            req.extra["text"]
        );
    }

    /// PR-9 (plan:437/427): the Copilot mock double-assertion — "accepts
    /// without 400". A request carrying the low-risk P2 fields that
    /// survive stripping (n/presence_penalty/frequency_penalty/seed, plus
    /// PR-8's parallel_tool_calls) must be accepted by Copilot with a
    /// 200. This is the "wiremock 200 接受" half: Copilot tolerates
    /// these fields (they are NOT stripped because the plan's Opus
    /// survey found Copilot accepts them), so the request succeeds.
    ///
    /// The conversion layer never injects these fields today (plan M3c
    /// "明确不写" — Anthropic has no source for them), so we build the
    /// ChatRequest by hand to simulate a future injection, strip the
    /// high-risk subset exactly as `complete_chat` does, and send it.
    /// The final provider request JSON is asserted via wiremock matchers
    /// (plan:538): `body_partial_json` locks the low-risk fields present,
    /// `JsonFieldAbsent` locks the stripped high-risk ones absent.
    #[tokio::test]
    async fn copilot_accepts_p2_fields_without_400() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer copilot-token"))
            // plan:538 — assert the *final provider request JSON* via
            // wiremock matchers, not just serde round-trip: body_partial_json
            // locks the low-risk P2 fields present on the wire (n is a
            // plan:433 "写" field, not stripped), JsonFieldAbsent locks the
            // stripped high-risk ones absent. A shape mismatch makes
            // wiremock return 404 (no matching mock) and fail the 200
            // assertion below.
            .and(body_partial_json(json!({
                "n": 2,
                "presence_penalty": 0.5,
                "frequency_penalty": -0.5,
                "seed": 12345,
                "parallel_tool_calls": false,
            })))
            .and(JsonFieldAbsent("service_tier"))
            .and(JsonFieldAbsent("prediction"))
            .and(JsonFieldAbsent("logit_bias"))
            .and(JsonFieldAbsent("logprobs"))
            .and(JsonFieldAbsent("top_logprobs"))
            .and(JsonFieldAbsent("metadata"))
            .and(JsonFieldAbsent("safety_identifier"))
            .and(JsonFieldAbsent("verbosity"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_response("ok")))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );

        let mut req = crate::openai::ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![crate::openai::ChatMessage::User {
                content: crate::openai::UserContent::Text("hi".into()),
                name: None,
            }],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: false,
            stream_options: None,
            tools: None,
            tool_choice: None,
            user: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            service_tier: Some("auto".into()),
            parallel_tool_calls: Some(false),
            safety_identifier: Some("user-42".into()),
            verbosity: Some("low".into()),
            n: Some(2),
            logit_bias: Some(std::collections::HashMap::from([("x".into(), 1)])),
            logprobs: Some(true),
            top_logprobs: Some(5),
            prediction: Some(json!({"type": "content", "content": "x"})),
            metadata: Some(json!({"trace": "abc"})),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(-0.5),
            seed: Some(12345),
            extra: json!({}),
        };
        // Mirror complete_chat: strip high-risk fields before sending.
        strip_high_risk_fields_chat(&mut req);
        let body = serde_json::to_value(&req).unwrap();

        // Wire shape is enforced by the mock's matchers above: a request
        // that leaks a stripped field or drops a low-risk one gets a 404
        // (no matching mock) and fails the 200 assertion below.
        let resp = provider
            .send_with_token(&provider.chat_url(), &body)
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    /// PR-9 (plan:437/427): the Copilot mock double-assertion — "rejects
    /// surfaces fallback". If Copilot 400s (e.g. a field it does NOT
    /// tolerate, or an unknown-model envelope), the provider must
    /// surface the upstream error unchanged (`ProxyError::Upstream`) so
    /// the router can decide fallback to the next provider in the chain.
    /// This is the "wiremock 400 验证 fallback" half.
    ///
    /// It goes through `complete()` (the production path) rather than
    /// `send_with_token` so the 400 → `ProxyError::Upstream` conversion
    /// is what's asserted (send_with_token returns a raw Response and
    /// never produces that error). The request carries no P2 fields
    /// because the conversion layer never injects them (plan M3c
    /// "明确不写"), so a field-triggered 400 cannot be produced through
    /// the production path — this test locks the error-surfacing
    /// mechanism the router depends on for fallback.
    #[tokio::test]
    async fn copilot_rejects_p2_fields_surfaces_fallback() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("unsupported parameter"))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        seed_chat_only_cache(&provider, "claude-model").await;

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("Copilot 400 must fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 400, ref body } if body == "unsupported parameter"
        ));
    }

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
            cached_models: RwLock::new(None),
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

    /// Pins the chat endpoint for `model`: these tests exercise the Chat
    /// Completions conversion path, which under responses-default routing
    /// needs the /models cache to advertise chat-only for the upstream
    /// model (otherwise a cold cache would send the request to /responses).
    async fn seed_chat_only_cache(provider: &CopilotProvider, model: &str) {
        *provider.state.cached_models.write().await = Some(vec![CopilotModel {
            id: model.to_string(),
            name: model.to_string(),
            vendor: "v".to_string(),
            supported_endpoints: vec!["/chat/completions".to_string()],
        }]);
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

    /// Regression for the operator-noise issue: GitHub's `/copilot_internal/v2/token`
    /// endpoint returns a full HTML page (the "Unicorn!" 502) on transient
    /// failures. Without `summarize_http_body`, that entire HTML document —
    /// inline stylesheets, image refs, links — would land verbatim in the
    /// warn log. The hint must drop everything but the first short,
    /// meaningful line.
    #[tokio::test]
    async fn fetch_copilot_token_5xx_html_body_is_summarized() {
        let github_502_html = "<!DOCTYPE html>\n\
            <!--\nHello future GitHubber! I bet you're here to remove those nasty inline styles,\n\
DRY up these templates and make 'em nice and re-usable, right?\n\
Please, don't. https://github.com/styleguide/templates/2.0\n-->\n\
<html>\n\
  <head>\n\
    <title>Unicorn! &middot; GitHub</title>\n\
    <style type=\"text/css\" media=\"screen\">\n\
      body { font-family: sans-serif; background: url(https://example.test/bg.png); }\n\
    </style>\n\
    <link rel=\"stylesheet\" href=\"https://example.test/main.css\">\n\
  </head>\n\
  <body>\n\
    <a href=\"https://status.github.com\"><img src=\"https://example.test/unicorn.png\" alt=\"Unicorn!\"></a>\n\
  </body>\n\
</html>\n";
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(502).set_body_string(github_502_html))
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(Some(&server), None);
        let err = provider.fetch_copilot_token("github").await.unwrap_err();
        let msg = err.to_string();
        // Status code preserved.
        assert!(msg.contains("502"), "missing 502 in: {msg}");
        // The "Unicorn!" page title is the only meaningful text.
        assert!(msg.contains("Unicorn!"), "expected page title in: {msg}");
        // None of the noise should leak through.
        assert!(!msg.contains("font-family"), "CSS leaked: {msg}");
        assert!(!msg.contains("example.test"), "URL leaked: {msg}");
        assert!(!msg.contains("background:"), "CSS url() leaked: {msg}");
        assert!(!msg.contains("github.com/styleguide"), "comment URL leaked: {msg}");
        assert!(!msg.contains(".css"), "stylesheet URL leaked: {msg}");
        assert!(!msg.contains(".png"), "image URL leaked: {msg}");
        assert!(!msg.contains("<html"), "raw HTML leaked: {msg}");
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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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
        seed_chat_only_cache(&provider, "copilot-model").await;
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
        // Cold-cache fallback: with no /models cache populated, GPT-5.x
        // falls back to /responses (the historical default). When the cache
        // IS populated and advertises /chat/completions, the cache-aware path
        // prefers chat — see complete_routes_gpt5_to_chat_when_cache_advertises_it.
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
        // Cold-cache regression guard: with no /models cache, GPT-5 falls
        // back to /responses and must NOT hit /chat/completions. Asserts
        // `expect(1)` on /responses AND `expect(0)` on /chat/completions via
        // a mock that 500s if hit. (When the cache advertises chat, the
        // cache-aware path does hit chat — covered by a separate test.)
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
    async fn complete_routes_gpt5_to_chat_when_cache_advertises_it() {
        // Cache-aware routing: when Copilot's /models cache lists GPT-5 with
        // /chat/completions in supported_endpoints, we MUST prefer chat over
        // /responses. This is the root-cause fix for EnterWorktree
        // "Invalid tool parameters" — the Responses path makes GPT-5.x emit
        // malformed EnterWorktree `name` values (empty strings / paths) that
        // fail Claude Code's Kor(name) validator; chat doesn't. Parity with
        // llmgateway. See docs/REVIEWS/llmproxy-vs-llmgateway-translation-diff.md.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({"model": "gpt-5", "stream": false})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello-from-chat"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        // Prime the cache: GPT-5 advertises BOTH endpoints → chat wins.
        *provider.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "gpt-5".to_string(),
            name: "GPT-5".to_string(),
            vendor: "openai".to_string(),
            supported_endpoints: vec![
                "/chat/completions".to_string(),
                "/responses".to_string(),
            ],
        }]);
        let mut gpt5_req = request(false);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider
            .complete(&gpt5_req, &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "hello-from-chat");
        });
    }

    #[tokio::test]
    async fn stream_routes_gpt5_to_chat_when_cache_advertises_it() {
        // Streaming twin of the above: cache advertises chat → stream goes
        // to /chat/completions, not /responses.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        *provider.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "gpt-5".to_string(),
            name: "GPT-5".to_string(),
            vendor: "openai".to_string(),
            supported_endpoints: vec!["/chat/completions".to_string()],
        }]);
        let mut gpt5_req = request(true);
        gpt5_req.model = "gpt-5".to_string();

        let output = provider.stream(&gpt5_req, &HashMap::new()).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut stream) => {
            let mut encoded = String::new();
            while let Some(item) = stream.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(encoded.contains("event: message_start"));
            assert!(encoded.contains("hi"));
        });
    }

    #[tokio::test]
    async fn complete_routes_gpt5_to_responses_when_cache_lists_only_responses() {
        // Cache lists only /responses for GPT-5 (no chat) → must use
        // /responses even though the cache is populated. Confirms the
        // cache-aware path doesn't blindly default to chat.
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
        *provider.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "gpt-5".to_string(),
            name: "GPT-5".to_string(),
            vendor: "openai".to_string(),
            supported_endpoints: vec!["/responses".to_string()],
        }]);
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
    async fn cold_cache_non_gpt5_routes_to_responses_endpoint() {
        // User-reported fix: with a cold /models cache (no endpoint data),
        // requests for non-gpt-5 models now default to /responses instead
        // of /chat/completions. grok-4.5 is responses-only on Copilot and
        // previously 400'd on the chat endpoint during the cold-cache
        // window. The chat-completions mock asserts it is never hit.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({"model": "grok-4.5"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                responses_response_json("via-responses"),
            ))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        let mut req = request(false);
        req.model = "grok-4.5".to_string();
        // No rewrite: upstream_model == req.model, cold cache -> responses.
        let rewrite = HashMap::new();

        let output = provider
            .complete(&req, &rewrite)
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "via-responses");
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
    fn prefers_chat_via_enterworktree_recognises_gpt5_prefix_only() {
        // The whitelist is deliberately narrow and exact-lowercase: it
        // protects models known to emit malformed EnterWorktree args under
        // /responses, nothing else. The reverse cases guard against the
        // whitelist silently widening (coverage + regression).
        assert!(prefers_chat_via_enterworktree("gpt-5"));
        assert!(prefers_chat_via_enterworktree("gpt-5-mini"));
        assert!(prefers_chat_via_enterworktree("gpt-5.5"));
        assert!(prefers_chat_via_enterworktree("gpt-5-2025-08-07"));
        assert!(!prefers_chat_via_enterworktree("o3-mini"));
        assert!(!prefers_chat_via_enterworktree("claude-sonnet-4.6"));
        assert!(!prefers_chat_via_enterworktree("gpt-4"));
        assert!(!prefers_chat_via_enterworktree(""));
        assert!(!prefers_chat_via_enterworktree("GPT-5"));
    }
    #[tokio::test]
    async fn endpoint_for_model_chat_only_routes_to_chat() {
        let (_dir, p) = test_provider(None, None);
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "claude-sonnet-4.6".to_string(),
            name: "claude-sonnet-4.6".to_string(),
            vendor: "anthropic".to_string(),
            supported_endpoints: vec!["/chat/completions".to_string()],
        }]);
        assert_eq!(p.endpoint_for_model("claude-sonnet-4.6").await, "chat_completions");
    }

    #[tokio::test]
    async fn endpoint_for_model_whitelisted_gpt5_dual_endpoint_prefers_chat() {
        // EnterWorktree whitelist: gpt-5 advertising both endpoints keeps
        // the chat preference (the Responses path risks malformed tool args).
        let (_dir, p) = test_provider(None, None);
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "gpt-5".to_string(),
            name: "gpt-5".to_string(),
            vendor: "openai".to_string(),
            supported_endpoints: vec!["/chat/completions".to_string(), "/responses".to_string()],
        }]);
        assert_eq!(p.endpoint_for_model("gpt-5").await, "chat_completions");
    }

    #[tokio::test]
    async fn endpoint_for_model_non_whitelist_dual_endpoint_routes_to_responses() {
        // Flip core: a non-whitelist model advertising BOTH endpoints
        // routes to /responses (responses-default), unlike the old chat
        // preference.
        let (_dir, p) = test_provider(None, None);
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "o3-mini".to_string(),
            name: "o3-mini".to_string(),
            vendor: "openai".to_string(),
            supported_endpoints: vec!["/chat/completions".to_string(), "/responses".to_string()],
        }]);
        assert_eq!(p.endpoint_for_model("o3-mini").await, "responses");
    }

    #[tokio::test]
    async fn endpoint_for_model_empty_supported_endpoints_routes_to_responses() {
        // Schema drift: entry present but no endpoints advertised. Treated
        // as a cold cache -> responses default (was chat for non-gpt-5).
        let (_dir, p) = test_provider(None, None);
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "grok-4.5".to_string(),
            name: "grok-4.5".to_string(),
            vendor: "xai".to_string(),
            supported_endpoints: vec![],
        }]);
        assert_eq!(p.endpoint_for_model("grok-4.5").await, "responses");
    }

    #[tokio::test]
    async fn endpoint_for_model_responses_only_routes_to_responses() {
        let (_dir, p) = test_provider(None, None);
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "grok-4.6".to_string(),
            name: "grok-4.6".to_string(),
            vendor: "xai".to_string(),
            supported_endpoints: vec!["responses".to_string()],
        }]);
        assert_eq!(p.endpoint_for_model("grok-4.6").await, "responses");
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
        seed_chat_only_cache(&provider, "claude-model").await;

        let output = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();

        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "retried");
        });
    }

    #[tokio::test]
    async fn unauthorized_responses_refreshes_and_retries_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
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
            .and(path("/responses"))
            .and(header("authorization", "Bearer new-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                responses_response_json("retried"),
            ))
            .expect(1)
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "old-token", 600)),
        );

        let mut req = request(false);
        req.model = "gpt-5".to_string();

        let output = provider
            .complete(&req, &HashMap::new())
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
        seed_chat_only_cache(&provider, "claude-model").await;

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
        seed_chat_only_cache(&provider, "claude-model").await;

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
    async fn complete_responses_preserves_402_quota() {
        // The Responses upstream path must surface the exact HTTP status
        // and body for a 402 (Copilot's monthly-quota signal) instead
        // of remapping or swallowing it. The router relies on this
        // preservation to fall back to the next provider in the chain.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(
                ResponseTemplate::new(402)
                    .set_body_string("You have exceeded your monthly quota"),
            )
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
            .expect("upstream 402 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 402, ref body }
                if body == "You have exceeded your monthly quota"
        ));
    }

    #[tokio::test]
    async fn stream_responses_preserves_402_quota() {
        // Streaming Responses variant: a 402 returned before any SSE
        // bytes are emitted must surface as ProxyError::Upstream so the
        // router can fall back; constructing the SSE stream here would
        // lock the client into a single provider with no recovery.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(
                ResponseTemplate::new(402)
                    .set_body_string("You have exceeded your monthly quota"),
            )
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
            .expect("upstream 402 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 402, ref body }
                if body == "You have exceeded your monthly quota"
        ));
    }

    #[tokio::test]
    async fn complete_chat_completions_preserves_402_quota() {
        // Chat Completions non-streaming path: same preservation
        // contract as Responses — the router's classification is
        // status-based, so a remapped or translated 402 would silently
        // bypass the cooldown path.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(402)
                    .set_body_string("You have exceeded your monthly quota"),
            )
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        seed_chat_only_cache(&provider, "claude-model").await;

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("upstream 402 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 402, ref body }
                if body == "You have exceeded your monthly quota"
        ));
    }

    #[tokio::test]
    async fn stream_chat_completions_preserves_402_quota() {
        // Chat Completions streaming variant: the HTTP status check
        // must run before the SSE stream is constructed so the router
        // still has a chance to fall back; emitting a 200-then-error
        // envelope here would bypass fallback entirely.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(402)
                    .set_body_string("You have exceeded your monthly quota"),
            )
            .mount(&server)
            .await;
        let (_dir, provider) = test_provider(
            Some(&server),
            Some(stored_tokens("github-token", "copilot-token", 600)),
        );
        seed_chat_only_cache(&provider, "claude-model").await;

        let error = provider
            .stream(&request(true), &HashMap::new())
            .await
            .err()
            .expect("upstream 402 should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 402, ref body }
                if body == "You have exceeded your monthly quota"
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
        seed_chat_only_cache(&provider, "claude-model").await;

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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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
        // complete_bootstrap calls cache_models_with_token → fetch_models.
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer bootstrap-copilot"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "boot-m1", "name": "Boot Model", "vendor": "v", "policy": {"state": "enabled"}}]
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
            cached_models: RwLock::new(None),
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
        // reaches its terminal state (or we time out). With real time +
        // interval=5s +1=6s poll, this should complete within ~6s.
        // Wait on BOTH tokens and the model cache: complete_bootstrap
        // populates tokens before its fetch_models round-trip + disk
        // write fill cached_models, so observing tokens alone races the
        // cache write (deterministically reproducible under slow /
        // instrumented runs, e.g. llvm-cov).
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            async {
                loop {
                    let tokens_set = provider.state.tokens.read().await.is_some();
                    let models_set = provider.cached_models().await.is_some();
                    if tokens_set && models_set {
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

        // complete_bootstrap calls cache_models_with_token, so the
        // model cache must be populated post-bootstrap.
        assert!(
            provider.cached_models().await.is_some(),
            "model cache must be populated after bootstrap completes"
        );
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
            cached_models: RwLock::new(None),
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
            cached_models: RwLock::new(None),
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

    #[test]
    fn models_url_returns_expected_path() {
        let (dir, p) = test_provider(None, None);
        drop(dir);
        assert_eq!(
            p.models_url(),
            "https://api.githubcopilot.com/models"
        );
    }

    #[tokio::test]
    async fn fetch_models_parses_valid_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "gpt-5", "name": "GPT-5", "object": "model", "vendor": "OpenAI",
                     "capabilities": {"supports_vision": true}, "policy": {"state": "enabled"}},
                    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "object": "model",
                     "vendor": "Anthropic", "capabilities": {}, "policy": {"state": "enabled"}}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        let result = p
            .fetch_models("test-token")
            .await
            .expect("fetch_models should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "gpt-5");
        assert_eq!(result[1].vendor, "Anthropic");
    }

    #[tokio::test]
    async fn cache_models_populates_cached_models_accessor() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "m1", "name": "Model 1", "vendor": "v", "policy": {"state": "enabled"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        assert!(
            p.cached_models().await.is_none(),
            "cache should be empty before fetch"
        );
        p.cache_models().await;
        let cached = p
            .cached_models()
            .await
            .expect("cached_models should be Some after cache_models");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "m1");
    }

    #[test]
    fn save_load_models_roundtrip_and_modes() {
        // Two entries out of id order: the disk write must sort by id so
        // refresh cycles do not churn the file.
        let (dir, p) = test_provider(None, None);
        let models = vec![
            CopilotModel {
                id: "z-model".to_string(),
                name: "Z".to_string(),
                vendor: "v".to_string(),
                supported_endpoints: vec![],
            },
            CopilotModel {
                id: "a-model".to_string(),
                name: "A".to_string(),
                vendor: "v".to_string(),
                supported_endpoints: vec!["/responses".to_string()],
            },
        ];
        p.save_models_to_disk(&models);

        let path = models_cache_path_for(&p.state.store);
        assert!(path.exists(), "models cache must be written next to token store");
        let loaded = load_models_from_disk(&path).expect("roundtrip load must succeed");
        let ids: Vec<_> = loaded.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["a-model", "z-model"], "disk snapshot is sorted by id");

        // No `.tmp` residue: write_atomic writes a temp file and renames.
        let leftover: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftover.is_empty(), "atomic write must leave no tmp files");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o644, "models cache holds no secrets; must be group-readable");
        }
    }

    #[test]
    fn load_models_from_disk_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            load_models_from_disk(&dir.path().join("copilot_models.json")).is_none(),
            "missing cache file must be treated as cold start"
        );
    }

    #[test]
    fn load_models_from_disk_corrupt_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("copilot_models.json");
        std::fs::write(&path, "{ definitely not json").unwrap();
        assert!(
            load_models_from_disk(&path).is_none(),
            "corrupt cache must warn and start cold, not block startup"
        );
    }

    #[tokio::test]
    async fn cold_start_loads_disk_cache_and_list_models_returns_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        let models = vec![CopilotModel {
            id: "grok-4.5".to_string(),
            name: "grok-4.5".to_string(),
            vendor: "xai".to_string(),
            supported_endpoints: vec!["/responses".to_string()],
        }];
        std::fs::write(
            models_cache_path_for(&store),
            serde_json::to_vec_pretty(&models).unwrap(),
        )
        .unwrap();

        let provider = CopilotProvider::new_with_store(store);
        let cached = provider
            .cached_models()
            .await
            .expect("cold start must load the persisted cache without a fetch");
        assert_eq!(cached[0].id, "grok-4.5");
        assert_eq!(cached[0].supported_endpoints, vec!["/responses"]);

        // Intentional behavior change (Opus P1-A): /v1/models is served
        // from the disk cache on cold start instead of returning None.
        let listed = provider
            .list_models()
            .await
            .expect("list_models must return disk-cache models on cold start");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["id"], "grok-4.5");
        assert_eq!(listed[0]["owned_by"], "xai");
    }

    #[tokio::test]
    async fn cache_models_persists_fetched_snapshot_to_disk() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "m1", "name": "Model 1", "vendor": "v", "policy": {"state": "enabled"}},
                    {"id": "m2", "name": "Model 2", "vendor": "v", "policy": {"state": "enabled"},
                     "supported_endpoints": ["/chat/completions", "/responses"]}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        p.cache_models().await;

        let path = models_cache_path_for(&p.state.store);
        let persisted =
            load_models_from_disk(&path).expect("a successful fetch must persist to disk");
        assert_eq!(persisted.len(), 2);
        let endpoints: Vec<_> = persisted
            .iter()
            .filter(|m| m.id == "m2")
            .flat_map(|m| m.supported_endpoints.iter().cloned())
            .collect();
        assert_eq!(endpoints, vec!["/chat/completions", "/responses"]);
    }

    #[tokio::test]
    async fn auth_failure_clears_memory_but_keeps_disk_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        let path = models_cache_path_for(&p.state.store);
        // Seed a disk snapshot as if a real fetch had persisted it.
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&vec![CopilotModel {
                id: "account-model".to_string(),
                name: "A".to_string(),
                vendor: "v".to_string(),
                supported_endpoints: vec![],
            }])
            .unwrap(),
        )
        .unwrap();
        assert!(path.exists(), "precondition: disk snapshot exists");

        p.cache_models().await;

        assert!(
            p.cached_models().await.is_none(),
            "memory cache must be cleared on 401"
        );
        assert!(
            path.exists(),
            "disk cache is endpoint-routing metadata, independent of auth state; kept on 401 (Opus P2-B)"
        );
    }

    #[tokio::test]
    async fn complete_bootstrap_removes_stale_account_disk_cache() {
        // A fresh OAuth cycle may belong to a different account whose model
        // set differs. The previous account's persisted snapshot must not
        // survive to be loaded by a cold restart (Opus P1-C). A transient
        // /models failure (5xx) keeps the fetch from re-persisting, so a
        // missing file here proves the removal inside complete_bootstrap.
        // Real tokio time: the device-flow poll sleeps on tokio::time::sleep.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "boot-github"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token boot-github"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "boot-copilot",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let (_dir, p) = test_provider(Some(&server), None);
        let path = models_cache_path_for(&p.state.store);
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&vec![CopilotModel {
                id: "old-account-model".to_string(),
                name: "Old".to_string(),
                vendor: "v".to_string(),
                supported_endpoints: vec![],
            }])
            .unwrap(),
        )
        .unwrap();
        assert!(path.exists(), "precondition: stale previous-account cache exists");

        let dc = DeviceCodeResponse {
            device_code: "sb-device".to_string(),
            user_code: "SB-CODE".to_string(),
            verification_uri: "https://example.test/device".to_string(),
            expires_in: 600,
            interval: 5,
        };
        p.complete_bootstrap(dc).await.expect("bootstrap must succeed despite /models 5xx");
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        assert!(
            !path.exists(),
            "stale account's disk cache must be removed on a fresh OAuth cycle"
        );
    }

    #[tokio::test]
    async fn fetch_models_filters_disabled_policy() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "enabled-model", "name": "E", "vendor": "v", "policy": {"state": "enabled"}},
                    {"id": "disabled-model", "name": "D", "vendor": "v", "policy": {"state": "disabled"}}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, p) = test_provider(Some(&server), None);
        let result = p.fetch_models("test-token").await.expect("fetch should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "enabled-model");
    }

    #[tokio::test]
    async fn fetch_models_drops_missing_policy() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "has-policy", "name": "A", "vendor": "v", "policy": {"state": "enabled"}},
                    {"id": "no-policy", "name": "B", "vendor": "v"}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, p) = test_provider(Some(&server), None);
        let result = p.fetch_models("test-token").await.expect("fetch should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "has-policy");
    }

    #[tokio::test]
    async fn cache_models_with_token_does_not_call_ensure_token_when_token_unexpired() {
        let server = MockServer::start().await;
        // /models endpoint returns valid data.
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer my-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "m1", "name": "M", "vendor": "v", "policy": {"state": "enabled"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        // Token refresh endpoint must NOT be called.
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"token": "unused"})))
            .expect(0)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "my-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        p.cache_models_with_token("my-token").await;
        let cached = p.cached_models().await.expect("cache should be populated");
        assert_eq!(cached.len(), 1);
    }

    #[tokio::test]
    async fn cache_models_keeps_stale_cache_on_failure() {
        let server = MockServer::start().await;
        // /models returns 503 (transient failure).
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));

        // Pre-populate the cache with a known entry.
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "cached-m1".to_string(),
            name: "Cached Model".to_string(),
            vendor: "v".to_string(),
            supported_endpoints: vec![],
        }]);

        // cache_models() will call ensure_token (succeeds, token unexpired),
        // then fetch_models which fails with 503. The cache must survive.
        p.cache_models().await;

        let cached = p
            .cached_models()
            .await
            .expect("stale cache should survive transient failure");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "cached-m1");
    }

    #[tokio::test]
    async fn fetch_models_warns_and_drops_entry_missing_required_field() {
        // One well-formed entry + one entry missing the `vendor` field.
        // The parse failure must result in the bad entry being dropped
        // (vec length 1, containing only the good id). The test does not
        // inspect log output — the assertion is on the returned vec.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "good-model", "name": "Good", "vendor": "v", "policy": {"state": "enabled"}},
                    {"id": "bad-model", "name": "Bad", "policy": {"state": "enabled"}}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, p) = test_provider(Some(&server), None);
        let result = p
            .fetch_models("test-token")
            .await
            .expect("fetch should succeed even with one bad entry");
        assert_eq!(
            result.len(),
            1,
            "entry missing 'vendor' must be dropped; got {} entries",
            result.len()
        );
        assert_eq!(result[0].id, "good-model");
    }

    #[tokio::test]
    async fn fetch_models_returns_auth_error_on_401_and_caller_clears_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));

        // Pre-populate the cache.
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "stale-1".to_string(),
            name: "Stale".to_string(),
            vendor: "v".to_string(),
            supported_endpoints: vec![],
        }]);

        p.cache_models().await;

        assert!(
            p.cached_models().await.is_none(),
            "cache must be cleared on 401 from /models"
        );
    }

    #[tokio::test]
    async fn fetch_models_returns_auth_error_on_403_and_caller_clears_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));

        // Pre-populate the cache.
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "stale-1".to_string(),
            name: "Stale".to_string(),
            vendor: "v".to_string(),
            supported_endpoints: vec![],
        }]);

        p.cache_models().await;

        assert!(
            p.cached_models().await.is_none(),
            "cache must be cleared on 403 from /models"
        );
    }

    #[tokio::test]
    async fn fetch_models_keeps_stale_cache_on_503() {
        // Mirror of cache_models_keeps_stale_cache_on_failure, verifying
        // that the new error-variant split still routes 503 through the
        // Transient branch (keeping the stale cache).
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "test-token".into(),
            copilot_expires_at: 9999999999,
            refresh_in: 3600,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));

        // Pre-populate the cache.
        *p.state.cached_models.write().await = Some(vec![CopilotModel {
            id: "stale-503".to_string(),
            name: "Stale 503".to_string(),
            vendor: "v".to_string(),
            supported_endpoints: vec![],
        }]);

        p.cache_models().await;

        let cached = p
            .cached_models()
            .await
            .expect("stale cache should survive 503");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "stale-503");
    }

    #[tokio::test]
    async fn spawn_refresh_loop_caches_models_when_memory_token_is_expired() {
        // Pre-seed an expired token. The new cold-start block checks the
        // on-disk store (not memory) for credentials and kicks
        // cache_models(), which calls ensure_token() to refresh the
        // expired copilot token and then fetches /models.
        let server = MockServer::start().await;
        // Token refresh endpoint: provide a fresh copilot token.
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
        // /models endpoint.
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "cold-start-model", "name": "M", "vendor": "v", "policy": {"state": "enabled"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        // Pre-seed an expired copilot token (expires_at = 3600 seconds in
        // the past).
        let expired = stored_tokens("github-token", "expired-copilot", -3600);
        let (_dir, p) = test_provider(Some(&server), Some(expired));
        let provider = Arc::new(p);
        let handle = provider.clone().spawn_refresh_loop();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async {
                loop {
                    if provider.cached_models().await.is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            },
        )
        .await;

        handle.abort();
        let _ = handle.await;

        assert!(
            result.is_ok(),
            "cold-start cache_models did not populate cache within 5s"
        );
        let cached = provider
            .cached_models()
            .await
            .expect("cache should be populated");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "cold-start-model");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn start_bootstrap_after_existing_token_populates_cache() {
        // Start a bootstrap while a token is already in memory.
        // complete_bootstrap calls cache_models_with_token, which must
        // fetch /models. Mock /models with .expect(1) so we know it was
        // called. Also serves as the round-3 deadlock regression test:
        // if complete_bootstrap ever goes back through ensure_token
        // while holding refresh_lock, this hangs until the timeout fires.
        let _env_guard = crate::oauth::device_flow::ENV_LOCK.lock().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "device_code": "sb3-device",
                "user_code": "SB3-CODE",
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
                "access_token": "bootstrap-github3"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("authorization", "token bootstrap-github3"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token": "bootstrap-copilot3",
                "expires_at": now() + 900,
                "refresh_in": 800
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer bootstrap-copilot3"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "boot-model", "name": "BM", "vendor": "v", "policy": {"state": "enabled"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::from_path(dir.path().join("github_token.json"));
        // Pre-seed a token so one is already in memory.
        let existing = stored_tokens("existing-gh", "existing-cp", 3600);
        store.save(&existing).unwrap();
        let state = Arc::new(CopilotState {
            tokens: RwLock::new(Some(existing)),
            store,
            refresh_lock: Mutex::new(()),
            cached_models: RwLock::new(None),
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

        let dc = provider.clone().start_bootstrap().await.unwrap();
        assert_eq!(dc.device_code, "sb3-device");

        // Poll for cache population. The spawned task's device-flow
        // poll loop sleeps interval+1 = 6 s between attempts, so allow
        // enough wall-clock time for bootstrap + cache_models to finish.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            async {
                loop {
                    if provider.cached_models().await.is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            },
        )
        .await;
        std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

        assert!(
            result.is_ok(),
            "bootstrap + cache_models did not complete within 15s"
        );
        let cached = provider.cached_models().await.unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "boot-model");
    }

    #[tokio::test]
    async fn spawn_refresh_loop_caches_models_on_first_iteration_with_unexpired_token() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer loop-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "loop-model", "name": "Loop Model", "vendor": "v", "policy": {"state": "enabled"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let initial = StoredTokens {
            github_access_token: "gh".into(),
            copilot_token: "loop-token".into(),
            copilot_expires_at: now() + 3600,
            refresh_in: 1500,
        };
        let (_dir, p) = test_provider(Some(&server), Some(initial));
        let provider = Arc::new(p);
        let handle = provider.clone().spawn_refresh_loop();

        // Poll until the initial cache_models() call completes (the
        // spawned task makes a real HTTP request to wiremock, which
        // needs actual wall-clock time — not just paused-time advances).
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async {
                loop {
                    if provider.cached_models().await.is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            },
        )
        .await;

        handle.abort();
        let _ = handle.await;

        assert!(
            result.is_ok(),
            "pre-sleep cache_models_with_token did not populate cache within 5s"
        );
        let cached = provider
            .cached_models()
            .await
            .expect("pre-sleep caching should populate models");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "loop-model");
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
