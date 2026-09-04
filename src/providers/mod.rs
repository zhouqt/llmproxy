//! Provider abstractions.
//!
//! A Provider is the unit of fallback: when the proxy receives a request, the
//! router picks a Provider (the primary unless it's cooling down), converts
//! the Anthropic-format request into whatever format the provider expects,
//! sends it, and returns either an Anthropic-format response or an SSE stream.

pub mod anthropic;
pub mod copilot;
pub mod openai_compat;
pub mod openai_responses;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::anthropic::MessagesRequest;
use crate::config::ProviderConfig;
use crate::error::Result;

/// Per-stream token usage captured from upstream response. Used by the
/// streaming log line to surface per-request token counts (`input`,
/// `output`, `cache_read`). `cache_read` is `None` when the upstream
/// did not report a cache hit. `errored` is set when the stream
/// terminated with an upstream error envelope (not a transport-level
/// `Err`) — `MappedStream` reads it to classify the stream as
/// `streaming aborted` instead of a clean `streaming completed`.
#[derive(Clone, Debug, Default)]
pub struct StreamUsage {
    pub input: u32,
    pub output: u32,
    pub cache_read: Option<u32>,
    pub errored: bool,
}

/// Shared, thread-safe cell that holds a `StreamUsage` written by an
/// SSE adapter and read by `MappedStream::with_callback` at terminal
/// `Ready(None)` time. The two ends share a single `Arc<Mutex<...>>`
/// so the value is materialized before the adapter is dropped.
#[derive(Clone, Default)]
pub struct StreamUsageSink {
    inner: Arc<Mutex<Option<StreamUsage>>>,
}

impl StreamUsageSink {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Wrap an existing `Arc` — used by `messages_handler` so it can
    /// pass one clone to the provider's `stream` call and another to
    /// `MappedStream::with_callback`. Both ends reference the same
    /// state cell.
    pub fn from_arc(inner: Arc<Mutex<Option<StreamUsage>>>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying `Arc` for cloning into another owner
    /// (e.g. `MappedStream`).
    pub fn arc(&self) -> Arc<Mutex<Option<StreamUsage>>> {
        self.inner.clone()
    }

    pub fn set(&self, usage: StreamUsage) {
        let mut g = self
            .inner
            .lock()
            .expect("StreamUsageSink mutex poisoned");
        *g = Some(usage);
    }

    /// Take the current value (consuming it). Returns `None` if the
    /// SSE adapter never wrote (no usage reported).
    pub fn take(&self) -> Option<StreamUsage> {
        let mut g = self
            .inner
            .lock()
            .expect("StreamUsageSink mutex poisoned");
        g.take()
    }
}

/// Output of a Provider call. Either a complete JSON response body or a
/// byte stream of SSE-encoded Anthropic events.
pub enum ProviderOutput {
    Json(serde_json::Value),
    Stream(Box<dyn Stream<Item = Result<Bytes>> + Send + Unpin>),
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, req: &MessagesRequest, model_rewrite: &std::collections::HashMap<String, String>) -> Result<ProviderOutput>;
    /// Streaming variant of `complete`. `usage_sink` is an optional
    /// cell into which the provider's SSE adapter writes the final
    /// `StreamUsage` (if the upstream reports usage). `None` disables
    /// capture; this is the value `messages_handler` passes when the
    /// caller doesn't need per-stream token counts.
    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &std::collections::HashMap<String, String>,
        usage_sink: Option<StreamUsageSink>,
    ) -> Result<ProviderOutput>;
    /// Whether this provider can serve `model` without the proxy sending an
    /// unmapped name upstream. Providers with an empty rewrite table accept
    /// any model name verbatim (they expose a model catalog of their own);
    /// providers with a non-empty rewrite table only accept names that are
    /// keys in that table. The router uses this to skip providers that
    /// would otherwise forward an unsupported model and trip a 400 from
    /// upstream — see fix-R11 in docs/TEST_ISSUES.md.
    fn can_serve_model(&self, model: &str) -> bool {
        let _ = model;
        true
    }
    /// Optionally spawn a background task (e.g. token refresh). Returns a
    /// handle the server can abort on shutdown.
    fn spawn_background(self: Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        let _ = self;
        None
    }
    /// Returns `Some(Arc<CopilotProvider>)` only when the concrete type is
    /// the GitHub Copilot provider. The default returns `None`. Used by
    /// `main.rs` to surface the provider to admin endpoints without
    /// downcasting through the trait object.
    fn as_any_copilot(self: Arc<Self>) -> Option<Arc<copilot::CopilotProvider>> {
        let _ = self;
        None
    }
    /// PR-12: combine the provider's configured `model_rewrite` with a
    /// runtime request-time override map (runtime wins on key
    /// collision). Providers that need model rewriting override this
    /// (AnthropicProvider had a private copy pre-PR-12; Copilot used
    /// a free function with 4 call sites). The default returns the
    /// runtime map unchanged for providers that don't configure any
    /// model rewriting.
    fn merged_rewrite<'a>(
        &'a self,
        runtime: &'a HashMap<String, String>,
    ) -> HashMap<String, String> {
        let _ = self;
        runtime.clone()
    }
    /// Return a best-effort list of models served by this upstream.
    ///
    /// Providers that have an upstream model-catalog endpoint implement
    /// this by issuing a GET to that endpoint and normalising the response
    /// into a stable shape (`id`, `object:"model"`, `created`, `owned_by`,
    /// `display_name`). Providers without a catalog endpoint return `None`.
    ///
    /// Individual provider failures are logged but not propagated — the
    /// aggregator in `/v1/models` collects whatever each provider can
    /// produce and moves on.
    async fn list_models(&self) -> Option<Vec<serde_json::Value>> {
        let _ = self;
        None
    }
}

pub type SharedProvider = Arc<dyn Provider>;

/// Convert an Anthropic-shaped `Usage` into the log-facing `StreamUsage`,
/// carrying the errored flag. Shared by the SSE adapters
/// (`OpenAiSseToAnthropic`, `ResponsesSseToAnthropic`, `PassthroughSse`)
/// so the `build_usage → StreamUsage` mapping lives in one place instead
/// of being duplicated per adapter (code-review F8).
pub(crate) fn usage_to_stream_usage(
    usage: &crate::anthropic::Usage,
    errored: bool,
) -> StreamUsage {
    StreamUsage {
        input: usage.input_tokens,
        output: usage.output_tokens,
        cache_read: usage.cache_read_input_tokens,
        errored,
    }
}

/// Detect whether `api_base` points at the OpenRouter gateway.
///
/// Returns `true` when the URL's host is `openrouter.ai` or any
/// `*.openrouter.ai` subdomain (case-insensitive). On parse failure
/// or a missing host, returns `false` (so non-URL garbage is never
/// classified as OpenRouter).
///
/// Used by `AnthropicProvider` and `OpenAiCompatProvider` to gate
/// the `provider: {ignore: [...]}` injection: the field is only
/// meaningful on OpenRouter's `/v1/messages` and
/// `/v1/chat/completions` endpoints, and strict Anthropic-compat
/// backends like DeepSeek would reject an unknown top-level field.
pub(crate) fn is_openrouter_api_base(api_base: &str) -> bool {
    reqwest::Url::parse(api_base)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_lowercase()))
        .map(|host| host == "openrouter.ai" || host.ends_with(".openrouter.ai"))
        .unwrap_or(false)
}

/// Build a provider instance from a ProviderConfig.
pub fn build(
    cfg: &ProviderConfig,
    http: reqwest::Client,
) -> Result<SharedProvider> {
    match cfg {
        ProviderConfig::GithubCopilot {
            name,
            vscode_version,
            account_type,
            model_rewrite,
            ..
        } => {
            let inner = copilot::CopilotProvider::new(
                name.clone(),
                vscode_version.clone(),
                account_type.clone(),
                model_rewrite.clone(),
                http,
            )?;
            Ok(Arc::new(inner))
        }
        ProviderConfig::Anthropic {
            name,
            api_key,
            api_base,
            model_rewrite,
            provider_ignore,
            ..
        } => {
            let inner = anthropic::AnthropicProvider::new(
                name.clone(),
                api_key.clone(),
                api_base.clone(),
                model_rewrite.clone(),
                provider_ignore.clone(),
                http,
            )?;
            Ok(Arc::new(inner))
        }
        ProviderConfig::OpenaiCompat {
            name,
            api_key,
            api_base,
            model_rewrite,
            provider_ignore,
            reasoning_echo,
            ..
        } => {
            let inner = openai_compat::OpenAiCompatProvider::new(
                name.clone(),
                api_base.clone(),
                api_key.clone(),
                model_rewrite.clone(),
                provider_ignore.clone(),
                reasoning_echo.clone(),
                http,
            )?;
            Ok(Arc::new(inner))
        }
        ProviderConfig::OpenaiResponses { name, api_key, api_base, model_rewrite, .. } => {
            let inner = openai_responses::OpenaiResponsesProvider::new(
                name.clone(),
                api_base.clone(),
                api_key.clone(),
                model_rewrite.clone(),
                http,
            )?;
            Ok(Arc::new(inner))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn builds_openai_compat_and_anthropic_providers() {
        let compat = build(
            &ProviderConfig::OpenaiCompat {
                name: "compat".to_string(),
                api_key: "key".to_string(),
                api_base: "https://example.test/v1".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
            reasoning_echo: false,
            },
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(compat.name(), "compat");
        assert!(compat.clone().spawn_background().is_none());

        let router = build(
            &ProviderConfig::Anthropic {
                name: "router".to_string(),
                api_key: "key".to_string(),
                api_base: "https://openrouter.ai/api/v1".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
            },
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(router.name(), "router");
        assert!(router.clone().spawn_background().is_none());
    }

    #[test]
    fn builds_openai_responses_provider() {
        let responses = build(
            &ProviderConfig::OpenaiResponses {
                name: "responses".to_string(),
                api_key: "key".to_string(),
                api_base: "https://example.test/v1".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(responses.name(), "responses");
    }

    #[tokio::test]
    async fn builds_copilot_without_reading_user_token_store() {
        let dir = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("XDG_DATA_HOME");
        std::env::set_var("XDG_DATA_HOME", dir.path());

        let copilot = build(
            &ProviderConfig::GithubCopilot {
                name: "copilot".to_string(),
                vscode_version: "1.95.0".to_string(),
                account_type: "individual".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            reqwest::Client::new(),
        )
        .unwrap();

        if let Some(previous) = previous {
            std::env::set_var("XDG_DATA_HOME", previous);
        } else {
            std::env::remove_var("XDG_DATA_HOME");
        }

        assert_eq!(copilot.name(), "copilot");
        let handle = copilot
            .clone()
            .spawn_background()
            .expect("copilot should spawn token refresh");
        handle.abort();
    }

    #[test]
    fn is_openrouter_api_base_matches_canonical_subdomains_and_case() {
        // Canonical OpenRouter host.
        assert!(is_openrouter_api_base("https://openrouter.ai/api/v1"));
        // www. subdomain.
        assert!(is_openrouter_api_base("https://www.openrouter.ai/api/v1"));
        // Generic subdomain (api.openrouter.ai etc.).
        assert!(is_openrouter_api_base("https://api.openrouter.ai/v1"));
        // Case-insensitive: scheme + host in mixed case.
        assert!(is_openrouter_api_base("HTTPS://OPENROUTER.AI/api/v1"));
        assert!(is_openrouter_api_base("Https://OpenRouter.Ai/api/v1"));

        // Negative cases: well-known Anthropic-compat / OpenAI-compat
        // backends must NOT be classified as OpenRouter — the field
        // would be rejected by strict backends like DeepSeek.
        assert!(!is_openrouter_api_base("https://api.deepseek.com/v1"));
        assert!(!is_openrouter_api_base("https://api.minimaxi.com/v1"));
        assert!(!is_openrouter_api_base("https://api.anthropic.com/v1"));
        // Host that merely contains the substring as a label prefix
        // (not as a real subdomain) — `evil-openrouter.ai.example.com`
        // ends in `.example.com`, so it must NOT match.
        assert!(!is_openrouter_api_base(
            "https://openrouter.ai.example.com/v1"
        ));
        // Unparseable / empty input is conservative `false`.
        assert!(!is_openrouter_api_base("not-a-url"));
        assert!(!is_openrouter_api_base(""));
    }

    // ──────────────────────────────────────────────────────────────────
    // StreamUsage + StreamUsageSink — covering the new types added for
    // streaming token capture. The two ends (SSE adapter writer +
    // MappedStream reader) share a single `Arc<Mutex<Option<...>>>`
    // so the value is materialized before the adapter is dropped
    // (Sonnet M5 — deterministic ordering).
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn stream_usage_sink_round_trips_through_set_and_take() {
        let sink = StreamUsageSink::empty();
        assert!(sink.take().is_none(), "fresh sink must be empty");
        sink.set(StreamUsage {
            input: 7,
            output: 3,
            cache_read: Some(2),
            ..Default::default()
        });
        let got = sink.take().expect("set value must be readable");
        assert_eq!(got.input, 7);
        assert_eq!(got.output, 3);
        assert_eq!(got.cache_read, Some(2));
        assert!(!got.errored, "default errored must be false");
        // take() is consuming: a second take returns None.
        assert!(sink.take().is_none());
    }

    #[test]
    fn stream_usage_sink_arc_shares_state_between_writer_and_reader() {
        // Two clones of the same sink must observe the same writes.
        // This is the wire shape that MappedStream relies on.
        let sink_a = StreamUsageSink::empty();
        let sink_b = sink_a.clone();
        sink_a.set(StreamUsage {
            input: 1,
            output: 2,
            cache_read: None,
            ..Default::default()
        });
        let got = sink_b.take().expect("writer's clone must see the value");
        assert_eq!(got.input, 1);
        assert_eq!(got.output, 2);
        assert!(got.cache_read.is_none());
    }

    #[test]
    fn stream_usage_sink_from_arc_wraps_existing_cell() {
        let cell: Arc<Mutex<Option<StreamUsage>>> = Arc::new(Mutex::new(None));
        let sink = StreamUsageSink::from_arc(cell.clone());
        sink.set(StreamUsage {
            input: 5,
            output: 5,
            cache_read: Some(0),
            ..Default::default()
        });
        // Direct read of the underlying cell must observe the same
        // write — confirms `from_arc` shares the cell by reference,
        // not by snapshot.
        let g = cell.lock().unwrap();
        let usage = g.clone().expect("cell must hold a value");
        assert_eq!(usage.input, 5);
        assert_eq!(usage.cache_read, Some(0));
    }

    #[test]
    fn stream_usage_sink_arc_accessor_round_trips() {
        let sink = StreamUsageSink::empty();
        let cell = sink.arc();
        sink.set(StreamUsage {
            input: 9,
            output: 1,
            cache_read: None,
            ..Default::default()
        });
        let g = cell.lock().unwrap();
        let usage = g.clone().expect("cell must hold a value");
        assert_eq!(usage.input, 9);
        assert_eq!(usage.output, 1);
    }

    #[test]
    fn stream_usage_default_is_zeros_and_none() {
        let u = StreamUsage::default();
        assert_eq!(u.input, 0);
        assert_eq!(u.output, 0);
        assert!(u.cache_read.is_none());
    }
}
