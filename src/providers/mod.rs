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
use std::sync::Arc;

use crate::anthropic::MessagesRequest;
use crate::config::ProviderConfig;
use crate::error::Result;

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
    async fn stream(&self, req: &MessagesRequest, model_rewrite: &std::collections::HashMap<String, String>) -> Result<ProviderOutput>;
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
            ..
        } => {
            let inner = openai_compat::OpenAiCompatProvider::new(
                name.clone(),
                api_base.clone(),
                api_key.clone(),
                model_rewrite.clone(),
                provider_ignore.clone(),
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
}
