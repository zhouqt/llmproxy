//! Provider abstractions.
//!
//! A Provider is the unit of fallback: when the proxy receives a request, the
//! router picks a Provider (the primary unless it's cooling down), converts
//! the Anthropic-format request into whatever format the provider expects,
//! sends it, and returns either an Anthropic-format response or an SSE stream.

pub mod anthropic;
pub mod copilot;
pub mod openai_compat;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
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
}

pub type SharedProvider = Arc<dyn Provider>;

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
            ..
        } => {
            let inner = anthropic::AnthropicProvider::new(
                name.clone(),
                api_key.clone(),
                api_base.clone(),
                model_rewrite.clone(),
                http,
            )?;
            Ok(Arc::new(inner))
        }
        ProviderConfig::OpenaiCompat { name, api_key, api_base, model_rewrite, .. } => {
            let inner = openai_compat::OpenAiCompatProvider::new(
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
            },
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(router.name(), "router");
        assert!(router.clone().spawn_background().is_none());
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
}
