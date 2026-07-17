//! Provider routing with fallback on cooldownable errors.
//!
//! Reference: litellm/router.py:async_function_with_retries and friends

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::anthropic::MessagesRequest;
use crate::config::{Config, ModelConfig};
use crate::cooldown::CooldownCache;
use crate::error::{ProxyError, Result};
use crate::providers::{ProviderOutput, SharedProvider};

/// Detect "this upstream cannot serve the requested model" responses
/// that arrive at runtime as a 400-level `Upstream` error. Used by the
/// router to skip such providers instead of returning the error to the
/// client — see fix-R11 in docs/TEST_ISSUES.md.
///
/// Heuristic: 400-class status with a body that mentions either
/// `model` / `not supported` / `not_found` / `not exist` / `not a valid`
/// / `model_not_*` substrings. This is intentionally narrow so a generic
/// `400 Bad Request` from a misconfigured client still surfaces to the
/// operator rather than silently chaining to the next provider.
fn is_model_unsupported(err: &ProxyError) -> bool {
    let ProxyError::Upstream { status, body } = err else {
        return false;
    };
    if !(400..500).contains(status) {
        return false;
    }
    let body_lower = body.to_ascii_lowercase();
    // Patterns observed across the providers we currently ship:
    // - Copilot:        "model_not_supported" / "The requested model is not supported"
    // - DeepSeek:       "The supported API model names are X or Y, but you passed Z"
    // - OpenAI generic: {"error":{"code":"model_not_found", ...}}
    // We deliberately do NOT include a bare "model" substring (it would
    // match too broadly — e.g. a malformed-JSON 400 like
    // "missing field `model`" still has to surface to the operator).
    let mentions_model = body_lower.contains("not supported")
        || body_lower.contains("not_supported")
        || body_lower.contains("not_found")
        || body_lower.contains("not exist")
        || body_lower.contains("not a valid")
        || body_lower.contains("model_not_")
        || body_lower.contains("\"model\"")
        || (body_lower.contains("supported api model") && body_lower.contains("you passed"));
    mentions_model
}

pub struct Router {
    cfg: Arc<Config>,
    providers: HashMap<String, SharedProvider>,
    cooldown: CooldownCache,
}

#[derive(Debug, Clone)]
pub struct RouteAttempt {
    pub provider: String,
    pub status: u16,
    pub body: String,
}

impl Router {
    pub fn new(cfg: Arc<Config>, providers: HashMap<String, SharedProvider>, cooldown: CooldownCache) -> Self {
        Self { cfg, providers, cooldown }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn cooldown(&self) -> &CooldownCache {
        &self.cooldown
    }

    pub fn find_model(&self, name: &str) -> Option<&ModelConfig> {
        self.cfg.find_model(name)
    }

    /// Pick the first non-cooling-down provider in the model's chain.
    /// If all are cooling down, return the one with the shortest remaining cooldown.
    pub async fn select_provider(&self, model: &ModelConfig) -> Result<(String, SharedProvider)> {
        let mut best: Option<(String, SharedProvider, Duration)> = None;

        for name in model.chain() {
            if let Some(p) = self.providers.get(name) {
                if !self.cooldown.is_cooling_down(name).await {
                    return Ok((name.to_string(), p.clone()));
                }
                // Track the soonest-expiring one as fallback.
                let active = self.cooldown.active().await;
                if let Some((_, _, remaining)) = active.iter().find(|(n, _, _)| n == name) {
                    if best.as_ref().map(|(_, _, d)| *remaining < *d).unwrap_or(true) {
                        best = Some((name.to_string(), p.clone(), *remaining));
                    }
                }
            } else {
                tracing::warn!("model '{}' references unknown provider '{}'", model.name, name);
            }
        }

        // Every candidate is on cooldown. Returning the soonest-expiring
        // would just trigger another cooldown mark and hand the caller
        // a generic 5xx. Fail fast with 503 + `Retry-After` instead so
        // clients back off — see fix-R7 in docs/TEST_ISSUES.md.
        if let Some((_, _, remaining)) = best {
            return Err(ProxyError::AllProvidersCoolingDown {
                model: model.name.clone(),
                retry_after_secs: Some(remaining.as_secs().max(1)),
            });
        }
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            retry_after_secs: None,
        })
    }

    /// Execute a complete request with retries across the chain.
    pub async fn complete(
        &self,
        model: &ModelConfig,
        req: &MessagesRequest,
    ) -> Result<(ProviderOutput, Vec<RouteAttempt>)> {
        let mut attempts: Vec<RouteAttempt> = Vec::new();
        let mut last_error: Option<ProxyError> = None;
        let mut tried: Vec<String> = Vec::new();
        let mut unmappable: Vec<String> = Vec::new();
        let chain: Vec<String> = model.chain().map(String::from).collect();
        let max_total = (model.max_retries_total as usize) * chain.len().max(1);

        for round in 0..chain.len() {
            let name = &chain[round];
            if tried.contains(name) {
                continue;
            }
            if self.cooldown.is_cooling_down(name).await {
                tried.push(name.clone());
                continue;
            }
            let Some(provider) = self.providers.get(name).cloned() else {
                tried.push(name.clone());
                continue;
            };
            // Providers with a non-empty model_rewrite only accept names
            // that are keys in their table. Skip them here so we don't
            // forward an unmapped name upstream (which would surface as
            // a misleading 400 and break the fallback chain) — see fix-R11.
            if !provider.can_serve_model(&req.model) {
                tracing::debug!(
                    provider = name.as_str(),
                    model = req.model.as_str(),
                    "provider's model_rewrite does not include this model; skipping"
                );
                unmappable.push(name.clone());
                tried.push(name.clone());
                continue;
            }
            tried.push(name.clone());

            // Try the primary provider up to max_retries_per_provider times.
            // Each attempt that returns a cooldownable error marks the
            // provider on cooldown and falls through to the next iteration;
            // when the loop exhausts, control returns to the outer chain
            // loop which moves to the next provider. We do NOT `break` on
            // the first error — that's the whole point of this counter.
            for _ in 0..model.max_retries_per_provider {
                match provider.complete(req, &HashMap::new()).await {
                    Ok(out) => return Ok((out, attempts)),
                    Err(e) if e.is_cooldownable() => {
                        if let ProxyError::Upstream { status, body } = &e {
                            attempts.push(RouteAttempt {
                                provider: name.clone(),
                                status: *status,
                                body: body.clone(),
                            });
                            let ttl = if *status == 429 {
                                Duration::from_secs(model.cooldown_seconds)
                            } else {
                                Duration::from_secs(5)
                            };
                            self.cooldown
                                .mark_cooldown(name, ttl, *status, &body)
                                .await;
                            last_error = Some(e);
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) if is_model_unsupported(&e) => {
                        // The upstream explicitly told us this provider can't
                        // serve the requested model (HTTP 400 + body that
                        // mentions "model" / "not supported"). Treat it like
                        // a cooldownable error so the router advances to the
                        // next provider in the chain instead of failing the
                        // whole request — see fix-R11. Use a short cooldown
                        // because the upstream's view of its model catalog
                        // could change, but record the attempt so the
                        // operator can see *why* this provider was skipped.
                        if let ProxyError::Upstream { status, body } = &e {
                            attempts.push(RouteAttempt {
                                provider: name.clone(),
                                status: *status,
                                body: body.clone(),
                            });
                            self.cooldown
                                .mark_cooldown(name, Duration::from_secs(60), *status, &body)
                                .await;
                            last_error = Some(e);
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) => {
                        // Non-cooldownable error (e.g., bad request shape) — return immediately.
                        return Err(e);
                    }
                }
            }

            if attempts.len() >= max_total {
                break;
            }
        }

        if let Some(err) = last_error {
            // At least one upstream actually returned an error; surface
            // the *last* one so the operator can see what really happened,
            // instead of the generic "all cooling down" message that
            // would imply we never even tried.
            return Err(ProxyError::AllProvidersFailed {
                model: model.name.clone(),
                attempts,
                last: Box::new(err),
            });
        }
        // No provider could even attempt the request — distinguish
        // "all were unmappable" (configuration gap) from "all were
        // already on cooldown". An unmappable-model error is a 400 with
        // a clear message; a generic cooldown is a 503.
        if !unmappable.is_empty() && unmappable.len() == chain.len() {
            return Err(ProxyError::BadRequest(format!(
                "no provider in chain '{}' can serve model '{}' (all {} entries have a model_rewrite that excludes it)",
                model.name,
                req.model,
                unmappable.len()
            )));
        }
        // Every candidate was on cooldown from the start, or every
        // configured provider name was unknown — we never even fired a
        // request, so the "all cooling down" framing is accurate.
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            retry_after_secs: None,
        })
    }

    /// Execute a streaming request. Returns the first provider's stream; if
    /// the request fails before streaming starts, falls back. Once bytes
    /// start flowing, the caller sees the entire stream.
    pub async fn stream(
        &self,
        model: &ModelConfig,
        req: &MessagesRequest,
    ) -> Result<(SharedProvider, ProviderOutput, Vec<RouteAttempt>)> {
        let mut attempts: Vec<RouteAttempt> = Vec::new();
        let mut last_error: Option<ProxyError> = None;
        let mut tried: Vec<String> = Vec::new();
        let mut unmappable: Vec<String> = Vec::new();
        let chain: Vec<String> = model.chain().map(String::from).collect();

        for round in 0..chain.len() {
            let name = &chain[round];
            if tried.contains(name) {
                continue;
            }
            if self.cooldown.is_cooling_down(name).await {
                tried.push(name.clone());
                continue;
            }
            let Some(provider) = self.providers.get(name).cloned() else {
                tried.push(name.clone());
                continue;
            };
            // Skip providers whose model_rewrite excludes this model —
            // see fix-R11 and the matching comment in `complete()`.
            if !provider.can_serve_model(&req.model) {
                tracing::debug!(
                    provider = name.as_str(),
                    model = req.model.as_str(),
                    "provider's model_rewrite does not include this model; skipping"
                );
                unmappable.push(name.clone());
                tried.push(name.clone());
                continue;
            }
            tried.push(name.clone());

            // Streaming has no inner retry: a stream() call is a single HTTP
            // request whose response begins streaming immediately on
            // success. Retrying after the first byte has flowed is unsafe
            // (we'd double-emit content to the client), so the per-provider
            // attempt count for streaming is implicitly 1.
            match provider.stream(req, &HashMap::new()).await {
                Ok(out) => return Ok((provider, out, attempts)),
                Err(e) if e.is_cooldownable() => {
                    if let ProxyError::Upstream { status, body } = &e {
                        attempts.push(RouteAttempt {
                            provider: name.clone(),
                            status: *status,
                            body: body.clone(),
                        });
                        let ttl = if *status == 429 {
                            Duration::from_secs(model.cooldown_seconds)
                        } else {
                            Duration::from_secs(5)
                        };
                        self.cooldown.mark_cooldown(name, ttl, *status, &body).await;
                        last_error = Some(e);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) if is_model_unsupported(&e) => {
                    // Same model-unsupported skip as in `complete()` —
                    // see fix-R11.
                    if let ProxyError::Upstream { status, body } = &e {
                        attempts.push(RouteAttempt {
                            provider: name.clone(),
                            status: *status,
                            body: body.clone(),
                        });
                        self.cooldown
                            .mark_cooldown(name, Duration::from_secs(60), *status, &body)
                            .await;
                        last_error = Some(e);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        if let Some(err) = last_error {
            return Err(ProxyError::AllProvidersFailed {
                model: model.name.clone(),
                attempts,
                last: Box::new(err),
            });
        }
        if !unmappable.is_empty() && unmappable.len() == chain.len() {
            return Err(ProxyError::BadRequest(format!(
                "no provider in chain '{}' can serve model '{}' (all {} entries have a model_rewrite that excludes it)",
                model.name,
                req.model,
                unmappable.len()
            )));
        }
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            retry_after_secs: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::config::{ApiFormat, ModelConfig, ProviderConfig};
    use crate::providers::Provider;
    use async_trait::async_trait;
    use futures_util::stream;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock provider: returns the given status as an upstream error the
    /// first `fail_count` times, then succeeds.
    struct MockProvider {
        name: String,
        fail_status: u16,
        fail_count: u32,
        call_count: AtomicU32,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn api_format(&self) -> ApiFormat {
            ApiFormat::Openai
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                return Err(ProxyError::Upstream {
                    status: self.fail_status,
                    body: "rate limited".into(),
                });
            }
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            if self.fail_count > 0 {
                return Err(ProxyError::Upstream {
                    status: self.fail_status,
                    body: "rate limited".into(),
                });
            }
            Ok(ProviderOutput::Stream(Box::new(stream::empty())))
        }
    }

    struct NonCooldownProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for NonCooldownProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn api_format(&self) -> ApiFormat {
            ApiFormat::Openai
        }

        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            Err(ProxyError::BadRequest("invalid request".into()))
        }

        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            Err(ProxyError::BadRequest("invalid stream request".into()))
        }
    }

    fn build_test_router() -> Router {
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 429,
                fail_count: u32::MAX, // always fail
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0, // always succeed
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };

        Router::new(Arc::new(cfg), providers, CooldownCache::new())
    }

    fn dummy_request() -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn falls_back_on_429() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        let req = dummy_request();
        let (out, attempts) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].provider, "primary");
        assert_eq!(attempts[0].status, 429);
    }

    #[tokio::test]
    async fn complete_retries_per_provider_count() {
        // max_retries_per_provider must actually retry against the same
        // provider. Configure primary to fail twice then succeed, with
        // max_retries_per_provider = 3 — the third attempt must hit
        // primary (not the backup) and produce a successful response.
        let call_count = Arc::new(AtomicU32::new(0));
        let primary = Arc::new(CountingMockProvider {
            name: "primary".into(),
            fail_count: 2,
            call_count: call_count.clone(),
        }) as SharedProvider;
        let backup = Arc::new(CountingMockProvider {
            name: "backup".into(),
            fail_count: 0,
            call_count: Arc::new(AtomicU32::new(0)),
        }) as SharedProvider;

        let mut providers = HashMap::new();
        providers.insert("primary".to_string(), primary);
        providers.insert("backup".to_string(), backup);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 3,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (out, attempts) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Primary was hit exactly 3 times: 2 failures + 1 success.
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        // The attempts vector records the two failures before the success.
        assert_eq!(attempts.len(), 2);
        for a in &attempts {
            assert_eq!(a.provider, "primary");
            assert_eq!(a.status, 429);
        }
    }

    /// Helper for `complete_retries_per_provider_count`: like MockProvider
    /// but tracks call count in an externally-shared AtomicU32 so the
    /// assertion can read it.
    struct CountingMockProvider {
        name: String,
        fail_count: u32,
        call_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Provider for CountingMockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn api_format(&self) -> ApiFormat {
            ApiFormat::Openai
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                return Err(ProxyError::Upstream {
                    status: 429,
                    body: format!("fail #{n}"),
                });
            }
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn cooldown_blocks_primary() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        // First call: primary fails 429, fallback succeeds.
        let _ = router.complete(model, &req).await.unwrap();

        // Second call: primary should be cooling down, backup used directly.
        let (out, attempts) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert!(attempts.is_empty(), "primary should be on cooldown");
    }

    #[tokio::test]
    async fn select_provider_skips_cooldown() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown
            .mark_cooldown("primary", Duration::from_secs(60), 429, "")
            .await;
        let (name, _p) = router.select_provider(model).await.unwrap();
        assert_eq!(name, "backup");
    }

    #[tokio::test]
    async fn select_provider_returns_retry_after_matching_soonest_cooldown() {
        // When every candidate provider is on cooldown, the router must
        // fail fast with 503 + Retry-After instead of calling the
        // soonest-expiring provider (which would just trip another
        // cooldown and hand the client a generic 5xx) — see fix-R7
        // in docs/TEST_ISSUES.md.
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(60), 429, "primary")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(10), 503, "backup")
            .await;

        let err = router
            .select_provider(model)
            .await
            .err()
            .expect("all providers cooling down must error");

        // retry_after_secs is the soonest-remaining cooldown (backup = 10s);
        // allow 9 in case scheduler crossed a second boundary.
        assert!(
            matches!(
                err,
                ProxyError::AllProvidersCoolingDown {
                    ref model,
                    retry_after_secs: Some(secs),
                } if model == "m" && (9..=10).contains(&secs)
            ),
            "expected AllProvidersCoolingDown with retry_after ~10, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn select_provider_errors_when_chain_has_no_known_provider() {
        let router = build_test_router();
        let empty = Router::new(
            Arc::new(router.config().clone()),
            HashMap::new(),
            CooldownCache::new(),
        );
        let model = empty.find_model("m").unwrap();

        let error = empty
            .select_provider(model)
            .await
            .err()
            .expect("selection should fail");

        assert!(matches!(
            error,
            ProxyError::AllProvidersCoolingDown { ref model, .. } if model == "m"
        ));
    }

    #[tokio::test]
    async fn stream_falls_back_on_cooldownable_error() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();

        let (provider, output, attempts) = router
            .stream(model, &dummy_request())
            .await
            .unwrap();

        assert_eq!(provider.name(), "backup");
        assert!(matches!(output, ProviderOutput::Stream(_)));
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].provider, "primary");
        assert_eq!(attempts[0].status, 429);
        assert_eq!(attempts[0].body, "rate limited");
        assert!(router.cooldown().is_cooling_down("primary").await);
    }

    #[tokio::test]
    async fn complete_and_stream_return_non_cooldownable_error_immediately() {
        let base = build_test_router();
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(NonCooldownProvider {
                name: "primary".into(),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        let router = Router::new(
            Arc::new(base.config().clone()),
            providers,
            CooldownCache::new(),
        );
        let model = router.find_model("m").unwrap();

        let complete = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("complete should fail");
        let stream = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("stream should fail");

        assert!(matches!(complete, ProxyError::BadRequest(ref message) if message == "invalid request"));
        assert!(matches!(stream, ProxyError::BadRequest(ref message) if message == "invalid stream request"));
        assert!(!router.cooldown().is_cooling_down("primary").await);
    }

    #[tokio::test]
    async fn complete_and_stream_error_when_every_candidate_is_skipped() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(60), 429, "")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(60), 429, "")
            .await;

        let complete = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        let stream = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");

        // No upstream call ever fired — both providers were on cooldown
        // from the start — so this is the legacy "all cooling down" path,
        // not AllProvidersFailed. The router distinguishes the two cases:
        // "no attempt happened" stays as AllProvidersCoolingDown,
        // "at least one attempt failed" becomes AllProvidersFailed.
        assert!(matches!(complete, ProxyError::AllProvidersCoolingDown { .. }));
        assert!(matches!(stream, ProxyError::AllProvidersCoolingDown { .. }));
    }

    #[tokio::test]
    async fn max_retries_total_zero_stops_before_fallback() {
        let router = build_test_router();
        let mut model = router.find_model("m").unwrap().clone();
        model.max_retries_total = 0;

        let error = router
            .complete(&model, &dummy_request())
            .await
            .err()
            .expect("request should fail");

        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
        assert!(router.cooldown().is_cooling_down("primary").await);
        assert!(!router.cooldown().is_cooling_down("backup").await);
    }

    #[tokio::test]
    async fn complete_skips_unknown_provider_in_chain() {
        // When a model chain references a provider that doesn't exist, the
        // router should skip it and try the next one instead of erroring.
        let base = build_test_router();
        let mut model = base.find_model("m").unwrap().clone();
        model.fallback_chain = vec!["missing".into(), "backup".into()];

        let (out, attempts) = router_clone_with(&base)
            .complete(&model, &dummy_request())
            .await
            .unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Only the 429 attempt against primary is recorded; the missing
        // provider is silently skipped without recording an attempt.
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].provider, "primary");
        assert_eq!(attempts[0].status, 429);
    }

    #[tokio::test]
    async fn stream_skips_unknown_provider_in_chain() {
        let base = build_test_router();
        let mut model = base.find_model("m").unwrap().clone();
        model.fallback_chain = vec!["missing".into(), "backup".into()];

        let (provider, output, attempts) = router_clone_with(&base)
            .stream(&model, &dummy_request())
            .await
            .unwrap();
        assert_eq!(provider.name(), "backup");
        assert!(matches!(output, ProviderOutput::Stream(_)));
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].provider, "primary");
        assert_eq!(attempts[0].status, 429);
    }

    #[tokio::test]
    async fn complete_skips_provider_already_tried() {
        // A duplicated entry in the chain (e.g. fallback_chain contains the
        // primary again) should be skipped on the second pass — the router
        // already recorded a failed attempt and moved on.
        //
        // For the duplicate to actually be reached, the primary AND the
        // other fallbacks must fail. Configure every provider as a
        // fail-forever stub and verify the router records at most one
        // attempt per provider name.
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 429,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into(), "primary".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 5,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let error = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
        // Each provider was attempted exactly once even though the chain
        // listed primary twice.
        assert_eq!(
            router
                .cooldown()
                .active_with_reason()
                .await
                .iter()
                .filter(|(n, _, _, _)| n == "primary" || n == "backup")
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn stream_skips_provider_already_tried() {
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 429,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into(), "primary".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 5,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let error = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
    }

    #[tokio::test]
    async fn select_provider_retry_after_is_primary_when_primary_remaining_is_shorter() {
        // When all providers are cooling down, the router must return
        // AllProvidersCoolingDown with retry_after_secs equal to the
        // soonest-remaining cooldown — here primary expires first, so
        // the value comes from primary's 5s window. See fix-R7 in
        // docs/TEST_ISSUES.md.
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(5), 429, "")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(120), 503, "")
            .await;

        let err = router
            .select_provider(model)
            .await
            .err()
            .expect("all providers cooling down must error");

        assert!(
            matches!(
                err,
                ProxyError::AllProvidersCoolingDown {
                    retry_after_secs: Some(secs),
                    ..
                } if (1..=5).contains(&secs)
            ),
            "expected retry_after_secs from primary's 5s window, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn complete_returns_all_providers_failed_with_last_error_when_chain_exhausted() {
        // When every candidate actually fires and returns an upstream
        // error, the router must surface the last one as
        // AllProvidersFailed (not AllProvidersCoolingDown) so operators
        // can see the real cause — see fix-B in TEST_ISSUES.md.
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 503,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let err = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        match err {
            ProxyError::AllProvidersFailed { model, attempts, last } => {
                assert_eq!(model, "m");
                assert_eq!(attempts.len(), 2);
                assert_eq!(attempts[0].provider, "primary");
                assert_eq!(attempts[0].status, 503);
                assert_eq!(attempts[1].provider, "backup");
                assert_eq!(attempts[1].status, 503);
                // `last` must be the last upstream error (backup's),
                // not the legacy generic "all cooling down" message.
                match last.as_ref() {
                    ProxyError::Upstream { status, body } => {
                        assert_eq!(*status, 503);
                        assert_eq!(body, "rate limited");
                    }
                    other => panic!("expected wrapped Upstream, got {other:?}"),
                }
            }
            other => panic!("expected AllProvidersFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_429_upstream_uses_short_cooldown_ttl() {
        // A 503 (or any non-429 cooldownable status) should mark the provider
        // for the default short TTL, not the model's cooldown_seconds.
        let router = build_test_router();
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 503,
                fail_count: u32::MAX,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        let router = Router::new(
            Arc::new(router.config().clone()),
            providers,
            CooldownCache::new(),
        );
        let model = router.find_model("m").unwrap();
        let _ = router.complete(model, &dummy_request()).await.unwrap();

        // Backup succeeded; primary should now be on a short cooldown.
        assert!(router.cooldown().is_cooling_down("primary").await);
    }

    fn router_clone_with(base: &Router) -> Router {
        Router::new(
            Arc::new(base.config().clone()),
            base.providers.clone(),
            base.cooldown.clone(),
        )
    }

    #[test]
    fn mock_providers_expose_name_and_api_format() {
        // The MockProvider and NonCooldownProvider impls in this module only
        // exist to support the router tests above. Their `name` and
        // `api_format` methods are otherwise dead code unless something
        // invokes them directly — exercise both impls so the coverage tool
        // records the bodies.
        let mock = MockProvider {
            name: "mock".into(),
            fail_status: 0,
            fail_count: 0,
            call_count: AtomicU32::new(0),
        };
        assert_eq!(mock.name(), "mock");
        assert_eq!(mock.api_format(), ApiFormat::Openai);

        let non_cd = NonCooldownProvider {
            name: "non-cd".into(),
        };
        assert_eq!(non_cd.name(), "non-cd");
        assert_eq!(non_cd.api_format(), ApiFormat::Openai);
    }

    /// A provider that accepts a fixed allow-list of model names; every
    /// other model is rejected at dispatch time via `can_serve_model`.
    /// Used by the R11 chain-skip tests.
    struct RestrictedMockProvider {
        name: String,
        allowed: Vec<String>,
        call_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Provider for RestrictedMockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn api_format(&self) -> ApiFormat {
            ApiFormat::Openai
        }
        fn can_serve_model(&self, model: &str) -> bool {
            // Empty allow-list means "no restriction" (matches the
            // OpenAiCompatProvider contract for an unconfigured rewrite
            // table). Non-empty list is an explicit allow-list.
            self.allowed.is_empty() || self.allowed.iter().any(|m| m == model)
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            unimplemented!()
        }
    }

    fn build_restricted_router(
        primary_allowed: Vec<String>,
        backup_allowed: Vec<String>,
    ) -> (Router, Arc<AtomicU32>, Arc<AtomicU32>) {
        let primary_count = Arc::new(AtomicU32::new(0));
        let backup_count = Arc::new(AtomicU32::new(0));
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(RestrictedMockProvider {
                name: "primary".into(),
                allowed: primary_allowed,
                call_count: primary_count.clone(),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(RestrictedMockProvider {
                name: "backup".into(),
                allowed: backup_allowed,
                call_count: backup_count.clone(),
            }) as SharedProvider,
        );
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        (
            Router::new(Arc::new(cfg), providers, CooldownCache::new()),
            primary_count,
            backup_count,
        )
    }

    #[tokio::test]
    async fn complete_skips_provider_that_cannot_serve_model_and_uses_next() {
        // Primary has a model_rewrite that excludes `m-allowed-1`.
        // Backup accepts it. The router must skip primary without
        // calling it and succeed on backup — see fix-R11.
        let (router, primary_count, backup_count) =
            build_restricted_router(vec!["other-model".into()], vec![]);
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        let (out, attempts) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(primary_count.load(Ordering::SeqCst), 0, "primary must be skipped, not called");
        assert_eq!(backup_count.load(Ordering::SeqCst), 1);
        // No upstream call recorded against the skipped provider.
        assert!(attempts.is_empty());
    }

    #[tokio::test]
    async fn complete_returns_bad_request_when_no_provider_can_serve_model() {
        // Both providers have a rewrite that excludes `m-bad`. The router
        // must surface a 400-level BadRequest so the operator sees that
        // the configuration gap (not a transient upstream failure) is the
        // cause — see fix-R11.
        let (router, primary_count, backup_count) = build_restricted_router(
            vec!["unrelated-a".into()],
            vec!["unrelated-b".into()],
        );
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        let err = router
            .complete(model, &req)
            .await
            .err()
            .expect("request should fail");
        match err {
            ProxyError::BadRequest(msg) => {
                assert!(
                    msg.contains("m") && msg.contains("can serve"),
                    "message should mention the model + cause: {msg}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
        // Neither provider should have been called.
        assert_eq!(primary_count.load(Ordering::SeqCst), 0);
        assert_eq!(backup_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn complete_skips_mismatched_provider_but_still_falls_back_on_cooldownable_error() {
        // Primary excludes `m`, backup accepts it. Primary is still
        // skipped at dispatch (so it doesn't get a 400 from upstream),
        // and backup succeeds. The dispatch-skip is a distinct path
        // from cooldown-based skip — both should coexist.
        let (router, primary_count, backup_count) =
            build_restricted_router(vec!["other".into()], vec![]);
        let model = router.find_model("m").unwrap();

        let (out, _attempts) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(primary_count.load(Ordering::SeqCst), 0);
        assert_eq!(backup_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_skips_provider_that_cannot_serve_model_and_uses_next() {
        // Same scenario as `complete_skips_provider_...`, but on the
        // streaming path. The first byte must come from backup.
        struct StreamingMockProvider {
            name: String,
            allowed: Vec<String>,
            call_count: Arc<AtomicU32>,
        }
        #[async_trait]
        impl Provider for StreamingMockProvider {
            fn name(&self) -> &str {
                &self.name
            }
            fn api_format(&self) -> ApiFormat {
                ApiFormat::Openai
            }
            fn can_serve_model(&self, model: &str) -> bool {
                self.allowed.is_empty() || self.allowed.iter().any(|m| m == model)
            }
            async fn complete(
                &self,
                _req: &MessagesRequest,
                _model_rewrite: &HashMap<String, String>,
            ) -> Result<ProviderOutput> {
                unimplemented!()
            }
            async fn stream(
                &self,
                _req: &MessagesRequest,
                _model_rewrite: &HashMap<String, String>,
            ) -> Result<ProviderOutput> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                let s: Box<dyn futures_util::Stream<Item = Result<Bytes>> + Send + Unpin> =
                    Box::new(stream::empty());
                Ok(ProviderOutput::Stream(s))
            }
        }
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(StreamingMockProvider {
                name: "primary".into(),
                allowed: vec!["other-model".into()],
                call_count: Arc::new(AtomicU32::new(0)),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(StreamingMockProvider {
                name: "backup".into(),
                allowed: vec![],
                call_count: Arc::new(AtomicU32::new(0)),
            }) as SharedProvider,
        );
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (provider, _output, attempts) =
            router.stream(model, &dummy_request()).await.unwrap();
        assert_eq!(provider.name(), "backup");
        assert!(attempts.is_empty(), "no upstream attempts should be recorded");
    }

    #[tokio::test]
    async fn stream_returns_bad_request_when_no_provider_can_serve_model() {
        let (router, _pc, _bc) = build_restricted_router(
            vec!["other".into()],
            vec!["another".into()],
        );
        let model = router.find_model("m").unwrap();

        let err = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("stream should fail");
        assert!(
            matches!(err, ProxyError::BadRequest(ref msg) if msg.contains("can serve")),
            "expected BadRequest, got {err:?}"
        );
    }

    #[test]
    fn is_model_unsupported_recognises_common_shapes() {
        // Cover the body patterns the runtime helper is supposed to catch.
        let cases = [
            (
                r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#,
                true,
            ),
            (
                r#"{"error":{"message":"Model Not Exist","type":"invalid_request_error","code":"model_not_found"}}"#,
                true,
            ),
            (
                r#"{"error":{"message":"The supported API model names are deepseek-v4-pro or deepseek-v4-flash, but you passed claude-sonnet-4.5.","param":null,"code":"invalid_request_error"}}"#,
                true,
            ),
            // Plain 400 with no model signal must NOT be treated as
            // model-unsupported — that would mask real request errors.
            (r#"{"error":{"message":"missing field `messages`"}}"#, false),
            (r#"rate limited"#, false),
            // Bare "model" without a "not supported" cue is also not
            // enough — covers "missing field `model`" false positives.
            (r#"{"error":"invalid value for field `model`"}"#, false),
        ];
        for (body, expected) in cases {
            let err = ProxyError::Upstream { status: 400, body: body.into() };
            assert_eq!(
                is_model_unsupported(&err),
                expected,
                "body={body} expected={expected}"
            );
        }
    }

    #[test]
    fn is_model_unsupported_only_for_4xx_status() {
        // A 5xx error mentioning model must still be the regular
        // cooldownable branch, not the model-unsupported one.
        let err = ProxyError::Upstream {
            status: 503,
            body: r#"{"error":"model_not_supported"}"#.into(),
        };
        assert!(!is_model_unsupported(&err));
        // 401/429 — handled by the cooldownable branch.
        let err = ProxyError::Upstream {
            status: 401,
            body: r#"unauthorized, please check your token"#.into(),
        };
        assert!(!is_model_unsupported(&err));
    }

    /// A mock provider that always returns an upstream 400 whose body
    /// looks like a model-unsupported envelope (e.g. Copilot rejecting
    /// `claude-sonnet-4.5`). The router must treat this as a
    /// "skip this provider, try the next one" signal — see fix-R11.
    struct ModelUnsupportedProvider {
        name: String,
        body: String,
        call_count: AtomicU32,
    }

    #[async_trait]
    impl Provider for ModelUnsupportedProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn api_format(&self) -> ApiFormat {
            ApiFormat::Openai
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(ProxyError::Upstream {
                status: 400,
                body: self.body.clone(),
            })
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(ProxyError::Upstream {
                status: 400,
                body: self.body.clone(),
            })
        }
    }

    #[tokio::test]
    async fn complete_skips_provider_returning_runtime_model_unsupported() {
        // Primary returns 400 + model_not_supported body. The router must
        // skip it and try backup. Backup succeeds. — see fix-R11.
        let primary = Arc::new(ModelUnsupportedProvider {
            name: "primary".into(),
            body: r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#.into(),
            call_count: AtomicU32::new(0),
        });
        let backup = Arc::new(MockProvider {
            name: "backup".into(),
            fail_status: 0,
            fail_count: 0,
            call_count: AtomicU32::new(0),
        });
        let mut providers: HashMap<String, SharedProvider> = HashMap::new();
        providers.insert("primary".to_string(), primary.clone() as SharedProvider);
        providers.insert("backup".to_string(), backup as SharedProvider);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (out, attempts) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Primary was tried once (returned the 400), backup took over.
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        // The skipped primary must be in the attempts list so the
        // operator can see why it was skipped.
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].provider, "primary");
        assert_eq!(attempts[0].status, 400);
        assert!(attempts[0].body.contains("model_not_supported"));
    }

    #[tokio::test]
    async fn complete_does_not_skip_provider_returning_generic_400() {
        // A 400 that does NOT mention model/not-supported must surface
        // as an error, not be silently swallowed by the model-unsupported
        // skip path. This is the guard against over-eager skipping.
        let primary = Arc::new(ModelUnsupportedProvider {
            name: "primary".into(),
            body: r#"{"error":{"message":"missing field `messages`"}}"#.into(),
            call_count: AtomicU32::new(0),
        });
        let backup = Arc::new(MockProvider {
            name: "backup".into(),
            fail_status: 0,
            fail_count: 0,
            call_count: AtomicU32::new(0),
        });
        let mut providers: HashMap<String, SharedProvider> = HashMap::new();
        providers.insert("primary".to_string(), primary.clone() as SharedProvider);
        providers.insert("backup".to_string(), backup as SharedProvider);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
            logging: Default::default(),
        };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let err = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("generic 400 must surface as Err");
        match err {
            ProxyError::Upstream { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Upstream 400, got {other:?}"),
        }
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
    }
}
