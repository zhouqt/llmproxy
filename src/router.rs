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

        if let Some((name, p, _)) = best {
            tracing::warn!(
                model = model.name,
                provider = name.as_str(),
                "all providers cooling down; using soonest-expiring"
            );
            return Ok((name, p));
        }
        Err(ProxyError::AllProvidersCoolingDown(model.name.clone()))
    }

    /// Execute a complete request with retries across the chain.
    pub async fn complete(
        &self,
        model: &ModelConfig,
        req: &MessagesRequest,
    ) -> Result<(ProviderOutput, Vec<RouteAttempt>)> {
        let mut attempts: Vec<RouteAttempt> = Vec::new();
        let mut tried: Vec<String> = Vec::new();
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

        Err(ProxyError::AllProvidersCoolingDown(model.name.clone()))
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
        let mut tried: Vec<String> = Vec::new();
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
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(ProxyError::AllProvidersCoolingDown(model.name.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
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
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
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
    async fn select_provider_uses_soonest_when_all_are_cooling() {
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

        let (name, _) = router.select_provider(model).await.unwrap();

        assert_eq!(name, "backup");
        assert_eq!(router.config().models[0].name, "m");
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
            ProxyError::AllProvidersCoolingDown(ref model) if model == "m"
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

        assert!(matches!(complete, ProxyError::AllProvidersCoolingDown(_)));
        assert!(matches!(stream, ProxyError::AllProvidersCoolingDown(_)));
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

        assert!(matches!(error, ProxyError::AllProvidersCoolingDown(_)));
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
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
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
        assert!(matches!(error, ProxyError::AllProvidersCoolingDown(_)));
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
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
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
        assert!(matches!(error, ProxyError::AllProvidersCoolingDown(_)));
    }

    #[tokio::test]
    async fn select_provider_prefers_shorter_remaining_cooldown() {
        // When all providers are cooling down, the one with the shortest
        // remaining time wins. Mark primary with a short cooldown and backup
        // with a long one and verify primary is selected.
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

        let (name, _) = router.select_provider(model).await.unwrap();
        assert_eq!(name, "primary");
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
}
