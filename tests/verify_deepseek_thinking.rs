//! Verification tests for the DeepSeek thinking strip-and-retry implementation.
//!
//! These tests exercise real-world cross-model fallback scenarios using
//! wiremock-simulated Anthropic-compatible endpoints. They complement the
//! existing unit and integration tests by covering the full Router +
//! fallback chain + AnthropicProvider strip-and-retry interaction, which
//! existing tests do not cover (they use single-provider chains).
//!
//! Run: `cargo test --test verify_deepseek_thinking`

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use llmproxy::anthropic::MessagesRequest;
use llmproxy::config::{Config, ModelConfig, ProviderConfig, ServerConfig};
use llmproxy::cooldown::CooldownCache;
use llmproxy::error::ProxyError;
use llmproxy::providers::anthropic::AnthropicProvider;
use llmproxy::providers::{ProviderOutput, SharedProvider};
use llmproxy::router::Router;

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn anthropic_ok(text: &str, model: &str) -> Value {
    json!({
        "id": "msg_verify",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 5, "output_tokens": 3}
    })
}

fn thinking_400_body() -> Value {
    json!({
        "error": {
            "message": "The `content[].thinking` in the thinking mode must be passed back to the API."
        }
    })
}

/// Multi-turn request whose history carries an assistant thinking block
/// (with an Anthropic signature) plus a text block. This is the canonical
/// cross-model fallback scenario: the client was previously talking to
/// Claude, and now the next request is being sent.
fn make_thinking_history_req(model: &str) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": model,
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig-1"},
                {"type": "text", "text": "previous answer"}
            ]},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap()
}

/// Request whose assistant message contains ONLY a thinking block (no text).
/// After stripping, this message becomes empty and should be removed entirely,
/// which may create a consecutive-same-role message sequence.
fn make_pure_thinking_history_req(model: &str) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": model,
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}
            ]},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap()
}

/// Build an AnthropicProvider pointing at a wiremock server.
fn make_anthropic_provider(name: &str, server: &MockServer) -> SharedProvider {
    Arc::new(
        AnthropicProvider::new(
            name.to_string(),
            "k".to_string(),
            format!("{}/v1", server.uri()),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap(),
    )
}

/// Build a ProviderConfig::Anthropic entry for the given wiremock server.
fn make_anthropic_config(name: &str, server: &MockServer) -> ProviderConfig {
    ProviderConfig::Anthropic {
        name: name.to_string(),
        api_key: "k".to_string(),
        api_base: format!("{}/v1", server.uri()),
        model_rewrite: HashMap::new(),
        use_proxy: false,
    }
}

/// Build a Router with the given providers and model chain (first = primary, rest = fallback).
fn build_router(
    providers: HashMap<String, SharedProvider>,
    provider_configs: Vec<ProviderConfig>,
    model_chain: Vec<&str>,
) -> Router {
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: provider_configs,
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: model_chain[0].to_string(),
            fallback_chain: model_chain[1..].iter().map(|s| s.to_string()).collect(),
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: model_chain.len() as u32,
        }],
    };
    Router::new(Arc::new(cfg), providers, CooldownCache::new())
}

// ────────────────────────────────────────────────────────────────────────
// Scenario 1: Cross-model fallback self-healing
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cross_model_fallback_self_healing() {
    // Simulate: primary (Claude) is rate-limited (429). The client has
    // thinking history with an Anthropic-native signature from a previous
    // Claude conversation. The Router falls back to deepseek, which
    // initially rejects the cross-model thinking signature with a 400.
    // The AnthropicProvider strips the thinking blocks and retries,
    // succeeding on the second attempt. The client sees a 200 response.
    //
    // Key verification points:
    //  (a) Router falls back from primary to deepseek (1 RouteAttempt for 429)
    //  (b) DeepSeek wiremock receives exactly 2 requests
    //  (c) The retried request body has no thinking/redacted_thinking blocks
    //  (d) The top-level `thinking` key is absent from the retried body
    //  (e) Non-thinking content (text block) is preserved

    // -- Wiremock: primary (Claude) returns 429 --------------------------
    let primary_srv = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate-limited"))
        .expect(1)
        .mount(&primary_srv)
        .await;

    // -- Wiremock: fallback (DeepSeek) returns 400 thinking → 200 --------
    let deepseek_srv = MockServer::start().await;
    let captured: Arc<std::sync::Mutex<Option<Value>>> =
        Arc::new(std::sync::Mutex::new(None));
    let captured_for_responder = captured.clone();
    let ds_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ds_counter_clone = ds_counter.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            let n = ds_counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(400).set_body_json(thinking_400_body())
            } else {
                *captured_for_responder.lock().unwrap() = Some(
                    serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                );
                ResponseTemplate::new(200)
                    .set_body_json(anthropic_ok("deepseek answer after strip", "claude-test"))
            }
        })
        .expect(2)
        .mount(&deepseek_srv)
        .await;

    // -- Build Router with primary=claude, fallback=deepseek -------------
    let mut providers = HashMap::new();
    providers.insert(
        "claude".to_string(),
        make_anthropic_provider("claude", &primary_srv),
    );
    providers.insert(
        "deepseek".to_string(),
        make_anthropic_provider("deepseek", &deepseek_srv),
    );
    let configs = vec![
        make_anthropic_config("claude", &primary_srv),
        make_anthropic_config("deepseek", &deepseek_srv),
    ];
    let router = build_router(providers, configs, vec!["claude", "deepseek"]);

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router
        .complete(model_cfg, &make_thinking_history_req("claude-test"))
        .await
        .unwrap();

    // -- Assert final output ----------------------------------------------
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(
        body["content"][0]["text"], "deepseek answer after strip",
        "client must see deepseek's successful response"
    );

    // -- Assert fallback happened (1 RouteAttempt for primary's 429) -----
    assert_eq!(attempts.len(), 1, "exactly one attempt (primary's 429)");
    assert_eq!(attempts[0].provider, "claude");
    assert_eq!(attempts[0].status, 429);

    // -- Assert deepseek was called exactly twice (strip+retry) -----------
    assert_eq!(
        ds_counter.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "deepseek must be called twice (initial 400 + stripped retry)"
    );

    // -- Assert the retried body is properly stripped ---------------------
    let sent = captured.lock().unwrap().clone().expect("second body captured");

    // Top-level thinking key must be ABSENT (not null).
    assert!(
        sent.as_object().unwrap().get("thinking").is_none(),
        "top-level thinking must be removed from the retried request"
    );

    // All 3 messages survive (assistant has text + thinking, strip removes
    // only the thinking block).
    let msgs = sent["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3, "all 3 messages must survive strip");

    // Assistant message (index 1) keeps its text block.
    let assistant = &msgs[1];
    assert_eq!(assistant["role"], "assistant");
    let text_block = assistant["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "text")
        .expect("assistant must keep its text block after strip");
    assert_eq!(text_block["text"], "previous answer");

    // No thinking or redacted_thinking blocks remain anywhere.
    for msg in msgs {
        if let Some(blocks) = msg["content"].as_array() {
            for b in blocks {
                let t = b["type"].as_str();
                assert_ne!(t, Some("thinking"), "thinking block must be stripped");
                assert_ne!(t, Some("redacted_thinking"), "redacted_thinking block must be stripped");
            }
        }
    }

    // User message (index 0) is unchanged (string content).
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "hello");

    // Third message (index 2) is unchanged (string content).
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(msgs[2]["content"], "continue");
}

// ────────────────────────────────────────────────────────────────────────
// Scenario 2: Strip exhausted, no deeper fallback
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn strip_exhausted_no_deeper_fallback() {
    // The fallback (deepseek) always returns 400 thinking. After one
    // strip+retry cycle (2 total requests), the AnthropicProvider
    // surfaces the raw 400. Because 400 is not cooldownable, the Router
    // returns the error immediately — no fallback to a third provider,
    // no RouteAttempt for deepseek, no x-llmproxy-failed-providers.
    //
    // Primary returns 429 (cooldownable) to force fallback.

    // -- Wiremock: primary (Claude) returns 429 --------------------------
    let primary_srv = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .mount(&primary_srv)
        .await;

    // -- Wiremock: fallback (DeepSeek) always returns 400 thinking --------
    let deepseek_srv = MockServer::start().await;
    let ds_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ds_counter_clone = ds_counter.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            ds_counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ResponseTemplate::new(400).set_body_json(thinking_400_body())
        })
        .expect(2)
        .mount(&deepseek_srv)
        .await;

    // -- Build Router with primary=claude, fallback=deepseek -------------
    let mut providers = HashMap::new();
    providers.insert(
        "claude".to_string(),
        make_anthropic_provider("claude", &primary_srv),
    );
    providers.insert(
        "deepseek".to_string(),
        make_anthropic_provider("deepseek", &deepseek_srv),
    );
    let configs = vec![
        make_anthropic_config("claude", &primary_srv),
        make_anthropic_config("deepseek", &deepseek_srv),
    ];
    let router = build_router(providers, configs, vec!["claude", "deepseek"]);

    let model_cfg = router.find_model("claude-test").unwrap();
    let err = router
        .complete(model_cfg, &make_thinking_history_req("claude-test"))
        .await
        .err()
        .expect("strip-exhausted must surface as Err");

    // -- Error is raw Upstream 400, not a friendly envelope ---------------
    match &err {
        ProxyError::Upstream { status, body } => {
            assert_eq!(*status, 400);
            let parsed: Value =
                serde_json::from_str(body).unwrap_or_else(|_| json!({}));
            assert_eq!(
                parsed["error"]["message"],
                "The `content[].thinking` in the thinking mode must be passed back to the API."
            );
            // Must NOT be the friendly thinking_not_supported envelope.
            assert_ne!(parsed["error"]["type"], "thinking_not_supported");
        }
        other => panic!("expected Upstream 400, got: {other:?}"),
    }

    // -- 400 is non-cooldownable: no failed-providers header --------------
    assert!(
        err.failed_providers_header().is_none(),
        "non-cooldownable 400 must not produce failed-providers header"
    );

    // -- DeepSeek was called exactly twice (initial + one strip-retry) ----
    assert_eq!(
        ds_counter.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "deepseek must be called exactly twice (strip+retry exhausted)"
    );

    // -- Confirm the error is NOT AllProvidersFailed (no deeper chain) ----
    assert!(
        !matches!(err, ProxyError::AllProvidersFailed { .. }),
        "400 is non-cooldownable; router must return Upstream directly, not advance chain"
    );

    // -- Primary must be on cooldown (429 is cooldownable) ----------------
    assert!(
        router.cooldown().is_cooling_down("claude").await,
        "primary must be on cooldown after 429"
    );

    // -- DeepSeek must NOT be on cooldown (400 is non-cooldownable) -----
    assert!(
        !router.cooldown().is_cooling_down("deepseek").await,
        "fallback must NOT be on cooldown for a non-cooldownable 400"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Scenario 3: Pure thinking history (no text) boundary
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn pure_thinking_history_message_removed_after_strip() {
    // The assistant message has ONLY a thinking block (no text). After
    // stripping, the message's content array becomes empty, so the entire
    // message is removed from the messages array. This creates a sequence
    // of [user, user] (consecutive same-role messages), which violates
    // the Anthropic Messages protocol.
    //
    // This test OBSERVES AND REPORTS the actual behavior. It does NOT
    // assert that the behavior is correct — see P1 in the deferred-issues
    // document. The test captures the retried request body and verifies
    // the structural consequence (message removal + consecutive roles).

    let server = MockServer::start().await;
    let captured: Arc<std::sync::Mutex<Option<Value>>> =
        Arc::new(std::sync::Mutex::new(None));
    let captured_for_responder = captured.clone();
    let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counter_clone = counter.clone();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(400).set_body_json(thinking_400_body())
            } else {
                *captured_for_responder.lock().unwrap() = Some(
                    serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                );
                // Return 200 so the test passes and we can inspect the
                // captured body. In production, a real upstream would
                // reject consecutive same-role messages with a new 400.
                ResponseTemplate::new(200)
                    .set_body_json(anthropic_ok("ok despite malformed messages", "claude-test"))
            }
        })
        .expect(2)
        .mount(&server)
        .await;

    let provider = make_anthropic_provider("deepseek", &server);
    let mut providers = HashMap::new();
    providers.insert("deepseek".to_string(), provider);
    let configs = vec![make_anthropic_config("deepseek", &server)];
    let router = build_router(providers, configs, vec!["deepseek"]);

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, _attempts) = router
        .complete(model_cfg, &make_pure_thinking_history_req("claude-test"))
        .await
        .unwrap();

    // -- Assert the strip-and-retry succeeded (our mock returned 200) -----
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "ok despite malformed messages");
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "provider must be called twice (strip+retry)"
    );

    // -- Inspect the retried body for structural consequences ------------
    let sent = captured.lock().unwrap().clone().expect("second body captured");

    // Top-level thinking is removed.
    assert!(sent.as_object().unwrap().get("thinking").is_none());

    // The assistant message (which had only a thinking block) was entirely
    // removed. We expect only 2 messages remaining: user("hello") and
    // user("continue").
    let msgs = sent["messages"].as_array().unwrap();
    assert_eq!(
        msgs.len(),
        2,
        "assistant message removed → 2 messages remain (was 3)"
    );

    // Both remaining messages are user-role — this is a consecutive
    // same-role violation of the Anthropic Messages protocol.
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "hello");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "continue");

    // No thinking block remains anywhere.
    for msg in msgs {
        if let Some(blocks) = msg["content"].as_array() {
            for b in blocks {
                let t = b["type"].as_str();
                assert_ne!(t, Some("thinking"));
                assert_ne!(t, Some("redacted_thinking"));
            }
        }
    }

    // NOTE: In production, a real Anthropic-compatible upstream would
    // likely reject the consecutive user-role messages with a 400. Our
    // mock returns 200 for observability. The observed behavior is:
    //   - Original: [user("hello"), assistant(thinking_only), user("continue")]
    //   - Stripped: [user("hello"), user("continue")]  ← consecutive users!
    // This is a known issue (P1 in the deferred-issues document).
}

// ────────────────────────────────────────────────────────────────────────
// Scenario 4: Top-level thinking param deletion
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn top_level_thinking_param_completely_absent_in_retry() {
    // The top-level `thinking` key must be removed (not nulled) from the
    // retried request body. DeepSeek is a strict validator — a spurious
    // `"thinking": null` on a request that never enabled thinking could
    // itself be rejected. This test verifies the key is entirely absent.
    //
    // This scenario is also covered by the unit test
    // `complete_thinking_error_strip_removes_top_level_thinking_param`,
    // but we replicate it here with a fresh fixture for completeness.

    let server = MockServer::start().await;
    let captured: Arc<std::sync::Mutex<Option<Value>>> =
        Arc::new(std::sync::Mutex::new(None));
    let captured_for_responder = captured.clone();
    let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counter_clone = counter.clone();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(400).set_body_json(thinking_400_body())
            } else {
                *captured_for_responder.lock().unwrap() = Some(
                    serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                );
                ResponseTemplate::new(200)
                    .set_body_json(anthropic_ok("ok", "claude-test"))
            }
        })
        .expect(2)
        .mount(&server)
        .await;

    let provider = make_anthropic_provider("deepseek", &server);
    let mut providers = HashMap::new();
    providers.insert("deepseek".to_string(), provider);
    let configs = vec![make_anthropic_config("deepseek", &server)];
    let router = build_router(providers, configs, vec!["deepseek"]);

    let model_cfg = router.find_model("claude-test").unwrap();

    // Use a request with the top-level thinking param enabled + history.
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "claude-test",
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 4000, "display": "summarized"},
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning", "signature": "sig-1"},
                {"type": "text", "text": "answer"}
            ]},
            {"role": "user", "content": "next"}
        ]
    }))
    .unwrap();

    let (_out, _attempts) = router.complete(model_cfg, &req).await.unwrap();

    let sent = captured.lock().unwrap().clone().expect("second body captured");

    // The key assertion: `thinking` must be completely absent from the
    // top-level object. It must NOT be present as `null`.
    let obj = sent.as_object().unwrap();
    assert!(
        !obj.contains_key("thinking"),
        "top-level 'thinking' key must be ABSENT from retried body, but found: {:?}",
        obj.get("thinking")
    );

    // Double-check: obj.get("thinking") returns None.
    assert!(
        obj.get("thinking").is_none(),
        "obj.get(\"thinking\") must return None"
    );

    // Verify other expected keys are still present (sanity check).
    assert!(obj.contains_key("model"));
    assert!(obj.contains_key("messages"));
    assert!(obj.contains_key("max_tokens"));
    assert_eq!(obj.get("stream"), Some(&json!(false)));

    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "must be called twice"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Bonus: Verify that the stream path still produces friendly errors
// (cross-check with unit test stream_does_not_apply_strip_and_retry)
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn stream_thinking_400_with_thinking_history_returns_friendly_error() {
    // Even when the request carries assistant thinking history (the exact
    // scenario that would trigger strip-and-retry on the complete() path),
    // the streaming path must NOT retry. Instead, it returns the friendly
    // `thinking_not_supported` envelope. This is the stream() contract:
    // once SSE bytes start flowing, retry would double-emit content.

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(thinking_400_body())
        )
        .expect(1) // exactly one call — no retry
        .mount(&server)
        .await;

    let provider = make_anthropic_provider("deepseek", &server);
    let mut providers = HashMap::new();
    providers.insert("deepseek".to_string(), provider);
    let configs = vec![make_anthropic_config("deepseek", &server)];
    let router = build_router(providers, configs, vec!["deepseek"]);

    let _model_cfg = router.find_model("claude-test").unwrap();

    // Use thinking_history_request with stream=true. We call the
    // provider directly (not through Router::stream) because the Router's
    // complete() method is used — stream exercises the stream path.
    // Actually, Router::complete() calls provider.complete(), not
    // provider.stream(). So we test the provider directly.

    // Use a streaming request with thinking history.
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "claude-test",
        "max_tokens": 64,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig-1"},
                {"type": "text", "text": "previous answer"}
            ]},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap();

    // Call provider.stream() directly — not through Router.
    let ds_provider = router.providers().get("deepseek").unwrap();
    let err = ds_provider
        .stream(&req, &HashMap::new())
        .await
        .err()
        .expect("thinking mismatch on stream must surface as Err");

    match err {
        ProxyError::Upstream { status, body } => {
            assert_eq!(status, 400);
            let parsed: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                parsed["error"]["type"], "thinking_not_supported",
                "stream path must return friendly envelope"
            );
            assert_eq!(parsed["error"]["provider"], "deepseek");
            assert_eq!(parsed["error"]["client_model"], "claude-test");
        }
        other => panic!("expected Upstream 400 with friendly envelope, got: {other:?}"),
    }
}
