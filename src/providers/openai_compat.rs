//! Generic OpenAI Chat Completions provider.
//!
//! Used by DeepSeek, MiniMax, OpenCode Zen, and any other backend that
//! exposes an OpenAI-style /chat/completions endpoint. The provider
//! always converts Anthropic requests to OpenAI Chat Completions and
//! converts the response back; for native Anthropic passthrough use the
//! `anthropic` provider type instead.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use futures_util::Stream;
use serde_json::{json, Value};

use crate::anthropic::{MessagesRequest, StreamEvent};
use crate::conversion::{anthropic_to_openai_request, make_message_id, openai_to_anthropic_response};
use crate::error::{ProxyError, Result};
use crate::openai::{looks_like_error_envelope, ChatMessage, ChatRequest};
use crate::providers::{Provider, ProviderOutput};

pub struct OpenAiCompatProvider {
    name: String,
    api_base: String,
    api_key: String,
    model_rewrite: HashMap<String, String>,
    /// 要排除的 OpenRouter 后端提供商 slug 列表。
    /// 非空 + `is_openrouter == true` 时，注入 `provider: {ignore: [...]}`
    /// 到请求体；非 OpenRouter 后端上忽略该配置并产生启动警告。
    provider_ignore: Vec<String>,
    /// 缓存的 `api_base` host 检测结果：`true` 当且仅当 host 是
    /// `openrouter.ai`（大小写不敏感，允许 `www.openrouter.ai` 等子域）。
    /// 在 `new()` 中计算一次，避免每请求重复解析。
    is_openrouter: bool,
    /// 当为 `true` 时，thinking 请求里每条缺失 `reasoning_content` 的历史
    /// assistant 消息（跨模型 thinking 轮 / redacted_thinking / 纯文本轮）
    /// 由转换层以 `reasoning_content: ""` 填充，使 DeepSeek-V4 类上游在
    /// thinking 模式下接受而非 400。同时关闭 reasoning-echo 400 的
    /// strip-and-retry / 友好 envelope 兜底（400 不再可预期；若仍出现则
    /// 逐字透传）。默认 `false`（wire 字节与关闭前一致）。
    reasoning_echo: bool,
    http: reqwest::Client,
}

impl OpenAiCompatProvider {
    pub fn new(
        name: String,
        api_base: String,
        api_key: String,
        model_rewrite: HashMap<String, String>,
        provider_ignore: Vec<String>,
        reasoning_echo: bool,
        http: reqwest::Client,
    ) -> Result<Self> {
        let api_base = api_base.trim_end_matches('/').to_string();
        // Detect whether api_base points at OpenRouter. The detection
        // itself lives in `providers::is_openrouter_api_base` so the
        // Anthropic and OpenAI-compat providers share one canonical
        // implementation; the per-provider struct field just caches
        // the result to avoid re-parsing on every request.
        let is_openrouter = crate::providers::is_openrouter_api_base(&api_base);

        // 启动时一次性警告：避免每请求刷日志。误配到 DeepSeek/opencode
        // 等严格校验后端时，请求体保持干净（不注入 provider 字段）。
        if !provider_ignore.is_empty() && !is_openrouter {
            tracing::warn!(
                provider = %name,
                api_base = %api_base,
                "provider_ignore configured but api_base is not openrouter.ai; \
                 field will be ignored (strict OpenAI-compat backends like \
                 DeepSeek may reject unknown top-level fields)"
            );
        }

        Ok(Self {
            name,
            api_base,
            api_key,
            model_rewrite,
            provider_ignore,
            is_openrouter,
            reasoning_echo,
            http,
        })
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.api_base)
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.api_base)
    }

    /// 如果 `provider_ignore` 非空且 `api_base` 指向 OpenRouter，则将
    /// `provider: {ignore: [...]}` 注入到 `ChatRequest.extra`。OpenRouter
    /// 会用该字段在路由时跳过被排除的提供商。
    ///
    /// 三种门控：
    /// - `provider_ignore` 为空 → 无操作（默认行为）。
    /// - 非空且是 OpenRouter → 注入。
    /// - 非空且非 OpenRouter → 无操作（已在 `new()` 启动时记录警告）。
    ///
    /// 注入必须在 retry 循环之前，因为 `strip_reasoning_echo()` /
    /// `downgrade_response_format()` 不触碰 `extra.provider`，retry
    /// 请求体保持携带该字段（正确的语义——排除的提供商在每次尝试中都
    /// 应被排除）。
    fn inject_provider_ignore(&self, req: &mut ChatRequest) {
        if self.provider_ignore.is_empty() || !self.is_openrouter {
            return;
        }
        // `extra` 由 `anthropic_to_openai_request` 初始化为
        // `Value::Object(Map::new())`，所以 `as_object_mut()` 实际
        // 路径上必然成功。`Value::Null` 分支仅为防御性 guard。
        if let Some(obj) = req.extra.as_object_mut() {
            obj.insert(
                "provider".to_string(),
                json!({"ignore": self.provider_ignore}),
            );
        }
    }

    /// Test-only constructor that bypasses the `is_openrouter` host
    /// detection so wire-level tests can assert both the ON and OFF
    /// paths of the gate without depending on `openrouter.ai` DNS or
    /// constructing a real HTTP server at that hostname. The override
    /// mirrors the production semantics: when `force_openrouter` is
    /// `true`, the gate permits injection regardless of `api_base`;
    /// when `false`, the gate suppresses injection.
    #[cfg(test)]
    fn new_for_test(
        name: String,
        api_base: String,
        api_key: String,
        model_rewrite: HashMap<String, String>,
        provider_ignore: Vec<String>,
        reasoning_echo: bool,
        force_openrouter: bool,
        http: reqwest::Client,
    ) -> Result<Self> {
        let api_base = api_base.trim_end_matches('/').to_string();
        Ok(Self {
            name,
            api_base,
            api_key,
            model_rewrite,
            provider_ignore,
            is_openrouter: force_openrouter,
            reasoning_echo,
            http,
        })
    }

    /// Build a friendly Anthropic-shaped error body when an OpenAI-compat
    /// upstream (DeepSeek/opencode in thinking mode) rejects the request
    /// because `reasoning_content` could not be echoed back on every
    /// assistant message. Mirrors `AnthropicProvider::thinking_not_supported_error`
    /// (anthropic.rs) but with a type specific to the reasoning-echo gap, so
    /// statusline / logs can tell this failure apart from "model doesn't
    /// support thinking".
    ///
    /// `client_model` is the name the proxy received from the client;
    /// `upstream_model` is the wire name actually sent upstream (the
    /// ChatRequest.model the conversion produced after `model_rewrite` +
    /// `strip_date_suffix` fallback) — what the operator needs to look up
    /// in the upstream dashboard.
    fn reasoning_echo_error(
        &self,
        client_model: &str,
        upstream_model: &str,
        upstream_body: &str,
    ) -> ProxyError {
        // Truncate upstream body for the human-readable message so the
        // envelope stays compact; the full body is preserved in
        // `upstream_body` for debugging.
        let snippet: String = upstream_body.chars().take(200).collect();
        let friendly = json!({
            "type": "error",
            "error": {
                "type": "reasoning_content_not_passed_back",
                "message": format!(
                    "Provider '{}' rejected this request in thinking mode: the upstream \
                     requires `reasoning_content` to be passed back on every assistant \
                     message, but the request history's thinking content is missing or \
                     cannot be echoed back (e.g. a redacted_thinking block, or a \
                     signature-less thinking block). This provider/model does not \
                     support cross-model thinking echo. Reconfigure config.yaml: \
                     either remove this provider from chains whose primary uses \
                     thinking, or use a non-thinking request for model '{}'. \
                     Upstream response: {}",
                    self.name, upstream_model, snippet
                ),
                "provider": self.name,
                "client_model": client_model,
                "upstream_model": upstream_model,
                "upstream_body": upstream_body,
            }
        });
        ProxyError::Upstream {
            status: 400,
            body: friendly.to_string(),
        }
    }
}

/// Detect OpenAI-style thinking-echo rejection. opencode/DeepSeek require
/// `reasoning_content` to be passed back on every assistant message in
/// thinking mode; when a message is missing it they return e.g.
///   [invalid_request_error] The reasoning_content in the thinking mode must
///   be passed back to the API
/// Match the two substrings (mirror of `anthropic.rs::has_thinking_error`,
/// but for the OpenAI wording). Case-sensitive; upstream variants with a
/// space (`reasoning content`) would miss — accepted, matches litellm.
fn has_reasoning_echo_error(body: &str) -> bool {
    body.contains("reasoning_content") && body.contains("must be passed back")
}

/// Strip every thinking-echo signal from an OpenAI request so a retry is
/// sent in non-thinking mode: each assistant message's `reasoning_content`
/// → None, top-level `reasoning_effort` → None, and any `thinking` key in
/// `extra` is removed. Non-destructive — on a request with no reasoning
/// signals it is a pure no-op.
fn strip_reasoning_echo(req: &mut ChatRequest) {
    for m in &mut req.messages {
        if let ChatMessage::Assistant { reasoning_content, .. } = m {
            *reasoning_content = None;
        }
    }
    req.reasoning_effort = None;
    if let Some(obj) = req.extra.as_object_mut() {
        obj.remove("thinking");
    }
}

/// Detect DeepSeek/opencode rejecting `response_format` json_schema, e.g.
/// "[invalid_request_error] This response_format type is unavailable now".
/// Two-substring match (mirror of `has_reasoning_echo_error`); upstream
/// variant-miss (different wording) is accepted — matches litellm-style
/// matchers. Gated by status==400 upstream.
fn has_response_format_error(body: &str) -> bool {
    body.contains("response_format") && body.contains("unavailable")
}

/// Replace extra.response_format `{type:"json_schema",...}` with
/// `{"type":"json_object"}`. Best-effort: keeps JSON output
/// (stop hook must parse it), drops schema enforcement. No-op when
/// `extra` is Null / non-object / `response_format` is absent or not
/// `json_schema`. Returns whether a downgrade happened.
fn downgrade_response_format(req: &mut ChatRequest) -> bool {
    let Some(obj) = req.extra.as_object_mut() else {
        return false;
    };
    let Some(rf) = obj.get("response_format") else {
        return false;
    };
    let is_json_schema = rf
        .get("type")
        .and_then(Value::as_str)
        .map(|t| t == "json_schema")
        .unwrap_or(false);
    if !is_json_schema {
        return false;
    }
    obj.insert(
        "response_format".to_string(),
        json!({"type": "json_object"}),
    );
    true
}

/// Detect the OpenAI-style error envelope `{"error": {...}}` returned on
/// HTTP 200 by some upstreams. Must be a top-level object with a single
/// `error` key whose value is itself an object (so we don't confuse it
/// with a legitimate assistant message that happens to contain the word
/// "error").

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_serve_model(&self, model: &str) -> bool {
        // Empty rewrite table means "this provider exposes its own model
        // catalog — pass the name through verbatim". A non-empty table
        // is an explicit allow-list; the proxy must not forward names
        // that aren't in it, because doing so produces a misleading 400
        // from the upstream and breaks the fallback chain — see fix-R11.
        self.model_rewrite.is_empty() || self.model_rewrite.contains_key(model)
    }

    fn merged_rewrite<'a>(
        &'a self,
        runtime: &'a HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(runtime.iter().map(|(k, v)| (k.clone(), v.clone())));
        merged
    }

    async fn list_models(&self) -> Option<Vec<serde_json::Value>> {
        let url = self.models_url();
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .header("accept", "application/json")
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            tracing::warn!(
                status = %resp.status(),
                provider = %self.name,
                "list_models returned non-success"
            );
            return None;
        }
        let body: serde_json::Value = resp.json().await.ok()?;
        let data = body.get("data")?.as_array()?;
        Some(
            data.iter()
                .filter_map(|entry| {
                    let id = entry.get("id")?.as_str()?;
                    let owned_by = entry
                        .get("owned_by")
                        .and_then(|v| v.as_str())
                        .unwrap_or("openai_compat");
                    let display_name = entry
                        .get("display_name")
                        .or_else(|| entry.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(id);
                    Some(serde_json::json!({
                        "id": id,
                        "object": "model",
                        "created": entry.get("created").and_then(|v| v.as_i64()).unwrap_or(0),
                        "owned_by": owned_by,
                        "display_name": display_name,
                    }))
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

        let mut openai_req = anthropic_to_openai_request(req, &merged, self.reasoning_echo)?;
        openai_req.stream = false;
        openai_req.stream_options = None;
        self.inject_provider_ignore(&mut openai_req);

        // Strip-and-retry loop (mirrors `AnthropicProvider::complete`):
        // DeepSeek/opencode in thinking mode reject requests whose assistant
        // history can't echo `reasoning_content` back. On that specific 400
        // we downgrade the SAME request to non-thinking mode and retry once.
        // DeepSeek/opencode also reject `response_format: {type:"json_schema"}`
        // with a 400 ("response_format ... unavailable"); we downgrade to
        // `{"type":"json_object"}` and retry once. Both downgrades are
        // per-type-gated (each at most once) and can cascade, so the cap is
        // 3 attempts total: original + reasoning-strip + format-downgrade.
        // The retry reuses the mutated `openai_req` — never rebuilt via
        // `anthropic_to_openai_request`, which would lose the downgrades.
        let mut attempt: u32 = 0;
        const MAX_ATTEMPTS: u32 = 3;
        let mut reasoning_stripped = false;
        let mut format_downgraded = false;
        loop {
            attempt += 1;
            let resp = self
                .http
                .post(self.chat_url())
                .bearer_auth(&self.api_key)
                .header("content-type", "application/json")
                .json(&openai_req)
                .send()
                .await?;

            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                if status.as_u16() == 400 && attempt < MAX_ATTEMPTS {
                    // When reasoning_echo is enabled the conversion already
                    // fills every assistant message with `reasoning_content`,
                    // so this 400 is no longer expected; skip the strip-and-
                    // retry path entirely and surface any 400 verbatim (a real
                    // protocol violation rather than the known gap).
                    if !reasoning_stripped && !self.reasoning_echo && has_reasoning_echo_error(&body) {
                        tracing::warn!(
                            provider = %self.name,
                            client_model = %req.model,
                            attempt = attempt,
                            "stripping reasoning echo from request and retrying once (upstream demands reasoning_content be passed back)"
                        );
                        strip_reasoning_echo(&mut openai_req);
                        reasoning_stripped = true;
                        continue;
                    }
                    if !format_downgraded
                        && has_response_format_error(&body)
                        && downgrade_response_format(&mut openai_req)
                    {
                        tracing::info!(
                            provider = %self.name,
                            client_model = %req.model,
                            attempt = attempt,
                            "downgrading response_format json_schema -> json_object and retrying"
                        );
                        format_downgraded = true;
                        continue;
                    }
                    // downgrade_response_format is a no-op (request had no
                    // json_schema) or neither subtitle matched — surface
                    // the 400 verbatim; do NOT spin an identical retry.
                }
                return Err(ProxyError::Upstream {
                    status: status.as_u16(),
                    body,
                });
            }

            let parsed: Value = serde_json::from_str(&body)?;
            if looks_like_error_envelope(&parsed) {
                // Some upstreams (e.g. DeepSeek on unknown model) return HTTP 200
                // with an OpenAI error envelope instead of a chat response. Treat
                // it as a 400-class upstream failure so the client sees the real
                // message instead of a generic 500 "missing field `object`".
                return Err(ProxyError::Upstream {
                    status: 400,
                    body,
                });
            }
            let chat: crate::openai::ChatResponse = serde_json::from_value(parsed)?;
            let msg_id = make_message_id();
            let anthropic_resp = openai_to_anthropic_response(&chat, &req.model, &msg_id)?;
            // Downgrades are invisible to the client (response shape is
            // unchanged), but operators need observability — only warn on a
            // success that actually triggered a downgrade, not on ordinary
            // successes. Two independent warns so the log line accurately
            // describes which downgrade happened.
            if reasoning_stripped {
                tracing::warn!(
                    provider = %self.name,
                    client_model = %req.model,
                    "recovered by downgrading request to non-thinking mode after upstream reasoning-echo 400"
                );
            }
            if format_downgraded {
                tracing::info!(
                    provider = %self.name,
                    client_model = %req.model,
                    "recovered by downgrading response_format json_schema -> json_object after upstream 400"
                );
            }
            return Ok(ProviderOutput::Json(serde_json::to_value(anthropic_resp)?));
        }
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = self.merged_rewrite(model_rewrite);

        let mut openai_req = anthropic_to_openai_request(req, &merged, self.reasoning_echo)?;
        openai_req.stream = true;
        self.inject_provider_ignore(&mut openai_req);

        // Streaming retry: response_format 400 is safe to retry (the 400 is
        // a synchronous POST response — it arrives before any SSE byte
        // flows, so retrying does not double-emit to the client). The
        // Router's "no retry once streaming" contract is preserved;
        // reasoning-echo 400 still surfaces as a friendly envelope (no
        // retry) because by the time we see a 400, the conversion is
        // already idempotent and the same envelope is friendlier than a
        // raw retry.
        let mut attempt: u32 = 0;
        let mut format_downgraded = false;
        loop {
            attempt += 1;
            let resp = self
                .http
                .post(self.chat_url())
                .bearer_auth(&self.api_key)
                .header("content-type", "application/json")
                .json(&openai_req)
                .send()
                .await?;

            let status = resp.status();
            if status.is_success() {
                let byte_stream = resp.bytes_stream();
                let sse = OpenAiSseToAnthropic::new(byte_stream, &req.model);
                if format_downgraded {
                    tracing::info!(
                        provider = %self.name,
                        client_model = %req.model,
                        "recovered by downgrading response_format json_schema -> json_object after upstream 400"
                    );
                }
                return Ok(ProviderOutput::Stream(Box::new(sse)));
            }

            let text = resp.text().await?;
            if status.as_u16() == 400
                && attempt < 2
                && !format_downgraded
                && has_response_format_error(&text)
                && downgrade_response_format(&mut openai_req)
            {
                tracing::info!(
                    provider = %self.name,
                    client_model = %req.model,
                    attempt = attempt,
                    "downgrading response_format json_schema -> json_object and retrying"
                );
                format_downgraded = true;
                continue;
            }
            // When reasoning_echo is enabled the conversion already fills every
            // assistant message with `reasoning_content`, so this 400 is no
            // longer expected; skip the friendly envelope and surface the
            // 400 verbatim (a real protocol violation rather than the known
            // gap).
            if status.as_u16() == 400 && !self.reasoning_echo && has_reasoning_echo_error(&text) {
                return Err(self.reasoning_echo_error(
                    &req.model,
                    &openai_req.model, // wire name: real model sent upstream
                    &text,
                ));
            }
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
    }
}

/// Adapter: reads an OpenAI SSE byte stream and emits Anthropic SSE byte stream.
pub struct OpenAiSseToAnthropic<S> {
    inner: S,
    translator: Option<crate::conversion::stream::StreamTranslator>,
    pending: BytesMut,
    finished: bool,
    output_buffer: VecDeque<Bytes>,
}

impl<S> OpenAiSseToAnthropic<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    pub fn new(inner: S, model: &str) -> Self {
        Self {
            inner,
            translator: Some(crate::conversion::stream::StreamTranslator::new(
                make_message_id(),
                model,
            )),
            pending: BytesMut::new(),
            finished: false,
            output_buffer: VecDeque::new(),
        }
    }

    fn encode(ev: &StreamEvent) -> Bytes {
        let payload = serde_json::to_string(ev).unwrap_or_default();
        Bytes::from(format!("event: {}\ndata: {}\n\n", event_name(ev), payload))
    }

    fn process_lines(&mut self) {
        // Drain complete `\n`-terminated lines and feed to translator.
        loop {
            let Some(pos) = self.pending.iter().position(|&b| b == b'\n') else {
                break;
            };
            let line_bytes = self.pending.split_to(pos);
            self.pending.advance(1); // consume the '\n'
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches('\r');
            let Some(rest) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = rest.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                if let Some(mut t) = self.translator.take() {
                    for ev in t.finalize() {
                        self.output_buffer.push_back(Self::encode(&ev));
                    }
                }
                self.finished = true;
                return;
            }
            let parsed = match serde_json::from_str::<Value>(payload) {
                Ok(value) => value,
                Err(e) => {
                    tracing::debug!("skipping malformed SSE line: {} ({e})", crate::util::summarize_for_log(payload, "<empty payload>"));
                    continue;
                }
            };
            if looks_like_error_envelope(&parsed) {
                let message = parsed
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("upstream error");
                let event = StreamEvent::Error {
                    error: serde_json::json!({
                        "type": "upstream_error",
                        "message": message,
                    }),
                };
                self.output_buffer.push_back(Self::encode(&event));
                self.translator.take();
                self.finished = true;
                return;
            }
            match serde_json::from_value::<crate::openai::ChatChunk>(parsed) {
                Ok(c) => {
                    if c.extra.get("x-opencode-type").is_some() {
                        tracing::trace!(
                            extra = %c.extra,
                            "absorbing upstream metadata line (not a ChatChunk)"
                        );
                    }
                    if let Some(t) = self.translator.as_mut() {
                        for ev in t.push_chunk(&c) {
                            self.output_buffer.push_back(Self::encode(&ev));
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("skipping malformed SSE line: {} ({e})", crate::util::summarize_for_log(payload, "<empty payload>"));
                }
            }
        }
    }
}

impl<S> Stream for OpenAiSseToAnthropic<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(b) = self.output_buffer.pop_front() {
            return Poll::Ready(Some(Ok(b)));
        }
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.pending.extend_from_slice(&chunk);
                    self.process_lines();
                    if let Some(b) = self.output_buffer.pop_front() {
                        return Poll::Ready(Some(Ok(b)));
                    }
                    if self.finished {
                        return Poll::Ready(None);
                    }
                    // No events yet — keep reading.
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(ProxyError::Http(e))));
                }
                Poll::Ready(None) => {
                    // EOF: close translator if not already.
                    if let Some(mut t) = self.translator.take() {
                        for ev in t.finalize() {
                            self.output_buffer.push_back(Self::encode(&ev));
                        }
                    }
                    self.finished = true;
                    if let Some(b) = self.output_buffer.pop_front() {
                        return Poll::Ready(Some(Ok(b)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn event_name(ev: &StreamEvent) -> &'static str {
    match ev {
        StreamEvent::MessageStart { .. } => "message_start",
        StreamEvent::ContentBlockStart { .. } => "content_block_start",
        StreamEvent::Ping => "ping",
        StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        StreamEvent::MessageDelta { .. } => "message_delta",
        StreamEvent::MessageStop => "message_stop",
        StreamEvent::Error { .. } => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;
    use crate::openai::{ChatTool, FunctionCall, FunctionDef, ToolCall, UserContent};
    use futures_util::{stream, StreamExt};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Wire-level matcher + wire fixtures shared with openai_responses
    /// (PR-10 consolidated them into crate::test_support).
    use crate::test_support::{
        cache_request_with, chat_response, openai_request as request, JsonFieldAbsent,
    };

    /// The user-reported upstream 400: OpenAI-style wording on the
    /// /chat/completions path (opencode Zen → deepseek in thinking mode).
    const REASONING_ECHO_400: &str = r#"{"error":{"message":"Upstream request failed: [invalid_request_error] The reasoning_content in the thinking mode must be passed back to the API"}}"#;

    /// Multi-turn request with thinking enabled. After conversion the wire
    /// carries `reasoning_effort: "medium"` (budget 2000) and an assistant
    /// message with `reasoning_content: "let me think"` — the exact
    /// situation DeepSeek/opencode reject when the echo is missing.
    fn thinking_request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
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

    #[tokio::test]
    async fn complete_sends_rewritten_request_and_converts_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(json!({
                "model": "runtime-model",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let mut configured_rewrite = HashMap::new();
        configured_rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "configured-model".to_string(),
        );
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            configured_rewrite,
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();
        let mut runtime_rewrite = HashMap::new();
        runtime_rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "runtime-model".to_string(),
        );

        let output = provider.complete(&request(false), &runtime_rewrite).await.unwrap();

        assert_eq!(provider.name(), "test");
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["type"], "message");
            assert_eq!(body["model"], "claude-sonnet-4-20250514");
            assert_eq!(body["content"][0]["text"], "world");
            assert_eq!(body["stop_reason"], "end_turn");
            assert_eq!(body["usage"]["input_tokens"], 3);
            assert_eq!(body["usage"]["output_tokens"], 2);
        });
    }

    #[tokio::test]
    async fn complete_preserves_upstream_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("request should fail");

        expect_variant!(error, ProxyError::Upstream { status, body } => {
            assert_eq!(status, 429);
            assert_eq!(body, "rate limited");
        });
    }

    #[tokio::test]
    async fn complete_surfaces_error_envelope_on_http_200() {
        // Some OpenAI-compatible upstreams return HTTP 200 with an error
        // envelope (e.g. DeepSeek for an unknown model). Without this
        // detection, ChatResponse deserialization fails with
        // "missing field `object`" and the proxy returns a generic 500
        // that hides the real upstream message.
        let server = MockServer::start().await;
        let envelope = json!({
            "error": {
                "message": "Model Not Exist",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&envelope))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("error envelope should surface as Err");

        expect_variant!(error, ProxyError::Upstream { status, body } => {
            assert_eq!(status, 400);
            assert!(body.contains("Model Not Exist"), "body was: {body}");
            assert!(body.contains("model_not_found"), "body was: {body}");
        });
    }

    #[tokio::test]
    async fn stream_converts_openai_sse() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "model": "stream-model",
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();
        let mut rewrite = HashMap::new();
        rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "stream-model".to_string(),
        );

        let output = provider.stream(&request(true), &rewrite).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut encoded = String::new();
            while let Some(item) = output.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }

            assert!(encoded.contains("event: message_start"));
            assert!(encoded.contains("event: content_block_delta"));
            assert!(encoded.contains("\"text\":\"hello\""));
            assert!(encoded.contains("\"input_tokens\":4"));
            assert!(encoded.contains("event: message_stop"));
        });
    }

    #[tokio::test]
    async fn stream_maps_text_then_first_tool_to_distinct_anthropic_blocks() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"before\"},\"finish_reason\":null}]}\n\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"EnterWorktree\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/x\\\"}\"}}]},\"finish_reason\":null}]}\n\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut events = Vec::new();

        while let Some(item) = adapter.next().await {
            let encoded = String::from_utf8(item.unwrap().to_vec()).unwrap();
            let data = encoded
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .expect("encoded SSE event must contain data");
            events.push(serde_json::from_str::<Value>(data).unwrap());
        }

        let lifecycle: Vec<(&str, u64)> = events
            .iter()
            .filter_map(|event| match event["type"].as_str()? {
                "content_block_start" => Some(("start", event["index"].as_u64()?)),
                "content_block_stop" => Some(("stop", event["index"].as_u64()?)),
                _ => None,
            })
            .collect();
        assert_eq!(
            lifecycle,
            vec![("start", 0), ("stop", 0), ("start", 1), ("stop", 1)]
        );

        let tool_delta = events
            .iter()
            .find(|event| event["delta"]["type"] == "input_json_delta")
            .expect("tool arguments delta must be emitted");
        assert_eq!(tool_delta["index"], 1);
        assert_eq!(tool_delta["delta"]["partial_json"], "{\"path\":\"/tmp/x\"}");
    }

    #[tokio::test]
    async fn stream_maps_text_then_tool_wiremock_full_path() {
        // Wiremock-based end-to-end test: SSE stream with text followed by
        // tool call at tc.index=0 must produce distinct Anthropic blocks
        // (text at 0, tool_use at 1). This exercises the real
        // OpenAiCompatProvider::stream() HTTP -> SSE -> translation path.
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"before\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"Edit\",\"arguments\":\"{\\\"file_path\\\":\\\"/tmp/x\\\",\\\"old_string\\\":\\\"a\\\",\\\"new_string\\\":\\\"b\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5,\"total_tokens\":9}}\n\n",
            "data: [DONE]\n\n"
        );
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let mut rewrite = HashMap::new();
        rewrite.insert("claude-sonnet-4-20250514".to_string(), "m".to_string());
        let output = provider.stream(&request(true), &rewrite).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut events = Vec::new();
            while let Some(item) = output.next().await {
                let encoded = String::from_utf8(item.unwrap().to_vec()).unwrap();
                let data = encoded
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .expect("encoded SSE event must contain data");
                events.push(serde_json::from_str::<Value>(data).unwrap());
            }

            let lifecycle: Vec<(&str, u64)> = events
                .iter()
                .filter_map(|event| match event["type"].as_str()? {
                    "content_block_start" => Some(("start", event["index"].as_u64()?)),
                    "content_block_stop" => Some(("stop", event["index"].as_u64()?)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                lifecycle,
                vec![("start", 0), ("stop", 0), ("start", 1), ("stop", 1)],
                "block start/stop indices must be [0 text, 1 tool_use], got {lifecycle:?}"
            );

            let tool_delta = events
                .iter()
                .find(|event| event["delta"]["type"] == "input_json_delta")
                .expect("tool arguments delta must be emitted");
            assert_eq!(tool_delta["index"], 1);
            assert_eq!(
                tool_delta["delta"]["partial_json"],
                "{\"file_path\":\"/tmp/x\",\"old_string\":\"a\",\"new_string\":\"b\"}"
            );

            let msg_delta = events
                .iter()
                .find(|event| event["type"] == "message_delta")
                .expect("message_delta must be emitted");
            assert_eq!(msg_delta["delta"]["stop_reason"], "tool_use");
            assert_eq!(msg_delta["delta"]["stop_sequence"], serde_json::Value::Null);
        });
    }

    #[tokio::test]
    async fn stream_preserves_upstream_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .stream(&request(true), &HashMap::new())
            .await
            .err()
            .expect("request should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 503, ref body } if body == "unavailable"
        ));
    }

    #[tokio::test]
    async fn adapter_handles_fragmented_lines_malformed_data_and_eof() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(b"event: ignored\ndata: not-json\ndata: {\"id\":\"c\",")),
            Ok(Bytes::from_static(b"\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n")),
        ];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("message_start"));
        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("message_delta"));
        assert!(encoded.contains("message_stop"));
    }

    #[tokio::test]
    async fn adapter_skips_empty_data_lines_and_comment_lines() {
        // Empty `data:` payloads and `:` comment lines must be ignored without
        // producing any events.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: \n: this is a comment\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("\"text\":\"hi\""));
        // No event should be emitted for empty payloads or comments.
        assert!(!encoded.contains("data: \n\n"));
    }

    #[tokio::test]
    async fn adapter_surfaces_error_envelope_in_successful_sse_response() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: {\"error\":{\"message\":\"model rejected request\",\"type\":\"invalid_request_error\"}}\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("event: error"));
        assert!(encoded.contains("model rejected request"));
        assert!(!encoded.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn adapter_surfaces_inner_stream_errors() {
        // A reqwest error from the inner stream is wrapped in ProxyError::Http.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(
                b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            )),
            Err(reqwest::Error::from(
                reqwest::Client::new()
                    .get("http://[invalid")
                    .build()
                    .unwrap_err(),
            )),
        ];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");

        let mut items = Vec::new();
        while let Some(item) = adapter.next().await {
            items.push(item);
        }

        // First event payload decodes successfully, then the error surfaces.
        assert!(items[0].is_ok());
        let err = items
            .iter()
            .find(|i| i.is_err())
            .expect("expected a stream error");
        assert!(matches!(err, Err(ProxyError::Http(_))));
    }

    #[tokio::test]
    async fn adapter_finalizes_on_eof_when_no_done_marker() {
        // If the upstream stream closes without sending `data: [DONE]`,
        // the adapter still flushes the translator's pending events.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final\"},\"finish_reason\":null}]}\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("\"text\":\"final\""));
        assert!(encoded.contains("message_stop"));
    }

    #[test]
    fn event_names_cover_all_variants() {
        let message = crate::anthropic::MessagesResponse {
            id: "m".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: "model".to_string(),
            stop_reason: None,
            stop_sequence: None,
            stop_details: None,
            container: None,
            usage: Default::default(),
            extra: Default::default(),
        };
        let events = [
            StreamEvent::MessageStart { message },
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: crate::anthropic::ResponseBlock::Text {
                    text: String::new(),
                    citations: None,
                },
            },
            StreamEvent::Ping,
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: crate::anthropic::BlockDelta::TextDelta {
                    text: "x".to_string(),
                },
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageDelta {
                delta: crate::anthropic::MessageDeltaPayload {
                    stop_reason: None,
                    stop_sequence: None,
                    stop_details: None,
                    container: None,
                },
                usage: None,
            },
            StreamEvent::MessageStop,
            StreamEvent::Error { error: json!({}) },
        ];

        assert_eq!(
            events.iter().map(event_name).collect::<Vec<_>>(),
            vec![
                "message_start",
                "content_block_start",
                "ping",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
                "error",
            ]
        );
    }

    #[tokio::test]
    async fn adapter_returns_pending_when_inner_is_pending() {
        // When the inner stream returns Pending, poll_next must also return
        // Pending without flipping finished. Using a noop waker makes this
        // deterministic.
        use futures_util::stream;
        let mut adapter = OpenAiSseToAnthropic::new(
            stream::pending::<reqwest::Result<Bytes>>(),
            "model",
        );
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = std::pin::Pin::new(&mut adapter).poll_next(&mut cx);
        assert!(
            matches!(poll, std::task::Poll::Pending),
            "expected Poll::Pending from a pending inner stream"
        );
    }

    fn provider_with_rewrite(rewrite: HashMap<String, String>) -> OpenAiCompatProvider {
        OpenAiCompatProvider::new(
            "p".to_string(),
            "https://x/v1/".to_string(),
            "k".to_string(),
            rewrite,
            Vec::new(),
            false,
            reqwest::Client::new(),
        )
        .unwrap()
    }

    #[test]
    fn can_serve_model_accepts_anything_when_rewrite_is_empty() {
        // An empty model_rewrite means "this provider exposes its own
        // model catalog; pass names through verbatim" — see fix-R11.
        let p = provider_with_rewrite(HashMap::new());
        assert!(p.can_serve_model("any-random-name"));
        assert!(p.can_serve_model("claude-sonnet-4.5"));
        assert!(p.can_serve_model(""));
    }

    #[test]
    fn can_serve_model_matches_keys_when_rewrite_is_non_empty() {
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-haiku-4.6".to_string(), "deepseek-v4-flash".to_string());
        rewrite.insert("claude-sonnet-4.6".to_string(), "deepseek-v4-flash".to_string());
        let p = provider_with_rewrite(rewrite);

        // Mapped names are accepted.
        assert!(p.can_serve_model("claude-haiku-4.6"));
        assert!(p.can_serve_model("claude-sonnet-4.6"));

        // Unmapped names are rejected — forwarding them would surface
        // as a misleading 400 from upstream and break the fallback chain.
        assert!(!p.can_serve_model("claude-sonnet-4.5"));
        assert!(!p.can_serve_model("gpt-4o"));
        assert!(!p.can_serve_model(""));
    }

    #[test]
    fn merged_rewrite_combines_configured_and_runtime_maps() {
        let mut configured = HashMap::new();
        configured.insert("claude-a".to_string(), "configured-model".to_string());
        configured.insert("claude-c".to_string(), "configured-only-model".to_string());
        let p = provider_with_rewrite(configured);

        let mut runtime = HashMap::new();
        runtime.insert("claude-a".to_string(), "runtime-model".to_string());
        runtime.insert("claude-b".to_string(), "runtime-b".to_string());

        let merged = p.merged_rewrite(&runtime);
        // runtime wins on key collision; configured-only entries survive.
        assert_eq!(merged.get("claude-a").map(String::as_str), Some("runtime-model"));
        assert_eq!(merged.get("claude-b").map(String::as_str), Some("runtime-b"));
        assert_eq!(
            merged.get("claude-c").map(String::as_str),
            Some("configured-only-model")
        );
        assert_eq!(merged.len(), 3);
    }

    #[tokio::test]
    async fn complete_emits_prompt_cache_key_and_in_memory_when_cache_control_ephemeral() {
        // Anthropic cache_control.ephemeral + metadata.user_id → wire
        // must carry prompt_cache_key=user_id and
        // prompt_cache_retention=in_memory. The whole point: the
        // client-side cache hint reaches the upstream verbatim (after
        // the type mapping) so OpenAI actually applies caching.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "prompt_cache_key": "u-42",
                "prompt_cache_retention": "in_memory"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral", Some("u-42")), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_emits_24h_retention_when_cache_control_ephemeral_1h() {
        // cache_control.ephemeral_1h → prompt_cache_retention="24h"
        // on the wire. Longest TTL tier both APIs offer (Anthropic
        // charges the 1h tier; OpenAI's nearest equivalent is 24h).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "prompt_cache_key": "u-9",
                "prompt_cache_retention": "24h"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral_1h", Some("u-9")), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_omits_prompt_cache_key_when_cache_control_without_user_id() {
        // cache_control present + no metadata.user_id → wire must
        // emit retention (client wants caching) but NOT
        // prompt_cache_key (no namespace to scope to; emitting an
        // empty key would lump unrelated requests into one cache
        // bucket).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "prompt_cache_retention": "in_memory"
            })))
            .and(JsonFieldAbsent("prompt_cache_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral", None), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_omits_cache_fields_when_request_has_no_cache_control() {
        // Default Anthropic request (no cache_control, with or
        // without metadata.user_id) → wire body must NOT carry
        // prompt_cache_key / prompt_cache_retention. The proxy must
        // not pollute requests with cache hints when the client
        // didn't ask — caching is opt-in on the client side and
        // affects billing.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(JsonFieldAbsent("prompt_cache_key"))
            .and(JsonFieldAbsent("prompt_cache_retention"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    /// T19: P1-4 — the openai_compat SSE malformed-line debug log uses
    /// `crate::util::summarize_for_log` so an HTML error page (e.g.
    /// Copilot 502) does not dump a full document into the log output.
    #[test]
    fn openai_compat_malformed_sse_payload_is_summarized_in_debug_log() {
        // We can't easily capture tracing output in a unit test, so
        // verify the function is wired up correctly by calling it with
        // HTML that the summarizer must strip.
        let html = "<html><head><style>body{color:red}</style></head><body>rate limited</body></html>";
        let result = crate::util::summarize_for_log(html, "<empty payload>");
        assert!(!result.contains("style"), "CSS stripped: {result}");
        assert!(!result.contains("<html"), "HTML stripped: {result}");
        assert!(result.contains("rate limited"), "text preserved: {result}");

        // Also check the empty case returns the placeholder.
        let empty = crate::util::summarize_for_log("", "<empty payload>");
        assert_eq!(empty, "<empty payload>");
    }

    #[tokio::test]
    async fn adapter_absorbs_opencode_metadata_without_events_or_logs() {
        // SSE stream containing: 1 content chunk + metadata line + [DONE]
        // Output should contain normal events but NOT extra events from metadata.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\ndata:{\"choices\":[],\"x-opencode-type\":\"inference-cost\",\"cost\":\"0.00\"}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("message_start"));
        assert!(encoded.contains("\"text\":\"hello\""));
        assert!(encoded.contains("message_stop"));
        // Count message_start — should be exactly 1 (metadata line must not trigger a second start)
        assert_eq!(encoded.matches("event: message_start").count(), 1, "metadata line should not emit extra message_start");
    }

    #[tokio::test]
    async fn metadata_line_does_not_disturb_standard_usage() {
        // Content chunk + standard usage chunk + metadata line + [DONE]
        // MessageDelta usage should reflect the standard usage chunk, not metadata.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\ndata:{\"choices\":[],\"x-opencode-type\":\"inference-cost\"}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        // Verify usage from the standard chunk is reflected
        assert!(encoded.contains("\"input_tokens\":4"));
        assert!(encoded.contains("\"output_tokens\":2"));
    }

    #[tokio::test]
    async fn metadata_line_before_content_still_yields_valid_stream() {
        // Metadata line comes BEFORE the content chunk.
        // Stream should still produce a valid message_start → ... → message_stop.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"choices\":[],\"x-opencode-type\":\"inference-cost\"}\n\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"late\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("message_start"));
        assert!(encoded.contains("\"text\":\"late\""));
        assert!(encoded.contains("message_stop"));
        // Exactly one message_start
        assert_eq!(encoded.matches("event: message_start").count(), 1);
    }

    #[tokio::test]
    async fn list_models_returns_normalized_entries_on_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "model-a", "display_name": "Model A", "owned_by": "org1", "created": 1000},
                    {"id": "model-b", "name": "Model B", "created": 2000},
                    {"id": "model-c"}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let models = provider.list_models().await;
        let models = models.expect("expected Some(_)");
        assert_eq!(models.len(), 3);

        assert_eq!(models[0]["id"], "model-a");
        assert_eq!(models[0]["display_name"], "Model A");
        assert_eq!(models[0]["owned_by"], "org1");
        assert_eq!(models[0]["created"], 1000);

        assert_eq!(models[1]["id"], "model-b");
        assert_eq!(models[1]["display_name"], "Model B");
        assert_eq!(models[1]["owned_by"], "openai_compat");

        assert_eq!(models[2]["id"], "model-c");
        assert_eq!(models[2]["display_name"], "model-c");
        assert_eq!(models[2]["owned_by"], "openai_compat");
    }

    #[tokio::test]
    async fn list_models_returns_none_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_on_network_error() {
        let server = MockServer::start().await;
        let uri = server.uri();
        drop(server);

        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            uri,
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_on_malformed_json() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_when_data_field_is_missing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "test".to_string(),
            server.uri(),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    // ─── A group: has_reasoning_echo_error ────────────────────────────────

    #[test]
    fn has_reasoning_echo_error_matches_reasoning_content_wording() {
        // The user-reported body (exact wording) must match.
        assert!(has_reasoning_echo_error(REASONING_ECHO_400));
        // Bare literal, no JSON envelope — pins that matching is
        // substring-based, not envelope-dependent.
        assert!(has_reasoning_echo_error(
            "The reasoning_content in the thinking mode must be passed back to the API."
        ));
        // Neither substring alone matches; bare 400 / model-not-found /
        // rate-limited / empty bodies never trigger a strip.
        assert!(!has_reasoning_echo_error("400 Bad Request"));
        assert!(!has_reasoning_echo_error(r#"{"error":{"message":"model not found"}}"#));
        assert!(!has_reasoning_echo_error("rate limited"));
        assert!(!has_reasoning_echo_error(""));
        assert!(!has_reasoning_echo_error("must be passed back"));
        assert!(!has_reasoning_echo_error("reasoning_content"));
    }

    // ─── B group: strip_reasoning_echo ────────────────────────────────────

    #[test]
    fn strip_reasoning_echo_clears_reasoning_signals_only() {
        // ChatRequest has no Deserialize/PartialEq, so construct it with a
        // struct literal and assert field-by-field after the strip.
        let mut req = ChatRequest {
            model: "m".to_string(),
            messages: vec![
                ChatMessage::User {
                    content: UserContent::Text("hello".to_string()),
                    name: None,
                },
                ChatMessage::Assistant {
                    content: Some("previous answer".to_string()),
                    tool_calls: Some(vec![ToolCall {
                        id: "t1".to_string(),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: "f".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                    reasoning_content: Some("let me think".to_string()),
                },
            ],
            max_tokens: Some(64),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: false,
            stream_options: None,
            tools: Some(vec![ChatTool {
                kind: "function".to_string(),
                function: FunctionDef {
                    name: "f".to_string(),
                    description: String::new(),
                    parameters: json!({"type": "object"}),
                },
            }]),
            tool_choice: None,
            user: None,
            reasoning_effort: Some("medium".to_string()),
            prompt_cache_key: None,
            prompt_cache_retention: None,
            service_tier: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            verbosity: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prediction: None,
            metadata: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            extra: json!({"thinking": {"type": "enabled"}}),
        };

        strip_reasoning_echo(&mut req);

        // Every assistant message's reasoning_content is cleared.
        for m in &req.messages {
            if let ChatMessage::Assistant { reasoning_content, .. } = m {
                assert!(reasoning_content.is_none());
            }
        }
        // Top-level reasoning_effort cleared.
        assert!(req.reasoning_effort.is_none());
        // extra no longer carries a `thinking` key.
        assert!(req.extra.as_object().unwrap().get("thinking").is_none());
        // Everything else is untouched (field-by-field; ChatRequest has no
        // PartialEq so we cannot assert_eq! the whole object).
        assert!(matches!(
            &req.messages[0],
            ChatMessage::User { content: UserContent::Text(t), .. } if t == "hello"
        ));
        let (content, tool_calls, reasoning_content) = match &req.messages[1] {
            ChatMessage::Assistant { content, tool_calls, reasoning_content } => {
                (content, tool_calls, reasoning_content)
            }
            other => panic!("expected Assistant, got {other:?}"),
        };
        assert_eq!(content.as_deref(), Some("previous answer"));
        assert_eq!(tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(tool_calls.as_ref().unwrap()[0].function.name, "f");
        assert!(reasoning_content.is_none());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.model, "m");
        assert!(!req.stream);
    }

    #[test]
    fn strip_reasoning_echo_is_noop_without_reasoning_signals() {
        // A plain request converted from `request(false)` has no reasoning
        // signals — stripping must leave it untouched (a pure no-op).
        let mut req = anthropic_to_openai_request(&request(false), &HashMap::new(), false)
            .expect("conversion must succeed");
        strip_reasoning_echo(&mut req);
        assert!(req.reasoning_effort.is_none());
        assert!(req.extra.as_object().unwrap().is_empty());
        assert_eq!(req.messages.len(), 1);
        assert!(matches!(
            &req.messages[0],
            ChatMessage::User { content: UserContent::Text(t), .. } if t == "hello"
        ));
    }

    #[test]
    fn strip_reasoning_echo_handles_null_extra_without_panicking() {
        // `extra: Value::Null` → `as_object_mut()` returns None and the
        // thinking-key removal is skipped; the message loop still clears
        // reasoning_content. Must not panic.
        let mut req = ChatRequest {
            model: "m".to_string(),
            messages: vec![
                ChatMessage::User {
                    content: UserContent::Text("hi".to_string()),
                    name: None,
                },
                ChatMessage::Assistant {
                    content: Some("answer".to_string()),
                    tool_calls: None,
                    reasoning_content: Some("let me think".to_string()),
                },
            ],
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
            reasoning_effort: Some("medium".to_string()),
            prompt_cache_key: None,
            prompt_cache_retention: None,
            service_tier: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            verbosity: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prediction: None,
            metadata: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            extra: Value::Null,
        };

        strip_reasoning_echo(&mut req);

        assert!(req.extra.is_null());
        assert!(req.reasoning_effort.is_none());
        for m in &req.messages {
            if let ChatMessage::Assistant { reasoning_content, .. } = m {
                assert!(reasoning_content.is_none());
            }
        }
    }

    // ─── C group: complete() strip-and-retry ──────────────────────────────

    #[tokio::test]
    async fn complete_reasoning_echo_error_strips_and_retries() {
        // C1: first 400 (reasoning_content wording) + second 200 → the proxy
        // strips the reasoning signals from the SAME request and retries once
        // (wiremock expect(2)). The retried body must keep the full message
        // structure (3 messages, assistant content == "previous answer", user
        // messages intact) and drop ONLY the reasoning signals — no
        // reasoning_content / reasoning_effort / thinking — so a "strip
        // everything / strip nothing" false green is impossible.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_string(REASONING_ECHO_400)
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(chat_response())
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&thinking_request(false), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "world");
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Retried body: only reasoning signals stripped, structure intact.
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        let messages = sent["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3, "must not drop/merge messages: {sent}");
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "previous answer");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "continue");
        for m in messages {
            assert!(
                m.get("reasoning_content").is_none(),
                "assistant reasoning_content must be gone after strip: {m}"
            );
        }
        assert!(
            sent.as_object().unwrap().get("reasoning_effort").is_none(),
            "top-level reasoning_effort must be gone after strip: {sent}"
        );
        assert!(
            sent.as_object().unwrap().get("thinking").is_none(),
            "top-level thinking must be gone after strip: {sent}"
        );
    }

    #[tokio::test]
    async fn complete_reasoning_echo_error_surfaces_second_attempt_body() {
        // C2: both attempts return the reasoning-echo 400. The proxy strips
        // and retries exactly once (expect(2)), then surfaces the SECOND
        // attempt's raw body verbatim — not the first, not a friendly
        // envelope. The retried body is captured and asserted with the same
        // "only reasoning signals stripped" checks as C1.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The reasoning_content in the thinking mode must be passed back to the API", "attempt": 1}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The reasoning_content in the thinking mode must be passed back to the API", "attempt": 2}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &HashMap::new())
            .await
            .err()
            .expect("second 400 must surface as Err");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["error"]["attempt"], 2,
            "must surface the SECOND attempt's body, got: {body}"
        );

        let sent = captured.lock().unwrap().clone().expect("second body captured");
        let messages = sent["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3, "must not drop/merge messages: {sent}");
        assert_eq!(messages[0]["content"], "hello");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "previous answer");
        assert_eq!(messages[2]["content"], "continue");
        for m in messages {
            assert!(m.get("reasoning_content").is_none());
        }
        assert!(sent.as_object().unwrap().get("reasoning_effort").is_none());
        assert!(sent.as_object().unwrap().get("thinking").is_none());
    }

    #[tokio::test]
    async fn complete_reasoning_echo_error_does_not_strip_non_reasoning_400() {
        // C3: a non-reasoning 400 ("model not found") on a request that DOES
        // carry reasoning_effort must NOT trigger strip-and-retry — the
        // two-substring gate rejects it — so the upstream is hit exactly once
        // (expect(1)) and the raw body passes through verbatim.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "model not found"}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &HashMap::new())
            .await
            .err()
            .expect("should fail");
        assert!(matches!(
            err,
            ProxyError::Upstream { status: 400, ref body } if body.contains("model not found")
        ));
    }

    #[tokio::test]
    async fn complete_success_path_with_reasoning_effort_unchanged() {
        // C4: a plain 200 success on a thinking request keeps the existing
        // behavior — exactly one request, reasoning_effort present on the
        // wire, normal response conversion. Strip-and-retry must not fire on
        // a success.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "reasoning_effort": "medium",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&thinking_request(false), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["model"], "claude-model");
            assert_eq!(body["content"][0]["text"], "world");
        });
    }

    // ─── D group: stream() friendly error ─────────────────────────────────

    #[tokio::test]
    async fn stream_reasoning_echo_error_returns_friendly_envelope() {
        // D: streaming must NOT retry on the reasoning-echo 400 (expect(1)
        // pins the single request) — once SSE flows, retrying would
        // double-emit. Instead it returns the friendly
        // `reasoning_content_not_passed_back` envelope with
        // provider/client_model/upstream_model/upstream_body. upstream_model
        // is the WIRE name after rewrite (openai_req.model), what the
        // operator looks up in the upstream dashboard.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string(REASONING_ECHO_400))
            .expect(1)
            .mount(&server)
            .await;

        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-claude".to_string());
        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            rewrite,
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .stream(&thinking_request(true), &HashMap::new())
            .await
            .err()
            .expect("reasoning-echo must surface as Err");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["error"]["type"], "reasoning_content_not_passed_back");
        assert_eq!(parsed["error"]["provider"], "p");
        assert_eq!(parsed["error"]["client_model"], "claude-model");
        assert_eq!(parsed["error"]["upstream_model"], "rewritten-claude");
        let message = parsed["error"]["message"].as_str().unwrap();
        assert!(message.contains("reasoning_content"));
        assert!(message.contains("p"));
        assert!(message.contains("rewritten-claude"));
    }

    // ─── E group: has_response_format_error ────────────────────────────────

    /// The user-reported upstream body for the DeepSeek/opencode_zen
    /// json_schema rejection. Wording copied verbatim from the bug
    /// report — must match `has_response_format_error` so the proxy
    /// downgrades the SAME request and retries once.
    const RESPONSE_FORMAT_400: &str = r#"{"error":{"message":"Upstream request failed: [invalid_request_error] This response_format type is unavailable now"}}"#;

    #[test]
    fn has_response_format_error_matches_user_reported_wording() {
        // The user-reported body (exact wording) must match.
        assert!(has_response_format_error(RESPONSE_FORMAT_400));
        // Bare literal, no JSON envelope — pins that matching is
        // substring-based, not envelope-dependent.
        assert!(has_response_format_error(
            "This response_format type is unavailable now"
        ));
        // Neither substring alone matches; bare 400 / model-not-found /
        // rate-limited / empty bodies never trigger a downgrade.
        assert!(!has_response_format_error("400 Bad Request"));
        assert!(!has_response_format_error(r#"{"error":{"message":"model not found"}}"#));
        assert!(!has_response_format_error("rate limited"));
        assert!(!has_response_format_error(""));
        assert!(!has_response_format_error("unavailable"));
        assert!(!has_response_format_error("response_format"));
    }

    // ─── F group: downgrade_response_format ────────────────────────────────

    #[test]
    fn downgrade_response_format_replaces_json_schema_with_json_object() {
        // The full DeepSeek-stop-hook shape: response_format carries
        // {type:"json_schema", json_schema:{name, strict, schema}}. The
        // downgrade must collapse it to {"type":"json_object"} (dropping
        // name/strict/schema) while preserving other extra keys (e.g.
        // web_search_options) — a "drop the whole request" false green
        // is impossible because the sibling key survives.
        let mut req = ChatRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: UserContent::Text("hi".to_string()),
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
            service_tier: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            verbosity: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prediction: None,
            metadata: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            extra: json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "my_schema",
                        "strict": true,
                        "schema": {"type": "object", "properties": {"q": {"type": "string"}}}
                    }
                },
                "web_search_options": {},
            }),
        };

        let downgraded = downgrade_response_format(&mut req);
        assert!(downgraded, "a json_schema response_format must trigger downgrade");

        let obj = req.extra.as_object().unwrap();
        assert_eq!(
            obj.get("response_format"),
            Some(&json!({"type": "json_object"})),
            "response_format must be replaced with plain json_object"
        );
        assert_eq!(
            obj.get("web_search_options"),
            Some(&json!({})),
            "sibling extra keys must be preserved"
        );
    }

    #[test]
    fn downgrade_response_format_noop_cases() {
        // json_object — already the target shape, nothing to do.
        let mut req = ChatRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: UserContent::Text("hi".to_string()),
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
            service_tier: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            verbosity: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prediction: None,
            metadata: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            extra: json!({"response_format": {"type": "json_object"}}),
        };
        assert!(!downgrade_response_format(&mut req));
        assert_eq!(req.extra, json!({"response_format": {"type": "json_object"}}));

        // text — must pass through verbatim.
        req.extra = json!({"response_format": {"type": "text"}});
        assert!(!downgrade_response_format(&mut req));
        assert_eq!(req.extra, json!({"response_format": {"type": "text"}}));

        // No response_format key at all — no-op.
        req.extra = json!({"thinking": {"type": "enabled"}});
        assert!(!downgrade_response_format(&mut req));
        assert_eq!(req.extra, json!({"thinking": {"type": "enabled"}}));

        // extra is Value::Null — must not panic.
        req.extra = Value::Null;
        assert!(!downgrade_response_format(&mut req));
        assert!(req.extra.is_null());
    }

    /// Build a MessagesRequest that mirrors the Claude Code stop-hook
    /// shape: thinking history AND `output_config.format: json_schema`.
    /// After conversion this carries both `reasoning_effort` and
    /// `extra.response_format = {type:"json_schema", ...}` — the
    /// cascade test exercises both downgrade paths against the same
    /// request.
    fn thinking_and_format_request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {"q": {"type": "string"}}}
                }
            },
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

    /// Build a MessagesRequest carrying only `output_config.format` (no
    /// thinking) — isolated response_format downgrade path.
    fn format_request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {"q": {"type": "string"}}}
                }
            },
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    // ─── G group: complete() response_format downgrade ─────────────────────

    #[tokio::test]
    async fn complete_response_format_error_downgrades_and_retries() {
        // G1: first 400 (response_format ... unavailable) + second 200 → the
        // proxy downgrades `response_format` from json_schema to json_object
        // and retries once (wiremock expect(2)). The retried body must keep
        // the full message structure and carry the new response_format.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_string(RESPONSE_FORMAT_400)
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(chat_response())
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&format_request(false), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "world");
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Retried body: response_format must be downgraded to json_object;
        // messages and model must be intact.
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["response_format"],
            json!({"type": "json_object"}),
            "downgraded response_format must be json_object: {sent}"
        );
        assert_eq!(sent["model"], "claude-model");
        assert_eq!(sent["stream"], false);
        let messages = sent["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[tokio::test]
    async fn complete_response_format_error_surfaces_second_attempt_body() {
        // G2: both attempts return the response_format 400. The proxy
        // downgrades and retries exactly once (expect(2)), then surfaces
        // the SECOND attempt's body verbatim — not the first.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "This response_format type is unavailable now", "attempt": 1}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "This response_format type is unavailable now", "attempt": 2}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&format_request(false), &HashMap::new())
            .await
            .err()
            .expect("second 400 must surface as Err");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["error"]["attempt"], 2,
            "must surface the SECOND attempt's body, got: {body}"
        );

        // The second body must already carry the downgraded response_format.
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["response_format"],
            json!({"type": "json_object"}),
            "second attempt must carry the downgraded response_format: {sent}"
        );
    }

    #[tokio::test]
    async fn complete_response_format_error_does_not_retry_other_400() {
        // G3: a non-matching 400 (no "response_format" / no "unavailable"
        // substring) must NOT trigger the downstream retry — the
        // two-substring gate rejects it. Wiremock counter == 1, raw body
        // passes through verbatim.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "model not found"}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&format_request(false), &HashMap::new())
            .await
            .err()
            .expect("should fail");
        assert!(matches!(
            err,
            ProxyError::Upstream { status: 400, ref body } if body.contains("model not found")
        ));
    }

    #[tokio::test]
    async fn complete_both_reasoning_echo_and_format_downgrade_cascade() {
        // G4: a request carrying BOTH thinking history AND json_schema
        // format hits a 400 with reasoning_echo wording first, then a
        // 400 with response_format wording, then 200. The retry loop must
        // cascade: strip reasoning → downgrade response_format → succeed.
        // Counter == 3, final body must have no reasoning signals AND
        // response_format == {"type":"json_object"}.
        let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let parsed: Value = serde_json::from_slice(&req.body)
                    .unwrap_or_else(|_| json!({}));
                captured_for_responder.lock().unwrap().push(parsed);
                match n {
                    0 => ResponseTemplate::new(400).set_body_string(REASONING_ECHO_400),
                    1 => ResponseTemplate::new(400).set_body_string(RESPONSE_FORMAT_400),
                    _ => ResponseTemplate::new(200).set_body_json(chat_response()),
                }
            })
            .expect(3)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&thinking_and_format_request(false), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "world");
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);

        let history = captured.lock().unwrap().clone();
        assert_eq!(history.len(), 3, "must capture all 3 attempts");

        // Third body: reasoning signals stripped AND response_format
        // downgraded — both cascades applied.
        let sent = &history[2];
        for m in sent["messages"].as_array().unwrap() {
            assert!(
                m.get("reasoning_content").is_none(),
                "assistant reasoning_content must be gone after strip: {m}"
            );
        }
        assert!(
            sent.as_object().unwrap().get("reasoning_effort").is_none(),
            "top-level reasoning_effort must be gone after strip: {sent}"
        );
        assert!(
            sent.as_object().unwrap().get("thinking").is_none(),
            "top-level thinking must be gone after strip: {sent}"
        );
        assert_eq!(
            sent["response_format"],
            json!({"type": "json_object"}),
            "response_format must be downgraded on third attempt: {sent}"
        );
    }

    // ─── H group: stream() response_format downgrade ──────────────────────

    #[tokio::test]
    async fn stream_response_format_error_downgrades_and_retries() {
        // H1: first 400 (response_format ... unavailable) + second SSE 200
        // → the proxy downgrades and retries once. Counter == 2, the
        // retried body must carry the downgraded response_format, and the
        // stream must convert normally.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_string(RESPONSE_FORMAT_400)
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .stream(&format_request(true), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut stream) => {
            let mut encoded = String::new();
            while let Some(item) = stream.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            assert!(encoded.contains("event: message_start"));
            assert!(encoded.contains("event: content_block_delta"));
            assert!(encoded.contains("\"text\":\"hello\""));
            assert!(encoded.contains("event: message_stop"));
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["response_format"],
            json!({"type": "json_object"}),
            "downgraded response_format must be json_object: {sent}"
        );
        assert_eq!(sent["stream"], true);
    }

    #[tokio::test]
    async fn stream_response_format_error_surfaces_second_attempt_body() {
        // H2: both attempts return the response_format 400. The proxy
        // downgrades and retries exactly once (expect(2)), then surfaces
        // the SECOND attempt's body as Upstream (status 400). Surfaces
        // the raw body verbatim — NOT a friendly envelope (that's the
        // reasoning-echo path).
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "This response_format type is unavailable now", "attempt": 1}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "This response_format type is unavailable now", "attempt": 2}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .stream(&format_request(true), &HashMap::new())
            .await
            .err()
            .expect("second 400 must surface as Err");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["error"]["attempt"], 2,
            "must surface the SECOND attempt's body, got: {body}"
        );

        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["response_format"],
            json!({"type": "json_object"}),
            "second attempt must carry the downgraded response_format: {sent}"
        );
    }

    #[tokio::test]
    async fn stream_response_format_error_does_not_retry_other_400() {
        // H3: a non-matching 400 must NOT trigger the downstream retry
        // (counter == 1). The raw body passes through verbatim.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "model not found"}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),

            false,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .stream(&format_request(true), &HashMap::new())
            .await
            .err()
            .expect("should fail");
        assert!(matches!(
            err,
            ProxyError::Upstream { status: 400, ref body } if body.contains("model not found")
        ));
    }

    // ─── I group: provider_ignore (OpenRouter routing) ────────────────────
    //
    // The tests use `new_for_test` (a `#[cfg(test)]` constructor) to
    // override the production `is_openrouter` host detection so we can
    // assert both the ON and OFF paths of the gate without depending
    // on a real DNS lookup of `openrouter.ai` or a wiremock server
    // bound to that hostname.

    /// `provider_ignore` + `is_openrouter=true` → wire body carries
    /// `provider: {ignore: [...]}`. Mirrors the exact shape OpenRouter's
    /// API reference documents for client-side provider routing.
    #[tokio::test]
    async fn complete_injects_provider_ignore_for_openrouter() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "provider": {"ignore": ["Azure"]}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            vec!["Azure".to_string()],
            false,
            true,  // force_openrouter
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    /// `provider_ignore` + `is_openrouter=false` (non-OpenRouter) →
    /// wire body must NOT carry `provider`. Strict OpenAI-compat
    /// backends like DeepSeek reject unknown top-level fields with
    /// 400; the gate must keep the wire clean.
    #[tokio::test]
    async fn complete_omits_provider_field_on_non_openrouter_api_base() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(JsonFieldAbsent("provider"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "deepseek-proxy".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),
            false,
            false, // force_openrouter
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    /// `provider_ignore` is non-empty but `is_openrouter=false` → wire
    /// body must STILL NOT carry `provider` (the non-empty branch of
    /// the gate is suppressed). This is the load-bearing safety path:
    /// an operator misconfiguring `provider_ignore` on a DeepSeek /
    /// opencode provider must not inject an unknown top-level field
    /// that would 400 the strict upstream. Empty-list case is covered
    /// by `complete_omits_provider_field_on_non_openrouter_api_base`
    /// above; this test exercises the *non-empty* list specifically.
    /// (Plan: "non-empty + non-OpenRouter host → no injection + startup
    /// warn"; we assert the wire-side guarantee directly.)
    #[tokio::test]
    async fn complete_omits_provider_when_nonempty_and_not_openrouter() {
        let server = MockServer::start().await;
        // `provider` is a top-level field, so the existing
        // JsonFieldAbsent matcher IS correct here (unlike
        // `reasoning_content` which lives under `messages[]`).
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(JsonFieldAbsent("provider"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "deepseek-proxy".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            vec!["Azure".to_string()], // non-empty list
            false,
            false, // force_openrouter=false → gate suppresses injection
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    /// Empty `provider_ignore` + `is_openrouter=true` → wire body must
    /// NOT carry a `provider` key (field-omission-is-correctness — no
    /// empty objects or null pollution).
    #[tokio::test]
    async fn complete_omits_provider_field_when_ignore_list_empty() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(JsonFieldAbsent("provider"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            Vec::new(),
            false,
            true,  // force_openrouter (even so, empty list means no injection)
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    /// `provider_ignore` must survive the strip-and-retry loop: when
    /// the upstream first 400s on reasoning-echo wording and then 200s,
    /// the second attempt's body must still carry
    /// `provider: {ignore:[…]}`. (Tests the plan's §6.1 guarantee that
    /// the retry downgrades don't clear `extra` keys.)
    #[tokio::test]
    async fn complete_provider_ignore_survives_reasoning_strip_retry() {
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_string(REASONING_ECHO_400)
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(chat_response())
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            vec!["Azure".to_string()],
            false,
            true,  // force_openrouter
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&thinking_request(false), &HashMap::new())
            .await
            .unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Retried body: provider ignore list survived the
        // reasoning-echo strip (only reasoning signals are removed,
        // not extra keys).
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["provider"]["ignore"],
            json!(["Azure"]),
            "provider.ignore must survive the strip-and-retry, got: {sent}"
        );
        // And reasoning signals really were stripped, as a sanity check.
        assert!(
            sent.as_object().unwrap().get("reasoning_effort").is_none(),
            "reasoning_effort must be gone after strip: {sent}"
        );
    }

    /// Companion to the test above: `provider_ignore` must ALSO
    /// survive the response_format downgrade retry. The downgrade
    /// rewrites `extra.response_format` from json_schema to
    /// json_object, but must not touch `extra.provider` — otherwise
    /// the retry request loses the ignore list and OpenRouter routes
    /// to the excluded backend on the second attempt. (Plan §5.2.)
    #[tokio::test]
    async fn complete_provider_ignore_survives_response_format_downgrade_retry() {
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    // First response: response_format 400 (DeepSeek
                    // wording). The proxy will then run
                    // `downgrade_response_format` and retry.
                    ResponseTemplate::new(400).set_body_string(RESPONSE_FORMAT_400)
                } else {
                    // Second response: 200, body captured for the
                    // assertion below. The retry body must STILL carry
                    // `provider: {ignore:[...]}` — that's the load-
                    // bearing guarantee.
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(chat_response())
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            vec!["Azure".to_string()],
            false,
            true, // force_openrouter
            reqwest::Client::new(),
        )
        .unwrap();

        // Build a request that exercises the response_format downgrade
        // path: `output_config.format = json_schema` is converted to
        // `extra.response_format = {type:"json_schema",...}`.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4.6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            }
        }))
        .unwrap();
        let _ = provider.complete(&req, &HashMap::new()).await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Retried body: provider.ignore survived the downgrade, AND
        // response_format was actually rewritten to json_object (the
        // downgrade path actually ran, otherwise this test is no-op).
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(
            sent["provider"]["ignore"],
            json!(["Azure"]),
            "provider.ignore must survive the response_format downgrade, got: {sent}"
        );
        assert_eq!(
            sent["response_format"]["type"],
            json!("json_object"),
            "response_format must be downgraded to json_object, got: {sent}"
        );
    }

    /// Streaming path must also inject `provider: {ignore: [...]}` onto
    /// the wire when the gate is open.
    #[tokio::test]
    async fn stream_injects_provider_ignore_for_openrouter() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "provider": {"ignore": ["Azure", "Together"]}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new_for_test(
            "p".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            vec!["Azure".to_string(), "Together".to_string()],
            false,
            true,  // force_openrouter
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .stream(&request(true), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut stream) => {
            // Drain the stream so wiremock records the matched
            // request — without consuming, the mock counter wouldn't
            // increment and the assertion below would race.
            while let Some(_) = stream.next().await {}
        });
    }

    /// `is_openrouter` is computed in `new()` by lowercasing the host
    /// string and comparing to `openrouter.ai` / `*.openrouter.ai`.
    /// The detection itself is unit-tested exhaustively in
    /// `src/providers/mod.rs` (see
    /// `is_openrouter_api_base_matches_canonical_subdomains_and_case`).
    /// This test pins the integration: the production `new()` with a
    /// canonical OpenRouter URL must cache `is_openrouter = true`, so a
    /// subsequent `inject_provider_ignore` actually injects.
    #[test]
    fn openrouter_host_detection_covers_subdomains_and_case() {
        let p = OpenAiCompatProvider::new(
            "p".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            "k".to_string(),
            HashMap::new(),
            vec!["Azure".to_string()],
            false,
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(p.name(), "p");
        // The flag is private; the strongest public-surface check is
        // that `inject_provider_ignore` actually mutates the request
        // when the production constructor detected an OpenRouter
        // api_base.
        let mut req = minimal_chat_request();
        p.inject_provider_ignore(&mut req);
        assert_eq!(
            req.extra["provider"]["ignore"],
            json!(["Azure"]),
            "production constructor with an OpenRouter api_base must \
             wire the gate so provider.ignore is injected"
        );
    }

    /// Build a `ChatRequest` carrying only the fields exercised by the
    /// `provider_ignore` tests (everything else is `None` / empty).
    /// Keeping the constructor inline avoids the ~25-field literal
    /// that previously lived in `openrouter_host_detection_covers_subdomains_and_case`.
    fn minimal_chat_request() -> crate::openai::ChatRequest {
        crate::openai::ChatRequest {
            model: "m".to_string(),
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
            service_tier: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            verbosity: None,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prediction: None,
            metadata: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            extra: serde_json::json!({}),
        }
    }

    /// Cross-model assistant history: redacted_thinking (Anthropic
    /// encrypted blob → dropped by convert_blocks) + plain text. With
    /// `reasoning_echo=true` this request would 400 on a DeepSeek/opencode
    /// upstream without the fill; the wiremock tests below pin the
    /// correct wire shape and the strip-retry suppression.
    fn thinking_request_with_redacted(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                    {"type": "text", "text": "previous answer"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap()
    }

    /// reasoning_echo=true + cross-model redacted-think history → wire
    /// carries `reasoning_content: ""` on the assistant message, and the
    /// upstream returns 200 on the FIRST request (no strip-and-retry).
    /// The counter assertion is the load-bearing one: if reasoning_echo
    /// were inactive, the wire would be missing reasoning_content and
    /// upstream would 400, which we deliberately do NOT simulate here.
    #[tokio::test]
    async fn complete_reasoning_echo_on_accepts_on_first_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant",
                     "content": "previous answer",
                     "reasoning_content": ""},
                    {"role": "user", "content": "continue"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "opencode_zen".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),
            true, // reasoning_echo on
            reqwest::Client::new(),
        )
        .unwrap();

        let out = provider
            .complete(&thinking_request_with_redacted(false), &HashMap::new())
            .await
            .expect("reasoning_echo=true must succeed on first POST");
        let _ = out;
    }

    /// reasoning_echo=true + upstream STILL 400s with the reasoning-echo
    /// error wording (e.g. a future upstream changes its validation to
    /// reject `reasoning_content: ""` because it requires non-empty).
    /// The proxy must:
    ///   - NOT retry (reasoning_echo is on, so the strip-and-retry path
    ///     is gated off — `!self.reasoning_echo` short-circuits).
    ///   - NOT wrap in the friendly `reasoning_content_not_passed_back`
    ///     envelope (the same gate suppresses it in `stream()`).
    ///   - Surface the 400 VERBATIM as `ProxyError::Upstream { status,
    ///     body }` so the client sees the real upstream error.
    /// This is the surface-path test that pins the `&& !self.reasoning_echo`
    /// guards on lines ~396 and ~534.
    #[tokio::test]
    async fn complete_reasoning_echo_on_surfaces_400_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string(REASONING_ECHO_400))
            .expect(1) // exactly ONE call: no retry with reasoning_echo on
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "opencode_zen".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),
            true, // reasoning_echo on
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request_with_redacted(false), &HashMap::new())
            .await
            .err()
            .expect("reasoning_echo=true + 400 must surface as error");

        // Must be a verbatim Upstream 400 (NOT the friendly envelope).
        // The friendly envelope uses `error: { type: "reasoning_content_not_passed_back" }`
        // inside the body; we assert the body bytes are the literal
        // upstream response, untouched.
        match err {
            ProxyError::Upstream { status, ref body } => {
                assert_eq!(status, 400);
                assert_eq!(body, REASONING_ECHO_400);
            }
            other => panic!(
                "reasoning_echo=true + 400 must surface as ProxyError::Upstream, got: {other:?}"
            ),
        }
    }

    /// reasoning_echo=false: the existing strip-and-retry behavior for
    /// the `reasoning_content ... must be passed back` 400 must be
    /// unchanged. This pins the OFF path so a future change can't
    /// accidentally suppress the safety net for non-echo providers.
    #[tokio::test]
    async fn complete_reasoning_echo_off_preserves_strip_retry() {
        let server = MockServer::start().await;
        // First request: 400 with the reasoning-echo error envelope,
        // triggered regardless of body shape (the strip path matches
        // only on the error MESSAGE in the body, not on wire shape).
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "The reasoning_content in the thinking mode must be passed back to the API"}
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Retry after strip: reasoning_content must be absent on every
        // assistant message. The `JsonFieldAbsent` matcher only inspects
        // the top-level body object, but `reasoning_content` lives inside
        // `messages[]`; we need a custom matcher that walks the array and
        // confirms no inner message carries the key.
        struct MessagesReasoningContentAbsent;
        impl wiremock::Match for MessagesReasoningContentAbsent {
            fn matches(&self, request: &wiremock::Request) -> bool {
                let Ok(body) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
                    return false;
                };
                !messages.iter().any(|m| m.get("reasoning_content").is_some())
            }
        }
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(MessagesReasoningContentAbsent)
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "deepseek".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),
            false, // reasoning_echo off → strip-and-retry path active
            reqwest::Client::new(),
        )
        .unwrap();

        let out = provider
            .complete(&thinking_request_with_redacted(false), &HashMap::new())
            .await
            .expect("strip-and-retry must recover the request");
        let _ = out;
    }

    /// reasoning_echo=true + streaming: upstream 200 on first request,
    /// no friendly `reasoning_content_not_passed_back` envelope is
    /// ever surfaced. The wire body still carries
    /// `reasoning_content: ""` on the assistant message.
    #[tokio::test]
    async fn stream_reasoning_echo_on_accepts_without_envelope() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "stream": true,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant",
                     "content": "previous answer",
                     "reasoning_content": ""},
                    {"role": "user", "content": "continue"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(
            "opencode_zen".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            HashMap::new(),
            Vec::new(),
            true, // reasoning_echo on
            reqwest::Client::new(),
        )
        .unwrap();

        let out = provider
            .stream(&thinking_request_with_redacted(true), &HashMap::new())
            .await
            .expect("reasoning_echo=true must succeed without envelope");
        // The output must be a Stream (not an error). Wire the stream
        // body just enough to drain it — wiremock recorded the request.
        match out {
            ProviderOutput::Stream(mut s) => {
                use futures_util::StreamExt;
                let _ = s.next().await;
            }
            ProviderOutput::Json(v) => panic!("expected Stream, got JSON: {v}"),
        }
    }
}
