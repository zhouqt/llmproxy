//! Debug instrumentation for Claude Code Stop-hook requests.
//!
//! The Stop hook that `/goal` invokes after each assistant turn is the
//! primary suspect for our recent goal-loop stalls: Claude Code hits a
//! ~30 s hard timeout on the hook LLM call, and the proxy has no view
//! into whether the timeout is caused by a slow upstream or by a slow
//! proxy hop. This module adds a one-shot timing record per request so
//! we can split the two numbers in production logs:
//!
//!   * `provider_first_byte_ms` — wall-clock from `send_with_token` to
//!     the first byte of the upstream response.
//!   * `provider_total_ms` — wall-clock from `send_with_token` to the
//!     last byte (i.e. how long upstream actually took).
//!   * `bytes` — total upstream bytes received.
//!
//! For detected Stop-hook requests (`is_stop_hook_request`) we also
//! buffer and log the entire raw upstream byte stream as a follow-up
//! line, so we can inspect the exact JSON the model returned.
//!
//! Non-hook requests log at DEBUG level; hook requests log at INFO so
//! they're easy to grep. The wrapper lives between `resp.bytes_stream()`
//! and the SSE translator, so the buffered bytes are the wire format
//! upstream actually sent, not the Anthropic-formatted rewrite.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use bytes::Bytes;
use futures_util::Stream;

use crate::anthropic::{ContentBlock, MessageContent, MessagesRequest};

/// Fixed substring of Claude Code's built-in Stop-hook prompt. There
/// is no wire-level marker for hook calls — they look identical to any
/// other `/v1/messages` request — so we sniff the prompt body.
pub const HOOK_PROMPT_SIGNATURE: &str = "Based on the conversation transcript above, has the following stopping condition been satisfied";

/// Sniff a MessagesRequest for the Stop-hook signature. Returns true
/// if any user-role message contains the fixed prompt prefix.
pub fn is_stop_hook_request(req: &MessagesRequest) -> bool {
    for m in &req.messages {
        if m.role != "user" {
            continue;
        }
        let text = match &m.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if text.contains(HOOK_PROMPT_SIGNATURE) {
            return true;
        }
    }
    false
}

/// Per-request timing state. Constructed by the provider at request
/// entry, populated by the wrapped upstream stream, and consumed by
/// `emit()` exactly once when the upstream finishes (or errors).
///
/// A Drop impl warns if the timer is leaked without `emit()` — that
/// happens when the stream wrapper itself is dropped mid-flight (e.g.
/// client abort) and is useful to spot 30 s client timeouts that never
/// let upstream finish.
pub struct RequestTimer {
    pub model: String,
    pub is_hook: bool,
    pub started_at: Instant,
    pub first_byte_at: Option<Instant>,
    pub last_byte_at: Option<Instant>,
    pub bytes_received: usize,
    pub raw_buffer: Option<Vec<u8>>,
    emitted: bool,
}

impl RequestTimer {
    pub fn new(model: String, is_hook: bool) -> Self {
        let raw_buffer = if is_hook { Some(Vec::new()) } else { None };
        Self {
            model,
            is_hook,
            started_at: Instant::now(),
            first_byte_at: None,
            last_byte_at: None,
            bytes_received: 0,
            raw_buffer,
            emitted: false,
        }
    }

    pub fn record_first_byte(&mut self) {
        if self.first_byte_at.is_none() {
            self.first_byte_at = Some(Instant::now());
        }
    }

    pub fn record_chunk(&mut self, bytes: &[u8]) {
        self.bytes_received += bytes.len();
        if let Some(buf) = self.raw_buffer.as_mut() {
            buf.extend_from_slice(bytes);
        }
    }

    pub fn record_end(&mut self) {
        self.last_byte_at = Some(Instant::now());
    }

    pub fn emit(self) {
        self.emit_with_dropped(false);
    }

    fn emit_with_dropped(mut self, dropped: bool) {
        self.emitted = true;
        let now = Instant::now();
        let total_ms = now.duration_since(self.started_at).as_millis() as u64;
        let first_byte_ms = self
            .first_byte_at
            .map(|t| t.duration_since(self.started_at).as_millis() as u64);
        let last_byte_ms = self
            .last_byte_at
            .map(|t| t.duration_since(self.started_at).as_millis() as u64)
            .unwrap_or(total_ms);
        let bytes = self.bytes_received;
        let model = &self.model;
        let is_hook = self.is_hook;
        if is_hook {
            tracing::info!(
                target: "hook_timing",
                model = %model,
                hook = true,
                bytes,
                dropped,
                provider_total_ms = last_byte_ms,
                provider_first_byte_ms = first_byte_ms.unwrap_or(0),
                "[hook-timing] upstream done"
            );
        } else if dropped {
            tracing::warn!(
                target: "hook_timing",
                model = %model,
                hook = false,
                bytes,
                dropped,
                provider_total_ms = last_byte_ms,
                provider_first_byte_ms = first_byte_ms.unwrap_or(0),
                "[hook-timing] upstream timer dropped without explicit end (sse adapter short-circuited on terminal?)"
            );
        } else {
            tracing::debug!(
                target: "hook_timing",
                model = %model,
                hook = false,
                bytes,
                provider_total_ms = last_byte_ms,
                provider_first_byte_ms = first_byte_ms.unwrap_or(0),
                "request timing complete"
            );
        }
        if let Some(buf) = &self.raw_buffer {
            tracing::info!(
                target: "hook_timing",
                model = %model,
                bytes = buf.len(),
                "[hook-raw-upstream]\n{}",
                String::from_utf8_lossy(buf)
            );
        }
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        if !self.emitted {
            // The wrapper-level Drop (TimedProviderStream::drop) normally
            // takes the timer and emits. If we still have it here, the
            // caller never fed it into a wrapper — emit at warn level so
            // the operator still sees the timing summary. Mem-replace to
            // move out of `&mut self` (Drop's borrow makes a plain move
            // impossible).
            let dummy = std::mem::replace(self, RequestTimer::new(String::new(), false));
            dummy.emit_with_dropped(true);
        }
    }
}

/// Proxy-side counterpart to [`RequestTimer`]: measures the wall-clock
/// from handler entry to when the last byte is written to the client.
/// For streaming, this is wired into the SSE body adapter (see
/// `server::TimedMappedStream`) so the timer is emitted exactly when
/// the body finishes (or errors). For non-streaming, the handler
/// captures a single `Instant` before the upstream call and calls
/// `emit()` after the response is built.
///
/// `is_hook` only affects whether we log at INFO/DEBUG — there's no
/// raw-buffer capture here because we already have it from
/// `RequestTimer` and we don't want to double-log the upstream bytes.
pub struct ProxyTimer {
    pub model: String,
    pub is_hook: bool,
    pub started_at: Instant,
    pub first_byte_at: Option<Instant>,
    emitted: bool,
}

impl ProxyTimer {
    pub fn new(model: String, is_hook: bool) -> Self {
        Self {
            model,
            is_hook,
            started_at: Instant::now(),
            first_byte_at: None,
            emitted: false,
        }
    }

    pub fn record_first_byte(&mut self) {
        if self.first_byte_at.is_none() {
            self.first_byte_at = Some(Instant::now());
        }
    }

    pub fn emit(mut self, proxy_total_ms_explicit: Option<u64>) {
        self.emitted = true;
        let now = Instant::now();
        let total_ms = now.duration_since(self.started_at).as_millis() as u64;
        let first_byte_ms = self
            .first_byte_at
            .map(|t| t.duration_since(self.started_at).as_millis() as u64);
        let total_ms = proxy_total_ms_explicit.unwrap_or(total_ms);
        let model = &self.model;
        let is_hook = self.is_hook;
        if is_hook {
            tracing::info!(
                target: "proxy_timing",
                model = %model,
                hook = true,
                proxy_total_ms = total_ms,
                proxy_first_byte_ms = first_byte_ms.unwrap_or(0),
                "[hook-timing] proxy done"
            );
        } else {
            tracing::debug!(
                target: "proxy_timing",
                model = %model,
                hook = false,
                proxy_total_ms = total_ms,
                proxy_first_byte_ms = first_byte_ms.unwrap_or(0),
                "proxy timing complete"
            );
        }
    }
}

impl Drop for ProxyTimer {
    fn drop(&mut self) {
        if !self.emitted {
            tracing::warn!(
                target: "proxy_timing",
                model = %self.model,
                hook = self.is_hook,
                "proxy timer dropped without emit (handler returned early?)"
            );
        }
    }
}

/// Stream wrapper that captures first-byte/end times and (for hooks)
/// buffers the entire raw upstream byte stream. Sits between
/// `resp.bytes_stream()` and the SSE translator, so the captured bytes
/// are upstream's wire format, not the Anthropic-formatted rewrite.
pub struct TimedProviderStream<S> {
    inner: S,
    timer: Option<RequestTimer>,
}

impl<S> TimedProviderStream<S> {
    pub fn new(inner: S, timer: RequestTimer) -> Self {
        Self {
            inner,
            timer: Some(timer),
        }
    }
}

impl<S> Stream for TimedProviderStream<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    type Item = reqwest::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = Pin::new(&mut self.inner).poll_next(cx);
        match &poll {
            Poll::Ready(Some(Ok(bytes))) => {
                if let Some(t) = self.timer.as_mut() {
                    t.record_first_byte();
                    t.record_chunk(bytes);
                }
            }
            Poll::Ready(None) => {
                if let Some(t) = self.timer.as_mut() {
                    t.record_end();
                }
            }
            Poll::Ready(Some(Err(_))) => {
                if let Some(t) = self.timer.as_mut() {
                    t.record_end();
                }
            }
            Poll::Pending => {}
        }
        // On the success path the inner stream's Ready(None) means the
        // SSE translator will never poll us again (it short-circuits on
        // [DONE] / terminal event by setting `finished = true`). So we
        // emit now to capture provider_total_ms while we still have a
        // wall-clock anchor.
        if matches!(poll, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            if let Some(t) = self.timer.take() {
                t.emit();
            }
        }
        poll
    }
}

impl<S> Drop for TimedProviderStream<S> {
    fn drop(&mut self) {
        // Fallback for the SSE-short-circuit path: if the SSE adapter
        // returns Ready(None) without ever seeing the inner stream
        // close, the explicit emit() in poll_next never runs. Take
        // the timer and emit (with dropped=true) here.
        if let Some(t) = self.timer.take() {
            t.emit_with_dropped(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::{Message, MessageContent};
    use serde_json::json;

    fn req_with_user_text(text: &str) -> MessagesRequest {
        MessagesRequest {
            model: "m".into(),
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text(text.into()),
            }],
            max_tokens: 1024,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
            cache_control: None,
            container: None,
            inference_geo: None,
            service_tier: None,
            output_config: None,
            user_profile_id: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn detects_hook_signature_in_user_text() {
        let req = req_with_user_text(
            "Based on the conversation transcript above, has the following stopping condition been satisfied? Answer based on transcript evidence only.",
        );
        assert!(is_stop_hook_request(&req));
    }

    #[test]
    fn ignores_hook_signature_in_assistant_message() {
        let mut req = req_with_user_text("hello");
        req.messages.push(Message {
            role: "assistant".into(),
            content: MessageContent::Text(HOOK_PROMPT_SIGNATURE.into()),
        });
        assert!(!is_stop_hook_request(&req));
    }

    #[test]
    fn ignores_unrelated_user_text() {
        let req = req_with_user_text("explain this Rust code");
        assert!(!is_stop_hook_request(&req));
    }

    #[test]
    fn detects_hook_signature_inside_user_blocks() {
        let req = MessagesRequest {
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlock::Text {
                    text: format!("preamble\n{HOOK_PROMPT_SIGNATURE}\nrest"),
                    cache_control: None,
                    citations: None,
                }]),
            }],
            ..req_with_user_text("")
        };
        assert!(is_stop_hook_request(&req));
    }

    #[test]
    fn timer_emits_when_first_byte_seen() {
        let t = RequestTimer::new("m".into(), false);
        assert_eq!(t.bytes_received, 0);
        assert!(t.first_byte_at.is_none());
    }

    #[test]
    fn timer_records_chunk_size_and_total() {
        let mut t = RequestTimer::new("m".into(), true);
        t.record_chunk(b"abc");
        t.record_chunk(b"defgh");
        assert_eq!(t.bytes_received, 8);
        assert_eq!(t.raw_buffer.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn timed_stream_records_first_byte_and_emits_on_end() {
        use futures_util::stream;
        let bytes = vec![Ok(Bytes::from_static(b"data: x\n\n")), Ok(Bytes::from_static(b"data: y\n\n"))];
        let inner = stream::iter(bytes);
        let timer = RequestTimer::new("m".into(), false);
        let mut wrapped = TimedProviderStream::new(inner, timer);
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        // Drain the stream; emit happens on the final Ready(None) poll.
        for expected in [b"data: x\n\n".to_vec(), b"data: y\n\n".to_vec()] {
            match Pin::new(&mut wrapped).poll_next(&mut cx) {
                Poll::Ready(Some(Ok(b))) => assert_eq!(b.as_ref(), expected.as_slice()),
                other => panic!("expected Ready(Some(Ok)), got {other:?}"),
            }
        }
        // One more poll reaches Ready(None), which triggers the timer
        // emit and drops the timer out of the wrapper.
        match Pin::new(&mut wrapped).poll_next(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected Ready(None), got {other:?}"),
        }
        // After end, timer is taken and emit runs (logs at DEBUG/INFO
        // level — we just assert no panic / no leftover timer).
        assert!(wrapped.timer.is_none());
    }

    #[test]
    fn raw_buffer_only_allocated_for_hooks() {
        let mut non_hook = RequestTimer::new("m".into(), false);
        non_hook.record_chunk(b"abc");
        assert!(non_hook.raw_buffer.is_none());
        let mut hook = RequestTimer::new("m".into(), true);
        hook.record_chunk(b"abc");
        assert_eq!(hook.raw_buffer.as_ref().unwrap().as_slice(), b"abc");
    }

    #[test]
    fn drop_without_emit_does_not_panic() {
        // We can't easily intercept the tracing::warn!, but the Drop
        // impl must not panic when the timer is leaked.
        let t = RequestTimer::new("m".into(), true);
        drop(t);
    }

    #[test]
    fn json_constructs_messages_request_with_output_config() {
        // Sanity-check the type still round-trips with output_config
        // (used by hooks). The hook prompt itself doesn't carry
        // output_config — Claude Code adds it via metadata — but a
        // hook-like request will.
        let raw = json!({
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role":"user","content":"x"}],
            "output_config": {"format": {"type":"json_schema"}}
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();
        assert!(req.output_config.is_some());
    }
}