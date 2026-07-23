//! In-memory cooldown cache for providers.
//!
//! Reference: litellm/router_utils/cooldown_cache.py

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

#[derive(Debug, Clone)]
struct CooldownEntry {
    until: Instant,
    status: u16,
}

/// Tracks per-provider cooldown windows.
#[derive(Clone)]
pub struct CooldownCache {
    inner: Arc<RwLock<HashMap<String, CooldownEntry>>>,
}

/// Maximum characters of upstream body we emit in a single
/// `tracing::warn!`. The body itself is NOT stored — it is only
/// truncated for the warn log so operator can see *why* a provider
/// was cooled down without flooding the log or holding onto KB-sized
/// upstream payloads.
const LOG_REASON_MAX_CHARS: usize = 200;

/// Truncate `s` to `max_chars` characters, appending an ellipsis marker
/// when truncated. Operates on char boundaries so multi-byte UTF-8 is
/// never split mid-codepoint.
fn truncate_for_log(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}… [+{} chars]", s.chars().count() - max_chars)
}

impl CooldownCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn is_cooling_down(&self, provider: &str) -> bool {
        let guard = self.inner.read().await;
        match guard.get(provider) {
            Some(entry) => Instant::now() < entry.until,
            None => false,
        }
    }

    pub async fn mark_cooldown(
        &self,
        provider: &str,
        duration: Duration,
        status: u16,
        reason: &str,
    ) {
        let mut guard = self.inner.write().await;
        guard.insert(
            provider.to_string(),
            CooldownEntry {
                until: Instant::now() + duration,
                status,
            },
        );
        // Some upstreams (e.g. DeepSeek on auth failure) return bodies
        // longer than a kilobyte. Emit the body in the cooldown warn
        // (truncated) so operator can see *why* a provider was cooled,
        // but do NOT keep the full body in the cache — there is no
        // reader for it. See issue #1.
        tracing::warn!(
            provider = provider,
            status = status,
            duration_secs = duration.as_secs(),
            reason = %crate::util::summarize_for_log(reason, "<empty body>"),
            "provider marked cooldown"
        );
    }

    /// Garbage-collect expired entries. Cheap to call.
    pub async fn cleanup(&self) {
        let now = Instant::now();
        let mut guard = self.inner.write().await;
        guard.retain(|_, e| e.until > now);
    }

    /// Snapshot of currently-cooling providers for debugging/headers.
    pub async fn active(&self) -> Vec<(String, u16, Duration)> {
        let now = Instant::now();
        let guard = self.inner.read().await;
        guard
            .iter()
            .filter_map(|(k, v)| {
                if v.until > now {
                    Some((k.clone(), v.status, v.until - now))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for CooldownCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mark_and_expire() {
        let c = CooldownCache::new();
        assert!(!c.is_cooling_down("p").await);
        c.mark_cooldown("p", Duration::from_millis(50), 429, "rate limit").await;
        assert!(c.is_cooling_down("p").await);
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(!c.is_cooling_down("p").await);
    }

    #[tokio::test]
    async fn different_providers_independent() {
        let c = CooldownCache::new();
        c.mark_cooldown("a", Duration::from_secs(10), 429, "").await;
        assert!(c.is_cooling_down("a").await);
        assert!(!c.is_cooling_down("b").await);
    }

    #[tokio::test]
    async fn cleanup_removes_expired() {
        let c = CooldownCache::new();
        c.mark_cooldown("a", Duration::from_millis(10), 500, "").await;
        c.mark_cooldown("b", Duration::from_secs(60), 429, "").await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        c.cleanup().await;
        let active = c.active().await;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, "b");
    }

    #[tokio::test]
    async fn active_returns_empty_when_no_entries() {
        let c = CooldownCache::new();
        assert!(c.active().await.is_empty());
    }

    #[tokio::test]
    async fn ttl_at_exact_boundary_counts_as_expired() {
        let c = CooldownCache::new();
        c.mark_cooldown("boundary", Duration::from_millis(10), 503, "")
            .await;
        assert!(c.is_cooling_down("boundary").await);
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(!c.is_cooling_down("boundary").await);
        assert!(c.active().await.is_empty());
    }

    #[tokio::test]
    async fn active_skips_expired_entries() {
        // When every entry has expired, active() should return an empty
        // vec (the None branch of the filter_map closure runs).
        let c = CooldownCache::new();
        c.mark_cooldown("expired", Duration::from_millis(5), 429, "x")
            .await;
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(c.active().await.is_empty());
    }

    #[tokio::test]
    async fn concurrent_marks_converge() {
        let c = std::sync::Arc::new(CooldownCache::new());
        let mut handles = Vec::new();
        for i in 0..32 {
            let cache = c.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .mark_cooldown(
                        "shared",
                        Duration::from_secs(60),
                        429,
                        &format!("attempt-{i}"),
                    )
                    .await;
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        assert!(c.is_cooling_down("shared").await);
        // The reason string is no longer cached — only one entry remains
        // after 32 concurrent writes. The reason is still consumed by
        // the warn log but is not stored in the cooldown entry itself.
        let snapshot = c.active().await;
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "shared");
        assert_eq!(snapshot[0].1, 429);
    }

    #[test]
    fn default_impl_matches_new() {
        // Default must behave the same as new() so the type can be used in
        // struct initializers and Default-derive contexts.
        let default_cache = CooldownCache::default();
        let new_cache = CooldownCache::new();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            assert!(default_cache.active().await.is_empty());
            assert!(new_cache.active().await.is_empty());
            assert!(!default_cache.is_cooling_down("anything").await);
            assert!(!new_cache.is_cooling_down("anything").await);
        });
    }

    #[test]
    fn truncate_for_log_passes_through_short_strings() {
        assert_eq!(truncate_for_log("short", 200), "short");
        assert_eq!(truncate_for_log("", 200), "");
    }

    #[test]
    fn truncate_for_log_truncates_long_strings_with_marker() {
        let s = "x".repeat(500);
        let out = truncate_for_log(&s, 200);
        // 200 chars of 'x' + ellipsis marker + extra count
        assert!(out.starts_with(&"x".repeat(200)), "got: {out}");
        assert!(out.contains("…"), "expected ellipsis marker in: {out}");
        assert!(
            out.contains("[+300 chars]"),
            "expected extra-count annotation in: {out}"
        );
    }

    #[test]
    fn truncate_for_log_respects_utf8_char_boundaries() {
        // 4-byte chars: if we sliced mid-codepoint the function would panic
        // or produce invalid UTF-8. Verify no panic and that the output
        // round-trips through String.
        let s = "🌀".repeat(300);
        let out = truncate_for_log(&s, 200);
        assert!(out.contains("…"));
        // Must be valid UTF-8 (String constructor enforces this; we just
        // assert the chars we kept are intact).
        assert!(out.starts_with(&"🌀".repeat(200)));
    }

    #[tokio::test]
    async fn mark_cooldown_does_not_cache_reason_body() {
        // After deleting the `reason` field on `CooldownEntry`, the cache
        // no longer holds the upstream body — but `truncate_for_log` is
        // still applied to the warn log. We can't easily intercept
        // `tracing::warn!` here without a subscriber; what we *can*
        // assert is that `active()` (the only remaining snapshot
        // interface) returns just (name, status, ttl) and never a body.
        let c = CooldownCache::new();
        c.mark_cooldown("p", Duration::from_secs(5), 503, "any body")
            .await;
        let snapshot = c.active().await;
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "p");
        assert_eq!(snapshot[0].1, 503);
        // Snapshot is a 3-tuple — no 4th element to leak body into.
        // (Type system enforces this; the comment is for readers.)
    }
}
