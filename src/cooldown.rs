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
    reason: String,
}

/// Tracks per-provider cooldown windows.
#[derive(Clone)]
pub struct CooldownCache {
    inner: Arc<RwLock<HashMap<String, CooldownEntry>>>,
}

/// Maximum characters of `reason` we emit in a single `tracing::warn!`.
/// CooldownEntry still stores the full body for debug snapshots.
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
                reason: reason.to_string(),
            },
        );
        // Some upstreams (e.g. DeepSeek on auth failure) return bodies
        // longer than a kilobyte. Emitting the full body in every
        // `provider marked cooldown` warn floods the log and breaks
        // log-line parsers that assume one event per line. Keep the
        // full body in CooldownEntry (for debug snapshots) but truncate
        // what hits the log.
        tracing::warn!(
            provider = provider,
            status = status,
            duration_secs = duration.as_secs(),
            reason = %truncate_for_log(reason, LOG_REASON_MAX_CHARS),
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

    /// Same as `active()` but also includes the cached reason string.
    pub async fn active_with_reason(&self) -> Vec<(String, u16, Duration, String)> {
        let now = Instant::now();
        let guard = self.inner.read().await;
        guard
            .iter()
            .filter_map(|(k, v)| {
                if v.until > now {
                    Some((k.clone(), v.status, v.until - now, v.reason.clone()))
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
    async fn active_with_reason_returns_reason_string() {
        let c = CooldownCache::new();
        c.mark_cooldown("copilot", Duration::from_secs(60), 429, "rate limited")
            .await;

        let snapshot = c.active_with_reason().await;

        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "copilot");
        assert_eq!(snapshot[0].1, 429);
        assert_eq!(snapshot[0].3, "rate limited");
        assert!(c.active().await.iter().all(|entry| entry.0 == "copilot"));
    }

    #[tokio::test]
    async fn active_returns_empty_when_no_entries() {
        let c = CooldownCache::new();
        assert!(c.active().await.is_empty());
        assert!(c.active_with_reason().await.is_empty());
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
    async fn active_with_reason_skips_expired_entries() {
        // When every entry has expired, active_with_reason should return an
        // empty vec (the None branch of the filter_map closure runs).
        let c = CooldownCache::new();
        c.mark_cooldown("expired", Duration::from_millis(5), 429, "x")
            .await;
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(c.active_with_reason().await.is_empty());
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
        let snapshot = c.active_with_reason().await;
        assert_eq!(snapshot.len(), 1);
        assert!(
            snapshot[0].3.starts_with("attempt-"),
            "reason should come from one of the writers: {}",
            snapshot[0].3
        );
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
    async fn mark_cooldown_keeps_full_reason_in_entry_but_logs_truncated() {
        // CooldownEntry stores the full reason for debug snapshots, but
        // the tracing::warn! macro receives a truncated version. We can't
        // easily intercept tracing::warn! here without a subscriber; what
        // we can check directly is that active_with_reason() still
        // returns the un-truncated reason — i.e. truncation is purely
        // a logging concern.
        let c = CooldownCache::new();
        let long_reason = "y".repeat(2000);
        c.mark_cooldown("p", Duration::from_secs(5), 503, &long_reason)
            .await;
        let snapshot = c.active_with_reason().await;
        assert_eq!(snapshot.len(), 1);
        assert_eq!(
            snapshot[0].3.len(),
            2000,
            "snapshot must keep the full reason for debug"
        );
    }
}
