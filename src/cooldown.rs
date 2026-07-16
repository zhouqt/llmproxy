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
        tracing::warn!(
            provider = provider,
            status = status,
            duration_secs = duration.as_secs(),
            reason = reason,
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
}
