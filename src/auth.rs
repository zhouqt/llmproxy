//! Bearer-token auth middleware.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use secrecy::ExposeSecret;
use tracing::warn;

use crate::state::AppState;

/// Length of the fixed window over which failed authentication attempts
/// are counted (audit L7).
pub const AUTH_FAILURE_WINDOW: Duration = Duration::from_secs(60);

/// How many failed attempts are answered with 401 inside one window;
/// every failure past this count gets 429 until the window closes.
pub const AUTH_FAILURE_LIMIT: u32 = 30;

pub async fn require_auth(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state
        .config
        .server
        .api_key
        .as_ref()
        .map(|k| k.expose_secret())
    else {
        // No api_key configured → auth disabled.
        return next.run(req).await;
    };

    let header_value = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            // Anthropic SDK uses x-api-key for the provider-side auth; allow
            // it as a fallback so existing clients work unchanged.
            req.headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
        });

    match header_value {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(req).await
        }
        _ => {
            // Audit L7: guessing the key must not be free. Failures share
            // one in-process fixed window; once [`AUTH_FAILURE_LIMIT`] is
            // exceeded, further bad attempts get 429 + Retry-After instead
            // of 401. Valid tokens never touch the limiter (arm above), and
            // with no api_key configured this arm is unreachable.
            let now = Instant::now();
            if state.auth_limiter.register_failure(now) {
                let retry = state.auth_limiter.retry_after_secs(now);
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::RETRY_AFTER, retry.to_string())],
                    axum::Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "too many failed authentication attempts; retry later"
                        }
                    })),
                )
                    .into_response();
            }
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "missing or invalid API key"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Per-process fixed-window counter of failed authentication attempts.
///
/// Deliberately in-process only (audit L7 is about making online
/// brute-force *slow*, not about multi-replica accounting): the proxy
/// runs as a single podman container in deployment, and a shared store
/// would add a failure mode to every authenticated request for little
/// gain. Successful authentications never touch this limiter, and with
/// no `server.api_key` configured the middleware never consults it.
///
/// Time is injected (`register_failure(now)`) so tests can walk the
/// window forward without sleeping.
#[derive(Clone, Debug, Default)]
pub struct AuthRateLimiter {
    inner: Arc<Mutex<Failures>>,
}

#[derive(Debug, Default)]
struct Failures {
    /// Start of the currently accumulating window; `None` before the
    /// first failure (and modelled as expired once `now` passes
    /// `start + AUTH_FAILURE_WINDOW`).
    window_start: Option<Instant>,
    count: u32,
    /// The crossing into throttled state is logged exactly once per
    /// window — repeated warnings on every subsequent brute-force
    /// attempt would flood the log with exactly the noise the attacker
    /// is producing.
    warned: bool,
}

impl AuthRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a failed attempt at time `now`. Returns `true` when the
    /// caller must answer 429 instead of 401, i.e. the fixed window has
    /// more than [`AUTH_FAILURE_LIMIT`] failures in it.
    pub fn register_failure(&self, now: Instant) -> bool {
        // A poisoned mutex must not take auth down with it: recover the
        // (possibly torn) state and keep counting.
        let mut f = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match f.window_start {
            Some(start) if now.duration_since(start) < AUTH_FAILURE_WINDOW => {
                f.count += 1;
            }
            _ => {
                // First failure ever, or the previous window expired.
                f.window_start = Some(now);
                f.count = 1;
                f.warned = false;
            }
        }
        if f.count > AUTH_FAILURE_LIMIT {
            if !f.warned {
                f.warned = true;
                warn!(
                    limit = AUTH_FAILURE_LIMIT,
                    window_secs = AUTH_FAILURE_WINDOW.as_secs(),
                    "auth failure threshold reached; further failed attempts get 429 until the window closes"
                );
            }
            true
        } else {
            false
        }
    }

    /// Seconds from `now` until the open window closes — the value for
    /// the `Retry-After` header on a 429. `0` when no window is open
    /// (the caller only asks this after `register_failure` returned
    /// `true`, so in practice a window is always open here).
    pub fn retry_after_secs(&self, now: Instant) -> u64 {
        let f = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        f.window_start
            .map(|start| {
                AUTH_FAILURE_WINDOW
                    .saturating_sub(now.duration_since(start))
                    .as_secs()
                    .max(1)
            })
            .unwrap_or(0)
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // Hash both sides before comparing (audit L3): the raw loop used to
    // return early on a length mismatch, which leaks the expected API
    // key's length through timing. SHA-256 digests are always 32 bytes,
    // so the comparison below has no length-dependent branch at all, and
    // SHA-256 collision resistance makes the extra step invisible to
    // correct-token traffic. `ring` is already a direct dependency (it
    // encrypts the Copilot token store), so this adds no crate.
    let da = ring::digest::digest(&ring::digest::SHA256, a);
    let db = ring::digest::digest(&ring::digest::SHA256, b);
    let mut diff: u8 = 0;
    for (x, y) in da.as_ref().iter().zip(db.as_ref().iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(!constant_time_eq(b"hello", b"helloo"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn constant_time_eq_has_no_length_shortcut() {
        // Audit L3: unequal lengths must still yield `false`, but through
        // the same 32-byte digest comparison as every other input — the
        // historical early `return false` on a length mismatch is gone.
        assert!(!constant_time_eq(b"", b"a"));
        assert!(!constant_time_eq(b"a", b""));
        assert!(!constant_time_eq(&[0u8; 64], &[0u8; 65]));

        // Long inputs: equal compares equal, a single flipped byte does not.
        let long = [7u8; 4096];
        assert!(constant_time_eq(&long, &long));
        let mut flipped = long;
        flipped[4095] ^= 1;
        assert!(!constant_time_eq(&long, &flipped));
    }

    #[test]
    fn limiter_throttles_from_the_31st_failure() {
        let lim = AuthRateLimiter::new();
        let t0 = Instant::now();
        for _ in 0..AUTH_FAILURE_LIMIT {
            assert!(!lim.register_failure(t0), "failures 1..=30 stay 401");
        }
        // 31st and every later failure inside the window: 429.
        assert!(lim.register_failure(t0), "31st failure throttles");
        assert!(lim.register_failure(t0 + Duration::from_secs(1)));

        // Retry-After counts down to the window edge (never below 1).
        assert_eq!(lim.retry_after_secs(t0), AUTH_FAILURE_WINDOW.as_secs());
        assert_eq!(
            lim.retry_after_secs(t0 + Duration::from_secs(59)),
            1,
            "clamped to at least one second"
        );
    }

    #[test]
    fn limiter_window_expiry_resets_count() {
        let lim = AuthRateLimiter::new();
        let t0 = Instant::now();
        for _ in 0..AUTH_FAILURE_LIMIT {
            lim.register_failure(t0);
        }
        assert!(lim.register_failure(t0), "window full → throttled");

        // Past the window the count resets: the next failure is a 401
        // again, and it takes a fresh full window to throttle again.
        let t1 = t0 + AUTH_FAILURE_WINDOW + Duration::from_secs(1);
        assert!(!lim.register_failure(t1)); // failure 1 of the new window
        assert_eq!(lim.retry_after_secs(t1), AUTH_FAILURE_WINDOW.as_secs());
        for _ in 0..(AUTH_FAILURE_LIMIT - 1) {
            assert!(!lim.register_failure(t1));
        }
        assert!(lim.register_failure(t1), "31st failure of the new window");
    }

    #[test]
    fn limiter_warns_exactly_once_per_window() {
        use crate::tracing_capture::{capture_tracing, CaptureWriter};

        let lim = AuthRateLimiter::new();
        let w = CaptureWriter::new();
        let t0 = Instant::now();
        let ((), captured) = capture_tracing(&w, || {
            for _ in 0..AUTH_FAILURE_LIMIT {
                lim.register_failure(t0);
            }
            // Crossing the limit warns once…
            assert!(lim.register_failure(t0));
            // …and every further failure in the same window stays silent.
            for _ in 0..5 {
                assert!(lim.register_failure(t0));
            }
        });
        assert_eq!(
            captured.matches("auth failure threshold reached").count(),
            1,
            "exactly one warning per window: {captured:?}"
        );

        // A fresh window arms the warning again.
        let ((), captured2) = capture_tracing(&w, || {
            let t1 = t0 + AUTH_FAILURE_WINDOW + Duration::from_secs(1);
            lim.register_failure(t1);
        });
        assert!(
            !captured2.contains("auth failure threshold reached"),
            "sub-threshold window stays silent: {captured2:?}"
        );
    }

    #[test]
    fn limiter_is_shareable_across_clones() {
        let lim = AuthRateLimiter::new();
        let clone = lim.clone();
        let t0 = Instant::now();
        for _ in 0..AUTH_FAILURE_LIMIT {
            lim.register_failure(t0);
        }
        // The clone observes the same window (Arc<Mutex> semantics the
        // AppState clones rely on).
        assert!(clone.register_failure(t0));
        assert_eq!(clone.retry_after_secs(t0), AUTH_FAILURE_WINDOW.as_secs());
    }
}
