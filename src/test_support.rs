//! Test-only helpers shared across modules.
//!
//! Currently holds:
//! - `JsonFieldAbsent` — wiremock matcher that asserts a JSON field is
//!   **absent** from the request body. wiremock's `body_partial_json`
//!   only checks presence; the complement is needed for "the proxy
//!   must NOT send field X when the client didn't ask" assertions
//!   (e.g. `prompt_cache_key` / `prompt_cache_retention`).
//! - `openai_request` / `chat_response` / `cache_request_with` — wire
//!   fixtures shared by the OpenAI-family provider tests
//!   (openai_compat / openai_responses); previously duplicated inline.
//! - `env_lock` / `EnvGuard` — serialization plus crash-safe restoration
//!   for tests that mutate process-global env vars (`XDG_DATA_HOME`,
//!   `HOME`, `LLMPROXY_TEST_GITHUB_BASE_URL`).
//!
//! Consolidating this from `openai_compat.rs` and `openai_responses.rs`
//! — both files previously carried identical inline copies (PR-10).
//!
//! Lives behind `#[cfg(test)]` so it does not bloat the release binary.

#![cfg(test)]

use crate::anthropic::MessagesRequest;
use serde_json::{json, Value};
use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard};
use wiremock::{Match, Request};

/// Wire-level "field X must NOT be present in the JSON request body"
/// matcher. See module docs for rationale.
pub struct JsonFieldAbsent(pub &'static str);

impl Match for JsonFieldAbsent {
    fn matches(&self, request: &Request) -> bool {
        let body: serde_json::Value = match serde_json::from_slice(&request.body) {
            Ok(v) => v,
            Err(_) => return false,
        };
        body.get(self.0).is_none()
    }
}

/// Minimal Anthropic Messages request — the shared shape the
/// OpenAI-family provider tests (openai_compat / openai_responses) send
/// through conversion. Consolidated from two identical inline `request`
/// copies (PR-10).
pub fn openai_request(streaming: bool) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "stream": streaming,
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap()
}

/// Minimal OpenAI Chat Completions response body in wire JSON shape.
/// Consolidated from the inline copy in `openai_compat.rs` (PR-10).
pub fn chat_response() -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "world"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5
        }
    })
}

/// Anthropic request with a `cache_control` block, optionally carrying
/// `metadata.user_id`. Consolidated from two identical inline copies in
/// `openai_compat.rs` / `openai_responses.rs` (PR-10).
pub fn cache_request_with(cache_type: &str, user_id: Option<&str>) -> MessagesRequest {
    let mut v = json!({
        "model": "claude-sonnet-4.6",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "long prefix", "cache_control": {"type": cache_type}},
                {"type": "text", "text": "actual question"}
            ]
        }]
    });
    if let Some(uid) = user_id {
        v["metadata"] = json!({"user_id": uid});
    }
    serde_json::from_value(v).unwrap()
}

/// Serializes every test that reads or mutates process-global environment
/// variables (`XDG_DATA_HOME`, `HOME`, `LLMPROXY_TEST_GITHUB_BASE_URL`).
///
/// Cargo runs tests in parallel inside one process, so without this lock two
/// tests can interleave `set_var` / `remove_var` and one of them will observe
/// the other's value — typically a mock-server URL whose server is gone, or a
/// `HOME` pointing at a deleted tempdir. `github_base_url()` re-reads the env
/// var on every call, so the lock must cover every call that can reach it,
/// not just the initial setup.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Locks a mutex, recovering from poisoning.
///
/// A test that panics with the guard alive poisons the mutex. `into_inner()`
/// keeps mutual exclusion intact — only the "an earlier holder panicked" flag
/// is discarded — which is exactly what we want: one broken test must not
/// cascade into every later test that needs the same env vars.
fn lock_tolerating_poison<T>(m: &'static Mutex<T>) -> MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// RAII guard for [`ENV_LOCK`] that restores every variable touched through it.
///
/// Restoration happens in `Drop::drop`, which runs before the struct's fields
/// are dropped (i.e. while the lock is still held) and also runs during panic
/// unwinding. That is what makes a failing test leave the process environment
/// clean for the tests that follow; the previous `remove_var` at the end of a
/// test body was skipped on panic and leaked a dead base URL into the next
/// lock holder.
///
/// Acquire exactly one guard at the top of a test and pass `&mut EnvGuard` to
/// helpers: [`env_lock`] is **not reentrant** (`std::sync::Mutex` is not), so
/// a helper that called it again would deadlock. `let _ = env_lock();` drops
/// the guard — and releases the lock — immediately, so bind it to a name. The
/// guard is `!Send` because it owns a `MutexGuard`, so it cannot be captured
/// by `tokio::spawn`.
#[must_use = "bind the guard to a variable (let mut env = env_lock()), or the lock is released immediately"]
pub(crate) struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

/// Take `ENV_LOCK` and return a guard that restores the env vars it changes.
///
/// Poison-tolerant: a panicking test does not make every later caller panic.
pub(crate) fn env_lock() -> EnvGuard {
    EnvGuard {
        _lock: lock_tolerating_poison(&ENV_LOCK),
        saved: Vec::new(),
    }
}

impl EnvGuard {
    /// Set `key`, remembering its pre-test value for restoration on drop.
    pub(crate) fn set_var(&mut self, key: &'static str, value: impl AsRef<OsStr>) {
        self.remember(key);
        std::env::set_var(key, value);
    }

    /// Remove `key`, remembering its pre-test value for restoration on drop.
    pub(crate) fn remove_var(&mut self, key: &'static str) {
        self.remember(key);
        std::env::remove_var(key);
    }

    fn remember(&mut self, key: &'static str) {
        // Snapshot on the first mutation only: a test may set the same key
        // twice, and it is the value from before the test that must come back.
        if !self.saved.iter().any(|(saved, _)| *saved == key) {
            self.saved.push((key, std::env::var_os(key)));
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, previous) in self.saved.drain(..) {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

#[cfg(test)]
mod env_guard_tests {
    use super::*;

    /// Private to this module, so no other test can race on it.
    const KEY: &str = "LLMPROXY_TEST_ENV_GUARD";

    /// Two more private keys, one per test that asserts *after* the guard
    /// is dropped.
    ///
    /// The assertion after `drop(env)` runs with the lock released, so it
    /// must not look at a variable another test might be mutating at that
    /// moment. The tests below all used the same key, so a sibling test that
    /// had just set it made the post-drop assertion read that value instead
    /// of the original one — a flake that only showed up when the suite ran
    /// the tests in a different order (it failed one PR run with
    /// `left: Some("set-before-panic")`, `right: None`).
    /// One key per test removes the overlap entirely: no test writes a
    /// variable that another test asserts on.
    const KEY_SECOND: &str = "LLMPROXY_TEST_ENV_GUARD_SECOND";
    const KEY_PANIC: &str = "LLMPROXY_TEST_ENV_GUARD_PANIC";

    #[test]
    fn env_guard_restores_the_previous_value_on_drop() {
        let mut env = env_lock();
        let original = std::env::var_os(KEY);
        env.set_var(KEY, "set-by-guard");
        assert_eq!(std::env::var_os(KEY).as_deref(), Some(OsStr::new("set-by-guard")));
        drop(env);

        assert_eq!(std::env::var_os(KEY), original);
    }

    #[test]
    fn env_guard_does_not_overwrite_its_snapshot_on_second_mutation() {
        let mut env = env_lock();
        let original = std::env::var_os(KEY_SECOND);
        env.set_var(KEY_SECOND, "first");
        env.set_var(KEY_SECOND, "second");
        drop(env);

        assert_eq!(std::env::var_os(KEY_SECOND), original);
    }

    #[test]
    fn env_guard_restores_the_previous_value_when_the_test_panics() {
        let mut env = env_lock();
        let original = std::env::var_os(KEY_PANIC);
        env.set_var(KEY_PANIC, "set-before-panic");

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _env = env;
            panic!("simulated assertion failure");
        }));
        assert!(outcome.is_err());

        // The guard is dropped while unwinding, so the variable is restored
        // even though this test never reached a cleanup statement.
        assert_eq!(std::env::var_os(KEY_PANIC), original);

        // This panic happened with the guard alive, so ENV_LOCK is now
        // poisoned for the rest of the process. That is intentional: it makes
        // the remainder of the suite exercise the poison-tolerant path. If a
        // later test starts failing on `.lock().unwrap()`, the de-poisoning
        // migration missed a site.
    }

    #[test]
    fn env_lock_recovers_from_a_poisoned_mutex() {
        static LOCAL: Mutex<u32> = Mutex::new(0);
        let outcome = std::panic::catch_unwind(|| {
            let _guard = LOCAL.lock().unwrap();
            panic!("poison the local mutex");
        });
        assert!(outcome.is_err());
        assert!(LOCAL.lock().is_err(), "the mutex should be poisoned");

        let mut guard = lock_tolerating_poison(&LOCAL);
        *guard += 1;
        assert_eq!(*guard, 1);
    }
}