//! llmproxy library — re-exports modules for integration tests.

pub mod anthropic;
pub mod auth;
pub mod config;
pub mod conversion;
pub mod cooldown;
pub mod error;
pub mod extractor;
pub mod oauth;
pub mod openai;
pub mod proxy_client;
pub mod providers;
pub mod responses;
pub mod router;
#[cfg(test)]
pub mod test_support;
pub mod server;
pub mod state;
pub mod tokenize;
pub mod tracing_capture;
pub mod usage;
pub mod util;

/// Test-only helper macro: match a value against a pattern, execute the
/// body on success, or panic with a single canonical message on failure.
/// Centralizing the panic message keeps each call site free of its own
/// missed panic-string line, which coverage treats as unreachable.
#[macro_export]
macro_rules! expect_variant {
    ($value:expr, $pattern:pat => $body:block) => {
        if let $pattern = $value {
            $body
        } else {
            panic!("expected variant match for {}", stringify!($pattern));
        }
    };
}

/// Install a benign global tracing subscriber at process start so lib
/// unit tests don't fight over the `tracing` subscriber slot and the
/// callsite `Interest` cache doesn't poison the first event with a
/// "max level = off" decision.
///
/// Why this ctor is necessary (Phase 2 of
/// `plans/feat-fallback-logging-rework.md`): without it, the very
/// first `tracing::event!` macro expanded in the lib-test binary
/// would see "no subscriber" and permanently cache
/// `Interest::never()` for that callsite's `TypeId`. Subsequent
/// `set_default` calls in `tracing_capture::capture_tracing` install a
/// subscriber that the lib-test events can no longer see, so test
/// assertions on captured bytes would be empty.
///
/// Cargo compiles the lib with `--cfg test` for the lib-test binary
/// only; integration tests (`tests/*.rs`) get the lib compiled without
/// `cfg(test)` and therefore do NOT pull in this ctor — they have
/// their own at the top of `tests/server.rs`.
#[cfg(test)]
#[ctor::ctor]
fn __install_benign_tracing_default_for_lib_tests() {
    let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
}
