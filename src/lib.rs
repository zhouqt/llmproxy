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
pub mod router;
pub mod server;
pub mod state;
pub mod tokenize;

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
