//! Shared utility predicates and helpers used across the proxy.

/// Returns `true` when `model` belongs to the GPT-5.x family (including
/// o-series reasoning models) that share specific API constraints:
///
/// - `/responses` endpoint required (Copilot rejects `/chat/completions`)
/// - `max_completion_tokens` instead of `max_tokens`
/// - Only `"24h"` prompt-cache retention is accepted (`"in_memory"` causes
///   a 400 from the upstream)
pub fn gpt5_family(model: &str) -> bool {
    model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt5_models_match() {
        assert!(gpt5_family("gpt-5"));
        assert!(gpt5_family("gpt-5-mini"));
        assert!(gpt5_family("gpt-5.5"));
        assert!(gpt5_family("gpt-5-2025-08-07"));
    }

    #[test]
    fn o_series_models_match() {
        assert!(gpt5_family("o1"));
        assert!(gpt5_family("o1-mini"));
        assert!(gpt5_family("o3"));
        assert!(gpt5_family("o3-mini"));
        assert!(gpt5_family("o4"));
        assert!(gpt5_family("o4-mini"));
    }

    #[test]
    fn non_gpt5_models_do_not_match() {
        assert!(!gpt5_family("gpt-4"));
        assert!(!gpt5_family("gpt-4o"));
        assert!(!gpt5_family("claude-sonnet-4-5"));
        assert!(!gpt5_family("deepseek-chat"));
        assert!(!gpt5_family(""));
    }
}
