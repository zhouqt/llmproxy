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

/// Reduce a payload string to a short, log-friendly hint.
///
/// The SSE and error-body debug logs used to dump the full payload, which
/// for a Copilot 502 HTML error page meant dumping a full HTML document
/// with stylesheets, image refs, and links. Strip tags, URLs, image refs,
/// scripts, and styles, then take the first ~200 chars of substantive text.
///
/// When the result would be empty, `empty_placeholder` is returned instead
/// (e.g. `"<empty body>"` or `"<empty payload>"`).
pub fn summarize_for_log(input: &str, empty_placeholder: &str) -> String {
    // Drop <script>...</script> and <style>...</style> blocks first —
    // those contain CSS / JS that we never want in the hint.
    let mut s = input.to_string();
    for tag in ["script", "style"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        while let Some(start) = s.find(&open) {
            let Some(end_rel) = s[start..].find(&close) else {
                // Unterminated block: drop everything from <tag to end.
                s.truncate(start);
                break;
            };
            let end = start + end_rel + close.len();
            s.replace_range(start..end, " ");
        }
    }
    // Strip HTML comments.
    while let Some(start) = s.find("<!--") {
        let Some(end_rel) = s[start..].find("-->") else {
            s.truncate(start);
            break;
        };
        s.replace_range(start..start + end_rel + 3, " ");
    }
    // Strip URLs (http/https/ftp), image refs, and CSS url(...) calls.
    let url_chars = |c: char| c.is_ascii_alphanumeric() || matches!(c, ':' | '/' | '.' | '-' | '_' | '?' | '&' | '=' | '%' | '#' | '+' | '~');
    let strip_token = |s: &mut String, token: &str| {
        let mut idx = 0;
        while let Some(rel) = s[idx..].find(token) {
            let start = idx + rel;
            let after = start + token.len();
            let mut end = after;
            while end < s.len() && url_chars(s.as_bytes()[end] as char) {
                end += 1;
            }
            s.replace_range(start..end, " ");
            idx = start + 1;
        }
    };
    for prefix in ["https://", "http://", "ftp://"] {
        strip_token(&mut s, prefix);
    }
    for token in ["src=\"", "src='", "url(", "href=\"", "href='"] {
        strip_token(&mut s, token);
    }
    // Strip any remaining HTML tags.
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    // Collapse whitespace and take the first non-empty line.
    let mut line = String::new();
    for c in out.chars() {
        if c.is_whitespace() {
            if !line.ends_with(' ') && !line.is_empty() {
                line.push(' ');
            }
        } else {
            line.push(c);
        }
        if line.len() > 200 {
            break;
        }
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        empty_placeholder.to_string()
    } else {
        trimmed.to_string()
    }
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

    /// T18: P1-4 — hoisted `summarize_for_log` handles HTML, plain text,
    /// empty body, and multibyte payloads.
    #[test]
    fn summarize_for_log_in_util_handles_html_plain_empty_multibyte() {
        // HTML with scripts, styles, URLs, and comments.
        let html = r#"<html><head><script>alert('x')</script><style>body{color:red}</style></head>
<body><!-- comment -->
<img src="https://example.com/img.png"/>
<p>rate limited</p></body></html>"#;
        let result = summarize_for_log(html, "<empty payload>");
        assert!(!result.contains("script"), "script content stripped");
        assert!(!result.contains("style"), "style content stripped");
        assert!(!result.contains("https://"), "URL stripped");
        assert!(!result.contains("<!--"), "comments stripped");
        assert!(result.contains("rate limited"), "text preserved");
        assert!(result.len() <= 210, "result truncated to ~200 chars");

        // Plain text passes through.
        assert_eq!(
            summarize_for_log("invalid grant: token expired", "<empty payload>"),
            "invalid grant: token expired"
        );

        // Empty input returns placeholder.
        assert_eq!(summarize_for_log("", "<empty payload>"), "<empty payload>");
        assert_eq!(summarize_for_log("   \n  \t ", "<empty payload>"), "<empty payload>");
        assert_eq!(summarize_for_log("<html></html>", "<empty payload>"), "<empty payload>");

        // Multibyte characters are preserved.
        let mb = "模型限流 请稍后重试";
        assert_eq!(summarize_for_log(mb, "<empty payload>"), mb);
    }
}
