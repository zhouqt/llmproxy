//! Secret overlay — the llmproxy half of the TPM-sealed `.env` plan.
//!
//! Historically every secret the proxy needed arrived through the
//! process environment: `config.rs` expands `${VAR}` against
//! `std::env`. That works, but it forces the plaintext key to live in
//! the process environment of whatever launches the proxy — and, in
//! the container deployment, `env_file:` puts it into
//! `podman inspect`'s `Config.Env` where any local user can read it.
//!
//! This module adds a second, **higher-priority** source for `${VAR}`:
//! a `.env`-format file on disk. The file is produced by the
//! container entrypoint (unsealing a TPM sealed object into the
//! container's tmpfs) or by hand (the plaintext fallback). The proxy
//! itself never learns that a TPM exists — it only learns "read this
//! file, resolve `${VAR}` from it first".
//!
//! See `plans/feat-vault-secret-backend.md` §5.2 (file format) and
//! §5.5 (`secrets:` block) for the specification this implements.

use secrecy::SecretString;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{ProxyError, Result};

/// Build a [`SecretString`] — the type every credential is stored in.
///
/// `secrecy::SecretString` keeps the value out of `{:?}` output and
/// zeroes it on drop, so a credential can never be printed by accident
/// (a `tracing::debug!("{cfg:?}")` used to dump every API key in the
/// process). See `plans/feat-vault-secret-backend.md` §A-Phase 5.
///
/// This constructor exists because the crate's API takes a
/// `SecretString` everywhere, and the construction sites (config
/// parsing, provider construction, test fixtures) are numerous.
pub fn secret_string(value: impl Into<String>) -> SecretString {
    SecretString::from(value.into())
}

/// Where `${VAR}` placeholders resolve from.
///
/// `None` and `Env` behave identically — the process environment is
/// the only source. They exist as two spellings of the same thing so
/// an operator can write down "environment only" explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretsProvider {
    /// Environment only. The default, and what a config without a
    /// `secrets:` block gets — keeping such a config byte-for-byte
    /// identical to the pre-feature behavior.
    #[default]
    None,
    /// Explicit spelling of "environment only".
    Env,
    /// Read the `.env`-format file named by `file.path` and resolve
    /// `${VAR}` from it before falling back to the environment.
    File,
}

impl SecretsProvider {
    /// Whether this provider consults the on-disk `.env` file.
    pub fn uses_file(self) -> bool {
        matches!(self, SecretsProvider::File)
    }
}

/// The `secrets.file` sub-block.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecretsFileConfig {
    /// Path of the `.env`-format file to read. Only consulted when
    /// `secrets.provider` is `file`. May itself contain a `${VAR}`
    /// reference, which is resolved from the process environment
    /// (the overlay cannot describe where the overlay lives).
    pub path: Option<String>,
    /// What to do when the file cannot be read.
    ///
    /// `true` (the default) aborts startup — fail fast, so a
    /// misconfigured or missing secret file never silently degrades
    /// into "the proxy started without keys". `false` logs a warning
    /// and continues with an empty overlay, falling back to the
    /// process environment.
    pub required: bool,
}

impl Default for SecretsFileConfig {
    fn default() -> Self {
        Self {
            path: None,
            required: true,
        }
    }
}

/// The top-level `secrets:` block of `config.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecretsConfig {
    /// Which source `${VAR}` resolves from.
    pub provider: SecretsProvider,
    /// File-source settings. Ignored unless `provider` is `file`.
    pub file: SecretsFileConfig,
}

impl Default for SecretsConfig {
    fn default() -> Self {
        Self {
            provider: SecretsProvider::None,
            file: SecretsFileConfig::default(),
        }
    }
}

impl SecretsConfig {
    /// The configured `.env` path, if one was set and it is not blank.
    pub fn file_path(&self) -> Option<&str> {
        self.file
            .path
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
    }

    /// Read the `.env` file at `path` and parse it into an overlay.
    ///
    /// A missing or unreadable file is an error when `file.required`
    /// is set (the default) and a warning plus an empty overlay
    /// otherwise. An **empty** file is never an error — it is a valid
    /// `.env` with no variables.
    pub fn read_overlay(&self, path: &Path) -> Result<HashMap<String, String>> {
        match std::fs::read_to_string(path) {
            Ok(raw) => Ok(parse_env_file(&raw)),
            Err(e) => {
                if self.file.required {
                    Err(ProxyError::Config(format!(
                        "read secrets file {}: {e}",
                        path.display()
                    )))
                } else {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "secrets file unreadable; continuing without overlay"
                    );
                    Ok(HashMap::new())
                }
            }
        }
    }
}

/// Parse a `.env`-format file into a flat name → value table.
///
/// The rules (plan §5.2) are deliberately *not* the ones
/// `docker compose`'s `env_file` uses — compose does not strip quotes
/// and does interpolate values, and a secret value should be passed
/// through byte-for-byte:
///
/// - each line is trimmed of leading/trailing whitespace;
/// - blank lines and lines starting with `#` are skipped;
/// - an optional `export` prefix is stripped;
/// - the line is split on the **first** `=` — the name is trimmed, the
///   value keeps any further `=`;
/// - a value wrapped in a *matching* pair of `"` or `'` is unquoted
///   (unmatched quotes are kept verbatim);
/// - **no** variable interpolation and **no** trailing-comment
///   stripping — a `#` inside a secret is a legal character;
/// - a repeated key replaces the earlier one;
/// - a line without `=` is skipped with a warning rather than
///   failing the whole file.
pub fn parse_env_file(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `export FOO=bar` is common in hand-written files; the
        // keyword is not part of the variable name.
        let line = match line.strip_prefix("export") {
            Some(rest) if rest.starts_with(char::is_whitespace) => rest.trim_start(),
            _ => line,
        };
        let Some((name, value)) = line.split_once('=') else {
            tracing::warn!(
                line = idx + 1,
                "secrets file line is not a NAME=VALUE assignment; skipped"
            );
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            tracing::warn!(
                line = idx + 1,
                "secrets file line has an empty name; skipped"
            );
            continue;
        }
        // Trim the whitespace around the value first, then unquote —
        // so `KEY="  padded  "` keeps its inner spaces while
        // `KEY=  padded  ` does not keep the outer ones.
        out.insert(
            name.to_string(),
            strip_matching_quotes(value.trim()).to_string(),
        );
    }
    out
}

/// Strip one matching pair of outer single or double quotes from `s`.
/// Unmatched quotes are preserved verbatim.
fn strip_matching_quotes(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

/// Deserialize the `secrets:` block, tolerating an explicitly null
/// value (`secrets:` with nothing under it) as "not configured".
///
/// `Option` in front of the concrete type is what makes `secrets:`
/// (null) distinct from `secrets: {}` (an empty but present block) —
/// serde would otherwise reject the null with "invalid type: unit
/// value", which is an unfriendly way to spell "I left the block
/// empty".
pub(crate) fn deserialize_secrets_config<'de, D>(
    d: D,
) -> std::result::Result<SecretsConfig, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<SecretsConfig>::deserialize(d)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    fn get<'a>(m: &'a HashMap<String, String>, k: &str) -> Option<&'a str> {
        m.get(k).map(String::as_str)
    }

    #[test]
    fn parses_plain_assignments() {
        let m = parse_env_file("A=1\nB=two\n");
        assert_eq!(get(&m, "A"), Some("1"));
        assert_eq!(get(&m, "B"), Some("two"));
    }

    #[test]
    fn skips_blank_lines_and_comments() {
        let m = parse_env_file("\n  \n# a comment\n   # indented comment\nA=1\n");
        assert_eq!(m.len(), 1);
        assert_eq!(get(&m, "A"), Some("1"));
    }

    #[test]
    fn strips_export_prefix() {
        let m = parse_env_file("export A=1\nexport\tB=2\nexported=3\n");
        assert_eq!(get(&m, "A"), Some("1"));
        assert_eq!(get(&m, "B"), Some("2"));
        // `exported` merely starts with "export" — not the keyword.
        assert_eq!(get(&m, "exported"), Some("3"));
    }

    #[test]
    fn splits_on_first_equals_only() {
        let m = parse_env_file("A=x=y=z\n");
        assert_eq!(get(&m, "A"), Some("x=y=z"));
    }

    #[test]
    fn trims_name_but_not_internal_value_spaces() {
        let m = parse_env_file("  A  =  hello world  \n");
        assert_eq!(get(&m, "A"), Some("hello world"));
    }

    #[test]
    fn strips_matching_quotes_only() {
        let m = parse_env_file("A=\"quoted\"\nB='single'\nC=\"unbalanced\nD=half\"\nE=\"\"\n");
        assert_eq!(get(&m, "A"), Some("quoted"));
        assert_eq!(get(&m, "B"), Some("single"));
        assert_eq!(get(&m, "C"), Some("\"unbalanced"));
        assert_eq!(get(&m, "D"), Some("half\""));
        assert_eq!(get(&m, "E"), Some(""));
    }

    #[test]
    fn preserves_spaces_inside_quotes() {
        let m = parse_env_file("A=\"  padded  \"\n");
        assert_eq!(get(&m, "A"), Some("  padded  "));
    }

    #[test]
    fn does_not_interpolate_variables() {
        let m = parse_env_file("A=${B}\nB=real\n");
        assert_eq!(get(&m, "A"), Some("${B}"));
        assert_eq!(get(&m, "B"), Some("real"));
    }

    #[test]
    fn does_not_strip_trailing_comments() {
        let m = parse_env_file("A=sk-123#not-a-comment\n");
        assert_eq!(get(&m, "A"), Some("sk-123#not-a-comment"));
    }

    #[test]
    fn later_duplicate_key_wins() {
        let m = parse_env_file("A=first\nA=second\n");
        assert_eq!(get(&m, "A"), Some("second"));
    }

    #[test]
    fn skips_lines_without_equals_and_empty_names() {
        let m = parse_env_file("JUST_A_WORD\n=orphan-value\nA=1\n");
        assert_eq!(m.len(), 1);
        assert_eq!(get(&m, "A"), Some("1"));
    }

    #[test]
    fn handles_crlf_line_endings() {
        let m = parse_env_file("A=1\r\nB=2\r\n");
        assert_eq!(get(&m, "A"), Some("1"));
        assert_eq!(get(&m, "B"), Some("2"));
    }

    #[test]
    fn handles_empty_value_and_empty_file() {
        let m = parse_env_file("A=\n");
        assert_eq!(get(&m, "A"), Some(""));
        assert!(parse_env_file("").is_empty());
    }

    #[test]
    fn provider_defaults_to_none() {
        let cfg = SecretsConfig::default();
        assert_eq!(cfg.provider, SecretsProvider::None);
        assert!(!cfg.provider.uses_file());
        assert!(cfg.file.required);
        assert!(cfg.file_path().is_none());
    }

    #[test]
    fn provider_deserializes_from_yaml() {
        let cfg: SecretsConfig =
            serde_saphyr::from_str("provider: file\nfile:\n  path: /x\n").unwrap();
        assert_eq!(cfg.provider, SecretsProvider::File);
        assert!(cfg.provider.uses_file());
        assert_eq!(cfg.file_path(), Some("/x"));
        assert!(cfg.file.required);
    }

    #[test]
    fn provider_none_and_env_are_distinct_but_both_non_file() {
        let none: SecretsConfig = serde_saphyr::from_str("provider: none\n").unwrap();
        let env: SecretsConfig = serde_saphyr::from_str("provider: env\n").unwrap();
        assert_eq!(none.provider, SecretsProvider::None);
        assert_eq!(env.provider, SecretsProvider::Env);
        assert!(!none.provider.uses_file());
        assert!(!env.provider.uses_file());
    }

    #[test]
    fn unknown_provider_is_rejected() {
        assert!(serde_saphyr::from_str::<SecretsConfig>("provider: vault\n").is_err());
    }

    #[test]
    fn unknown_key_in_secrets_block_is_rejected() {
        assert!(serde_saphyr::from_str::<SecretsConfig>("provider: file\nbogus: 1\n").is_err());
    }

    #[test]
    fn blank_path_is_treated_as_unset() {
        let cfg = SecretsConfig {
            provider: SecretsProvider::File,
            file: SecretsFileConfig {
                path: Some("   ".into()),
                required: true,
            },
        };
        assert!(cfg.file_path().is_none());
    }

    #[test]
    fn read_overlay_parses_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        std::fs::write(&path, "A=1\nB=2\n").unwrap();
        let cfg = SecretsConfig {
            provider: SecretsProvider::File,
            file: SecretsFileConfig {
                path: Some(path.display().to_string()),
                required: true,
            },
        };
        let overlay = cfg.read_overlay(&path).unwrap();
        assert_eq!(get(&overlay, "A"), Some("1"));
        assert_eq!(get(&overlay, "B"), Some("2"));
    }

    #[test]
    fn read_overlay_missing_file_fails_when_required() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.env");
        let cfg = SecretsConfig::default();
        let err = cfg.read_overlay(&path).unwrap_err();
        assert!(matches!(err, ProxyError::Config(_)), "got {err:?}");
    }

    #[test]
    fn read_overlay_missing_file_is_tolerated_when_not_required() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.env");
        let cfg = SecretsConfig {
            provider: SecretsProvider::File,
            file: SecretsFileConfig {
                path: Some(path.display().to_string()),
                required: false,
            },
        };
        let writer = crate::tracing_capture::CaptureWriter::default();
        let (overlay, captured) =
            crate::tracing_capture::capture_tracing(&writer, || cfg.read_overlay(&path));
        assert!(overlay.unwrap().is_empty());
        assert!(
            captured.contains("secrets file unreadable"),
            "expected a warning about the unreadable file, got: {captured}"
        );
    }

    #[test]
    fn malformed_line_is_reported_as_a_warning() {
        let writer = crate::tracing_capture::CaptureWriter::default();
        let (m, captured) = crate::tracing_capture::capture_tracing(&writer, || {
            parse_env_file("NOT_AN_ASSIGNMENT\nA=1\n")
        });
        assert_eq!(m.len(), 1);
        assert!(
            captured.contains("not a NAME=VALUE assignment"),
            "expected a warning about the skipped line, got: {captured}"
        );
    }

    #[test]
    fn read_overlay_empty_file_yields_empty_overlay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.env");
        std::fs::write(&path, "").unwrap();
        let cfg = SecretsConfig::default();
        assert!(cfg.read_overlay(&path).unwrap().is_empty());
    }

    #[test]
    fn null_block_deserializes_to_none() {
        // The `Option` wrapper in `deserialize_secrets_config` is what
        // turns an explicitly-null `secrets:` into "not configured".
        let parsed: Option<SecretsConfig> =
            serde_json::from_value(serde_json::Value::Null).unwrap();
        assert!(parsed.is_none());
    }

    #[test]
    fn deserialize_secrets_config_maps_null_to_default() {
        let cfg = deserialize_secrets_config(serde_json::Value::Null).unwrap();
        assert_eq!(cfg.provider, SecretsProvider::None);
        assert!(cfg.file.required);
    }

    #[test]
    fn secret_string_hides_its_value_in_debug_output() {
        let s = secret_string("sk-not-for-logs");
        let dumped = format!("{s:?}");
        assert!(
            !dumped.contains("sk-not-for-logs"),
            "the secret leaked into the Debug output: {dumped}"
        );
        assert!(dumped.contains("[REDACTED"), "{dumped}");
    }

    #[test]
    fn secret_string_exposes_its_value_on_demand() {
        let s = secret_string("sk-the-real-value");
        assert_eq!(s.expose_secret(), "sk-the-real-value");
    }
}
