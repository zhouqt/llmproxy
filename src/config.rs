use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{ProxyError, Result};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub proxy: ProxyConfig,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            proxy: ProxyConfig::default(),
            providers: Vec::new(),
            models: Vec::new(),
            logging: LoggingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_listen() -> String {
    "127.0.0.1:8080".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyConfig {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderConfig {
    GithubCopilot {
        name: String,
        #[serde(default = "default_vscode_version")]
        vscode_version: String,
        #[serde(default = "default_account_type")]
        account_type: String,
        /// Optional map: incoming-model-name → upstream-model-name.
        /// Empty (default) accepts any model name verbatim (the proxy
        /// forwards the Anthropic `model` value as-is to Copilot); a
        /// non-empty table is an explicit allow-list (the router skips
        /// this provider for unmapped names), with the upstream name
        /// used when the request is dispatched.
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        /// Whether this provider should route its outbound requests
        /// through the global SOCKS/HTTP proxy. Defaults to `false`
        /// because most providers in the example config work fine
        /// without it; enable per-provider when the upstream is
        /// regionally blocked.
        #[serde(default)]
        use_proxy: bool,
    },
    /// Native Anthropic Messages passthrough provider. Sends the
    /// Anthropic request body verbatim to `{api_base}/messages` and
    /// streams the response unchanged. Used for OpenRouter's native
    /// `/v1/messages` endpoint and any other gateway that speaks
    /// Anthropic Messages without translation.
    Anthropic {
        name: String,
        api_key: String,
        #[serde(default = "default_api_base_anthropic")]
        api_base: String,
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        /// Whether this provider should route its outbound requests
        /// through the global SOCKS/HTTP proxy. Defaults to `false`.
        #[serde(default)]
        use_proxy: bool,
    },
    #[serde(rename = "openai_compat")]
    OpenaiCompat {
        name: String,
        api_key: String,
        api_base: String,
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        #[serde(default)]
        use_proxy: bool,
    },
    /// OpenAI Responses API passthrough provider. Sends an
    /// Anthropic-converted request to `{api_base}/responses` and
    /// translates the response back. Use for upstreams that expose
    /// `/v1/responses` (OpenAI GPT-5.x, direct OpenAI reverse proxies,
    /// etc.). For Chat-Completions-style backends use
    /// `openai_compat` instead.
    #[serde(rename = "openai_responses")]
    OpenaiResponses {
        name: String,
        api_key: String,
        api_base: String,
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        #[serde(default)]
        use_proxy: bool,
    },
}

fn default_vscode_version() -> String {
    "1.95.0".to_string()
}

fn default_account_type() -> String {
    "individual".to_string()
}

fn default_api_base_anthropic() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

impl ProviderConfig {
    pub fn name(&self) -> &str {
        match self {
            ProviderConfig::GithubCopilot { name, .. } => name,
            ProviderConfig::Anthropic { name, .. } => name,
            ProviderConfig::OpenaiCompat { name, .. } => name,
            ProviderConfig::OpenaiResponses { name, .. } => name,
        }
    }

    /// Whether this provider should route its outbound requests through
    /// the global proxy. See the field doc-comment on each variant.
    pub fn use_proxy(&self) -> bool {
        match self {
            ProviderConfig::GithubCopilot { use_proxy, .. }
            | ProviderConfig::Anthropic { use_proxy, .. }
            | ProviderConfig::OpenaiCompat { use_proxy, .. }
            | ProviderConfig::OpenaiResponses { use_proxy, .. } => *use_proxy,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub primary: String,
    #[serde(default)]
    pub fallback_chain: Vec<String>,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries_per_provider: u32,
    #[serde(default = "default_max_retries")]
    pub max_retries_total: u32,
}

fn default_cooldown_seconds() -> u64 {
    300
}

fn default_max_retries() -> u32 {
    2
}

impl ModelConfig {
    pub fn chain(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.primary.as_str()).chain(self.fallback_chain.iter().map(String::as_str))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "pretty".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ProxyError::Config(format!("read {}: {e}", path.as_ref().display())))?;
        Self::parse(&raw)
    }

    pub fn parse(raw: &str) -> Result<Self> {
        let expanded = expand_env_vars(raw);
        let cfg: Config = serde_yaml::from_str(&expanded)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.providers.is_empty() {
            return Err(ProxyError::Config("at least one provider required".into()));
        }
        if self.models.is_empty() {
            return Err(ProxyError::Config("at least one model required".into()));
        }

        let provider_names: std::collections::HashSet<&str> =
            self.providers.iter().map(|p| p.name()).collect();

        for m in &self.models {
            if !provider_names.contains(m.primary.as_str()) {
                return Err(ProxyError::Config(format!(
                    "model '{}' primary '{}' not in providers list",
                    m.name, m.primary
                )));
            }
            for fb in &m.fallback_chain {
                if !provider_names.contains(fb.as_str()) {
                    return Err(ProxyError::Config(format!(
                        "model '{}' fallback '{}' not in providers list",
                        m.name, fb
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn find_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.name == name)
    }

    pub fn find_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name() == name)
    }
}

/// Expand ${VAR} / $VAR references in a string using process environment.
///
/// Supports bash-style `${VAR:-default}` for fallback values: when VAR is
/// unset or empty, the literal `default` is emitted (no `env var ... not set`
/// warning is logged). `default` may be wrapped in matching single or double
/// quotes (`${VAR:-"hello world"}`, `${VAR:-'hello world'}`); the wrapping
/// characters are stripped. Unmatched quotes are kept verbatim.
fn expand_env_vars(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'{' {
                if let Some(end) = input[i + 2..].find('}') {
                    let inner = &input[i + 2..i + 2 + end];
                    // Bash-style ${VAR:-default} falls back when VAR is
                    // unset OR empty; ${VAR-default} only when unset.
                    // We support the colon-prefixed form because that is
                    // what template literals and .env style configs reach
                    // for.
                    let (var_name, default) = match inner.split_once(":-") {
                        Some((v, d)) => (v, Some(strip_outer_quotes(d))),
                        None => (inner, None),
                    };

                    match std::env::var(var_name) {
                        Ok(val) if !val.is_empty() => {
                            out.push_str(&val);
                        }
                        Ok(_) => {
                            // Set but empty. Treat as unset for the
                            // purposes of `:-`. With no default, the
                            // original code emitted empty silently.
                            if let Some(d) = default {
                                out.push_str(d);
                            }
                        }
                        Err(_) => match default {
                            Some(d) => out.push_str(d),
                            None => {
                                tracing::warn!("env var {} not set", var_name);
                            }
                        },
                    }
                    i += 3 + end;
                    continue;
                }
            } else if bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_' {
                let start = i + 1;
                let mut end = start;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                let var = &input[start..end];
                if let Ok(val) = std::env::var(var) {
                    out.push_str(&val);
                } else {
                    tracing::warn!("env var {} not set", var);
                }
                i = end;
                continue;
            }
        }
        let ch = input[i..].chars().next().expect("valid UTF-8 boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Strip matching outer single or double quotes from `s`. Unmatched
/// quotes are preserved verbatim. Intended for the `default` half of
/// `${VAR:-default}` so users can write `${VAR:-"hello world"}` and
/// get `hello world`, mirroring shell behaviour.
fn strip_outer_quotes(s: &str) -> &str {
    let trimmed = s.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &trimmed[1..trimmed.len() - 1];
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_expansion() {
        std::env::set_var("LLMPROXY_TEST_VAR", "hello");
        let expanded = expand_env_vars("key=${LLMPROXY_TEST_VAR}-end");
        assert_eq!(expanded, "key=hello-end");
    }

    #[test]
    fn missing_env_var_expands_to_empty() {
        std::env::remove_var("LLMPROXY_DEFINITELY_NOT_SET");
        let expanded = expand_env_vars("foo=${LLMPROXY_DEFINITELY_NOT_SET}bar");
        assert_eq!(expanded, "foo=bar");
    }

    #[test]
    fn config_parses_minimal() {
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.models.len(), 1);
        assert_eq!(cfg.server.listen, "127.0.0.1:8080");
    }

    #[test]
    fn config_rejects_unknown_provider() {
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: openrouter
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(format!("{err}").contains("not in providers"));
    }

    #[test]
    fn env_expansion_supports_unbraced_names_and_unicode() {
        std::env::set_var("LLMPROXY_UNBRACED_VAR", "值");

        let expanded = expand_env_vars("前缀-$LLMPROXY_UNBRACED_VAR-后缀");

        assert_eq!(expanded, "前缀-值-后缀");
        assert_eq!(expand_env_vars("literal-$9-${UNCLOSED"), "literal-$9-${UNCLOSED");
    }

    #[test]
    fn env_expansion_bash_default_unset_var_uses_default() {
        std::env::remove_var("LLMPROXY_DEFINITELY_NOT_SET");
        let expanded =
            expand_env_vars("key=${LLMPROXY_DEFINITELY_NOT_SET:-fallback}-end");
        assert_eq!(expanded, "key=fallback-end");
    }

    #[test]
    fn env_expansion_bash_default_set_var_wins_over_default() {
        std::env::set_var("LLMPROXY_BASH_DEFAULT_SET", "actual");
        let expanded = expand_env_vars("key=${LLMPROXY_BASH_DEFAULT_SET:-fallback}");
        assert_eq!(expanded, "key=actual");
    }

    #[test]
    fn env_expansion_bash_default_empty_var_uses_default() {
        // Bash `:-` treats empty as null. Match that behaviour.
        std::env::set_var("LLMPROXY_BASH_DEFAULT_EMPTY", "");
        let expanded = expand_env_vars("key=${LLMPROXY_BASH_DEFAULT_EMPTY:-fallback}");
        assert_eq!(expanded, "key=fallback");
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_EMPTY");
    }

    #[test]
    fn env_expansion_bash_default_strips_double_quotes() {
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_QUOTED");
        let expanded =
            expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_QUOTED:-\"hello world\"}-e");
        assert_eq!(expanded, "k=hello world-e");
    }

    #[test]
    fn env_expansion_bash_default_strips_single_quotes() {
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_SQUOTED");
        let expanded =
            expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_SQUOTED:-'hello world'}-e");
        assert_eq!(expanded, "k=hello world-e");
    }

    #[test]
    fn env_expansion_bash_default_unmatched_quote_passes_through() {
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_UNMATCHED");
        // Bare brace without quotes — default is literal.
        let expanded = expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_UNMATCHED:-hello}");
        assert_eq!(expanded, "k=hello");
    }

    #[test]
    fn env_expansion_bash_default_empty_default_substitutes_empty() {
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_EMPTY_D");
        let expanded = expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_EMPTY_D:-}-e");
        assert_eq!(expanded, "k=-e");
    }

    #[test]
    fn validation_requires_providers_and_models() {
        let no_providers = Config::default().validate().unwrap_err();
        assert!(no_providers.to_string().contains("at least one provider"));

        let no_models = Config {
            providers: vec![ProviderConfig::GithubCopilot {
                name: "copilot".to_string(),
                vscode_version: default_vscode_version(),
                account_type: default_account_type(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            }],
            ..Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(no_models.to_string().contains("at least one model"));
    }

    #[test]
    fn validation_rejects_unknown_fallback() {
        let config = Config {
            providers: vec![ProviderConfig::GithubCopilot {
                name: "copilot".to_string(),
                vscode_version: default_vscode_version(),
                account_type: default_account_type(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            }],
            models: vec![ModelConfig {
                name: "model".to_string(),
                primary: "copilot".to_string(),
                fallback_chain: vec!["missing".to_string()],
                cooldown_seconds: 300,
                max_retries_per_provider: 2,
                max_retries_total: 2,
            }],
            ..Config::default()
        };

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains("fallback 'missing'"));
    }

    #[test]
    fn provider_defaults_names_lookup_and_chain_work() {
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
  - name: router
    type: anthropic
    api_key: key
  - name: compat
    type: openai_compat
    api_key: key
    api_base: https://example.test/v1
models:
  - name: model
    primary: copilot
    fallback_chain: [router, compat]
"#;

        let config = Config::parse(raw).unwrap();

        assert_eq!(config.providers[0].name(), "copilot");
        assert_eq!(config.providers[1].name(), "router");
        assert_eq!(config.providers[2].name(), "compat");
        match (&config.providers[0], &config.providers[1]) {
            (
                ProviderConfig::GithubCopilot {
                    vscode_version,
                    account_type,
                    ..
                },
                ProviderConfig::Anthropic { api_base, .. },
            ) => {
                assert_eq!(vscode_version, "1.95.0");
                assert_eq!(account_type, "individual");
                assert_eq!(api_base, "https://openrouter.ai/api/v1");
            }
            _ => panic!("expected [copilot, anthropic] providers"),
        }
        assert!(config.find_provider("router").is_some());
        assert!(config.find_provider("missing").is_none());
        assert!(config.find_model("model").is_some());
        assert!(config.find_model("missing").is_none());
        assert_eq!(
            config.models[0].chain().collect::<Vec<_>>(),
            vec!["copilot", "router", "compat"]
        );
    }

    #[test]
    fn load_reads_file_and_reports_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            "providers:\n  - name: p\n    type: github_copilot\nmodels:\n  - name: m\n    primary: p\n",
        )
        .unwrap();

        let config = Config::load(&path).unwrap();
        assert_eq!(config.models[0].name, "m");

        let missing = Config::load(dir.path().join("missing.yaml")).unwrap_err();
        assert!(missing.to_string().contains("read"));
        assert!(missing.to_string().contains("missing.yaml"));
    }

    #[test]
    fn parse_rejects_unknown_fields_and_invalid_yaml() {
        let unknown = Config::parse("unknown: true").unwrap_err();
        assert!(matches!(unknown, ProxyError::Yaml(_)));

        let invalid = Config::parse("providers: [").unwrap_err();
        assert!(matches!(invalid, ProxyError::Yaml(_)));
    }

    #[test]
    fn openai_responses_variant_parses_yaml_with_defaults() {
        // The new Responses API provider type must parse a minimal
        // YAML block — model_rewrite and use_proxy have defaults so
        // only the required fields (name, api_key, api_base) are
        // strictly necessary.
        let raw = r#"
providers:
  - name: openai_direct
    type: openai_responses
    api_key: "${OPENAI_API_KEY}"
    api_base: "https://api.openai.com/v1"
models:
  - name: gpt-5
    primary: openai_direct
"#;
        let cfg = Config::parse(raw).unwrap();
        match &cfg.providers[0] {
            ProviderConfig::OpenaiResponses {
                name,
                api_key,
                api_base,
                model_rewrite,
                use_proxy,
            } => {
                assert_eq!(name, "openai_direct");
                assert_eq!(api_base, "https://api.openai.com/v1");
                // api_key is the env-expanded value of ${OPENAI_API_KEY}
                // which is unset in test, so it expands to empty.
                assert_eq!(api_key, "");
                assert!(model_rewrite.is_empty());
                assert!(!use_proxy);
            }
            other => panic!("expected OpenaiResponses, got {other:?}"),
        }
    }

    #[test]
    fn openai_responses_variant_picks_up_model_rewrite_and_use_proxy() {
        // Non-default fields must survive the YAML round-trip and be
        // reachable via the accessor helpers.
        let raw = r#"
providers:
  - name: openai_direct
    type: openai_responses
    api_key: "k"
    api_base: "https://api.openai.com/v1"
    model_rewrite:
      "claude-sonnet-4.6": "gpt-5"
      "claude-opus-4.6": "gpt-5"
    use_proxy: true
models:
  - name: gpt-5
    primary: openai_direct
"#;
        let cfg = Config::parse(raw).unwrap();
        let p = &cfg.providers[0];
        assert_eq!(p.name(), "openai_direct");
        assert!(p.use_proxy());
        match p {
            ProviderConfig::OpenaiResponses {
                model_rewrite, ..
            } => {
                assert_eq!(model_rewrite.len(), 2);
                assert_eq!(model_rewrite.get("claude-sonnet-4.6").unwrap(), "gpt-5");
                assert_eq!(model_rewrite.get("claude-opus-4.6").unwrap(), "gpt-5");
            }
            _ => panic!("expected OpenaiResponses"),
        }
    }

    #[test]
    fn openai_responses_rejects_unknown_fields() {
        // deny_unknown_fields is on every variant; an unrecognized
        // key on openai_responses must surface as a YAML error.
        let raw = r#"
providers:
  - name: openai_direct
    type: openai_responses
    api_key: "k"
    api_base: "https://api.openai.com/v1"
    bogus_field: true
models:
  - name: gpt-5
    primary: openai_direct
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(matches!(err, ProxyError::Yaml(_)), "got: {err:?}");
    }
}
