use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{ProxyError, Result};
use crate::secrets::{deserialize_secrets_config, SecretsConfig};
use crate::usage::DEFAULT_USAGE_CAPACITY;

/// Placeholder written instead of a credential whenever a config is
/// serialized. `secrecy` deliberately does not implement `Serialize`
/// for `Secret<String>` (it would be an exfiltration channel), so the
/// redaction has to happen here — a serialized config is meant for
/// inspection, never for reuse.
const REDACTED: &str = "[REDACTED]";

/// Serialize a credential as [`REDACTED`].
fn serialize_redacted_secret<S: serde::Serializer>(
    _secret: &SecretString,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.serialize_str(REDACTED)
}

/// Serialize an optional credential: `null` stays `null`, a value
/// becomes [`REDACTED`].
fn serialize_redacted_secret_option<S: serde::Serializer>(
    secret: &Option<SecretString>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    match secret {
        Some(_) => serializer.serialize_str(REDACTED),
        None => serializer.serialize_none(),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub proxy: ProxyConfig,
    /// User-Agent presented to upstream LLM providers on outbound
    /// requests (all providers sharing the client pools in
    /// `proxy_client.rs`). Defaults to simulating the Claude Code
    /// client identity (`claude-cli/<ver> (...)` — see
    /// `default_user_agent`) so gateways classify the proxy's traffic
    /// as coming from Claude Code instead of an unknown client.
    /// Copilot is unaffected — it overrides this header per-request with
    /// `GitHubCopilotChat/<version>`. A top-level setting because it
    /// describes the client identity, not the `proxy:` outbound HTTP
    /// proxy.
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    /// Capacity of the in-memory ring buffer that backs `/admin/usage`.
    /// When the buffer is full, the oldest record is evicted (FIFO).
    /// **Set to `0` to disable the feature entirely** (`UsageStats::new`
    /// treats `0` as a no-op store — no allocation, `record` is a no-op,
    /// `snapshot` returns empty). Any positive value is hard-clamped at
    /// `usage::MAX_USAGE_CAPACITY` (one million rows) so a
    /// misconfiguration can't pre-allocate gigabytes. Default:
    /// `usage::DEFAULT_USAGE_CAPACITY` (100_000).
    #[serde(default = "default_usage_capacity")]
    pub usage_capacity: usize,
    /// Where `${VAR}` placeholders resolve from.
    ///
    /// Absent (the default) means "process environment only" — the
    /// pre-existing behavior, unchanged. `provider: file` adds a
    /// `.env`-format file as the **highest-priority** source, which is
    /// how the container deployment feeds the proxy keys that its
    /// entrypoint unsealed from a TPM sealed object. See
    /// `plans/feat-vault-secret-backend.md` §5.5.
    #[serde(default, deserialize_with = "deserialize_secrets_config")]
    pub secrets: SecretsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            proxy: ProxyConfig::default(),
            user_agent: default_user_agent(),
            providers: Vec::new(),
            models: Vec::new(),
            usage_capacity: default_usage_capacity(),
            secrets: SecretsConfig::default(),
        }
    }
}

fn default_usage_capacity() -> usize {
    DEFAULT_USAGE_CAPACITY
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    #[serde(default)]
    #[serde(serialize_with = "serialize_redacted_secret_option")]
    pub api_key: Option<SecretString>,
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

/// Outbound User-Agent default. Mirrors what the real Claude Code
/// client reports when driven as an external agent — the `claude-cli/`
/// format the binary emits for agent-sdk / vscode integrations
/// (`claude-cli/${VERSION} (external, ${CLAUDE_CODE_ENTRYPOINT}, ...)`,
/// extracted from its embedded JS — see
/// plans/simulate-claude-code-identity.md). Providers classify by this
/// exact string; the bare `claude-code/<ver>` CLI format is NOT
/// recognized (observed on OpenRouter). The value tracks the operator's
/// actual Claude Code identity and is overridable via the top-level
/// `user_agent` config.
pub fn default_user_agent() -> String {
    "claude-cli/2.1.224 (external, claude-vscode, agent-sdk/0.3.224)".to_string()
}

/// What to do when an outbound Anthropic request carries no `thinking`
/// field. See `ProviderConfig::Anthropic::thinking_when_absent`.
///
/// Deliberately an enum rather than a bool: each variant names the exact
/// wire change it makes, and an operator opting out needs the config line
/// to name that change. `disabled` is the default — the feature exists
/// precisely because omission lets compat backends reason by default —
/// and `omit` is the explicit opt-out that keeps the outbound wire
/// byte-identical to the pre-feature behavior. `effort_low` covers the
/// third case: backends that force thinking on (rejecting
/// `thinking:{"type":"disabled"}`) but expose an effort knob.
///
/// OpenAI-family upstreams need a different vocabulary entirely
/// (`reasoning_effort:"none"` vs. `thinking:{type:"disabled"}`, and
/// `none` is rejected outright by o-series / gpt-5 / Kimi K3). That
/// separate enum is deferred along with the `openai_compat` /
/// `openai_responses` phases of
/// `plans/feat-disable-thinking-when-absent.md` §3.7 — do not widen this
/// one to cover them; the legal value sets differ per provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingWhenAbsent {
    /// Leave the request body untouched. Opt-out for operators who need
    /// the outbound wire to stay byte-identical to the pre-feature
    /// behavior.
    Omit,
    /// Inject `{"thinking": {"type": "disabled"}}`. Default: the feature
    /// is on for every anthropic provider unless explicitly disabled.
    /// Only the *absence* (or explicit null) of `thinking` triggers it;
    /// client-sent `enabled`/`adaptive`/`disabled` pass through.
    #[default]
    Disabled,
    /// Send no `thinking` field at all; instead inject
    /// `{"output_config": {"effort": "low"}}` when the request carries
    /// neither `thinking` nor `output_config` (or both are explicit
    /// null). For upstreams that force thinking on at their default
    /// effort — typically `max` — and reject
    /// `thinking:{"type":"disabled"}` outright, but honor
    /// `output_config.effort` (e.g. MiniMax M3.1-Flash, which 400s on
    /// `disabled` while accepting `low`/`medium`/`high`/`xhigh`/`max`).
    /// Client-sent `thinking` or `output_config` passes through
    /// untouched — the trigger is the absence of both.
    EffortLow,
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
        #[serde(serialize_with = "serialize_redacted_secret")]
        api_key: SecretString,
        #[serde(default = "default_api_base_anthropic")]
        api_base: String,
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        /// Whether this provider should route its outbound requests
        /// through the global SOCKS/HTTP proxy. Defaults to `false`.
        #[serde(default)]
        use_proxy: bool,
        /// 要排除的 OpenRouter 后端提供商 slug 列表（如 `["Inceptron"]`）。
        /// 非空时，请求体会携带 `provider: {ignore: [...]}` 字段，
        /// 让 OpenRouter 跳过这些提供商。仅当 `api_base` 指向
        /// `openrouter.ai` 时生效；其他 Anthropic 兼容后端（如
        /// DeepSeek、Minimax、阿里云百炼）会忽略该配置并产生一条
        /// 启动警告，因为严格的 Anthropic 兼容后端会对未知顶层
        /// 字段返回 400。
        /// 空列表（默认）不注入任何 `provider` 字段。
        #[serde(default)]
        provider_ignore: Vec<String>,
        /// OpenRouter backend provider slugs to RESTRICT routing to.
        /// Non-empty → the request body carries `provider: {only: [...]}`,
        /// which OpenRouter treats as a hard ceiling: the request is only
        /// routed to providers in the list, and if none of them can serve
        /// the model it fails with 503 ("No available model provider meets
        /// your routing requirements").
        ///
        /// Only active when `api_base` points at `openrouter.ai`; other
        /// Anthropic-compatible backends ignore the config and emit a
        /// startup warning, because strict backends 400 on unknown
        /// top-level fields.
        ///
        /// Slugs are passed through verbatim — exact, operator-supplied
        /// strings, no normalization or validation. Note that a base slug
        /// (e.g. `deepinfra`) matches every endpoint of that provider,
        /// including variants like `deepinfra/turbo`; use a suffixed slug
        /// to pin a single endpoint.
        ///
        /// When both `provider_allow` and `provider_ignore` are set, both
        /// sub-fields are forwarded verbatim in the same `provider` object;
        /// the proxy does not compute the intersection locally.
        /// Empty list (default) injects no `provider.only` key.
        #[serde(default)]
        provider_allow: Vec<String>,
        /// Inject `{"thinking": {"type": "disabled"}}` into the request
        /// body when the outbound request has **no** `thinking` field
        /// (or an explicit `null`). This suppresses third-party
        /// Anthropic-compatible backends that read an omitted `thinking`
        /// as "reasoning ON by default". Typical case: Claude Code Auto
        /// Mode's non-streaming classifier request — it carries neither
        /// `thinking` nor `output_config`, so the upstream used to spend
        /// a whole thinking pass on it. See
        /// plans/feat-disable-thinking-when-absent.md.
        ///
        /// The trigger is strictly *field absence or null*: an explicit
        /// client-sent `enabled` / `adaptive` / `disabled` always passes
        /// through untouched. Note that Claude Code's main request sends
        /// `adaptive` (not `enabled`) — rewriting that to `disabled`
        /// would switch off the main model's reasoning.
        ///
        /// Default is `disabled` (the feature is on without any
        /// per-provider configuration). Set `omit` explicitly when the
        /// outbound wire must stay byte-identical to the pre-feature
        /// behavior. On the real Anthropic API `disabled` is a
        /// first-class variant equivalent to omitting the field, so the
        /// injection is a no-op there; the benefit comes entirely from
        /// compat backends.
        ///
        /// Harmless on the real Anthropic API, but an **unknown**
        /// third-party backend could still 400 on the field, so
        /// `AnthropicProvider` has a fallback: if the upstream 400s and
        /// the body blames the `thinking` parameter, the field is
        /// removed and the request is retried once.
        #[serde(default)]
        thinking_when_absent: ThinkingWhenAbsent,
    },
    #[serde(rename = "openai_compat")]
    OpenaiCompat {
        name: String,
        #[serde(serialize_with = "serialize_redacted_secret")]
        api_key: SecretString,
        api_base: String,
        #[serde(default)]
        model_rewrite: HashMap<String, String>,
        #[serde(default)]
        use_proxy: bool,
        /// 要排除的 OpenRouter 后端提供商 slug 列表（如 `["Azure"]`）。
        /// 非空时，请求体会携带 `provider: {ignore: [...]}` 字段，
        /// 让 OpenRouter 跳过这些提供商。仅当 `api_base` 指向
        /// `openrouter.ai` 时生效；其他 OpenAI-compat 后端会忽略该
        /// 配置并产生一条启动警告（因为严格校验的 OpenAI-compat 后端
        /// 如 DeepSeek 会对未知顶层字段返回 400）。
        /// 空列表（默认）不注入任何 `provider` 字段。
        #[serde(default)]
        provider_ignore: Vec<String>,
        /// OpenRouter backend provider slugs to RESTRICT routing to.
        /// Non-empty → the request body carries `provider: {only: [...]}`,
        /// which OpenRouter treats as a hard ceiling: the request is only
        /// routed to providers in the list, and if none of them can serve
        /// the model it fails with 503 ("No available model provider meets
        /// your routing requirements").
        ///
        /// Only active when `api_base` points at `openrouter.ai`; other
        /// OpenAI-compat backends ignore the config and emit a startup
        /// warning, because strict backends such as DeepSeek 400 on
        /// unknown top-level fields.
        ///
        /// Slugs are passed through verbatim — exact, operator-supplied
        /// strings, no normalization or validation. Note that a base slug
        /// (e.g. `deepinfra`) matches every endpoint of that provider,
        /// including variants like `deepinfra/turbo`; use a suffixed slug
        /// to pin a single endpoint.
        ///
        /// When both `provider_allow` and `provider_ignore` are set, both
        /// sub-fields are forwarded verbatim in the same `provider` object;
        /// the proxy does not compute the intersection locally.
        /// Empty list (default) injects no `provider.only` key.
        #[serde(default)]
        provider_allow: Vec<String>,
        /// 当为 `true` 时，thinking 请求里每条缺失 `reasoning_content`
        /// 的历史 assistant 消息（跨模型 thinking 轮 / redacted_thinking /
        /// 纯文本轮）以 `reasoning_content: ""` 发送，使 DeepSeek-V4 类
        /// 上游在 thinking 模式下接受而非 400。默认 `false`（严格按
        /// OpenAI spec 的上游可能拒绝该非标准字段，其它 provider wire
        /// 字节不变）。
        #[serde(default)]
        reasoning_echo: bool,
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
        #[serde(serialize_with = "serialize_redacted_secret")]
        api_key: SecretString,
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

    /// Stable machine-readable type label for this provider — the same
    /// string the serde `type` tag (`src/config.rs:63`) deserializes into,
    /// so the status endpoint and any consumer can identify provider kind
    /// without re-deriving it from the enum. See
    /// `plans/provider-status-endpoint.md`.
    pub fn type_label(&self) -> &'static str {
        match self {
            ProviderConfig::GithubCopilot { .. } => "github_copilot",
            ProviderConfig::Anthropic { .. } => "anthropic",
            ProviderConfig::OpenaiCompat { .. } => "openai_compat",
            ProviderConfig::OpenaiResponses { .. } => "openai_responses",
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

    /// The provider's configured model_rewrite table (client-facing name →
    /// upstream name). Empty = pass any name verbatim; non-empty = explicit
    /// allow-list. Used by the `/admin/models` endpoint to enumerate the
    /// statically-configured upstream models without contacting upstreams.
    pub fn model_rewrite(&self) -> &HashMap<String, String> {
        match self {
            ProviderConfig::GithubCopilot { model_rewrite, .. }
            | ProviderConfig::Anthropic { model_rewrite, .. }
            | ProviderConfig::OpenaiCompat { model_rewrite, .. }
            | ProviderConfig::OpenaiResponses { model_rewrite, .. } => model_rewrite,
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

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ProxyError::Config(format!("read {}: {e}", path.as_ref().display())))?;
        Self::parse(&raw)
    }

    /// Parse a config from YAML text.
    ///
    /// Two phases, because the `secrets:` block decides *where* the
    /// `${VAR}` values come from and the values decide the rest of the
    /// config:
    ///
    /// 1. pick the `secrets:` block out of the raw text and load the
    ///    overlay (an empty map unless `provider: file`);
    /// 2. expand `${VAR}` against **overlay first, process environment
    ///    second**, then deserialize and validate.
    ///
    /// With no `secrets:` block the overlay is empty and phase 2
    /// behaves exactly as it did before the overlay existed.
    pub fn parse(raw: &str) -> Result<Self> {
        let overlay = load_secrets_overlay(raw)?;
        let expanded = expand_env_vars_with_overlay(raw, &overlay);
        let cfg: Config = serde_yaml::from_str(&expanded)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        // An api_key that expanded to "" is not "no auth configured" --
        // it is auth configured with an empty token, which any client
        // can satisfy with a bare `Authorization: Bearer `. That is
        // exactly what a `${VAR}` pointing at a missing secret looks
        // like, so refuse it instead of starting in a weak state.
        if let Some(key) = &self.server.api_key {
            if key.expose_secret().is_empty() {
                return Err(ProxyError::Config(
                    "server.api_key is empty after ${VAR} expansion; remove the field to disable auth, or point the variable at a value"
                        .into(),
                ));
            }
        } else {
            // No api_key at all → the proxy serves every request without
            // authentication (see `auth::require_auth`). That is a
            // deliberate design choice, but it used to be *silent*, so an
            // operator who dropped the key (or never set one) could
            // publish the port and expose every configured provider key
            // without noticing.
            //
            // There is deliberately NO listen-address based refusal here:
            // the proxy runs containerized, where it must bind a
            // non-loopback address or published ports can never reach it.
            // The container-side `listen` is not a trust boundary — the
            // host-side port publication and the api_key are. The proxy
            // cannot observe the former from inside the container, so the
            // only thing it can do is make the weak state loud.
            tracing::warn!(
                "authentication is DISABLED (no server.api_key); any client that can reach the port can use every configured provider key"
            );
        }
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

/// Load the `${VAR}` overlay described by the config's `secrets:`
/// block.
///
/// Returns an empty map unless `secrets.provider` is `file` — the
/// common case, where the process environment remains the only source
/// and the expansion behavior is exactly what it always was.
fn load_secrets_overlay(raw: &str) -> Result<HashMap<String, String>> {
    let secrets_cfg = extract_secrets_config(raw)?;
    if !secrets_cfg.provider.uses_file() {
        return Ok(HashMap::new());
    }
    let Some(path) = secrets_cfg.file_path() else {
        return Err(ProxyError::Config(
            "secrets.provider is 'file' but secrets.file.path is not set".into(),
        ));
    };
    // The path itself may carry a `${VAR}` reference, resolved from
    // the process environment only: the overlay cannot describe where
    // the overlay lives.
    let path = expand_env_vars(path);
    secrets_cfg.read_overlay(Path::new(&path))
}

/// Pick the `secrets:` block out of the raw config text.
///
/// Deliberately lenient about everything *except* the block itself:
/// the raw text is parsed into a generic YAML value so that a
/// type error elsewhere in the config (say `max_retries: ${N}` where
/// `N` is not set) is still reported by the real parse in
/// `Config::parse`, with the message the operator expects. Only the
/// `secrets:` block is type-checked here.
fn extract_secrets_config(raw: &str) -> Result<SecretsConfig> {
    // A YAML syntax error is reported verbatim by the main parse; this
    // phase must not turn it into a different error.
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(raw) else {
        return Ok(SecretsConfig::default());
    };
    let Some(block) = value.get("secrets") else {
        return Ok(SecretsConfig::default());
    };
    if block.is_null() {
        return Ok(SecretsConfig::default());
    }
    serde_yaml::from_value::<SecretsConfig>(block.clone())
        .map_err(|e| ProxyError::Config(format!("secrets: {e}")))
}

/// Expand ${VAR} / $VAR references in a string using process environment.
///
/// Supports bash-style `${VAR:-default}` for fallback values: when VAR is
/// unset or empty, the literal `default` is emitted (no `env var ... not set`
/// warning is logged). `default` may be wrapped in matching single or double
/// quotes (`${VAR:-"hello world"}`, `${VAR:-'hello world'}`); the wrapping
/// characters are stripped. Unmatched quotes are kept verbatim.
fn expand_env_vars(input: &str) -> String {
    expand_env_vars_with_overlay(input, &HashMap::new())
}

/// Expand ${VAR} / $VAR references in a string.
///
/// Resolution order is **overlay first, process environment second**,
/// then the `${VAR:-default}` fallback, then empty. The overlay wins
/// because it is the current source of truth: an environment variable
/// left over from the previous deployment must never silently
/// override a key that the secret file just supplied.
///
/// The `${VAR:-default}` rules are the ones documented on
/// [`expand_env_vars`], which is this function with an empty overlay.
fn expand_env_vars_with_overlay(input: &str, overlay: &HashMap<String, String>) -> String {
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

                    match lookup_var(var_name, overlay) {
                        Some(val) if !val.is_empty() => {
                            out.push_str(&val);
                        }
                        Some(_) => {
                            // Set but empty. Treat as unset for the
                            // purposes of `:-`. With no default, the
                            // original code emitted empty silently.
                            if let Some(d) = default {
                                out.push_str(d);
                            }
                        }
                        None => match default {
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
                match lookup_var(var, overlay) {
                    Some(val) => {
                        out.push_str(&val);
                    }
                    None => {
                        tracing::warn!("env var {} not set", var);
                    }
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

/// Resolve `name` for expansion, overlay first.
///
/// Returns `None` when the name is set nowhere; `Some(value)` when it
/// is — possibly the empty string, which the `${VAR:-default}` form
/// treats as "unset" exactly like an empty environment variable does.
///
/// The source is logged, never the value: an operator needs to know
/// *where* a key came from, and the key itself is the secret.
fn lookup_var(name: &str, overlay: &HashMap<String, String>) -> Option<String> {
    if let Some(val) = overlay.get(name) {
        tracing::info!(
            var = name,
            source = "file",
            "secret resolved from secrets file"
        );
        return Some(val.clone());
    }
    match std::env::var(name) {
        Ok(val) => {
            tracing::debug!(
                var = name,
                source = "env",
                "secret resolved from environment"
            );
            Some(val)
        }
        Err(_) => None,
    }
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
    use crate::expect_variant;
    use crate::secrets::{secret_string, SecretsProvider};

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
    fn user_agent_defaults_to_claude_code() {
        // The whole point of the identity simulation: with no config
        // override, outbound requests must present the Claude Code UA so
        // providers don't label the proxy "Unknown". Both the manual
        // Default impl and the serde default for an absent `user_agent:`
        // key must yield it.
        assert_eq!(
            Config::default().user_agent,
            "claude-cli/2.1.224 (external, claude-vscode, agent-sdk/0.3.224)"
        );

        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert_eq!(
            cfg.user_agent,
            "claude-cli/2.1.224 (external, claude-vscode, agent-sdk/0.3.224)"
        );
    }

    #[test]
    fn user_agent_parses_from_config() {
        let raw = r#"
user_agent: "custom-agent/9.9"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert_eq!(cfg.user_agent, "custom-agent/9.9");
    }

    #[test]
    fn user_agent_rejects_unknown_fields() {
        // deny_unknown_fields is on Config; a typo'd key must surface as
        // a YAML error rather than being silently ignored.
        let raw = r#"
user_agentt: "typo"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(matches!(err, ProxyError::Yaml(_)), "got: {err:?}");
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
    fn validation_rejects_an_api_key_that_expanded_to_empty() {
        std::env::remove_var("LLMPROXY_TEST_UNSET_SERVER_KEY");
        let raw = r#"
server:
  api_key: "${LLMPROXY_TEST_UNSET_SERVER_KEY}"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: model
    primary: copilot
"#;

        let error = Config::parse(raw).unwrap_err();

        assert!(
            error.to_string().contains("server.api_key is empty"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn validation_accepts_an_absent_api_key() {
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: model
    primary: copilot
"#;

        let cfg = Config::parse(raw).unwrap();

        assert!(cfg.server.api_key.is_none());
    }

    #[test]
    fn validation_warns_loudly_when_auth_is_disabled() {
        // A missing api_key is a supported configuration, but it must not
        // be silent: the proxy then serves every provider key to whoever
        // can reach the port.
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: model
    primary: copilot
"#;

        let writer = crate::tracing_capture::CaptureWriter::new();
        let (_, captured) = crate::tracing_capture::capture_tracing(&writer, || {
            Config::parse(raw).expect("config without api_key is valid");
        });

        assert!(
            captured.contains("authentication is DISABLED"),
            "captured = {captured:?}"
        );
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
    fn type_label_matches_serde_tag_for_every_variant() {
        // `type_label()` must return exactly the serde `type` tag string
        // used when serializing each variant, so consumers (e.g. the
        // /admin/status endpoint) can key on a stable label without
        // re-deriving the tag.
        let cases: Vec<(ProviderConfig, &str)> = vec![
            (
                ProviderConfig::GithubCopilot {
                    name: "copilot".into(),
                    vscode_version: default_vscode_version(),
                    account_type: default_account_type(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                },
                "github_copilot",
            ),
            (
                ProviderConfig::Anthropic {
                    name: "router".into(),
                    api_key: secret_string("k"),
                    api_base: "https://example.test/v1".into(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                    provider_ignore: Vec::new(),
                    provider_allow: Vec::new(),
                    thinking_when_absent: ThinkingWhenAbsent::Omit,
                },
                "anthropic",
            ),
            (
                ProviderConfig::OpenaiCompat {
                    name: "compat".into(),
                    api_key: secret_string("k"),
                    api_base: "https://example.test/v1".into(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                    provider_ignore: Vec::new(),
                    provider_allow: Vec::new(),
                reasoning_echo: false,
                },
                "openai_compat",
            ),
            (
                ProviderConfig::OpenaiResponses {
                    name: "responses".into(),
                    api_key: secret_string("k"),
                    api_base: "https://example.test/v1".into(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                },
                "openai_responses",
            ),
        ];
        for (provider, expected) in cases {
            assert_eq!(provider.type_label(), expected);
            // And the label round-trips through the serde tag: the enum
            // serializes as {"type": <label>, ...}.
            let value = serde_json::to_value(&provider).unwrap();
            assert_eq!(value["type"], expected);
        }
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
                assert_eq!(api_key.expose_secret(), "");
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

    /// `reasoning_echo: true` parses correctly on `openai_compat`
    /// providers. Off by default when absent — the field is gated by
    /// `#[serde(default)]` so existing configs without the field still
    /// load.
    #[test]
    fn reasoning_echo_parses_and_defaults_to_false() {
        // Explicit `true` round-trips.
        let raw = r#"
providers:
  - name: opencode_zen
    type: openai_compat
    api_key: k
    api_base: https://opencode.ai/zen/v1
    reasoning_echo: true
models:
  - name: m
    primary: opencode_zen
"#;
        let cfg = Config::parse(raw).unwrap();
        expect_variant!(&cfg.providers[0], ProviderConfig::OpenaiCompat { reasoning_echo, .. } => {
            assert!(*reasoning_echo, "explicit true must parse as true");
        });

        // Field absent defaults to false.
        let raw = r#"
providers:
  - name: p
    type: openai_compat
    api_key: k
    api_base: https://example.test/v1
models:
  - name: m
    primary: p
"#;
        let cfg = Config::parse(raw).unwrap();
        expect_variant!(&cfg.providers[0], ProviderConfig::OpenaiCompat { reasoning_echo, .. } => {
            assert!(!*reasoning_echo, "absent field must default to false");
        });
    }

    /// `provider_allow` round-trips through YAML into both variants that
    /// support it, alongside `provider_ignore`. The list is preserved
    /// verbatim (no normalization) because slugs are operator-supplied and
    /// OpenRouter documents no case-folding.
    #[test]
    fn provider_allow_parses_from_yaml() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_allow: ["DeepInfra", "Together"]
    provider_ignore: ["Azure"]
  - name: or_compat
    type: openai_compat
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_allow: ["DeepInfra"]
models:
  - name: m
    primary: or_anthropic
"#;
        let cfg = Config::parse(raw).unwrap();
        expect_variant!(&cfg.providers[0], ProviderConfig::Anthropic { provider_allow, provider_ignore, .. } => {
            assert_eq!(
                provider_allow,
                &vec!["DeepInfra".to_string(), "Together".to_string()]
            );
            assert_eq!(provider_ignore, &vec!["Azure".to_string()]);
        });
        expect_variant!(&cfg.providers[1], ProviderConfig::OpenaiCompat { provider_allow, .. } => {
            assert_eq!(provider_allow, &vec!["DeepInfra".to_string()]);
        });
    }

    /// An omitted `provider_allow` defaults to an empty list on both
    /// variants, so every existing config keeps its current wire bytes
    /// (no `provider.only` key is injected).
    #[test]
    fn provider_allow_defaults_to_empty() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
  - name: or_compat
    type: openai_compat
    api_key: k
    api_base: https://openrouter.ai/api/v1
models:
  - name: m
    primary: or_anthropic
"#;
        let cfg = Config::parse(raw).unwrap();
        expect_variant!(&cfg.providers[0], ProviderConfig::Anthropic { provider_allow, .. } => {
            assert!(provider_allow.is_empty());
        });
        expect_variant!(&cfg.providers[1], ProviderConfig::OpenaiCompat { provider_allow, .. } => {
            assert!(provider_allow.is_empty());
        });
    }

    /// `deny_unknown_fields` on the variants means a typo like
    /// `provider_allowed:` is a hard YAML parse error rather than a
    /// silently-ignored key — which for a routing allowlist would be a
    /// silent under-constraint.
    #[test]
    fn provider_allow_rejects_unknown_fields() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_allowed: ["DeepInfra"]
models:
  - name: m
    primary: or_anthropic
"#;
        assert!(
            Config::parse(raw).is_err(),
            "a misspelled provider_allow key must fail to parse"
        );
    }

    // ── thinking_when_absent ───────────────────────────────────────

    /// The feature is on by default — an anthropic provider that omits
    /// the key must inject `disabled`, so operators don't have to repeat
    /// the option for every provider.
    #[test]
    fn thinking_when_absent_defaults_to_disabled() {
        let raw = r#"
providers:
  - name: qwen
    type: anthropic
    api_key: k
    api_base: https://example.com/anthropic
models:
  - name: m
    primary: qwen
"#;
        let cfg = Config::parse(raw).unwrap();
        match &cfg.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::Disabled),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    #[test]
    fn thinking_when_absent_parses_disabled() {
        let raw = r#"
providers:
  - name: qwen
    type: anthropic
    api_key: k
    api_base: https://example.com/anthropic
    thinking_when_absent: disabled
models:
  - name: m
    primary: qwen
"#;
        let cfg = Config::parse(raw).unwrap();
        match &cfg.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::Disabled),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    #[test]
    fn thinking_when_absent_parses_omit() {
        let raw = r#"
providers:
  - name: qwen
    type: anthropic
    api_key: k
    api_base: https://example.com/anthropic
    thinking_when_absent: omit
models:
  - name: m
    primary: qwen
"#;
        let cfg = Config::parse(raw).unwrap();
        match &cfg.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::Omit),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    #[test]
    fn thinking_when_absent_parses_effort_low() {
        let raw = r#"
providers:
  - name: mm
    type: anthropic
    api_key: k
    api_base: https://api.minimaxi.com/anthropic
    thinking_when_absent: effort_low
models:
  - name: m
    primary: mm
"#;
        let cfg = Config::parse(raw).unwrap();
        match &cfg.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::EffortLow),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    /// A misspelled value is a hard error. `thinking_when_absent`
    /// changes the outbound wire, so silently falling back to `disabled`
    /// (or to `omit`) would be a silent misconfiguration.
    #[test]
    fn thinking_when_absent_rejects_unknown_value() {
        for bad in ["disable", "none", "off", "true", "effort", "low"] {
            let raw = format!(
                r#"
providers:
  - name: qwen
    type: anthropic
    api_key: k
    api_base: https://example.com/anthropic
    thinking_when_absent: {bad}
models:
  - name: m
    primary: qwen
"#
            );
            assert!(
                Config::parse(&raw).is_err(),
                "thinking_when_absent: {bad} must fail to parse"
            );
        }
    }

    /// The setting must round-trip through serialization.
    #[test]
    fn thinking_when_absent_roundtrips() {
        let raw = r#"
providers:
  - name: qwen
    type: anthropic
    api_key: k
    api_base: https://example.com/anthropic
    thinking_when_absent: disabled
models:
  - name: m
    primary: qwen
"#;
        let cfg = Config::parse(raw).unwrap();
        let s = serde_yaml::to_string(&cfg).unwrap();
        assert!(s.contains("thinking_when_absent: disabled"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        match &reparsed.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::Disabled),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    /// The new `effort_low` value must round-trip through serialization
    /// as well — operators regenerate configs through `/admin/status`
    /// tooling and a variant that serializes differently from its YAML
    /// spelling would be silently rewritten.
    #[test]
    fn thinking_when_absent_effort_low_roundtrips() {
        let raw = r#"
providers:
  - name: mm
    type: anthropic
    api_key: k
    api_base: https://api.minimaxi.com/anthropic
    thinking_when_absent: effort_low
models:
  - name: m
    primary: mm
"#;
        let cfg = Config::parse(raw).unwrap();
        let s = serde_yaml::to_string(&cfg).unwrap();
        assert!(s.contains("thinking_when_absent: effort_low"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        match &reparsed.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent, ..
            } => assert_eq!(*thinking_when_absent, ThinkingWhenAbsent::EffortLow),
            other => panic!("expected an anthropic provider, got {other:?}"),
        }
    }

    // ── secrets overlay ────────────────────────────────────────────

    /// A minimal config whose provider key comes from `${<var>}`.
    ///
    /// Each test uses its own variable name: the process environment
    /// is global, and the test harness runs tests in parallel, so
    /// sharing one name would make these tests race each other.
    fn secret_config(var: &str, secrets_block: &str) -> String {
        format!(
            r#"
{secrets_block}
providers:
  - name: compat
    type: openai_compat
    api_key: "${{{var}}}"
    api_base: "https://example.test/v1"
models:
  - name: m
    primary: compat
"#
        )
    }

    fn api_key_of(cfg: &Config) -> String {
        match &cfg.providers[0] {
            ProviderConfig::OpenaiCompat { api_key, .. } => api_key.expose_secret().clone(),
            other => panic!("expected an openai_compat provider, got {other:?}"),
        }
    }

    #[test]
    fn no_secrets_block_keeps_env_behavior() {
        std::env::set_var("LLMPROXY_SECRET_NOBLOCK", "from-env");
        let cfg = Config::parse(&secret_config("LLMPROXY_SECRET_NOBLOCK", "")).unwrap();
        assert_eq!(api_key_of(&cfg), "from-env");
        assert_eq!(cfg.secrets.provider, SecretsProvider::None);
    }

    #[test]
    fn null_secrets_block_is_ignored() {
        std::env::set_var("LLMPROXY_SECRET_NULL", "from-env");
        let cfg = Config::parse(&secret_config("LLMPROXY_SECRET_NULL", "secrets:")).unwrap();
        assert_eq!(api_key_of(&cfg), "from-env");
        assert_eq!(cfg.secrets.provider, SecretsProvider::None);
    }

    #[test]
    fn file_overlay_wins_over_environment() {
        let var = "LLMPROXY_SECRET_OVERRIDE";
        std::env::set_var(var, "from-env");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        std::fs::write(&path, format!("{var}=from-file\n")).unwrap();
        let block = format!(
            "secrets:\n  provider: file\n  file:\n    path: {}\n    required: true",
            path.display()
        );
        let cfg = Config::parse(&secret_config(var, &block)).unwrap();
        assert_eq!(api_key_of(&cfg), "from-file");
    }

    #[test]
    fn file_overlay_supplies_keys_the_environment_lacks() {
        let var = "LLMPROXY_SECRET_FILEONLY";
        std::env::remove_var(var);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        std::fs::write(&path, format!("{var}=from-file\n")).unwrap();
        let block = format!(
            "secrets:\n  provider: file\n  file:\n    path: {}",
            path.display()
        );
        let cfg = Config::parse(&secret_config(var, &block)).unwrap();
        assert_eq!(api_key_of(&cfg), "from-file");
    }

    #[test]
    fn file_overlay_falls_back_to_environment_for_absent_keys() {
        let var = "LLMPROXY_SECRET_FALLBACK";
        std::env::set_var(var, "from-env");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        // The file does not mention the key at all.
        std::fs::write(&path, "SOMETHING_ELSE=1\n").unwrap();
        let block = format!(
            "secrets:\n  provider: file\n  file:\n    path: {}",
            path.display()
        );
        let cfg = Config::parse(&secret_config(var, &block)).unwrap();
        assert_eq!(api_key_of(&cfg), "from-env");
    }

    #[test]
    fn missing_required_file_fails_startup() {
        let var = "LLMPROXY_SECRET_REQUIRED";
        std::env::set_var(var, "from-env");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.env");
        let block = format!(
            "secrets:\n  provider: file\n  file:\n    path: {}\n    required: true",
            path.display()
        );
        let err = Config::parse(&secret_config(var, &block)).unwrap_err();
        assert!(format!("{err}").contains("read secrets file"), "{err}");
    }

    #[test]
    fn missing_optional_file_falls_back_to_environment() {
        let var = "LLMPROXY_SECRET_OPTIONAL";
        std::env::set_var(var, "from-env");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.env");
        let block = format!(
            "secrets:\n  provider: file\n  file:\n    path: {}\n    required: false",
            path.display()
        );
        let cfg = Config::parse(&secret_config(var, &block)).unwrap();
        assert_eq!(api_key_of(&cfg), "from-env");
    }

    #[test]
    fn file_provider_without_path_is_a_config_error() {
        let var = "LLMPROXY_SECRET_NOPATH";
        let block = "secrets:\n  provider: file\n";
        let err = Config::parse(&secret_config(var, block)).unwrap_err();
        assert!(format!("{err}").contains("secrets.file.path"), "{err}");
    }

    #[test]
    fn overlay_value_beats_the_default_form() {
        let overlay = HashMap::from([("LLMPROXY_D".to_string(), "from-file".to_string())]);
        assert_eq!(
            expand_env_vars_with_overlay("k=${LLMPROXY_D:-fallback}", &overlay),
            "k=from-file"
        );
    }

    #[test]
    fn overlay_empty_value_falls_through_to_the_default_form() {
        // An empty overlay value behaves exactly like an empty
        // environment variable: `${VAR:-default}` still uses the
        // default.
        let overlay = HashMap::from([("LLMPROXY_D".to_string(), String::new())]);
        assert_eq!(
            expand_env_vars_with_overlay("k=${LLMPROXY_D:-fallback}", &overlay),
            "k=fallback"
        );
        // Without the `:-` form it expands to empty, like an empty
        // environment variable does.
        assert_eq!(
            expand_env_vars_with_overlay("k=${LLMPROXY_D}", &overlay),
            "k="
        );
    }

    #[test]
    fn overlay_covers_the_unbraced_form_too() {
        let overlay = HashMap::from([("LLMPROXY_D".to_string(), "from-file".to_string())]);
        assert_eq!(
            expand_env_vars_with_overlay("k=$LLMPROXY_D", &overlay),
            "k=from-file"
        );
    }

    #[test]
    fn secrets_path_may_come_from_the_environment() {
        let var = "LLMPROXY_SECRET_VIAPATH";
        let path_var = "LLMPROXY_SECRETS_PATH_VIAENV";
        std::env::set_var(var, "from-env");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        std::fs::write(&path, format!("{var}=from-file\n")).unwrap();
        std::env::set_var(path_var, &path.display().to_string());
        let quote = format!("\"${{{path_var}}}\"");
        let block = format!("secrets:\n  provider: file\n  file:\n    path: {quote}\n");
        let cfg = Config::parse(&secret_config(var, &block)).unwrap();
        assert_eq!(api_key_of(&cfg), "from-file");
    }

    #[test]
    fn secrets_block_round_trips_through_serde() {
        let raw = secret_config(
            "LLMPROXY_SECRET_ROUNDTRIP",
            "secrets:\n  provider: file\n  file:\n    path: /run/secrets/llmproxy.env\n    required: false\n",
        );
        let cfg = Config::parse(&raw).unwrap();
        let s = serde_yaml::to_string(&cfg).unwrap();
        assert!(s.contains("provider: file"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        assert_eq!(reparsed.secrets.provider, SecretsProvider::File);
        assert_eq!(
            reparsed.secrets.file_path(),
            Some("/run/secrets/llmproxy.env")
        );
        assert!(!reparsed.secrets.file.required);
    }

    #[test]
    fn malformed_secrets_block_is_rejected() {
        let var = "LLMPROXY_SECRET_BAD";
        let block = "secrets:\n  provider: vault\n";
        let err = Config::parse(&secret_config(var, block)).unwrap_err();
        assert!(format!("{err}").contains("secrets:"), "{err}");
    }

    #[test]
    fn unknown_key_in_secrets_block_is_rejected() {
        let var = "LLMPROXY_SECRET_BOGUS";
        let block = "secrets:\n  bogus: 1\n";
        let err = Config::parse(&secret_config(var, block)).unwrap_err();
        assert!(format!("{err}").contains("secrets:"), "{err}");
    }

    // ------------------------------------------------------------------
    // A-Phase 5 — credentials must never appear in a `{:?}` dump or in a
    // serialized config. Both are inspection formats: a log line, a
    // `/admin/status` snapshot, a bug report. A credential in one of
    // them is a credential in a log file that outlives the process.
    // ------------------------------------------------------------------

    /// A provider config carrying a recognizable credential.
    fn provider_with_secret() -> ProviderConfig {
        ProviderConfig::OpenaiCompat {
            name: "compat".to_string(),
            api_key: secret_string("sk-super-secret-value"),
            api_base: "https://example.test/v1".to_string(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
            provider_ignore: Vec::new(),
            provider_allow: Vec::new(),
            reasoning_echo: false,
        }
    }

    #[test]
    fn provider_debug_output_hides_the_api_key() {
        let cfg = provider_with_secret();
        let dumped = format!("{cfg:?}");
        assert!(
            !dumped.contains("sk-super-secret-value"),
            "the api key leaked into the Debug output: {dumped}"
        );
        // The redaction marker is still there, so a reader can tell the
        // field was populated rather than empty.
        assert!(dumped.contains("[REDACTED"), "{dumped}");
    }

    #[test]
    fn serialized_provider_redacts_the_api_key() {
        let cfg = provider_with_secret();
        let dumped = serde_yaml::to_string(&cfg).unwrap();
        assert!(
            !dumped.contains("sk-super-secret-value"),
            "the api key leaked into the serialized config: {dumped}"
        );
        assert!(dumped.contains("[REDACTED]"), "{dumped}");
    }

    #[test]
    fn serialized_config_redacts_every_credential() {
        let cfg = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: Some(secret_string("sk-server-secret-value")),
            },
            providers: vec![provider_with_secret()],
            ..Config::default()
        };
        let dumped = serde_yaml::to_string(&cfg).unwrap();
        assert!(
            !dumped.contains("sk-server-secret-value"),
            "server.api_key leaked into the serialized config: {dumped}"
        );
        assert!(
            !dumped.contains("sk-super-secret-value"),
            "provider api_key leaked into the serialized config: {dumped}"
        );
    }

    #[test]
    fn config_debug_output_hides_every_credential() {
        let cfg = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: Some(secret_string("sk-server-secret-value")),
            },
            providers: vec![provider_with_secret()],
            ..Config::default()
        };
        let dumped = format!("{cfg:?}");
        assert!(
            !dumped.contains("sk-server-secret-value"),
            "server.api_key leaked into the Debug output: {dumped}"
        );
        assert!(
            !dumped.contains("sk-super-secret-value"),
            "provider api_key leaked into the Debug output: {dumped}"
        );
    }

    #[test]
    fn an_absent_server_api_key_stays_absent_when_serialized() {
        // Redaction must not turn "auth disabled" into "auth configured
        // with a placeholder" — the serialized form has to keep the
        // `null` / absent distinction.
        let cfg = Config::default();
        let dumped = serde_yaml::to_string(&cfg).unwrap();
        assert!(
            !dumped.contains("[REDACTED]"),
            "an unset api_key must not serialize as a redacted value: {dumped}"
        );
    }

    #[test]
    fn a_redacted_config_can_still_be_reparsed() {
        // The redacted form is meant for inspection, but it must stay
        // parseable — otherwise "dump the config to paste into an issue"
        // would produce something that no longer loads.
        let cfg = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: Some(secret_string("sk-server-secret-value")),
            },
            providers: vec![provider_with_secret()],
            models: vec![ModelConfig {
                name: "m".to_string(),
                primary: "compat".to_string(),
                fallback_chain: Vec::new(),
                cooldown_seconds: default_cooldown_seconds(),
                max_retries_per_provider: default_max_retries(),
                max_retries_total: default_max_retries(),
            }],
            ..Config::default()
        };
        let dumped = serde_yaml::to_string(&cfg).unwrap();
        let reparsed = Config::parse(&dumped).unwrap();
        assert_eq!(
            reparsed
                .server
                .api_key
                .as_ref()
                .map(|k| k.expose_secret().as_str()),
            Some("[REDACTED]")
        );
    }
}
