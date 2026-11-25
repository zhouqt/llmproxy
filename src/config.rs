use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{ProxyError, Result};
use crate::oauth::token_crypto::parse_key;
use crate::secrets::{deserialize_secrets_config, SecretsConfig};
use crate::usage::DEFAULT_USAGE_CAPACITY;

/// Placeholder written instead of a credential whenever a config is
/// serialized. `secrecy` deliberately does not implement `Serialize`
/// for `SecretString` (it would be an exfiltration channel), so the
/// redaction has to happen here — a serialized config is meant for
/// inspection, never for reuse.
///
/// Since secrecy 0.10 the reason is stricter, not looser: `Serialize`
/// is only implemented for `SecretBox<T>` where `T` opts in via the
/// crate's `SerializableSecret` marker trait. `str` is a foreign type
/// and cannot implement that marker, so `SecretString` stays
/// non-serializable by construction.
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
    /// The **default** User-Agent presented to upstream LLM providers on
    /// outbound requests (all providers sharing the client pools in
    /// `proxy_client.rs`). Defaults to simulating the Claude Code
    /// client identity (`claude-cli/<ver> (...)` — see
    /// `default_user_agent`) so gateways classify the proxy's traffic
    /// as coming from Claude Code instead of an unknown client.
    /// Copilot is unaffected — it overrides this header per-request with
    /// `GitHubCopilotChat/<version>`. A top-level setting because it
    /// describes the client identity, not the `proxy:` outbound HTTP
    /// proxy.
    ///
    /// A provider may override it for itself with its own `user_agent:`
    /// field (see `ProviderConfig`); an unset or illegal value falls back
    /// to the built-in default rather than failing startup.
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
    /// Whether upstream details are exposed to clients.
    ///
    /// `true` (the default) keeps today's behaviour byte for byte: the
    /// upstream error body is passed through to the client, the
    /// `x-llmproxy-failed-providers` header names the backends that
    /// failed, and an in-stream `event: error` embeds the upstream
    /// text. That is a debugging convenience, and it is also a
    /// disclosure surface: anyone holding the shared bearer token can
    /// read the backend topology and the upstream accounts' quota and
    /// auth details.
    ///
    /// `false` replaces all of that with generic wording. The operator
    /// still gets the full detail in the logs and in `/admin/usage` —
    /// only the *client* stops seeing it.
    ///
    /// "All of that" includes the error frames the SSE adapters
    /// synthesise themselves (`OpenAiSseToAnthropic`'s error-envelope
    /// branch, `ResponsesStreamTranslator`'s `response.failed` /
    /// `error` frames) and the error variants that merely *carry*
    /// upstream text without being `ProxyError::Upstream`
    /// (`UpstreamText`, `Http`, `Internal`). Errors that are purely
    /// client-side (`ProxyError::Other`) keep their own wording: they
    /// describe the request, not the backend.
    #[serde(default = "default_expose_upstream_details")]
    pub expose_upstream_details: bool,

    /// Maximum accepted inbound request body, in bytes.
    ///
    /// axum's `DefaultBodyLimit` defaults to 2 MiB, and every extractor
    /// built on `Bytes` (which includes `Json`) honours it. A long
    /// Claude Code session — large context, base64 images — exceeds that
    /// and is then rejected *before* it ever reaches a provider, so the
    /// proxy silently becomes the component that decides the session has
    /// grown too large. This limit is sized for a large session while
    /// still bounding how much a single request can make the proxy
    /// buffer.
    ///
    /// `0` is **not** "unlimited" and is refused by `Config::validate`:
    /// axum has no numeric unlimited mode, and `DefaultBodyLimit::max(0)`
    /// rejects every body that carries data. The only truly unlimited
    /// mode is `DefaultBodyLimit::disable()`, which this proxy does not
    /// offer — it would turn the proxy into an unbounded-memory target
    /// for anyone holding the bearer token.
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes: usize,
}

fn default_listen() -> String {
    "127.0.0.1:8080".to_string()
}

/// Upstream details are exposed unless the operator says otherwise:
/// the switch exists to *reduce* what a client can see, so it must
/// not silently change the current behaviour when it is absent.
fn default_expose_upstream_details() -> bool {
    true
}

/// 32 MiB — large enough for a long Claude Code session, small enough to
/// bound what one request can make the proxy buffer.
pub fn default_max_request_bytes() -> usize {
    32 * 1024 * 1024
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            api_key: None,
            expose_upstream_details: default_expose_upstream_details(),
            max_request_bytes: default_max_request_bytes(),
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

/// Client-side rate limiting for one provider (the `rate_limit:` block on
/// a provider). The proxy paces its own outbound requests so the upstream
/// never sees a limit-violating request — the point is to avoid the
/// 429 → cooldown → fallback cascade entirely, not to react to it.
///
/// Industry limits come in three locally-modelable shapes, so three
/// independent knobs are offered (any combination, each optional):
///
/// - `rpm` / `tpm` — per-minute quotas (MiniMax Token Plan: paid tier
///   200 requests / 10,000,000 input+output tokens per minute; Aliyun
///   Bailian per-model quotas; OpenRouter free-model 20 RPM). Both are
///   enforced over a rolling 60-second window, which is strictly
///   stronger than any fixed-minute window an upstream may use.
/// - `max_concurrency` — cap on in-flight upstream requests (Zhipu
///   GLM and DeepSeek publish concurrency, not RPM/TPM; a request
///   counts as concurrent from dispatch until its response completes).
///
/// Token estimates are reconciled with the upstream-reported usage
/// after each response, so estimation drift self-corrects. Unset
/// (the default) = no pacing, byte-identical dispatch behavior.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Maximum upstream requests in any rolling 60-second window.
    /// `None` = no request-count limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u32>,
    /// Maximum input+output tokens in any rolling 60-second window.
    /// The request is charged with an estimate (string-leaf heuristic
    /// plus a `max_tokens` reserve) *before* dispatch, then the
    /// estimate is replaced by the upstream-reported usage after the
    /// response. `None` = no token limit.
    ///
    /// The pre-dispatch charge includes the `max_tokens` reserve (the
    /// output size is unknowable before dispatch and the upstream
    /// counts input+output), so the budget must hold
    /// `typical input + max_tokens` per concurrent request. A `tpm`
    /// below a single request's estimate skips that request outright
    /// — the proxy logs `rate_limit.tpm is below the estimated request
    /// size` once per minute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>,
    /// Maximum concurrent in-flight upstream requests. A request
    /// holds its slot from dispatch until its response completes
    /// (for streams: until the stream terminates, including a client
    /// disconnect). `None` = no concurrency limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<u32>,
    /// Longest a request may wait at the gate before the router gives
    /// up on this provider for that request and moves to the next
    /// chain entry (a local skip — no cooldown is recorded, the
    /// provider is not penalized). Default 2000 ms; `0` disables
    /// waiting entirely (never wait, always skip). Capped at
    /// `MAX_WAIT_MS_CEILING`: waiting longer than one window slide can
    /// never buy more capacity, and larger values overflow the
    /// monotonic clock arithmetic inside the gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wait_ms: Option<u64>,
}

/// Sanity ceiling for `rate_limit.max_wait_ms`. The windows are 60 s:
/// a wait beyond one slide can never help (the whole window cycles),
/// so anything past a few minutes is an operator typo rather than a
/// tuning choice — and values near `u64::MAX` ms would panic inside
/// `ProviderLimiter::acquire` on `now + max_wait` clock overflow.
pub const MAX_WAIT_MS_CEILING: u64 = 600_000; // 10 minutes

impl RateLimitConfig {
    /// Effective wait cap: the configured `max_wait_ms` or the 2 s
    /// default.
    pub fn max_wait(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.max_wait_ms.unwrap_or(2000))
    }

    /// Whether at least one limit knob is set. A block with none of
    /// `rpm`/`tpm`/`max_concurrency` is a configuration mistake —
    /// `Config::validate` rejects it.
    fn has_a_limit(&self) -> bool {
        self.rpm.is_some() || self.tpm.is_some() || self.max_concurrency.is_some()
    }
}

/// OpenRouter provider-routing for one provider (the `provider_routing:`
/// block). The block is serialized 1:1 into the upstream `provider`
/// request field (`#/components/schemas/ProviderPreferences`) and only
/// takes effect when the provider's `api_base` points at OpenRouter —
/// other backends ignore it with a startup warning, because strict
/// Anthropic-/OpenAI-compatible backends answer 400 on unknown
/// top-level fields.
///
/// Design notes:
///
/// - Every key is optional and independent. Absence is what keeps the
///   outbound body byte-identical to a config without the block; an
///   explicit `false` (`allow_fallbacks`, `zdr`, ...) is meaningful and
///   is therefore emitted.
/// - `only` / `ignore` absorb the former top-level `provider_allow` /
///   `provider_ignore` fields using OpenRouter's own keywords
///   (plans/openrouter-provider-routing-plan.md §3.3).
/// - The hard filters (`max_price`, `zdr`, `data_collection`,
///   `quantizations`, `require_parameters`) can leave zero eligible
///   endpoints; OpenRouter then answers 503, which the router cools
///   down. The router warns about that case when a provider carries a
///   routing block (§6.1).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProviderRouting {
    /// Provider slugs routing is restricted to (OpenRouter's `only`).
    /// A hard ceiling; merged upstream with account-wide dashboard
    /// settings. Absent = no restriction; an empty list is a config
    /// error (it can only mean "no filter", which is spelled by
    /// omitting the key).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Provider slugs to exclude (OpenRouter's `ignore`). Absent = no
    /// exclusions; empty is a config error, as for `only`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Ranking strategy: `price` | `throughput` | `latency` | `exacto`,
    /// or `{by: <same>, partition: model|none}`. When set, OpenRouter
    /// performs no load balancing across the ranked endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<SortPreference>,
    /// Explicit provider order. Takes precedence over `sort` upstream
    /// (which is then ignored), so configuring both is rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// `false` → use only the primary/`order` provider and return its
    /// upstream error instead of falling back to another endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Only route to endpoints that declare support for every parameter
    /// in the request. Can shrink the candidate set drastically for our
    /// tool-heavy bodies; off unless an experiment needs it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// `allow` | `deny` — whether endpoints that store or train on
    /// prompts may be used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    /// Restrict routing to Zero Data Retention endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// Restrict routing to models that allow text distillation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    /// Allow-list of quantization levels. Empty is a config error (it
    /// can only mean "matches nothing").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
    /// Per-unit price ceiling. Values are **strings** on the wire and
    /// are forwarded exactly as the operator wrote them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
    /// Soft preference: endpoints below the threshold are deprioritized,
    /// not excluded. Scalar = p50; or `{p50,p75,p90,p99}` (all must be
    /// met to be preferred).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<ThroughputCutoffs>,
    /// Soft preference, same shape as `preferred_min_throughput`, in
    /// **seconds**.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<LatencyCutoffs>,
}

impl ProviderRouting {
    /// `true` when no key is set — `provider_routing: {}` is a config
    /// error (same rule as an empty `rate_limit: {}`).
    pub fn is_empty(&self) -> bool {
        self.only.is_none()
            && self.ignore.is_none()
            && self.sort.is_none()
            && self.order.is_none()
            && self.allow_fallbacks.is_none()
            && self.require_parameters.is_none()
            && self.data_collection.is_none()
            && self.zdr.is_none()
            && self.enforce_distillable_text.is_none()
            && self.quantizations.is_none()
            && self.max_price.is_none()
            && self.preferred_min_throughput.is_none()
            && self.preferred_max_latency.is_none()
    }
}

/// OpenRouter `sort`: either one of the named strategies or the object
/// form `{by, partition}`.
///
/// `untagged` is used only here (never on the wire types' config path);
/// the config tests pin that a typo still produces an actionable
/// message — if serde-saphyr cannot do that through `untagged`, this
/// type is replaced by the raw value plus validation (plan §3.1).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SortPreference {
    Name(SortKey),
    Config(SortConfig),
}

/// Object form of `sort`. A named struct, not an inline enum variant:
/// `deny_unknown_fields` is a serde *container* attribute and cannot be
/// applied to a variant, so `partiton:` would otherwise be accepted
/// silently.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SortConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<SortKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<Partition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SortKey {
    Price,
    Throughput,
    Latency,
    /// OpenRouter-defined strategy listed alongside the other three;
    /// its exact semantics are not in the spec (plan §3.1).
    Exacto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    /// Group endpoints by model before sorting (default upstream).
    Model,
    /// Sort all endpoints together, which may promote a fallback model
    /// ahead of the requested one — documented as unsupported for us
    /// (the echoed `model` would disagree with what the client asked).
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataCollection {
    Allow,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    Int4,
    Int8,
    Fp4,
    Mxfp4,
    Nvfp4,
    Fp6,
    Fp8,
    Mxfp8,
    Fp16,
    Bf16,
    Fp32,
    Unknown,
}

/// Per-unit price ceiling, **USD**; `prompt`/`completion` are USD per
/// million tokens, the other keys are per image / audio unit / request.
/// Values are strings on the wire and are forwarded verbatim.
///
/// `deny_unknown_fields`: the upstream schema tolerates junk keys, but a
/// config typo like `completiont:` would then silently drop the output
/// ceiling — an overspend with no 503 and no warning.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MaxPrice {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
}

impl MaxPrice {
    pub fn is_empty(&self) -> bool {
        self.prompt.is_none()
            && self.completion.is_none()
            && self.image.is_none()
            && self.audio.is_none()
            && self.request.is_none()
    }

    fn values(&self) -> impl Iterator<Item = (&'static str, &String)> {
        [
            ("prompt", self.prompt.as_ref()),
            ("completion", self.completion.as_ref()),
            ("image", self.image.as_ref()),
            ("audio", self.audio.as_ref()),
            ("request", self.request.as_ref()),
        ]
        .into_iter()
        .filter_map(|(k, v)| v.map(|v| (k, v)))
    }
}

/// Scalar (p50) or per-percentile throughput thresholds, tokens/second.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ThroughputCutoffs {
    Number(f64),
    Percentiles(PercentileMap),
}

/// Scalar (p50) or per-percentile latency thresholds, **seconds**.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LatencyCutoffs {
    Number(f64),
    Percentiles(PercentileMap),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PercentileMap {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

impl PercentileMap {
    pub fn is_empty(&self) -> bool {
        self.p50.is_none() && self.p75.is_none() && self.p90.is_none() && self.p99.is_none()
    }

    fn values(&self) -> impl Iterator<Item = (&'static str, f64)> {
        [
            ("p50", self.p50),
            ("p75", self.p75),
            ("p90", self.p90),
            ("p99", self.p99),
        ]
        .into_iter()
        .filter_map(|(k, v)| v.map(|v| (k, v)))
    }
}

/// Shared validation of a `preferred_*` cutoff: positive and finite,
/// whether scalar or percentile map.
fn validate_cutoffs(name: &str, field: &str, value: Option<f64>) -> Result<()> {
    if let Some(v) = value {
        if !v.is_finite() || v <= 0.0 {
            return Err(ProxyError::Config(format!(
                "provider '{name}': provider_routing.{field} must be a positive, finite number (got {v})"
            )));
        }
    }
    Ok(())
}

/// Startup validation of one provider's `provider_routing` block.
///
/// Everything here is a local-shape rule; satisfiability (a filter
/// combination no endpoint can meet, a ceiling below the cheapest
/// endpoint) is deliberately not checked — that depends on OpenRouter's
/// live catalogue and surfaces as the router's 503 warning.
fn validate_provider_routing(name: &str, routing: &ProviderRouting) -> Result<()> {
    if routing.is_empty() {
        return Err(ProxyError::Config(format!(
            "provider '{name}': provider_routing block carries no setting; \
             remove the block or configure at least one key"
        )));
    }
    if routing.sort.is_some() && routing.order.is_some() {
        return Err(ProxyError::Config(format!(
            "provider '{name}': provider_routing.sort and provider_routing.order cannot be \
             combined (OpenRouter ignores sort when order is set)"
        )));
    }
    if let Some(SortPreference::Config(cfg)) = &routing.sort {
        if cfg.by.is_none() {
            return Err(ProxyError::Config(format!(
                "provider '{name}': provider_routing.sort.by is required in the object form \
                 (use `sort: <strategy>` or `sort: {{by: <strategy>}}`)"
            )));
        }
    }
    for (field, list) in [
        ("only", routing.only.as_ref()),
        ("ignore", routing.ignore.as_ref()),
        ("order", routing.order.as_ref()),
    ] {
        let Some(list) = list else { continue };
        if list.is_empty() {
            return Err(ProxyError::Config(format!(
                "provider '{name}': provider_routing.{field} is empty; \
                 an empty list cannot mean anything — omit the key instead"
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for slug in list {
            if slug.trim().is_empty() {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': provider_routing.{field} contains an empty slug"
                )));
            }
            if !seen.insert(slug.as_str()) {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': provider_routing.{field} lists '{slug}' more than once"
                )));
            }
        }
    }
    if let Some(q) = &routing.quantizations {
        if q.is_empty() {
            return Err(ProxyError::Config(format!(
                "provider '{name}': provider_routing.quantizations is empty; \
                 an empty allow-list matches no endpoint"
            )));
        }
    }
    if let Some(price) = &routing.max_price {
        if price.is_empty() {
            return Err(ProxyError::Config(format!(
                "provider '{name}': provider_routing.max_price carries no key"
            )));
        }
        for (key, raw) in price.values() {
            let parsed: f64 = raw.parse().map_err(|_| {
                ProxyError::Config(format!(
                    "provider '{name}': provider_routing.max_price.{key} must be a decimal \
                     number of USD (got '{raw}')"
                ))
            })?;
            if !parsed.is_finite() || parsed < 0.0 {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': provider_routing.max_price.{key} must be a finite, \
                     non-negative number (got '{raw}')"
                )));
            }
        }
    }
    for (field, cutoffs) in [
        (
            "preferred_min_throughput",
            routing.preferred_min_throughput.as_ref().map(Cutoffs::from_throughput),
        ),
        (
            "preferred_max_latency",
            routing.preferred_max_latency.as_ref().map(Cutoffs::from_latency),
        ),
    ] {
        let Some(cutoffs) = cutoffs else { continue };
        match cutoffs {
            Cutoffs::Number(v) => validate_cutoffs(name, field, Some(v))?,
            Cutoffs::Percentiles(map) => {
                if map.is_empty() {
                    return Err(ProxyError::Config(format!(
                        "provider '{name}': provider_routing.{field} has no percentile set"
                    )));
                }
                for (key, v) in map.values() {
                    validate_cutoffs(name, &format!("{field}.{key}"), Some(v))?;
                }
            }
        }
    }
    Ok(())
}

/// Borrowing view over the two cutoff types, which differ only in unit.
enum Cutoffs<'a> {
    Number(f64),
    Percentiles(&'a PercentileMap),
}

impl<'a> Cutoffs<'a> {
    fn from_throughput(c: &'a ThroughputCutoffs) -> Self {
        match c {
            ThroughputCutoffs::Number(v) => Cutoffs::Number(*v),
            ThroughputCutoffs::Percentiles(m) => Cutoffs::Percentiles(m),
        }
    }

    fn from_latency(c: &'a LatencyCutoffs) -> Self {
        match c {
            LatencyCutoffs::Number(v) => Cutoffs::Number(*v),
            LatencyCutoffs::Percentiles(m) => Cutoffs::Percentiles(m),
        }
    }
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
        /// Key used to encrypt the on-disk Copilot token store
        /// (`github_token.json`) at rest.
        ///
        /// The store holds a live GitHub OAuth token and a live Copilot
        /// API token. It lives in the `llmproxy-data` volume, which is
        /// usually on an unencrypted host disk — so a TPM-sealed `.env`
        /// protecting the provider keys while the token store sits in
        /// the clear next to it is only half a story. With this key set,
        /// the store is sealed with AES-256-GCM and the disk only ever
        /// holds ciphertext.
        ///
        /// Format: **64 hex characters = 32 bytes** (generate one with
        /// `openssl rand -hex 32`). Deliver it the same way as the
        /// other secrets — `${LLMPROXY_TOKEN_STORE_KEY}` resolved from
        /// the sealed `.env` through the `secrets.provider: file`
        /// overlay.
        ///
        /// Unset (the default) → the store is written as plaintext
        /// JSON, exactly as before this feature existed. A store
        /// written in plaintext is still readable after the key is
        /// added — the next save rewrites it encrypted.
        #[serde(default, serialize_with = "serialize_redacted_secret_option")]
        token_store_key: Option<SecretString>,
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
        /// Client-side rate limiting for this provider. The proxy paces
        /// its own outbound requests so the upstream never sees a
        /// limit-violating request — no 429, no cooldown, no fallback.
        /// Absent (the default) = no pacing. See `RateLimitConfig` for
        /// the three knobs (`rpm` / `tpm` / `max_concurrency`) and which
        /// upstream limit style each models.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limit: Option<RateLimitConfig>,
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
        /// OpenRouter provider routing for this provider: the
        /// `provider` request object — allow-list (`only`), deny-list
        /// (`ignore`), ranking (`sort` / `order`), the
        /// fallback/parameter switches and the hard filters
        /// (`max_price`, `zdr`, `data_collection`, `quantizations`,
        /// `require_parameters`). See [`ProviderRouting`].
        ///
        /// Only active when `api_base` points at `openrouter.ai`; on
        /// other backends the block is ignored with a startup warning,
        /// because strict Anthropic-compatible backends (e.g. DeepSeek,
        /// Minimax, Alibaba Bailian) 400 on unknown top-level fields.
        /// Absent (the default) injects no `provider` field at all.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_routing: Option<ProviderRouting>,
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
        /// Optional User-Agent override for THIS provider.
        ///
        /// Absent (the default) ⇒ the global top-level `user_agent:` applies
        /// (the shared reqwest clients' default header). Set it to present
        /// this provider as a different client, e.g. `omp/18.14.0`.
        ///
        /// An empty or illegal value is dropped with a warning and the
        /// global default applies — never fatal.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_agent: Option<String>,
        /// Optional app attribution, sent verbatim as the `HTTP-Referer`
        /// header — the header an app registry (OpenRouter et al.) credits
        /// traffic on. Without it the request shows up as "Unknown" on the
        /// registry's activity page, whatever the User-Agent says.
        ///
        /// Absent (the default) ⇒ nothing is sent. An empty or illegal
        /// value is dropped with a warning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        referer: Option<String>,
        /// Optional custom headers (name → value) sent on every outbound
        /// request of this provider.
        ///
        /// The provider block IS the scope: these headers go to this
        /// provider only. Names the proxy sets itself are rejected with a
        /// warning (`user-agent`, `authorization`, `content-type`,
        /// `accept`, `x-api-key`, `anthropic-version`, `x-app`, …), and an
        /// empty value omits the header entirely.
        ///
        /// Two value sentinels: `"auto"` is one id generated at startup and
        /// reused for every request of this provider; `"auto-request"` is a
        /// fresh id per outbound request.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        /// Client-side rate limiting for this provider. The proxy paces
        /// its own outbound requests so the upstream never sees a
        /// limit-violating request — no 429, no cooldown, no fallback.
        /// Absent (the default) = no pacing. See `RateLimitConfig` for
        /// the three knobs (`rpm` / `tpm` / `max_concurrency`) and which
        /// upstream limit style each models.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limit: Option<RateLimitConfig>,
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
        /// OpenRouter provider routing for this provider: the
        /// `provider` request object — allow-list (`only`), deny-list
        /// (`ignore`), ranking (`sort` / `order`), the
        /// fallback/parameter switches and the hard filters
        /// (`max_price`, `zdr`, `data_collection`, `quantizations`,
        /// `require_parameters`). See [`ProviderRouting`].
        ///
        /// Only active when `api_base` points at `openrouter.ai`; on
        /// other OpenAI-compatible backends the block is ignored with a
        /// startup warning, because strict backends such as DeepSeek
        /// 400 on unknown top-level fields. Absent (the default)
        /// injects no `provider` field at all.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_routing: Option<ProviderRouting>,
        /// When `true`, every historical assistant message that lacks `reasoning_content`
        /// in a thinking request (a cross-model thinking turn, a redacted_thinking
        /// or a plain-text turn) is sent with `reasoning_content: ""`, so that
        /// DeepSeek-V4-like upstreams accept it in thinking mode instead of answering
        /// 400. Default `false` (upstreams that follow the OpenAI spec strictly may
        /// reject this non-standard field; other providers keep the wire bytes unchanged).
        /// Optional User-Agent override for THIS provider.
        ///
        /// Absent (the default) ⇒ the global top-level `user_agent:` applies
        /// (the shared reqwest clients' default header). Set it to present
        /// this provider as a different client, e.g. `omp/18.14.0`.
        ///
        /// An empty or illegal value is dropped with a warning and the
        /// global default applies — never fatal.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_agent: Option<String>,
        /// Optional app attribution, sent verbatim as the `HTTP-Referer`
        /// header — the header an app registry (OpenRouter et al.) credits
        /// traffic on. Without it the request shows up as "Unknown" on the
        /// registry's activity page, whatever the User-Agent says.
        ///
        /// Absent (the default) ⇒ nothing is sent. An empty or illegal
        /// value is dropped with a warning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        referer: Option<String>,
        /// Optional custom headers (name → value) sent on every outbound
        /// request of this provider.
        ///
        /// The provider block IS the scope: these headers go to this
        /// provider only. Names the proxy sets itself are rejected with a
        /// warning (`user-agent`, `authorization`, `content-type`,
        /// `accept`, `x-api-key`, `anthropic-version`, `x-app`, …), and an
        /// empty value omits the header entirely.
        ///
        /// Two value sentinels: `"auto"` is one id generated at startup and
        /// reused for every request of this provider; `"auto-request"` is a
        /// fresh id per outbound request.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        #[serde(default)]
        reasoning_echo: bool,
        /// Client-side rate limiting for this provider. The proxy paces
        /// its own outbound requests so the upstream never sees a
        /// limit-violating request — no 429, no cooldown, no fallback.
        /// Absent (the default) = no pacing. See `RateLimitConfig` for
        /// the three knobs (`rpm` / `tpm` / `max_concurrency`) and which
        /// upstream limit style each models.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limit: Option<RateLimitConfig>,
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
        /// Optional User-Agent override for THIS provider.
        ///
        /// Absent (the default) ⇒ the global top-level `user_agent:` applies
        /// (the shared reqwest clients' default header). Set it to present
        /// this provider as a different client, e.g. `omp/18.14.0`.
        ///
        /// An empty or illegal value is dropped with a warning and the
        /// global default applies — never fatal.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_agent: Option<String>,
        /// Optional app attribution, sent verbatim as the `HTTP-Referer`
        /// header — the header an app registry (OpenRouter et al.) credits
        /// traffic on. Without it the request shows up as "Unknown" on the
        /// registry's activity page, whatever the User-Agent says.
        ///
        /// Absent (the default) ⇒ nothing is sent. An empty or illegal
        /// value is dropped with a warning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        referer: Option<String>,
        /// Optional custom headers (name → value) sent on every outbound
        /// request of this provider.
        ///
        /// The provider block IS the scope: these headers go to this
        /// provider only. Names the proxy sets itself are rejected with a
        /// warning (`user-agent`, `authorization`, `content-type`,
        /// `accept`, `x-api-key`, `anthropic-version`, `x-app`, …), and an
        /// empty value omits the header entirely.
        ///
        /// Two value sentinels: `"auto"` is one id generated at startup and
        /// reused for every request of this provider; `"auto-request"` is a
        /// fresh id per outbound request.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        /// Client-side rate limiting for this provider. The proxy paces
        /// its own outbound requests so the upstream never sees a
        /// limit-violating request — no 429, no cooldown, no fallback.
        /// Absent (the default) = no pacing. See `RateLimitConfig` for
        /// the three knobs (`rpm` / `tpm` / `max_concurrency`) and which
        /// upstream limit style each models.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limit: Option<RateLimitConfig>,
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

    /// The provider's optional client-side rate-limit block (`rate_limit:`).
    /// `None` when absent — the provider then dispatches with no pacing,
    /// exactly as before the knob existed.
    pub fn rate_limit(&self) -> Option<&RateLimitConfig> {
        match self {
            ProviderConfig::GithubCopilot { rate_limit, .. }
            | ProviderConfig::Anthropic { rate_limit, .. }
            | ProviderConfig::OpenaiCompat { rate_limit, .. }
            | ProviderConfig::OpenaiResponses { rate_limit, .. } => rate_limit.as_ref(),
        }
    }

    /// The provider's upstream base URL, when it has one. `github_copilot`
    /// derives its endpoints internally, so it has none.
    pub fn api_base(&self) -> Option<&str> {
        match self {
            ProviderConfig::GithubCopilot { .. } => None,
            ProviderConfig::Anthropic { api_base, .. }
            | ProviderConfig::OpenaiCompat { api_base, .. }
            | ProviderConfig::OpenaiResponses { api_base, .. } => Some(api_base),
        }
    }

    /// The provider's optional OpenRouter routing block
    /// (`provider_routing:`). `None` on the variants that do not accept
    /// one (`github_copilot`, `openai_responses`) and when the key is
    /// absent.
    pub fn provider_routing(&self) -> Option<&ProviderRouting> {
        match self {
            ProviderConfig::GithubCopilot { .. } | ProviderConfig::OpenaiResponses { .. } => None,
            ProviderConfig::Anthropic {
                provider_routing, ..
            }
            | ProviderConfig::OpenaiCompat {
                provider_routing, ..
            } => provider_routing.as_ref(),
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
        let cfg: Config = serde_saphyr::from_str_with_options(&expanded, yaml_options())?;
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
        // A zero body limit is not "unlimited" -- axum has no numeric
        // unlimited mode, and `DefaultBodyLimit::max(0)` wraps the body
        // in `Limited::new(body, 0)`, which rejects *every* body that
        // carries data. Wiring 0 through would make the proxy refuse
        // every non-empty request, so refuse the configuration instead.
        if self.server.max_request_bytes == 0 {
            return Err(ProxyError::Config(
                "server.max_request_bytes must be greater than 0 (0 is not 'unlimited' -- \
                 axum rejects every non-empty body under a zero limit)"
                    .into(),
            ));
        }
        if self.providers.is_empty() {
            return Err(ProxyError::Config("at least one provider required".into()));
        }
        if self.models.is_empty() {
            return Err(ProxyError::Config("at least one model required".into()));
        }

        // The Copilot token store key must be present-and-well-formed or
        // absent. An empty value is NOT "no key" — it is a
        // `${VAR}` pointing at a missing secret, and starting with it
        // would silently downgrade the store to plaintext.
        for p in &self.providers {
            let ProviderConfig::GithubCopilot {
                name,
                token_store_key,
                ..
            } = p
            else {
                continue;
            };
            let Some(key) = token_store_key else {
                continue;
            };
            let raw = key.expose_secret();
            if raw.is_empty() {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': token_store_key is empty after ${{VAR}} expansion; \
                     remove the field to keep the token store in plaintext, or point the \
                     variable at a 64-hex-character value"
                )));
            }
            parse_key(raw)
                .map_err(|msg| ProxyError::Config(format!("provider '{name}': {msg}")))?;
        }

        // `account_type` is interpolated into the token-host URL
        // (`https://api.{account_type}.githubcopilot.com`). A typo here
        // would send the GitHub token to an unintended subdomain, so pin
        // it to the three hostnames GitHub actually operates (audit L4).
        for p in &self.providers {
            let ProviderConfig::GithubCopilot {
                name, account_type, ..
            } = p
            else {
                continue;
            };
            if !matches!(
                account_type.as_str(),
                "individual" | "business" | "enterprise"
            ) {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': account_type must be one of \
                     individual|business|enterprise (got '{account_type}')"
                )));
            }
        }

        let provider_names: std::collections::HashSet<&str> =
            self.providers.iter().map(|p| p.name()).collect();

        // Provider names end up in HTTP response headers
        // (`x-llmproxy-failed-providers`) and in log lines. A name with
        // a control character or a non-ASCII byte is not a legal HTTP
        // header value, so writing it used to panic on the request
        // path. Pin the names to a conservative character set instead
        // of sanitizing them at every write site.
        for p in &self.providers {
            let name = p.name();
            let ok = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
            if !ok {
                return Err(ProxyError::Config(format!(
                    "provider name '{name}' must be non-empty and contain only ASCII \
                     letters, digits, '.', '_' or '-'"
                )));
            }
        }

        // A `rate_limit` block must carry at least one knob, and every
        // knob present must be positive. `0` is never "unlimited" — a
        // zero rpm/tpm/concurrency would reject every request to the
        // provider, which is a configuration mistake, not a throttle.
        for p in &self.providers {
            let Some(rl) = p.rate_limit() else {
                continue;
            };
            let name = p.name();
            if rl.rpm == Some(0) {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': rate_limit.rpm must be at least 1 (0 would reject every request)"
                )));
            }
            if rl.tpm == Some(0) {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': rate_limit.tpm must be at least 1 (0 would reject every request)"
                )));
            }
            if rl.max_concurrency == Some(0) {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': rate_limit.max_concurrency must be at least 1 (0 would reject every request)"
                )));
            }
            if let Some(wait) = rl.max_wait_ms {
                if wait > MAX_WAIT_MS_CEILING {
                    return Err(ProxyError::Config(format!(
                        "provider '{name}': rate_limit.max_wait_ms must be at most {MAX_WAIT_MS_CEILING} (got {wait}); \
                         waiting past one window slide can never buy more capacity"
                    )));
                }
            }
            if !rl.has_a_limit() {
                return Err(ProxyError::Config(format!(
                    "provider '{name}': rate_limit block carries no limit; \
                     set at least one of rpm, tpm or max_concurrency"
                )));
            }
        }

        // A `provider_routing` block must say something (an empty block
        // is a mistake, same rule as an empty `rate_limit: {}`), and
        // every list/price/threshold it does carry must be meaningful.
        // None of this is a *reachability* check: whether the policy can
        // be satisfied is OpenRouter-side data we do not have offline —
        // the router warns on a 503 instead (plan §6.1).
        for p in &self.providers {
            let Some(routing) = p.provider_routing() else {
                continue;
            };
            validate_provider_routing(p.name(), routing)?;
        }

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

/// Parser options shared by every YAML entry point.
///
/// Each field pins a behavior that `serde-saphyr` does not default to
/// but that this config format relies on:
///
/// - `LastWins` reproduces `serde_yaml`'s handling of duplicate keys in
///   a mapping (the later value wins). `config.example.yaml` promises
///   exactly that for `model_rewrite`; `serde-saphyr` errors instead.
/// - `AsOrdinary` keeps YAML merge keys (`<<`) as ordinary keys, as
///   `serde_yaml` did. The default (`Merge`) expands them, which would
///   silently let one mapping inject fields into another.
/// - `with_snippet = false` keeps source lines out of error messages. A
///   parse error next to an `api_key:` line must never echo the secret
///   into the startup log.
fn yaml_options() -> serde_saphyr::Options {
    let mut options = serde_saphyr::Options::default();
    options.duplicate_keys = serde_saphyr::DuplicateKeyPolicy::LastWins;
    options.merge_keys = serde_saphyr::MergeKeyPolicy::AsOrdinary;
    options.with_snippet = false;
    options
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
    // phase must not turn it into a different error. Parsing into
    // `serde_json::Value` (serde-saphyr has no generic YAML value) also
    // means a file with non-string mapping keys lands here; the main
    // parse still rejects it loudly.
    let Ok(value) = serde_saphyr::from_str_with_options::<serde_json::Value>(raw, yaml_options())
    else {
        return Ok(SecretsConfig::default());
    };
    let Some(block) = value.get("secrets") else {
        return Ok(SecretsConfig::default());
    };
    if block.is_null() {
        return Ok(SecretsConfig::default());
    }
    serde_json::from_value::<SecretsConfig>(block.clone())
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
    fn rate_limit_parses_on_all_four_provider_types() {
        // The knob must deserialize identically on every variant — the
        // `deny_unknown_fields` tagged enum repeats the field per variant
        // (model_rewrite precedent), so a field declared on only one
        // variant would surface here as a parse error on the others.
        let blocks = [
            r#"
providers:
  - name: p
    type: github_copilot
    rate_limit:
      rpm: 190
      tpm: 9500000
      max_concurrency: 8
      max_wait_ms: 1500
models:
  - name: gpt-4
    primary: p
"#,
            r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit:
      rpm: 190
      tpm: 9500000
      max_concurrency: 8
      max_wait_ms: 1500
models:
  - name: gpt-4
    primary: p
"#,
            r#"
providers:
  - name: p
    type: openai_compat
    api_key: sk-test
    api_base: https://x.example/v1
    rate_limit:
      rpm: 190
      tpm: 9500000
      max_concurrency: 8
      max_wait_ms: 1500
models:
  - name: gpt-4
    primary: p
"#,
            r#"
providers:
  - name: p
    type: openai_responses
    api_key: sk-test
    api_base: https://x.example/v1
    rate_limit:
      rpm: 190
      tpm: 9500000
      max_concurrency: 8
      max_wait_ms: 1500
models:
  - name: gpt-4
    primary: p
"#,
        ];
        for raw in blocks {
            let cfg = Config::parse(raw).unwrap_or_else(|e| panic!("parse failed: {e}"));
            let rl = cfg.providers[0].rate_limit().expect("rate_limit present");
            assert_eq!(rl.rpm, Some(190), "rpm on {}", cfg.providers[0].type_label());
            assert_eq!(rl.tpm, Some(9_500_000), "tpm");
            assert_eq!(rl.max_concurrency, Some(8), "max_concurrency");
            assert_eq!(rl.max_wait_ms, Some(1500), "max_wait_ms");
            assert_eq!(rl.max_wait(), std::time::Duration::from_millis(1500));
        }
    }

    #[test]
    fn rate_limit_accessor_reads_every_variant() {
        // Keep the shared accessor honest for the variants the parse
        // test above can't pattern-match: copilot (concurrency knob),
        // openai_compat, openai_responses.
        let raws = [
            ("github_copilot", "", 190u32),
            ("openai_compat", "    api_key: sk-test\n    api_base: https://x.example/v1\n", 20u32),
            ("openai_responses", "    api_key: sk-test\n    api_base: https://x.example/v1\n", 20u32),
        ];
        for (kind, extra, rpm) in raws {
            let raw = format!(
                r#"
providers:
  - name: p
    type: {kind}
{extra}    rate_limit:
      rpm: {rpm}
models:
  - name: gpt-4
    primary: p
"#
            );
            let cfg = Config::parse(&raw).unwrap_or_else(|e| panic!("parse failed: {e}"));
            let rl = cfg.providers[0].rate_limit().expect("rate_limit present");
            assert_eq!(rl.rpm, Some(rpm), "variant {kind}");
        }
    }

    #[test]
    fn rate_limit_block_without_any_limit_is_rejected() {
        // A block with none of rpm/tpm/max_concurrency does nothing but
        // cost a lock per request — refuse it instead of silently
        // accepting a no-op.
        let raw = r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit: {}
models:
  - name: gpt-4
    primary: p
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(
            format!("{err}").contains("set at least one of rpm, tpm or max_concurrency"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rate_limit_zero_values_are_rejected() {
        // 0 is not "unlimited" — a zero knob would reject every request
        // to the provider.
        for (field, yaml) in [
            ("rpm", "      rpm: 0\n"),
            ("tpm", "      tpm: 0\n"),
            ("max_concurrency", "      max_concurrency: 0\n"),
        ] {
            let raw = format!(
                r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit:
{yaml}models:
  - name: gpt-4
    primary: p
"#
            );
            let err = Config::parse(&raw).unwrap_err();
            assert!(
                format!("{err}").contains(&format!("rate_limit.{field} must be at least 1")),
                "field {field}: unexpected error: {err}"
            );
        }
    }

    #[test]
    fn rate_limit_absurd_max_wait_is_rejected() {
        // `Instant::now() + max_wait` panics on clock overflow, and a
        // wait past one window slide can never buy more capacity — both
        // make an absurd value an operator typo, not a tuning choice.
        let raw = r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit:
      rpm: 1
      max_wait_ms: 99999999999999999
models:
  - name: gpt-4
    primary: p
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(
            format!("{err}").contains("rate_limit.max_wait_ms must be at most"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rate_limit_max_wait_at_the_ceiling_is_accepted() {
        // The boundary itself is fine — the ceiling rejects absurd
        // values, not the whole upper range.
        let raw = format!(
            r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit:
      rpm: 1
      max_wait_ms: {MAX_WAIT_MS_CEILING}
models:
  - name: gpt-4
    primary: p
"#
        );
        let cfg = Config::parse(&raw).expect("ceiling value parses");
        expect_variant!(&cfg.providers[0], ProviderConfig::Anthropic { rate_limit, .. } => {
            let rl = rate_limit.as_ref().expect("rate_limit present");
            assert_eq!(rl.max_wait_ms, Some(MAX_WAIT_MS_CEILING));
        });
    }

    #[test]
    fn rate_limit_unknown_subfield_is_a_parse_error() {
        // `deny_unknown_fields` must extend into the rate_limit block:
        // a typo'd knob (`rps`) must not silently vanish.
        let raw = r#"
providers:
  - name: p
    type: anthropic
    api_key: sk-test
    rate_limit:
      rps: 5
models:
  - name: gpt-4
    primary: p
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(format!("{err}").contains("rps"), "unexpected error: {err}");
    }

    #[test]
    fn rate_limit_max_wait_defaults_to_2000ms() {
        let rl = RateLimitConfig {
            rpm: Some(1),
            tpm: None,
            max_concurrency: None,
            max_wait_ms: None,
        };
        assert_eq!(rl.max_wait(), std::time::Duration::from_millis(2000));
        let rl = RateLimitConfig {
            max_wait_ms: Some(0),
            ..rl
        };
        assert_eq!(rl.max_wait(), std::time::Duration::ZERO);
    }

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
    fn max_request_bytes_defaults_to_32_mib_and_can_be_overridden() {
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert_eq!(cfg.server.max_request_bytes, default_max_request_bytes());
        assert_eq!(cfg.server.max_request_bytes, 32 * 1024 * 1024);

        let raw = r#"
server:
  max_request_bytes: 1048576
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert_eq!(cfg.server.max_request_bytes, 1_048_576);
    }

    #[test]
    fn validation_rejects_a_zero_body_limit() {
        // 0 is NOT "unlimited" in axum: `DefaultBodyLimit::max(0)` wraps
        // the body in `Limited::new(body, 0)`, which rejects every body
        // that carries data. Wiring it through would brick the proxy,
        // so the configuration is refused at startup instead.
        let raw = r#"
server:
  max_request_bytes: 0
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(
            format!("{err}").contains("max_request_bytes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn upstream_details_are_exposed_unless_the_config_says_otherwise() {
        // Audit R0: the switch exists to *reduce* what a client can
        // see, so its default must keep today's behaviour.
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert!(cfg.server.expose_upstream_details);

        let raw = r#"
server:
  expose_upstream_details: false
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        assert!(!cfg.server.expose_upstream_details);
    }

    #[test]
    fn config_rejects_a_provider_name_that_is_not_header_safe() {
        // Audit R2: a provider name lands in the
        // `x-llmproxy-failed-providers` header, so it must be a legal
        // header value.
        let raw = r#"
providers:
  - name: "bad name"
    type: github_copilot
models:
  - name: gpt-4
    primary: "bad name"
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(
            format!("{err}").contains("must be non-empty and contain only ASCII"),
            "{err}"
        );
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
    fn config_rejects_unknown_copilot_account_type() {
        // Audit L4: `account_type` is interpolated into the token host
        // (`https://api.{account_type}.githubcopilot.com`); a typo must
        // fail validation — naming the provider — instead of pointing
        // the GitHub token at an unintended subdomain.
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
    account_type: indivdual
models:
  - name: gpt-4
    primary: copilot
"#;
        let err = Config::parse(raw).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("account_type must be one of"), "{msg}");
        assert!(msg.contains("indivdual"), "{msg}");
        assert!(msg.contains("'copilot'"), "{msg}");
    }

    #[test]
    fn config_accepts_all_documented_copilot_account_types() {
        for account_type in ["individual", "business", "enterprise"] {
            let raw = format!(
                r#"
providers:
  - name: copilot
    type: github_copilot
    account_type: {account_type}
models:
  - name: gpt-4
    primary: copilot
"#
            );
            let cfg = Config::parse(&raw).unwrap();
            let ProviderConfig::GithubCopilot {
                account_type: got, ..
            } = cfg.find_provider("copilot").unwrap()
            else {
                panic!("expected a github_copilot provider");
            };
            assert_eq!(got, account_type);
        }

        // Absent field → the documented default (`individual`) is
        // unaffected by the whitelist.
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let cfg = Config::parse(raw).unwrap();
        let ProviderConfig::GithubCopilot { account_type, .. } =
            cfg.find_provider("copilot").unwrap()
        else {
            panic!("expected a github_copilot provider");
        };
        assert_eq!(account_type, "individual");
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
        assert_eq!(
            expand_env_vars("literal-$9-${UNCLOSED"),
            "literal-$9-${UNCLOSED"
        );
    }

    #[test]
    fn env_expansion_bash_default_unset_var_uses_default() {
        std::env::remove_var("LLMPROXY_DEFINITELY_NOT_SET");
        let expanded = expand_env_vars("key=${LLMPROXY_DEFINITELY_NOT_SET:-fallback}-end");
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
        let expanded = expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_QUOTED:-\"hello world\"}-e");
        assert_eq!(expanded, "k=hello world-e");
    }

    #[test]
    fn env_expansion_bash_default_strips_single_quotes() {
        std::env::remove_var("LLMPROXY_BASH_DEFAULT_SQUOTED");
        let expanded = expand_env_vars("k=${LLMPROXY_BASH_DEFAULT_SQUOTED:-'hello world'}-e");
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
                token_store_key: None,
                model_rewrite: HashMap::new(),
                use_proxy: false,
            rate_limit: None,
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

    /// The Copilot token-store key: present-and-well-formed or absent.
    /// An empty value is a missing secret, not "no key".
    mod token_store_key_validation {
        use super::*;

        const VALID_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        fn config_with_key(value: &str) -> Config {
            Config {
                providers: vec![ProviderConfig::GithubCopilot {
                    name: "copilot".to_string(),
                    vscode_version: default_vscode_version(),
                    account_type: default_account_type(),
                    token_store_key: Some(secret_string(value)),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                rate_limit: None,
                }],
                models: vec![ModelConfig {
                    name: "model".to_string(),
                    primary: "copilot".to_string(),
                    fallback_chain: Vec::new(),
                    cooldown_seconds: 300,
                    max_retries_per_provider: 2,
                    max_retries_total: 2,
                }],
                ..Config::default()
            }
        }

        #[test]
        fn an_absent_key_is_fine() {
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
            match &cfg.providers[0] {
                ProviderConfig::GithubCopilot {
                    token_store_key, ..
                } => {
                    assert!(token_store_key.is_none());
                }
                other => panic!("expected a copilot provider, got {other:?}"),
            }
        }

        #[test]
        fn a_64_hex_char_key_is_accepted() {
            config_with_key(VALID_KEY).validate().unwrap();
        }

        #[test]
        fn an_empty_key_is_rejected() {
            let err = config_with_key("").validate().unwrap_err();
            assert!(
                err.to_string().contains("token_store_key is empty"),
                "unexpected error: {err}"
            );
        }

        #[test]
        fn a_63_char_key_is_rejected() {
            let err = config_with_key(&VALID_KEY[..63]).validate().unwrap_err();
            assert!(
                err.to_string().contains("must be 64 characters long"),
                "unexpected error: {err}"
            );
        }

        #[test]
        fn a_non_hex_key_is_rejected() {
            let err = config_with_key(&"z".repeat(64)).validate().unwrap_err();
            assert!(
                err.to_string().contains("must be hexadecimal"),
                "unexpected error: {err}"
            );
        }

        #[test]
        fn the_key_is_redacted_when_the_config_is_serialized() {
            let cfg = config_with_key(VALID_KEY);
            let dumped = serde_saphyr::to_string(&cfg).expect("config serializes");
            assert!(!dumped.contains(VALID_KEY), "{dumped}");
            assert!(dumped.contains("[REDACTED]"), "{dumped}");
        }
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
                token_store_key: None,
                model_rewrite: HashMap::new(),
                use_proxy: false,
            rate_limit: None,
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
                    token_store_key: None,
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                rate_limit: None,
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
                    provider_routing: None,
                    thinking_when_absent: ThinkingWhenAbsent::Omit,

                    user_agent: None,
                    referer: None,
                    headers: std::collections::BTreeMap::new(),
                    rate_limit: None,
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
                    provider_routing: None,
                    reasoning_echo: false,

                    user_agent: None,
                    referer: None,
                    headers: std::collections::BTreeMap::new(),
                    rate_limit: None,
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

                    user_agent: None,
                    referer: None,
                    headers: std::collections::BTreeMap::new(),
                    rate_limit: None,
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
    fn duplicate_mapping_keys_keep_the_last_value() {
        // serde_yaml kept the later value for duplicate keys in a mapping,
        // and `config.example.yaml` documents that for `model_rewrite`.
        // serde-saphyr rejects duplicates by default; `yaml_options()`
        // opts back into last-wins so an existing config keeps loading.
        let raw = r#"
providers:
  - name: oa
    type: openai_compat
    api_key: "k"
    api_base: "https://example.test/v1"
    model_rewrite:
      "claude-a": "first"
      "claude-a": "second"
models:
  - name: m
    primary: oa
"#;
        let cfg = Config::parse(raw).unwrap();
        let rewrite = cfg.find_provider("oa").unwrap().model_rewrite();
        assert_eq!(rewrite.len(), 1);
        assert_eq!(rewrite.get("claude-a").unwrap(), "second");
    }

    #[test]
    fn yaml_merge_keys_are_not_expanded() {
        // serde_yaml had no YAML 1.1 merge-key support: `<<` was an
        // ordinary key. serde-saphyr expands merges by default, which
        // would let one mapping silently inject entries into another
        // inside a security-relevant config. `AsOrdinary` keeps the old
        // behavior, so the merge key holds a mapping where a string is
        // expected and the whole config is rejected. With the default
        // policy this parse would succeed with the merged entry present.
        let raw = r#"
providers:
  - name: oa
    type: openai_compat
    api_key: "k"
    api_base: "https://example.test/v1"
    model_rewrite: &base
      "claude-a": "upstream-a"
  - name: oa2
    type: openai_compat
    api_key: "k"
    api_base: "https://example.test/v1"
    model_rewrite:
      <<: *base
      "claude-b": "upstream-b"
models:
  - name: m
    primary: oa
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(matches!(err, ProxyError::Yaml(_)), "got: {err:?}");
    }

    #[test]
    fn parse_errors_never_echo_the_source_line() {
        // `with_snippet` defaults to true in serde-saphyr: the error
        // would quote the offending line *and its neighbours*, so an
        // error on the line after `api_key:` writes the key into the
        // startup log. `yaml_options()` turns snippets off; this test
        // fails if that ever regresses.
        let raw = r#"
server:
  api_key: "sk-SUPERSECRET-SENTINEL"
bogus: 1
providers:
  - name: copilot
    type: github_copilot
models:
  - name: gpt-4
    primary: copilot
"#;
        let err = Config::parse(raw).unwrap_err();
        let message = err.to_string();
        assert!(
            !message.contains("sk-SUPERSECRET-SENTINEL"),
            "the parse error leaked the api_key value: {message}"
        );
        assert!(
            message.contains("bogus"),
            "expected the offending field to be named: {message}"
        );
    }

    #[test]
    fn non_string_mapping_key_fails_loudly() {
        // The generic pre-pass parses into `serde_json::Value`, which
        // only has string keys, so a numeric top-level key makes it fall
        // back to the default `secrets:` config. The main parse must
        // still reject the file rather than start with the block
        // silently ignored.
        let raw = r#"
1: nope
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
                ..
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
            ProviderConfig::OpenaiResponses { model_rewrite, .. } => {
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

    /// `provider_routing` round-trips through YAML into both variants
    /// that support it. Lists are preserved verbatim (no normalization)
    /// because slugs are operator-supplied and OpenRouter documents no
    /// case-folding.
    #[test]
    fn provider_routing_parses_from_yaml() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      only: ["DeepInfra", "Together"]
      ignore: ["Azure"]
      sort: throughput
      allow_fallbacks: false
      data_collection: deny
      zdr: true
      quantizations: ["fp8", "bf16"]
      max_price: {prompt: "0.30", completion: "1.20"}
      preferred_min_throughput: {p50: 40, p90: 20}
  - name: or_compat
    type: openai_compat
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      only: ["DeepInfra"]
      sort: {by: latency, partition: model}
models:
  - name: m
    primary: or_anthropic
"#;
        let cfg = Config::parse(raw).unwrap();
        let routing = cfg.providers[0]
            .provider_routing()
            .expect("anthropic routing present");
        assert_eq!(
            routing.only.as_deref(),
            Some(["DeepInfra".to_string(), "Together".to_string()].as_slice())
        );
        assert_eq!(
            routing.ignore.as_deref(),
            Some(["Azure".to_string()].as_slice())
        );
        assert_eq!(routing.sort, Some(SortPreference::Name(SortKey::Throughput)));
        assert_eq!(routing.allow_fallbacks, Some(false));
        assert_eq!(routing.data_collection, Some(DataCollection::Deny));
        assert_eq!(routing.zdr, Some(true));
        assert_eq!(
            routing.quantizations.as_deref(),
            Some([Quantization::Fp8, Quantization::Bf16].as_slice())
        );
        let price = routing.max_price.as_ref().expect("max_price");
        assert_eq!(price.prompt.as_deref(), Some("0.30"));
        assert_eq!(price.completion.as_deref(), Some("1.20"));
        match routing
            .preferred_min_throughput
            .as_ref()
            .expect("cutoffs")
        {
            ThroughputCutoffs::Percentiles(map) => {
                assert_eq!(map.p50, Some(40.0));
                assert_eq!(map.p90, Some(20.0));
            }
            other => panic!("expected percentile map, got {other:?}"),
        }

        let routing = cfg.providers[1]
            .provider_routing()
            .expect("compat routing present");
        assert_eq!(
            routing.sort,
            Some(SortPreference::Config(SortConfig {
                by: Some(SortKey::Latency),
                partition: Some(Partition::Model),
            }))
        );
    }

    /// An omitted block yields `None` on both variants, so every existing
    /// config keeps its current wire bytes (no `provider` key injected).
    #[test]
    fn provider_routing_defaults_to_none() {
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
        assert!(cfg.providers[0].provider_routing().is_none());
        assert!(cfg.providers[1].provider_routing().is_none());
    }

    /// The pre-`provider_routing` top-level keys are removed, not
    /// aliased: a stale config refuses to start instead of silently
    /// running unconstrained (plan §3.3). This is the migration guard.
    #[test]
    fn removed_top_level_routing_keys_are_a_parse_error() {
        for key in ["provider_allow", "provider_ignore"] {
            let raw = format!(
                r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    {key}: ["DeepInfra"]
models:
  - name: m
    primary: or_anthropic
"#
            );
            let err = Config::parse(&raw)
                .expect_err("a removed top-level routing key must fail to parse");
            assert!(
                format!("{err}").contains(key),
                "error should name the offending key ({key}): {err}"
            );
        }
    }

    /// A misspelled key inside the block is a hard parse error, not a
    /// silently-ignored constraint.
    #[test]
    fn provider_routing_rejects_unknown_fields() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      provider_allowed: ["DeepInfra"]
models:
  - name: m
    primary: or_anthropic
"#;
        assert!(
            Config::parse(raw).is_err(),
            "a misspelled provider_routing key must fail to parse"
        );
    }

    /// `provider_routing: {}` is a mistake — it means nothing and can
    /// only mislead. Same rule as an empty `rate_limit: {}`.
    #[test]
    fn provider_routing_empty_block_is_rejected() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing: {}
models:
  - name: m
    primary: or_anthropic
"#;
        let err = Config::parse(raw).expect_err("empty provider_routing");
        assert!(format!("{err}").contains("provider_routing"), "{err}");
    }

    /// `sort: {by, partition}` needs `by`; a bare `partition` cannot
    /// rank anything.
    #[test]
    fn provider_routing_sort_object_requires_by() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      sort: {partition: model}
models:
  - name: m
    primary: or_anthropic
"#;
        let err = Config::parse(raw).expect_err("sort without by");
        assert!(format!("{err}").contains("sort.by"), "{err}");
    }

    /// A typo inside the object form must not be silently accepted —
    /// `deny_unknown_fields` on `SortConfig` is what catches it.
    #[test]
    fn provider_routing_sort_object_typo_is_a_parse_error() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      sort: {by: throughput, partiton: model}
models:
  - name: m
    primary: or_anthropic
"#;
        let err = Config::parse(raw).expect_err("partiton typo");
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains("sort") || msg.contains("partiton"),
            "error should be actionable: {err}"
        );
    }

    /// `order` makes `sort` irrelevant upstream; configuring both is
    /// refused rather than silently dropping one of them.
    #[test]
    fn provider_routing_sort_and_order_together_are_rejected() {
        let raw = r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      sort: price
      order: ["DeepInfra"]
models:
  - name: m
    primary: or_anthropic
"#;
        let err = Config::parse(raw).expect_err("sort + order");
        assert!(format!("{err}").contains("sort"), "{err}");
    }

    /// An empty list can only mean "no filter", which is spelled by
    /// omitting the key — accepting `only: []` would silently do nothing.
    /// Blank and duplicated slugs are operator mistakes too.
    #[test]
    fn provider_routing_list_rules() {
        let base = |body: &str| {
            format!(
                r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
{body}
models:
  - name: m
    primary: or_anthropic
"#
            )
        };
        for body in [
            "      only: []",
            "      ignore: []",
            "      order: []",
            "      only: [\"DeepInfra\", \"DeepInfra\"]",
            "      ignore: [\"  \"]",
        ] {
            assert!(
                Config::parse(&base(body)).is_err(),
                "provider_routing with `{body}` must be rejected"
            );
        }
        assert!(Config::parse(&base("      only: [\"DeepInfra\"]")).is_ok());
    }

    /// `max_price` values are USD strings forwarded verbatim; only their
    /// shape is validated locally. `"0"` is meaningful (free endpoints),
    /// `nan`/`inf`/negative are not.
    #[test]
    fn provider_routing_max_price_validation() {
        let base = |body: &str| {
            format!(
                r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      max_price: {body}
models:
  - name: m
    primary: or_anthropic
"#
            )
        };
        for body in ["{}", "{prompt: cheap}", "{prompt: \"-1\"}", "{prompt: \"inf\"}", "{prompt: \"nan\"}"] {
            assert!(
                Config::parse(&base(body)).is_err(),
                "max_price: {body} must be rejected"
            );
        }
        // A typo'd key is rejected, not silently dropped (an unfiltered
        // output-price ceiling is exactly the silent-overcharge case).
        assert!(Config::parse(&base("{completiont: \"1.20\"}")).is_err());
        // "0" is legal and is forwarded as the same string.
        let cfg = Config::parse(&base("{prompt: \"0\"}")).unwrap();
        let routing = cfg.providers[0].provider_routing().expect("routing");
        let price = routing.max_price.as_ref().expect("max_price");
        assert_eq!(price.prompt.as_deref(), Some("0"));
    }

    /// `quantizations` is an allow-list; an empty one matches nothing,
    /// an unknown level is a typo. Every documented level parses.
    #[test]
    fn provider_routing_quantizations_validation() {
        let base = |body: &str| {
            format!(
                r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      quantizations: {body}
models:
  - name: m
    primary: or_anthropic
"#
            )
        };
        assert!(Config::parse(&base("[]")).is_err());
        assert!(Config::parse(&base("[\"fp64\"]")).is_err());
        let cfg = Config::parse(&base(
            r#"["int4","int8","fp4","mxfp4","nvfp4","fp6","fp8","mxfp8","fp16","bf16","fp32","unknown"]"#,
        ))
        .unwrap();
        let routing = cfg.providers[0].provider_routing().expect("routing");
        assert_eq!(
            routing.quantizations.as_ref().map(Vec::len),
            Some(12),
            "every documented quantization level must parse"
        );
    }

    /// `preferred_*` cutoffs are positive and finite, scalar or
    /// percentile map; an object with no percentile set is a mistake.
    #[test]
    fn provider_routing_preferred_cutoffs_validation() {
        let base = |body: &str| {
            format!(
                r#"
providers:
  - name: or_anthropic
    type: anthropic
    api_key: k
    api_base: https://openrouter.ai/api/v1
    provider_routing:
      {body}
models:
  - name: m
    primary: or_anthropic
"#
            )
        };
        assert!(Config::parse(&base("preferred_min_throughput: 40")).is_ok());
        assert!(Config::parse(&base("preferred_max_latency: 2.5")).is_ok());
        assert!(Config::parse(&base("preferred_max_latency: {p50: 1, p99: 4}")).is_ok());
        for body in [
            "preferred_min_throughput: 0",
            "preferred_min_throughput: -5",
            "preferred_max_latency: {}",
            "preferred_max_latency: {p50: 0}",
            "preferred_max_latency: {p50: -1}",
        ] {
            assert!(
                Config::parse(&base(body)).is_err(),
                "`{body}` must be rejected"
            );
        }
    }

    /// The routing block is only accepted on the two variants that can
    /// drive OpenRouter's `provider` object. `github_copilot` and
    /// `openai_responses` reject it (pins the Phase-1 surface; Phase 2
    /// may extend `openai_responses`).
    #[test]
    fn provider_routing_is_rejected_on_variants_without_it() {
        for ty in [
            "type: github_copilot",
            "type: openai_responses\n    api_base: https://api.example.test/v1",
        ] {
            let raw = format!(
                r#"
providers:
  - name: p
    {ty}
    api_key: k
    provider_routing:
      only: ["DeepInfra"]
models:
  - name: m
    primary: p
"#
            );
            assert!(
                Config::parse(&raw).is_err(),
                "provider_routing must not parse on `{ty}`"
            );
        }
    }

    /// A routing block on a non-OpenRouter host still parses (the
    /// provider ignores it with a startup warning); this pins that the
    /// config layer does not try to police upstream hosts.
    #[test]
    fn provider_routing_on_non_openrouter_host_parses() {
        let raw = r#"
providers:
  - name: ds
    type: anthropic
    api_key: k
    api_base: https://api.deepseek.com/anthropic
    provider_routing:
      only: ["DeepInfra"]
models:
  - name: m
    primary: ds
"#;
        let cfg = Config::parse(raw).unwrap();
        assert!(cfg.providers[0].provider_routing().is_some());
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
                thinking_when_absent,
                ..
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
                thinking_when_absent,
                ..
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
                thinking_when_absent,
                ..
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
                thinking_when_absent,
                ..
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
        let s = serde_saphyr::to_string(&cfg).unwrap();
        assert!(s.contains("thinking_when_absent: disabled"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        match &reparsed.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent,
                ..
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
        let s = serde_saphyr::to_string(&cfg).unwrap();
        assert!(s.contains("thinking_when_absent: effort_low"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        match &reparsed.providers[0] {
            ProviderConfig::Anthropic {
                thinking_when_absent,
                ..
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
            ProviderConfig::OpenaiCompat { api_key, .. } => api_key.expose_secret().to_string(),
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
            "secrets:\n  provider: file\n  file:\n    path: /run/llmproxy/llmproxy.env\n    required: false\n",
        );
        let cfg = Config::parse(&raw).unwrap();
        let s = serde_saphyr::to_string(&cfg).unwrap();
        assert!(s.contains("provider: file"), "{s}");
        let reparsed = Config::parse(&s).unwrap();
        assert_eq!(reparsed.secrets.provider, SecretsProvider::File);
        assert_eq!(
            reparsed.secrets.file_path(),
            Some("/run/llmproxy/llmproxy.env")
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
            provider_routing: None,
            reasoning_echo: false,

            user_agent: None,
            referer: None,
            headers: std::collections::BTreeMap::new(),
            rate_limit: None,
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
        let dumped = serde_saphyr::to_string(&cfg).unwrap();
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
                expose_upstream_details: true,
                max_request_bytes: crate::config::default_max_request_bytes(),
            },
            providers: vec![provider_with_secret()],
            ..Config::default()
        };
        let dumped = serde_saphyr::to_string(&cfg).unwrap();
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
                expose_upstream_details: true,
                max_request_bytes: crate::config::default_max_request_bytes(),
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
        let dumped = serde_saphyr::to_string(&cfg).unwrap();
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
                expose_upstream_details: true,
                max_request_bytes: crate::config::default_max_request_bytes(),
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
        let dumped = serde_saphyr::to_string(&cfg).unwrap();
        let reparsed = Config::parse(&dumped).unwrap();
        assert_eq!(
            reparsed.server.api_key.as_ref().map(|k| k.expose_secret()),
            Some("[REDACTED]")
        );
    }

    // ── per-provider identity fields ──────────────────────────────────
    //
    // `user_agent` / `referer` / `headers` are per-provider and optional.
    // A missing key is not an error; a wrong-typed value is a startup
    // error (serde is type-driven, and a mistyped key must never be
    // silently ignored).

    const IDENTITY_PROVIDER_YAML: &str = r#"
providers:
  - name: compat
    type: openai_compat
    api_key: "key"
    api_base: "https://example.invalid"
models:
  - name: m
    primary: compat
"#;

    #[test]
    fn the_provider_identity_fields_are_absent_by_default() {
        let cfg = Config::parse(IDENTITY_PROVIDER_YAML).unwrap();
        let provider = &cfg.providers[0];
        expect_variant!(
            provider,
            ProviderConfig::OpenaiCompat { user_agent, referer, headers, .. } => {
                assert!(user_agent.is_none());
                assert!(referer.is_none());
                assert!(headers.is_empty());
            }
        );
    }

    #[test]
    fn the_provider_identity_fields_parse_on_every_supported_provider() {
        let raw = r#"
providers:
  - name: router
    type: anthropic
    api_key: "key"
    api_base: "https://example.invalid"
    user_agent: "omp/18.14.0"
    referer: "https://omp.sh/"
    headers:
      X-Title: "omp"
  - name: compat
    type: openai_compat
    api_key: "key"
    api_base: "https://example.invalid"
    user_agent: "omp/18.14.0"
    referer: "https://omp.sh/"
    headers:
      X-Title: "omp"
  - name: responses
    type: openai_responses
    api_key: "key"
    api_base: "https://example.invalid"
    user_agent: "omp/18.14.0"
    referer: "https://omp.sh/"
    headers:
      X-Title: "omp"
models:
  - name: m
    primary: router
"#;
        let cfg = Config::parse(raw).unwrap();
        for provider in &cfg.providers {
            let (user_agent, referer, headers) = match provider {
                ProviderConfig::Anthropic {
                    user_agent,
                    referer,
                    headers,
                    ..
                }
                | ProviderConfig::OpenaiCompat {
                    user_agent,
                    referer,
                    headers,
                    ..
                }
                | ProviderConfig::OpenaiResponses {
                    user_agent,
                    referer,
                    headers,
                    ..
                } => (user_agent, referer, headers),
                ProviderConfig::GithubCopilot { .. } => continue,
            };
            assert_eq!(user_agent.as_deref(), Some("omp/18.14.0"));
            assert_eq!(referer.as_deref(), Some("https://omp.sh/"));
            assert_eq!(headers.get("X-Title").map(String::as_str), Some("omp"));
        }
    }

    #[test]
    fn a_wrong_typed_provider_identity_field_is_a_startup_error() {
        let raw = r#"
providers:
  - name: compat
    type: openai_compat
    api_key: "key"
    api_base: "https://example.invalid"
    user_agent: 18
models:
  - name: m
    primary: compat
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(matches!(err, ProxyError::Yaml(_)), "got: {err:?}");
    }

    #[test]
    fn the_copilot_provider_rejects_the_identity_fields() {
        // GitHub gates on the client version, so the copilot provider
        // owns its identity. `deny_unknown_fields` makes a stray
        // `user_agent:` there a config error rather than a silent no-op.
        let raw = r#"
providers:
  - name: copilot
    type: github_copilot
    user_agent: "omp/18.14.0"
models:
  - name: m
    primary: copilot
"#;
        let err = Config::parse(raw).unwrap_err();
        assert!(matches!(err, ProxyError::Yaml(_)), "got: {err:?}");
    }

    #[test]
    fn the_absent_identity_fields_are_not_serialized() {
        // Field omission is correctness: the serialized config must not
        // grow empty identity keys.
        let cfg = Config::parse(IDENTITY_PROVIDER_YAML).unwrap();
        let dumped = serde_saphyr::to_string(&cfg).unwrap();
        // The top-level `user_agent:` is always present; the provider's
        // own identity fields must not add a second one.
        assert_eq!(dumped.matches("user_agent").count(), 1, "{dumped}");
        assert!(!dumped.contains("referer"), "{dumped}");
        assert!(!dumped.contains("headers"), "{dumped}");
    }
}
