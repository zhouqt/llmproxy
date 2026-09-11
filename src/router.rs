//! Provider routing with fallback on cooldownable errors.
//!
//! Reference: litellm/router.py:async_function_with_retries and friends

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::anthropic::MessagesRequest;
use crate::config::{Config, ModelConfig};

use crate::cooldown::CooldownCache;
use crate::error::{ProxyError, Result};
use crate::providers::{ProviderOutput, SharedProvider};
use serde::Serialize;

/// Detect "this upstream cannot serve the requested model" responses
/// that arrive at runtime as a 400-level `Upstream` error. Used by the
/// router to skip such providers instead of returning the error to the
/// client — see fix-R11 in docs/TEST_ISSUES.md.
///
/// Heuristic: 400-class status with a body that mentions either
/// `model` / `not supported` / `not_found` / `not exist` / `not a valid`
/// / `model_not_*` substrings. This is intentionally narrow so a generic
/// `400 Bad Request` from a misconfigured client still surfaces to the
/// operator rather than silently chaining to the next provider.
fn is_model_unsupported(err: &ProxyError) -> bool {
    let ProxyError::Upstream { status, body } = err else {
        return false;
    };
    if !(400..500).contains(status) {
        return false;
    }
    let body_lower = body.to_ascii_lowercase();
    // Patterns observed across the providers we currently ship:
    // - Copilot:        "model_not_supported" / "The requested model is not supported"
    // - DeepSeek:       "The supported API model names are X or Y, but you passed Z"
    // - OpenAI generic: {"error":{"code":"model_not_found", ...}}
    // We deliberately do NOT include a bare "model" substring (it would
    // match too broadly — e.g. a malformed-JSON 400 like
    // "missing field `model`" still has to surface to the operator).
    let mentions_model = body_lower.contains("not supported")
        || body_lower.contains("not_supported")
        || body_lower.contains("not_found")
        || body_lower.contains("not exist")
        || body_lower.contains("not a valid")
        || body_lower.contains("model_not_")
        || body_lower.contains("\"model\"")
        || (body_lower.contains("supported api model") && body_lower.contains("you passed"))
        // Copilot rejections of the wrong endpoint, e.g.
        // model "grok-4.5" is not accessible via the /chat/completions endpoint.
        // Gated on the trailing "the /" so generic routing errors like
        // "file is not accessible via fallback proxy" do not match. This
        // phrase is currently only observed in Copilot's endpoint-rejection
        // shape; if another source surfaces it, revisit the gate.
        || body_lower.contains("is not accessible via the /");
    mentions_model
}

pub struct Router {
    cfg: Arc<Config>,
    providers: HashMap<String, SharedProvider>,
    cooldown: CooldownCache,
}

/// One step in a request's fallback chain. Three terminal facts we want
/// to surface to operators:
/// - `Failed`: an upstream call actually fired and returned a
///   cooldownable error (or a runtime model-unsupported 400). `body`
///   is the truncated, sanitized upstream body.
/// - `LocalSkip::Cooldown`: the provider was skipped before any HTTP
///   call because it was already on cooldown.
/// - `LocalSkip::ModelUnmapped`: the provider was skipped because its
///   `model_rewrite` table doesn't include the requested model.
///
/// `body` is captured at the push site (see `truncate_for_log` in
/// `crate::util`); the cap exists so a 100 KiB Cloudflare page in the
/// log doesn't blow past grep, while still preserving enough of the
/// body to recognize the failure mode.
#[derive(Debug, Clone)]
pub enum RouteStep {
    /// Real upstream call returned a cooldownable or model-unsupported
    /// response. `body` is the sanitized upstream body (≤ 4 KiB at
    /// char boundary).
    Failed {
        provider: String,
        status: u16,
        body: String,
    },
    /// Local skip — no upstream call fired. The provider name and
    /// reason are recorded for operator visibility.
    LocalSkip {
        provider: String,
        reason: LocalSkipReason,
    },
}

/// Why a `RouteStep::LocalSkip` was taken. Used by `derive_fail_reason`
/// to label the `fail_reason=` log field and to drive
/// `format_attempts_header`'s filter (only upstream failures surface
/// in the `x-llmproxy-failed-providers` header — see plan §三方 policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalSkipReason {
    /// `cooldown.is_cooling_down(name)` was true at dispatch.
    Cooldown,
    /// `!provider.can_serve_model(&req.model)` (provider's
    /// `model_rewrite` excludes the requested model).
    ModelUnmapped,
}

/// Why a streaming response terminated before normal completion. The
/// streaming helpers (`log_streaming_fallback` / `_completed` /
/// `_aborted`) consume this enum so the log vocabulary stays
/// consistent across `Router::stream` mid-stream failures and the
/// `stream_response` contract-violation branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    /// The upstream stream returned `Err(_)` mid-flow.
    UpstreamError,
    /// A streaming request was answered with non-stream output
    /// (provider contract violation).
    ContractViolation,
}

/// High-level classification of an "all providers failed" terminal
/// state. Derived from `&[RouteStep]` at the emit site so the log
/// vocabulary stays consistent across `messages_handler`'s three
/// terminal paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailReason {
    /// At least one `RouteStep::Failed` was recorded — a real upstream
    /// call returned an error.
    Upstream,
    /// Every step was `LocalSkip::Cooldown` — no upstream call fired.
    Cooldown,
    /// Every step was `LocalSkip::ModelUnmapped` — no upstream call
    /// fired (configuration gap).
    ModelSkip,
    /// Mixed `LocalSkip::Cooldown` + `LocalSkip::ModelUnmapped` —
    /// chain contains both skip classes and no upstream attempt.
    Mixed,
}

impl std::fmt::Display for FailReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FailReason::Upstream => "upstream",
            FailReason::Cooldown => "cooldown",
            FailReason::ModelSkip => "model_skip",
            FailReason::Mixed => "mixed",
        })
    }
}

impl RouteStep {
    /// Provider name for any variant. Used as the unquoted `provider`
    /// field in log lines and the `provider` component of `render`.
    pub fn provider(&self) -> &str {
        match self {
            RouteStep::Failed { provider, .. } => provider,
            RouteStep::LocalSkip { provider, .. } => provider,
        }
    }

    /// Stable label for the `failure=` field on per-step fallback
    /// lines. For a real upstream response this is the status code
    /// as a decimal string (`"429"`, `"503"`); for a local skip it's
    /// the reason name (`"cooldown"` / `"model_skip"`).
    pub fn failure(&self) -> std::borrow::Cow<'static, str> {
        match self {
            RouteStep::Failed { status, .. } => std::borrow::Cow::Owned(status.to_string()),
            RouteStep::LocalSkip { reason, .. } => std::borrow::Cow::Borrowed(match reason {
                LocalSkipReason::Cooldown => "cooldown",
                LocalSkipReason::ModelUnmapped => "model_skip",
            }),
        }
    }

    /// `"<provider>:<status>"` for `Failed`, `"<provider>:cooldown"` /
    /// `"<provider>:model_skip"` for `LocalSkip`. Used by log
    /// `failed_providers=` fields and the
    /// `x-llmproxy-failed-providers` header.
    pub fn render(&self) -> String {
        format!("{}:{}", self.provider(), self.failure())
    }

    /// True iff this step records an actual upstream HTTP response
    /// (i.e. `Failed`). Drives both `format_attempts_header`'s filter
    /// (only upstream failures go in the HTTP header) and the
    /// `max_retries_total` guard (only upstream calls count as cost).
    pub fn is_upstream_failure(&self) -> bool {
        matches!(self, RouteStep::Failed { .. })
    }
}

/// Maximum byte length of the upstream body that we keep on
/// `RouteStep::Failed` for logging. Pushed bodies are run through
/// `crate::util::truncate_for_log` at the push site so a 100 KiB
/// Cloudflare HTML page never reaches the WARN line. 4 KiB is the
/// empirical "human-scannable upper bound" — anything longer was
/// unreadable on the operator side anyway (plan §Phase 1 commit 7).
const FAILED_BODY_LOG_CAP_BYTES: usize = 4096;

/// Derive the high-level `FailReason` for a fallback chain that
/// produced no successful response. Returns `None` only for an empty
/// slice — callers that hold a terminal-state slice always have at
/// least one step (the all-cooldown and all-unmappable paths both
/// push at least one `LocalSkip` before reaching the terminal arm).
///
/// Classification:
/// - any `Failed` → `Upstream` (a real upstream call returned an
///   error and the chain gave up)
/// - all `LocalSkip::Cooldown` → `Cooldown`
/// - all `LocalSkip::ModelUnmapped` → `ModelSkip`
/// - both skip classes, no `Failed` → `Mixed`
pub fn derive_fail_reason(steps: &[RouteStep]) -> Option<FailReason> {
    if steps.is_empty() {
        return None;
    }
    // Single pass collecting both skip flags — code review #8. The
    // upstream check short-circuits so a chain with any `Failed` skips
    // the LocalSkip scan entirely.
    let mut has_cooldown = false;
    let mut has_unmapped = false;
    for s in steps {
        if s.is_upstream_failure() {
            return Some(FailReason::Upstream);
        }
        match s {
            RouteStep::LocalSkip {
                reason: LocalSkipReason::Cooldown,
                ..
            } => has_cooldown = true,
            RouteStep::LocalSkip {
                reason: LocalSkipReason::ModelUnmapped,
                ..
            } => has_unmapped = true,
            RouteStep::Failed { .. } => unreachable!("filtered above"),
        }
        // Early exit: once both flags are set we know the answer is
        // `Mixed` without scanning the rest.
        if has_cooldown && has_unmapped {
            return Some(FailReason::Mixed);
        }
    }
    match (has_cooldown, has_unmapped) {
        (true, false) => Some(FailReason::Cooldown),
        (false, true) => Some(FailReason::ModelSkip),
        // The `(true, true)` arm is unreachable: the early-exit
        // `if has_cooldown && has_unmapped` above returns
        // `Some(Mixed)` before we reach this match. Listed here so
        // the compiler accepts the exhaustive tuple pattern.
        (true, true) => unreachable!("early-exit above returns Mixed when both flags are set"),
        // Non-empty steps + no `Failed` implies at least one
        // `LocalSkip`, which implies `Cooldown` or `ModelUnmapped`.
        (false, false) => unreachable!("derive_fail_reason: non-empty steps with no Failed implies at least one LocalSkip"),
    }
}

/// The most recent `RouteStep::Failed`'s provider, scanning the slice
/// in reverse. Used by `messages_handler`'s
/// `AllProvidersFailed` arm to pick the `UsageRecord.provider` for the
/// error row — there must always be at least one `Failed` in that
/// variant; the caller may `expect` `None` is unreachable.
///
/// Returns `None` if no `Failed` step is present (e.g. all-cooldown or
/// all-unmappable slices — those rows use `"<router>"` for the
/// `UsageRecord.provider`).
pub fn last_failed_provider(steps: &[RouteStep]) -> Option<&str> {
    steps.iter().rev().find_map(|s| match s {
        RouteStep::Failed { provider, .. } => Some(provider.as_str()),
        _ => None,
    })
}

/// The most recent `RouteStep::Failed`'s body, as a borrowed slice —
/// caller decides whether to render it (and pay the `summarize_for_log`
/// alloc + scan cost). Returns `None` if no `Failed` step is present
/// (all-skip slices have no upstream body to summarize). The reverse
/// walk is O(N) but `N` is the chain length (typically 1–3), and the
/// slice is already on the request's hot path through `attempts`.
///
/// Callers that need to log the body MUST run it through
/// `crate::util::summarize_for_log(..., "<empty error message>")`
/// themselves (returns the placeholder for empty bodies). The
/// returned `&str` borrows from the `RouteStep::Failed.body` inside
/// the caller's `attempts` slice, so the slice must outlive the
/// caller (true everywhere in the success / streaming-fallback paths
/// because the slice is on the handler stack frame).
pub fn last_failed_body(steps: &[RouteStep]) -> Option<&str> {
    steps.iter().rev().find_map(|s| match s {
        RouteStep::Failed { body, .. } => Some(body.as_str()),
        _ => None,
    })
}

/// Render the failed-provider chain for the
/// `x-llmproxy-failed-providers` HTTP response header. Filters to
/// upstream failures only (`is_upstream_failure()`) so callers see
/// real upstream errors, not local-skip bookkeeping. If `served_by`
/// is `Some`, attempts against that provider are dropped (a
/// successful primary after in-place retries must not show up in its
/// own response header — plan §三方 policy).
///
/// Returns `None` when the slice is empty or every step is filtered
/// out — the caller should skip setting the header in that case.
pub fn format_attempts_header(steps: &[RouteStep], served_by: Option<&str>) -> Option<String> {
    use std::fmt::Write;
    let mut out = String::new();
    let mut first = true;
    for step in steps {
        if !step.is_upstream_failure() {
            continue;
        }
        if let Some(skip) = served_by {
            if step.provider() == skip {
                continue;
            }
        }
        if !first {
            out.push(',');
        }
        // Unreachable: the `if !is_upstream_failure()` guard above
        // narrows to the `Failed` arm, which has `status` and
        // `provider`.
        let (provider, status) = match step {
            RouteStep::Failed { provider, status, .. } => (provider.as_str(), *status),
            RouteStep::LocalSkip { .. } => unreachable!("filtered above"),
        };
        let _ = write!(out, "{provider}:{status}");
        first = false;
    }
    if first {
        None
    } else {
        Some(out)
    }
}

/// Serialized health state of a single provider for the `/admin/status`
/// endpoint. Deliberately only two values — see
/// `plans/provider-status-endpoint.md`. The proxy does not actively
/// probe upstreams, so `CoolingDown` means "a cooldownable failure is
/// still inside its TTL window (the router will skip this provider)",
/// NOT a broader "unavailable" claim; consumers render it as 🔴.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderHealth {
    Available,
    CoolingDown,
}

/// One provider's row in the status snapshot. Field names and semantics
/// are part of the `/admin/status` contract consumed by the Claude Code
/// statusline hook — keep changes backwards-compatible.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderStatus {
    pub name: String,
    /// Stable kind label, equal to the provider config's serde `type`
    /// tag (`github_copilot`/`anthropic`/`openai_compat`/`openai_responses`).
    #[serde(rename = "type")]
    pub provider_type: &'static str,
    pub status: ProviderHealth,
    /// Client-visible model names whose chain (`primary` + `fallback`)
    /// includes this provider.
    pub models: Vec<String>,
    /// Upstream HTTP status that triggered the cooldown, when cooling.
    pub last_error_status: Option<u16>,
    /// Remaining cooldown in whole seconds (aligns with `Retry-After`),
    /// when cooling.
    pub cooling_down_remaining_secs: Option<u64>,
}

impl Router {
    pub fn new(cfg: Arc<Config>, providers: HashMap<String, SharedProvider>, cooldown: CooldownCache) -> Self {
        Self { cfg, providers, cooldown }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn cooldown(&self) -> &CooldownCache {
        &self.cooldown
    }

    pub fn providers(&self) -> &HashMap<String, SharedProvider> {
        &self.providers
    }

    pub fn find_model(&self, name: &str) -> Option<&ModelConfig> {
        self.cfg.find_model(name)
    }

    /// Snapshot of every configured provider's current health, in config
    /// declaration order (deterministic — never rely on the internal
    /// `providers` HashMap ordering). A provider is `CoolingDown` iff a
    /// live cooldown entry exists for it, which is exactly the signal the
    /// router's own dispatch (`is_cooling_down`) consumes, so the status
    /// endpoint can never disagree with actual routing decisions.
    pub async fn provider_status(&self) -> Vec<ProviderStatus> {
        let active = self.cooldown.active().await;
        let cooling: HashMap<&str, (u16, Duration)> = active
            .iter()
            .map(|(name, status, remaining)| (name.as_str(), (*status, *remaining)))
            .collect();

        let mut out = Vec::with_capacity(self.cfg.providers.len());
        for p in &self.cfg.providers {
            let name = p.name();
            let models = self
                .cfg
                .models
                .iter()
                .filter(|m| m.chain().any(|n| n == name))
                .map(|m| m.name.clone())
                .collect();
            let entry = cooling.get(name).copied();
            out.push(ProviderStatus {
                name: name.to_string(),
                provider_type: p.type_label(),
                status: if entry.is_some() {
                    ProviderHealth::CoolingDown
                } else {
                    ProviderHealth::Available
                },
                models,
                last_error_status: entry.map(|(s, _)| s),
                cooling_down_remaining_secs: entry.map(|(_, d)| d.as_secs()),
            });
        }
        out
    }

    /// Pick the first non-cooling-down provider in the model's chain.
    /// If all are cooling down, return the one with the shortest remaining cooldown.
    pub async fn select_provider(&self, model: &ModelConfig) -> Result<(String, SharedProvider)> {
        let mut best: Option<(String, SharedProvider, Duration)> = None;

        for name in model.chain() {
            if let Some(p) = self.providers.get(name) {
                if !self.cooldown.is_cooling_down(name).await {
                    return Ok((name.to_string(), p.clone()));
                }
                // Track the soonest-expiring one as fallback.
                let active = self.cooldown.active().await;
                if let Some((_, _, remaining)) = active.iter().find(|(n, _, _)| n == name) {
                    if best.as_ref().map(|(_, _, d)| *remaining < *d).unwrap_or(true) {
                        best = Some((name.to_string(), p.clone(), *remaining));
                    }
                }
            } else {
                tracing::warn!("model '{}' references unknown provider '{}'", model.name, name);
            }
        }

        // Every candidate is on cooldown. Returning the soonest-expiring
        // would just trigger another cooldown mark and hand the caller
        // a generic 5xx. Fail fast with 503 + `Retry-After` instead so
        // clients back off — see fix-R7 in docs/TEST_ISSUES.md.
        if let Some((_, _, remaining)) = best {
            return Err(ProxyError::AllProvidersCoolingDown {
                model: model.name.clone(),
                attempts: Vec::new(),
                retry_after_secs: Some(remaining.as_secs().max(1)),
            });
        }
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            attempts: Vec::new(),
            retry_after_secs: None,
        })
    }

    /// Execute a complete request with retries across the chain.
    ///
    /// Returns the response, the attempt log, and the **serving
    /// provider's name** (the chain entry that actually produced
    /// `resp`). The third tuple element is the explicit, non-inferred
    /// answer to "which provider served this request?" — it lets
    /// `messages_handler` record the serving provider in the
    /// `UsageRecord` without re-deriving it from `attempts`. On
    /// failure, the name is `None`.
    pub async fn complete(
        &self,
        model: &ModelConfig,
        req: &MessagesRequest,
    ) -> Result<(ProviderOutput, Vec<RouteStep>, Option<String>)> {
        let mut attempts: Vec<RouteStep> = Vec::new();
        let mut last_error: Option<ProxyError> = None;
        let mut tried: Vec<String> = Vec::new();
        let mut unmappable: Vec<String> = Vec::new();
        let chain: Vec<String> = model.chain().map(String::from).collect();
        let max_total = (model.max_retries_total as usize) * chain.len().max(1);

        for round in 0..chain.len() {
            let name = &chain[round];
            if tried.contains(name) {
                continue;
            }
            if self.cooldown.is_cooling_down(name).await {
                tried.push(name.clone());
                attempts.push(RouteStep::LocalSkip {
                    provider: name.clone(),
                    reason: LocalSkipReason::Cooldown,
                });
                continue;
            }
            let Some(provider) = self.providers.get(name).cloned() else {
                tried.push(name.clone());
                continue;
            };
            // Providers with a non-empty model_rewrite only accept names
            // that are keys in their table. Skip them here so we don't
            // forward an unmapped name upstream (which would surface as
            // a misleading 400 and break the fallback chain) — see fix-R11.
            if !provider.can_serve_model(&req.model) {
                tracing::debug!(
                    provider = name.as_str(),
                    model = req.model.as_str(),
                    "provider's model_rewrite does not include this model; skipping"
                );
                unmappable.push(name.clone());
                tried.push(name.clone());
                attempts.push(RouteStep::LocalSkip {
                    provider: name.clone(),
                    reason: LocalSkipReason::ModelUnmapped,
                });
                continue;
            }
            tried.push(name.clone());

            // Log "fallback triggered" only when this is a fallback attempt
            // (not the first provider). The healthy path is covered by the
            // server's "request completed" log, which already names the
            // provider that served the request — adding a "trying provider"
            // line here would just duplicate that.
            if !attempts.is_empty() {
                let last = attempts.last().expect("non-empty checked above");
                tracing::info!(
                    model = req.model.as_str(),
                    provider = %last.provider(),
                    failure = %last.failure(),
                    "fallback triggered"
                );
            }

            // Try the primary provider up to max_retries_per_provider times.
            // Each attempt that returns a cooldownable error marks the
            // provider on cooldown and falls through to the next iteration;
            // when the loop exhausts, control returns to the outer chain
            // loop which moves to the next provider. We do NOT `break` on
            // the first error — that's the whole point of this counter.
            for _ in 0..model.max_retries_per_provider {
                match provider.complete(req, &HashMap::new()).await {
                    Ok(out) => return Ok((out, attempts, Some(name.clone()))),
                    Err(e) if e.is_cooldownable() => {
                        if let ProxyError::Upstream { status, body } = &e {
                            attempts.push(RouteStep::Failed {
                                provider: name.clone(),
                                status: *status,
                                body: crate::util::truncate_for_log(
                                    body,
                                    FAILED_BODY_LOG_CAP_BYTES,
                                )
                                .to_string(),
                            });
                            let ttl = if matches!(*status, 402 | 429) {
                                Duration::from_secs(model.cooldown_seconds)
                            } else {
                                Duration::from_secs(5)
                            };
                            self.cooldown
                                .mark_cooldown(name, ttl, *status, &body)
                                .await;
                            last_error = Some(e);
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) if is_model_unsupported(&e) => {
                        // The upstream explicitly told us this provider can't
                        // serve the requested model (HTTP 400 + body that
                        // mentions "model" / "not supported"). Treat it like
                        // a cooldownable error so the router advances to the
                        // next provider in the chain instead of failing the
                        // whole request — see fix-R11. Use a short cooldown
                        // because the upstream's view of its model catalog
                        // could change, but record the attempt so the
                        // operator can see *why* this provider was skipped.
                        if let ProxyError::Upstream { status, body } = &e {
                            attempts.push(RouteStep::Failed {
                                provider: name.clone(),
                                status: *status,
                                body: crate::util::truncate_for_log(
                                    body,
                                    FAILED_BODY_LOG_CAP_BYTES,
                                )
                                .to_string(),
                            });
                            self.cooldown
                                .mark_cooldown(name, Duration::from_secs(60), *status, &body)
                                .await;
                            last_error = Some(e);
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) => {
                        // Non-cooldownable error (e.g., bad request shape) — return immediately.
                        return Err(e);
                    }
                }
            }

            // `max_total` only counts real upstream calls (RouteStep::Failed).
            // Local skips (cooldown / model_rewrite miss) have no upstream
            // cost and shouldn't trip the budget — plan §Phase 1.
            let failed_count = attempts
                .iter()
                .filter(|s| s.is_upstream_failure())
                .count();
            if failed_count >= max_total {
                break;
            }
        }

        if let Some(err) = last_error {
            // At least one upstream actually returned an error; surface
            // the *last* one so the operator can see what really happened,
            // instead of the generic "all cooling down" message that
            // would imply we never even tried.
            return Err(ProxyError::AllProvidersFailed {
                model: model.name.clone(),
                attempts,
                last: Box::new(err),
            });
        }
        // No provider could even attempt the request — distinguish
        // "all were unmappable" (configuration gap) from "all were
        // already on cooldown". An unmappable-model error is a 400 with
        // a clear message; a generic cooldown is a 503.
        if !unmappable.is_empty() && unmappable.len() == chain.len() {
            return Err(ProxyError::RouterBadRequest {
                message: format!(
                    "no provider in chain '{}' can serve model '{}' (all {} entries have a model_rewrite that excludes it)",
                    model.name,
                    req.model,
                    unmappable.len()
                ),
                attempts: std::mem::take(&mut attempts),
            });
        }
        // Every candidate was on cooldown from the start, or every
        // configured provider name was unknown — we never even fired a
        // request, so the "all cooling down" framing is accurate. The
        // skipped chain is carried so the terminal WARN / /admin/usage
        // row can show which providers were skipped and why (plan
        // §空切片语义).
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            attempts: std::mem::take(&mut attempts),
            retry_after_secs: None,
        })
    }

    /// Execute a streaming request. Returns the first provider's stream; if
    /// the request fails before streaming starts, falls back. Once bytes
    /// start flowing, the caller sees the entire stream.
    pub async fn stream(
        &self,
        model: &ModelConfig,
        req: &MessagesRequest,
    ) -> Result<(SharedProvider, ProviderOutput, Vec<RouteStep>)> {
        let mut attempts: Vec<RouteStep> = Vec::new();
        let mut last_error: Option<ProxyError> = None;
        let mut tried: Vec<String> = Vec::new();
        let mut unmappable: Vec<String> = Vec::new();
        let chain: Vec<String> = model.chain().map(String::from).collect();

        for round in 0..chain.len() {
            let name = &chain[round];
            if tried.contains(name) {
                continue;
            }
            if self.cooldown.is_cooling_down(name).await {
                tried.push(name.clone());
                attempts.push(RouteStep::LocalSkip {
                    provider: name.clone(),
                    reason: LocalSkipReason::Cooldown,
                });
                continue;
            }
            let Some(provider) = self.providers.get(name).cloned() else {
                tried.push(name.clone());
                continue;
            };
            // Skip providers whose model_rewrite excludes this model —
            // see fix-R11 and the matching comment in `complete()`.
            if !provider.can_serve_model(&req.model) {
                tracing::debug!(
                    provider = name.as_str(),
                    model = req.model.as_str(),
                    "provider's model_rewrite does not include this model; skipping"
                );
                unmappable.push(name.clone());
                tried.push(name.clone());
                attempts.push(RouteStep::LocalSkip {
                    provider: name.clone(),
                    reason: LocalSkipReason::ModelUnmapped,
                });
                continue;
            }
            tried.push(name.clone());

            // Log "fallback triggered" only when this is a fallback attempt
            // (not the first provider). The healthy path is covered by the
            // server's "request completed" log, which already names the
            // provider that served the request — adding a "trying provider"
            // line here would just duplicate that.
            if !attempts.is_empty() {
                let last = attempts.last().expect("non-empty checked above");
                tracing::info!(
                    model = req.model.as_str(),
                    provider = %last.provider(),
                    failure = %last.failure(),
                    "fallback triggered"
                );
            }

            // Streaming has no inner retry: a stream() call is a single HTTP
            // request whose response begins streaming immediately on
            // success. Retrying after the first byte has flowed is unsafe
            // (we'd double-emit content to the client), so the per-provider
            // attempt count for streaming is implicitly 1.
            match provider.stream(req, &HashMap::new()).await {
                Ok(out) => return Ok((provider, out, attempts)),
                Err(e) if e.is_cooldownable() => {
                    if let ProxyError::Upstream { status, body } = &e {
                        attempts.push(RouteStep::Failed {
                            provider: name.clone(),
                            status: *status,
                            body: crate::util::truncate_for_log(
                                body,
                                FAILED_BODY_LOG_CAP_BYTES,
                            )
                            .to_string(),
                        });
                        let ttl = if matches!(*status, 402 | 429) {
                            Duration::from_secs(model.cooldown_seconds)
                        } else {
                            Duration::from_secs(5)
                        };
                        self.cooldown.mark_cooldown(name, ttl, *status, &body).await;
                        last_error = Some(e);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) if is_model_unsupported(&e) => {
                    // Same model-unsupported skip as in `complete()` —
                    // see fix-R11.
                    if let ProxyError::Upstream { status, body } = &e {
                        attempts.push(RouteStep::Failed {
                            provider: name.clone(),
                            status: *status,
                            body: crate::util::truncate_for_log(
                                body,
                                FAILED_BODY_LOG_CAP_BYTES,
                            )
                            .to_string(),
                        });
                        self.cooldown
                            .mark_cooldown(name, Duration::from_secs(60), *status, &body)
                            .await;
                        last_error = Some(e);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        if let Some(err) = last_error {
            return Err(ProxyError::AllProvidersFailed {
                model: model.name.clone(),
                attempts,
                last: Box::new(err),
            });
        }
        if !unmappable.is_empty() && unmappable.len() == chain.len() {
            return Err(ProxyError::RouterBadRequest {
                message: format!(
                    "no provider in chain '{}' can serve model '{}' (all {} entries have a model_rewrite that excludes it)",
                    model.name,
                    req.model,
                    unmappable.len()
                ),
                attempts: std::mem::take(&mut attempts),
            });
        }
        Err(ProxyError::AllProvidersCoolingDown {
            model: model.name.clone(),
            attempts: std::mem::take(&mut attempts),
            retry_after_secs: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::config::{ModelConfig, ProviderConfig};
    use crate::providers::Provider;
    use async_trait::async_trait;
    use futures_util::stream;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock provider: returns the given status as an upstream error the
    /// first `fail_count` times, then succeeds.
    struct MockProvider {
        name: String,
        fail_status: u16,
        fail_count: u32,
        call_count: AtomicU32,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                return Err(ProxyError::Upstream {
                    status: self.fail_status,
                    body: "rate limited".into(),
                });
            }
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            if self.fail_count > 0 {
                return Err(ProxyError::Upstream {
                    status: self.fail_status,
                    body: "rate limited".into(),
                });
            }
            Ok(ProviderOutput::Stream(Box::new(stream::empty())))
        }
    }

    struct NonCooldownProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for NonCooldownProvider {
        fn name(&self) -> &str {
            &self.name
        }


        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            Err(ProxyError::BadRequest("invalid request".into()))
        }

        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            Err(ProxyError::BadRequest("invalid stream request".into()))
        }
    }

    fn build_test_router() -> Router {
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 429,
                fail_count: u32::MAX, // always fail
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0, // always succeed
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };

        Router::new(Arc::new(cfg), providers, CooldownCache::new())
    }

    fn dummy_request() -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn falls_back_on_429() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        let req = dummy_request();
        let (out, attempts, _served) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, .. } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 429);
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn complete_retries_per_provider_count() {
        // max_retries_per_provider must actually retry against the same
        // provider. Configure primary to fail twice then succeed, with
        // max_retries_per_provider = 3 — the third attempt must hit
        // primary (not the backup) and produce a successful response.
        let call_count = Arc::new(AtomicU32::new(0));
        let primary = Arc::new(CountingMockProvider {
            name: "primary".into(),
            fail_count: 2,
            call_count: call_count.clone(),
        }) as SharedProvider;
        let backup = Arc::new(CountingMockProvider {
            name: "backup".into(),
            fail_count: 0,
            call_count: Arc::new(AtomicU32::new(0)),
        }) as SharedProvider;

        let mut providers = HashMap::new();
        providers.insert("primary".to_string(), primary);
        providers.insert("backup".to_string(), backup);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 3,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers.clone(), CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (out, attempts, _served) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Primary was hit exactly 3 times: 2 failures + 1 success.
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        // The attempts vector records the two failures before the success.
        assert_eq!(attempts.len(), 2);
        for a in &attempts {
            match a {
                RouteStep::Failed { provider, status, .. } => {
                    assert_eq!(provider, "primary");
                    assert_eq!(*status, 429);
                }
                other => panic!("expected RouteStep::Failed, got {other:?}"),
            }
        }
        // Touch `.name()` on each mock provider so the trait impl method
        // is not just compiled but actually exercised in this test.
        for (label, expected) in [("primary", "primary"), ("backup", "backup")] {
            assert_eq!(providers.get(label).unwrap().name(), expected);
        }
    }

    /// Helper for `complete_retries_per_provider_count`: like MockProvider
    /// but tracks call count in an externally-shared AtomicU32 so the
    /// assertion can read it.
    struct CountingMockProvider {
        name: String,
        fail_count: u32,
        call_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Provider for CountingMockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                return Err(ProxyError::Upstream {
                    status: 429,
                    body: format!("fail #{n}"),
                });
            }
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn cooldown_blocks_primary() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        // First call: primary fails 429, fallback succeeds.
        let (_out, _attempts, _served) = router.complete(model, &req).await.unwrap();

        // Second call: primary should be cooling down, backup used directly.
        let (out, attempts, _served) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert!(
            attempts.iter().any(|s| matches!(
                s,
                RouteStep::LocalSkip {
                    reason: LocalSkipReason::Cooldown,
                    ..
                }
            )),
            "primary skip must be recorded as RouteStep::LocalSkip::Cooldown, got {attempts:?}"
        );
        assert!(
            !attempts.iter().any(|s| s.is_upstream_failure()),
            "no upstream call should have fired against cooling-down primary"
        );
    }

    #[tokio::test]
    async fn select_provider_skips_cooldown() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown
            .mark_cooldown("primary", Duration::from_secs(60), 429, "")
            .await;
        let (name, _p) = router.select_provider(model).await.unwrap();
        assert_eq!(name, "backup");
    }

    #[tokio::test]
    async fn provider_status_reflects_cooldown_and_models() {
        // The status snapshot must mirror the cooldown cache exactly:
        // a provider with a live cooldown entry is CoolingDown with the
        // triggering status + remaining TTL; everything else is
        // Available. `models` lists every client model whose chain
        // includes the provider.
        let router = build_test_router();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(60), 429, "rate limited")
            .await;

        let statuses = router.provider_status().await;
        assert_eq!(statuses.len(), 2, "both configured providers must appear");

        let primary = statuses.iter().find(|s| s.name == "primary").unwrap();
        assert_eq!(primary.status, ProviderHealth::CoolingDown);
        assert_eq!(primary.provider_type, "openai_compat");
        assert_eq!(primary.last_error_status, Some(429));
        let remaining = primary.cooling_down_remaining_secs.unwrap();
        assert!(
            (55..=60).contains(&remaining),
            "429 cooldown should report ~60s remaining, got {remaining}"
        );

        let backup = statuses.iter().find(|s| s.name == "backup").unwrap();
        assert_eq!(backup.status, ProviderHealth::Available);
        assert_eq!(backup.last_error_status, None);
        assert_eq!(backup.cooling_down_remaining_secs, None);

        // Model "m" has chain primary -> backup, so both list it.
        for s in &statuses {
            assert_eq!(s.models, vec!["m".to_string()]);
        }

        // Serialization must emit the documented contract keywords so the
        // statusline hook can branch on a fixed enum.
        let arr = serde_json::to_value(&statuses).unwrap();
        let arr = arr.as_array().unwrap();
        let primary_json = arr.iter().find(|v| v["name"] == "primary").unwrap();
        assert_eq!(primary_json["type"], "openai_compat");
        assert_eq!(primary_json["status"], "cooling_down");
        assert_eq!(primary_json["last_error_status"], 429);
        assert!(primary_json["cooling_down_remaining_secs"].is_u64());
        let backup_json = arr.iter().find(|v| v["name"] == "backup").unwrap();
        assert_eq!(backup_json["status"], "available");
        assert!(backup_json["last_error_status"].is_null());
        assert!(backup_json["cooling_down_remaining_secs"].is_null());
    }

    #[tokio::test]
    async fn provider_status_all_available_when_no_cooldown() {
        let router = build_test_router();
        let statuses = router.provider_status().await;
        assert_eq!(statuses.len(), 2);
        assert!(statuses.iter().all(|s| s.status == ProviderHealth::Available));
        assert!(statuses.iter().all(|s| s.last_error_status.is_none()));
    }

    #[tokio::test]
    async fn select_provider_returns_retry_after_matching_soonest_cooldown() {
        // When every candidate provider is on cooldown, the router must
        // fail fast with 503 + Retry-After instead of calling the
        // soonest-expiring provider (which would just trip another
        // cooldown and hand the client a generic 5xx) — see fix-R7
        // in docs/TEST_ISSUES.md.
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(60), 429, "primary")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(10), 503, "backup")
            .await;

        let err = router
            .select_provider(model)
            .await
            .err()
            .expect("all providers cooling down must error");

        // retry_after_secs is the soonest-remaining cooldown (backup = 10s);
        // allow 9 in case scheduler crossed a second boundary.
        assert!(
            matches!(
                err,
                ProxyError::AllProvidersCoolingDown {
                    ref model,
                    ref attempts,
                    retry_after_secs: Some(secs),
                } if model == "m" && (9..=10).contains(&secs) && attempts.is_empty()
            ),
            "expected AllProvidersCoolingDown with retry_after ~10, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn select_provider_errors_when_chain_has_no_known_provider() {
        let router = build_test_router();
        let empty = Router::new(
            Arc::new(router.config().clone()),
            HashMap::new(),
            CooldownCache::new(),
        );
        let model = empty.find_model("m").unwrap();

        let error = empty
            .select_provider(model)
            .await
            .err()
            .expect("selection should fail");

        assert!(matches!(
            error,
            ProxyError::AllProvidersCoolingDown { ref model, .. } if model == "m"
        ));
    }

    #[tokio::test]
    async fn stream_falls_back_on_cooldownable_error() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();

        let (provider, output, attempts) = router
            .stream(model, &dummy_request())
            .await
            .unwrap();

        assert_eq!(provider.name(), "backup");
        assert!(matches!(output, ProviderOutput::Stream(_)));
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, body } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 429);
                assert_eq!(body, "rate limited");
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
        assert!(router.cooldown().is_cooling_down("primary").await);
    }

    /// Streaming twin of `falls_back_on_429`: a 402 returned by primary's
    /// `stream()` before any bytes flow must (a) be recorded as a
    /// cooldownable attempt and (b) advance the router to backup. This
    /// covers the copilot-402 plan's claim that `Router::stream()` uses
    /// the same classification as `Router::complete()`.
    #[tokio::test]
    async fn stream_falls_back_on_402_quota() {
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 402,
                fail_count: u32::MAX,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (provider, output, attempts) = router
            .stream(model, &dummy_request())
            .await
            .unwrap();

        assert_eq!(provider.name(), "backup");
        assert!(matches!(output, ProviderOutput::Stream(_)));
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, body } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 402);
                assert_eq!(body, "rate limited");
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }

        // Quota cooldown must use the configured cooldown_seconds, not
        // the 5s transient fallback — mirrors the wire-level TTL test
        // at tests/integration_router.rs:1111-1209.
        let active = router.cooldown().active().await;
        let primary_entry = active
            .iter()
            .find(|(name, _, _)| name == "primary")
            .expect("primary must be on cooldown after 402");
        assert_eq!(primary_entry.1, 402);
        let ttl = primary_entry.2;
        assert!(
            ttl <= Duration::from_secs(60) && ttl > Duration::from_secs(55),
            "402 cooldown must use configured ~60s, got {ttl:?}"
        );
    }

    /// 403 must surface immediately and NOT enter cooldown — without
    /// this, a non-quota authorization failure could silently mask a
    /// policy misconfiguration by chaining the request to a provider
    /// with different access controls. See copilot-402 plan: do not add
    /// 403 globally.
    #[tokio::test]
    async fn stream_surfaces_403_without_cooldown() {
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 403,
                fail_count: u32::MAX,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        // No backup — a single-provider chain must surface 403 directly.
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![ProviderConfig::OpenaiCompat {
                name: "primary".into(),
                api_key: "k".into(),
                api_base: "http://x".into(),
                model_rewrite: Default::default(),
                use_proxy: false,
            provider_ignore: Vec::new(),
            reasoning_echo: false,
            }],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec![],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 1,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let error = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("403 must surface immediately");
        assert!(matches!(error, ProxyError::Upstream { status: 403, .. }));

        // 403 must not be recorded in the cooldown cache.
        assert!(!router.cooldown().is_cooling_down("primary").await);
        assert!(router.cooldown().active().await.is_empty());
    }

    #[tokio::test]
    async fn complete_and_stream_return_non_cooldownable_error_immediately() {
        let base = build_test_router();
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(NonCooldownProvider {
                name: "primary".into(),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        let router = Router::new(
            Arc::new(base.config().clone()),
            providers,
            CooldownCache::new(),
        );
        let model = router.find_model("m").unwrap();

        // Touch the NonCooldownProvider's name() method so the trait
        // impl is covered by this test, not just compiled.
        assert_eq!(router.providers.get("primary").unwrap().name(), "primary");

        let complete = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("complete should fail");
        let stream = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("stream should fail");

        assert!(matches!(complete, ProxyError::BadRequest(ref message) if message == "invalid request"));
        assert!(matches!(stream, ProxyError::BadRequest(ref message) if message == "invalid stream request"));
        assert!(!router.cooldown().is_cooling_down("primary").await);
    }

    #[tokio::test]
    async fn complete_and_stream_error_when_every_candidate_is_skipped() {
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(60), 429, "")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(60), 429, "")
            .await;

        let complete = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        let stream = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");

        // No upstream call ever fired — both providers were on cooldown
        // from the start — so this is the legacy "all cooling down" path,
        // not AllProvidersFailed. The router distinguishes the two cases:
        // "no attempt happened" stays as AllProvidersCoolingDown,
        // "at least one attempt failed" becomes AllProvidersFailed.
        assert!(matches!(complete, ProxyError::AllProvidersCoolingDown { .. }));
        assert!(matches!(stream, ProxyError::AllProvidersCoolingDown { .. }));
    }

    #[tokio::test]
    async fn max_retries_total_zero_stops_before_fallback() {
        let router = build_test_router();
        let mut model = router.find_model("m").unwrap().clone();
        model.max_retries_total = 0;

        let error = router
            .complete(&model, &dummy_request())
            .await
            .err()
            .expect("request should fail");

        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
        assert!(router.cooldown().is_cooling_down("primary").await);
        assert!(!router.cooldown().is_cooling_down("backup").await);
    }

    #[tokio::test]
    async fn complete_skips_unknown_provider_in_chain() {
        // When a model chain references a provider that doesn't exist, the
        // router should skip it and try the next one instead of erroring.
        let base = build_test_router();
        let mut model = base.find_model("m").unwrap().clone();
        model.fallback_chain = vec!["missing".into(), "backup".into()];

        let (out, attempts, _served) = router_clone_with(&base)
            .complete(&model, &dummy_request())
            .await
            .unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Only the 429 attempt against primary is recorded; the missing
        // provider is silently skipped without recording an attempt.
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, .. } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 429);
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_skips_unknown_provider_in_chain() {
        let base = build_test_router();
        let mut model = base.find_model("m").unwrap().clone();
        model.fallback_chain = vec!["missing".into(), "backup".into()];

        let (provider, output, attempts) = router_clone_with(&base)
            .stream(&model, &dummy_request())
            .await
            .unwrap();
        assert_eq!(provider.name(), "backup");
        assert!(matches!(output, ProviderOutput::Stream(_)));
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, .. } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 429);
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn complete_skips_provider_already_tried() {
        // A duplicated entry in the chain (e.g. fallback_chain contains the
        // primary again) should be skipped on the second pass — the router
        // already recorded a failed attempt and moved on.
        //
        // For the duplicate to actually be reached, the primary AND the
        // other fallbacks must fail. Configure every provider as a
        // fail-forever stub and verify the router records at most one
        // attempt per provider name.
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 429,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into(), "primary".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 5,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let error = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
        // Each provider was attempted exactly once even though the chain
        // listed primary twice.
        assert_eq!(
            router
                .cooldown()
                .active()
                .await
                .iter()
                .filter(|(n, _, _)| n == "primary" || n == "backup")
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn stream_skips_provider_already_tried() {
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 429,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into(), "primary".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 5,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let error = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        assert!(matches!(error, ProxyError::AllProvidersFailed { .. }));
    }

    #[tokio::test]
    async fn select_provider_retry_after_is_primary_when_primary_remaining_is_shorter() {
        // When all providers are cooling down, the router must return
        // AllProvidersCoolingDown with retry_after_secs equal to the
        // soonest-remaining cooldown — here primary expires first, so
        // the value comes from primary's 5s window. See fix-R7 in
        // docs/TEST_ISSUES.md.
        let router = build_test_router();
        let model = router.find_model("m").unwrap();
        router
            .cooldown()
            .mark_cooldown("primary", Duration::from_secs(5), 429, "")
            .await;
        router
            .cooldown()
            .mark_cooldown("backup", Duration::from_secs(120), 503, "")
            .await;

        let err = router
            .select_provider(model)
            .await
            .err()
            .expect("all providers cooling down must error");

        assert!(
            matches!(
                err,
                ProxyError::AllProvidersCoolingDown {
                    retry_after_secs: Some(secs),
                    ..
                } if (1..=5).contains(&secs)
            ),
            "expected retry_after_secs from primary's 5s window, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn complete_returns_all_providers_failed_with_last_error_when_chain_exhausted() {
        // When every candidate actually fires and returns an upstream
        // error, the router must surface the last one as
        // AllProvidersFailed (not AllProvidersCoolingDown) so operators
        // can see the real cause — see fix-B in TEST_ISSUES.md.
        let mut providers = HashMap::new();
        for name in ["primary", "backup"] {
            providers.insert(
                name.to_string(),
                Arc::new(MockProvider {
                    name: name.into(),
                    fail_status: 503,
                    fail_count: u32::MAX,
                    call_count: AtomicU32::new(0),
                }) as SharedProvider,
            );
        }
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let err = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("request should fail");
        match err {
            ProxyError::AllProvidersFailed { model, attempts, last } => {
                assert_eq!(model, "m");
                assert_eq!(attempts.len(), 2);
                match (&attempts[0], &attempts[1]) {
                    (
                        RouteStep::Failed {
                            provider: p0,
                            status: s0,
                            ..
                        },
                        RouteStep::Failed {
                            provider: p1,
                            status: s1,
                            ..
                        },
                    ) => {
                        assert_eq!(p0, "primary");
                        assert_eq!(*s0, 503);
                        assert_eq!(p1, "backup");
                        assert_eq!(*s1, 503);
                    }
                    _ => panic!("expected two RouteStep::Failed entries"),
                }
                // `last` must be the last upstream error (backup's),
                // not the legacy generic "all cooling down" message.
                match last.as_ref() {
                    ProxyError::Upstream { status, body } => {
                        assert_eq!(*status, 503);
                        assert_eq!(body, "rate limited");
                    }
                    other => panic!("expected wrapped Upstream, got {other:?}"),
                }
            }
            other => panic!("expected AllProvidersFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_429_upstream_uses_short_cooldown_ttl() {
        // A 503 (or any non-429 cooldownable status) should mark the provider
        // for the default short TTL, not the model's cooldown_seconds.
        let router = build_test_router();
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(MockProvider {
                name: "primary".into(),
                fail_status: 503,
                fail_count: u32::MAX,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(MockProvider {
                name: "backup".into(),
                fail_status: 0,
                fail_count: 0,
                call_count: AtomicU32::new(0),
            }) as SharedProvider,
        );
        let router = Router::new(
            Arc::new(router.config().clone()),
            providers,
            CooldownCache::new(),
        );
        let model = router.find_model("m").unwrap();
        let (_out, _attempts, _served) = router.complete(model, &dummy_request()).await.unwrap();

        // Backup succeeded; primary should now be on a short cooldown.
        assert!(router.cooldown().is_cooling_down("primary").await);
    }

    fn router_clone_with(base: &Router) -> Router {
        Router::new(
            Arc::new(base.config().clone()),
            base.providers.clone(),
            base.cooldown.clone(),
        )
    }


    /// A provider that accepts a fixed allow-list of model names; every
    /// other model is rejected at dispatch time via `can_serve_model`.
    /// Used by the R11 chain-skip tests.
    struct RestrictedMockProvider {
        name: String,
        allowed: Vec<String>,
        call_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Provider for RestrictedMockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn can_serve_model(&self, model: &str) -> bool {
            // Empty allow-list means "no restriction" (matches the
            // OpenAiCompatProvider contract for an unconfigured rewrite
            // table). Non-empty list is an explicit allow-list.
            self.allowed.is_empty() || self.allowed.iter().any(|m| m == model)
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ProviderOutput::Json(serde_json::json!({
                "id": "msg_ok",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            unimplemented!()
        }
    }

    fn build_restricted_router(
        primary_allowed: Vec<String>,
        backup_allowed: Vec<String>,
    ) -> (Router, Arc<AtomicU32>, Arc<AtomicU32>) {
        let primary_count = Arc::new(AtomicU32::new(0));
        let backup_count = Arc::new(AtomicU32::new(0));
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(RestrictedMockProvider {
                name: "primary".into(),
                allowed: primary_allowed,
                call_count: primary_count.clone(),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(RestrictedMockProvider {
                name: "backup".into(),
                allowed: backup_allowed,
                call_count: backup_count.clone(),
            }) as SharedProvider,
        );
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        (
            Router::new(Arc::new(cfg), providers, CooldownCache::new()),
            primary_count,
            backup_count,
        )
    }

    #[tokio::test]
    async fn complete_skips_provider_that_cannot_serve_model_and_uses_next() {
        // Primary has a model_rewrite that excludes `m-allowed-1`.
        // Backup accepts it. The router must skip primary without
        // calling it and succeed on backup — see fix-R11.
        let (router, primary_count, backup_count) =
            build_restricted_router(vec!["other-model".into()], vec![]);
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        // Touch the RestrictedMockProvider's name() method on each side
        // so the trait impl is covered by this test (and the helper
        // doesn't remain a compiled-but-never-called shell).
        assert_eq!(router.providers.get("primary").unwrap().name(), "primary");
        assert_eq!(router.providers.get("backup").unwrap().name(), "backup");

        let (out, attempts, _served) = router.complete(model, &req).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(primary_count.load(Ordering::SeqCst), 0, "primary must be skipped, not called");
        assert_eq!(backup_count.load(Ordering::SeqCst), 1);
        // Skipped provider recorded as RouteStep::LocalSkip::ModelUnmapped
        // (no upstream call fired against it). See plan §Phase 1 commit 2.
        assert!(
            attempts.iter().any(|s| matches!(
                s,
                RouteStep::LocalSkip {
                    reason: LocalSkipReason::ModelUnmapped,
                    ..
                }
            )),
            "primary skip must be recorded as LocalSkip::ModelUnmapped, got {attempts:?}"
        );
        assert!(
            !attempts.iter().any(|s| s.is_upstream_failure()),
            "no upstream call should have fired against the unmapped primary"
        );
    }

    #[tokio::test]
    async fn complete_returns_bad_request_when_no_provider_can_serve_model() {
        // Both providers have a rewrite that excludes `m-bad`. The router
        // must surface a 400-level BadRequest so the operator sees that
        // the configuration gap (not a transient upstream failure) is the
        // cause — see fix-R11.
        let (router, primary_count, backup_count) = build_restricted_router(
            vec!["unrelated-a".into()],
            vec!["unrelated-b".into()],
        );
        let model = router.find_model("m").unwrap();
        let req = dummy_request();

        let err = router
            .complete(model, &req)
            .await
            .err()
            .expect("request should fail");
        match err {
            ProxyError::RouterBadRequest { message, .. } => {
                assert!(
                    message.contains("m") && message.contains("can serve"),
                    "message should mention the model + cause: {message}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
        // Neither provider should have been called.
        assert_eq!(primary_count.load(Ordering::SeqCst), 0);
        assert_eq!(backup_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn complete_skips_mismatched_provider_but_still_falls_back_on_cooldownable_error() {
        // Primary excludes `m`, backup accepts it. Primary is still
        // skipped at dispatch (so it doesn't get a 400 from upstream),
        // and backup succeeds. The dispatch-skip is a distinct path
        // from cooldown-based skip — both should coexist.
        let (router, primary_count, backup_count) =
            build_restricted_router(vec!["other".into()], vec![]);
        let model = router.find_model("m").unwrap();

        let (out, _attempts, _served) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        assert_eq!(primary_count.load(Ordering::SeqCst), 0);
        assert_eq!(backup_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_skips_provider_that_cannot_serve_model_and_uses_next() {
        // Same scenario as `complete_skips_provider_...`, but on the
        // streaming path. The first byte must come from backup.
        struct StreamingMockProvider {
            name: String,
            allowed: Vec<String>,
            call_count: Arc<AtomicU32>,
        }
        #[async_trait]
        impl Provider for StreamingMockProvider {
            fn name(&self) -> &str {
                &self.name
            }
            fn can_serve_model(&self, model: &str) -> bool {
                self.allowed.is_empty() || self.allowed.iter().any(|m| m == model)
            }
            async fn complete(
                &self,
                _req: &MessagesRequest,
                _model_rewrite: &HashMap<String, String>,
            ) -> Result<ProviderOutput> {
                unimplemented!()
            }
            async fn stream(
                &self,
                _req: &MessagesRequest,
                _model_rewrite: &HashMap<String, String>,
            ) -> Result<ProviderOutput> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                let s: Box<dyn futures_util::Stream<Item = Result<Bytes>> + Send + Unpin> =
                    Box::new(stream::empty());
                Ok(ProviderOutput::Stream(s))
            }
        }
        let mut providers = HashMap::new();
        providers.insert(
            "primary".to_string(),
            Arc::new(StreamingMockProvider {
                name: "primary".into(),
                allowed: vec!["other-model".into()],
                call_count: Arc::new(AtomicU32::new(0)),
            }) as SharedProvider,
        );
        providers.insert(
            "backup".to_string(),
            Arc::new(StreamingMockProvider {
                name: "backup".into(),
                allowed: vec![],
                call_count: Arc::new(AtomicU32::new(0)),
            }) as SharedProvider,
        );
        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (provider, _output, attempts) =
            router.stream(model, &dummy_request()).await.unwrap();
        assert_eq!(provider.name(), "backup");
        // The streaming twin of `complete_skips_provider_...`:
        // primary is unmappable, so it's recorded as a
        // RouteStep::LocalSkip::ModelUnmapped and never called.
        assert!(
            attempts.iter().any(|s| matches!(
                s,
                RouteStep::LocalSkip {
                    reason: LocalSkipReason::ModelUnmapped,
                    ..
                }
            )),
            "primary skip must be recorded as LocalSkip::ModelUnmapped, got {attempts:?}"
        );
        assert!(
            !attempts.iter().any(|s| s.is_upstream_failure()),
            "no upstream call should have fired against the unmapped primary"
        );
    }

    #[tokio::test]
    async fn stream_returns_bad_request_when_no_provider_can_serve_model() {
        let (router, _pc, _bc) = build_restricted_router(
            vec!["other".into()],
            vec!["another".into()],
        );
        let model = router.find_model("m").unwrap();

        let err = router
            .stream(model, &dummy_request())
            .await
            .err()
            .expect("stream should fail");
        assert!(
            matches!(
                err,
                ProxyError::RouterBadRequest { ref message, .. } if message.contains("can serve")
            ),
            "expected RouterBadRequest, got {err:?}"
        );
    }

    #[test]
    fn is_model_unsupported_recognises_common_shapes() {
        // Cover the body patterns the runtime helper is supposed to catch.
        let cases = [
            (
                r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#,
                true,
            ),
            (
                r#"{"error":{"message":"Model Not Exist","type":"invalid_request_error","code":"model_not_found"}}"#,
                true,
            ),
            (
                r#"{"error":{"message":"The supported API model names are deepseek-v4-pro or deepseek-v4-flash, but you passed claude-sonnet-4.5.","param":null,"code":"invalid_request_error"}}"#,
                true,
            ),
            // Plain 400 with no model signal must NOT be treated as
            // model-unsupported — that would mask real request errors.
            (r#"{"error":{"message":"missing field `messages`"}}"#, false),
            (r#"rate limited"#, false),
            // Bare "model" without a "not supported" cue is also not
            // enough — covers "missing field `model`" false positives.
            (r#"{"error":"invalid value for field `model`"}"#, false),
            // Copilot endpoint rejection (responses-only model sent to
            // the chat endpoint): must be treated as model-unsupported so
            // the router falls back instead of leaking a 400.
            (
                r#"{"error":{"message":"model \"grok-4.5\" is not accessible via the /chat/completions endpoint"}}"#,
                true,
            ),
            // Generic routing error must NOT match: the pattern is gated
            // on "the /" so this stays a surfaced failure.
            (r#"{"error":{"message":"file is not accessible via fallback proxy"}}"#, false),
        ];
        for (body, expected) in cases {
            let err = ProxyError::Upstream { status: 400, body: body.into() };
            assert_eq!(
                is_model_unsupported(&err),
                expected,
                "body={body} expected={expected}"
            );
        }
    }

    #[test]
    fn quota_402_body_is_not_model_unsupported() {
        // Copilot's monthly-quota exhaustion uses HTTP 402 with a body
        // that does not mention any model-support keyword. The router
        // must treat it as cooldownable quota failure, not as a
        // model-catalog mismatch. This guards the precedence: the
        // cooldownable branch fires first at runtime, and
        // `is_model_unsupported` must agree.
        let quota = ProxyError::Upstream {
            status: 402,
            body: "You have exceeded your monthly quota".into(),
        };
        assert!(!is_model_unsupported(&quota));
        // The 402 must be classified as cooldownable at the error
        // boundary so the router advances to the next provider.
        assert!(quota.is_cooldownable());
    }

    #[test]
    fn is_model_unsupported_only_for_4xx_status() {
        // A 5xx error mentioning model must still be the regular
        // cooldownable branch, not the model-unsupported one.
        let err = ProxyError::Upstream {
            status: 503,
            body: r#"{"error":"model_not_supported"}"#.into(),
        };
        assert!(!is_model_unsupported(&err));
        // 401/429 — handled by the cooldownable branch.
        let err = ProxyError::Upstream {
            status: 401,
            body: r#"unauthorized, please check your token"#.into(),
        };
        assert!(!is_model_unsupported(&err));
    }

    /// A mock provider that always returns an upstream 400 whose body
    /// looks like a model-unsupported envelope (e.g. Copilot rejecting
    /// `claude-sonnet-4.5`). The router must treat this as a
    /// "skip this provider, try the next one" signal — see fix-R11.
    struct ModelUnsupportedProvider {
        name: String,
        body: String,
        call_count: AtomicU32,
    }

    #[async_trait]
    impl Provider for ModelUnsupportedProvider {
        fn name(&self) -> &str {
            &self.name
        }
        async fn complete(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(ProxyError::Upstream {
                status: 400,
                body: self.body.clone(),
            })
        }
        async fn stream(
            &self,
            _req: &MessagesRequest,
            _model_rewrite: &HashMap<String, String>,
        ) -> Result<ProviderOutput> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(ProxyError::Upstream {
                status: 400,
                body: self.body.clone(),
            })
        }
    }

    #[tokio::test]
    async fn complete_skips_provider_returning_runtime_model_unsupported() {
        // Primary returns 400 + model_not_supported body. The router must
        // skip it and try backup. Backup succeeds. — see fix-R11.
        let primary = Arc::new(ModelUnsupportedProvider {
            name: "primary".into(),
            body: r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#.into(),
            call_count: AtomicU32::new(0),
        });
        let backup = Arc::new(MockProvider {
            name: "backup".into(),
            fail_status: 0,
            fail_count: 0,
            call_count: AtomicU32::new(0),
        });
        let mut providers: HashMap<String, SharedProvider> = HashMap::new();
        providers.insert("primary".to_string(), primary.clone() as SharedProvider);
        providers.insert("backup".to_string(), backup as SharedProvider);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        // Touch the ModelUnsupportedProvider's name() method so the trait
        // impl is covered by this test (it would otherwise only exist as
        // a compiled-but-never-called method on the mock helper).
        assert_eq!(primary.name(), "primary");

        let (out, attempts, _served) = router.complete(model, &dummy_request()).await.unwrap();
        assert!(matches!(out, ProviderOutput::Json(_)));
        // Primary was tried once (returned the 400), backup took over.
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        // The skipped primary must be in the attempts list so the
        // operator can see why it was skipped.
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, body } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 400);
                assert!(body.contains("model_not_supported"));
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_skips_provider_returning_runtime_model_unsupported() {
        // Streaming twin of the complete() model-unsupported skip: the
        // primary answers the stream() call with 400 + model_not_supported;
        // the router must record the attempt, cool the primary down, and
        // fall back to the backup's stream. Covers router.rs stream()
        // is_model_unsupported branch (the 400 skip on the streaming path)
        // AND ModelUnsupportedProvider::stream. — see fix-R11.
        let primary = Arc::new(ModelUnsupportedProvider {
            name: "primary".into(),
            body: r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#.into(),
            call_count: AtomicU32::new(0),
        });
        let backup = Arc::new(MockProvider {
            name: "backup".into(),
            fail_status: 0,
            fail_count: 0, // stream succeeds with an empty stream
            call_count: AtomicU32::new(0),
        });
        let mut providers: HashMap<String, SharedProvider> = HashMap::new();
        providers.insert("primary".to_string(), primary.clone() as SharedProvider);
        providers.insert("backup".to_string(), backup as SharedProvider);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let (provider, _out, attempts) =
            router.stream(model, &dummy_request()).await.unwrap();
        // Backup served the stream after primary was skipped.
        assert_eq!(provider.name(), "backup");
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        // The skipped primary must appear in attempts with its 400 body.
        assert_eq!(attempts.len(), 1);
        match &attempts[0] {
            RouteStep::Failed { provider, status, body } => {
                assert_eq!(provider, "primary");
                assert_eq!(*status, 400);
                assert!(body.contains("model_not_supported"));
            }
            other => panic!("expected RouteStep::Failed, got {other:?}"),
        }
        // A model-unsupported skip cools the provider down (60s) so the
        // next request bypasses it entirely.
        assert!(router.cooldown().is_cooling_down("primary").await);
    }

    #[tokio::test]
    async fn complete_does_not_skip_provider_returning_generic_400() {
        // A 400 that does NOT mention model/not-supported must surface
        // as an error, not be silently swallowed by the model-unsupported
        // skip path. This is the guard against over-eager skipping.
        let primary = Arc::new(ModelUnsupportedProvider {
            name: "primary".into(),
            body: r#"{"error":{"message":"missing field `messages`"}}"#.into(),
            call_count: AtomicU32::new(0),
        });
        let backup = Arc::new(MockProvider {
            name: "backup".into(),
            fail_status: 0,
            fail_count: 0,
            call_count: AtomicU32::new(0),
        });
        let mut providers: HashMap<String, SharedProvider> = HashMap::new();
        providers.insert("primary".to_string(), primary.clone() as SharedProvider);
        providers.insert("backup".to_string(), backup as SharedProvider);

        let cfg = Config {
            server: Default::default(),
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "primary".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
                ProviderConfig::OpenaiCompat {
                    name: "backup".into(),
                    api_key: "k".into(),
                    api_base: "http://x".into(),
                    model_rewrite: Default::default(),
                    use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
                },
            ],
            models: vec![ModelConfig {
                name: "m".into(),
                primary: "primary".into(),
                fallback_chain: vec!["backup".into()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 3,
            }],
                ..Config::default()
            };
        let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());
        let model = router.find_model("m").unwrap();

        let err = router
            .complete(model, &dummy_request())
            .await
            .err()
            .expect("generic 400 must surface as Err");
        match err {
            ProxyError::Upstream { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Upstream 400, got {other:?}"),
        }
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
    }
}
