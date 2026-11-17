//! Client-side per-provider rate limiting (the `rate_limit:` provider
//! block).
//!
//! The proxy paces its *own* outbound requests so the upstream never
//! sees a limit-violating request. This is the difference between
//! avoiding the upstream 429 and reacting to it: without pacing, a
//! bursty Claude Code session (parallel subagents + background
//! classifier calls) exceeds provider quotas, the 429 marks the
//! provider on cooldown, and the request falls back to a chain entry
//! that may serve the model worse. With pacing, requests wait a few
//! hundred milliseconds at the gate and the primary keeps serving.
//!
//! Upstream limits come in three locally-modelable shapes, mirrored by
//! the three knobs in `RateLimitConfig`:
//!
//! - **RPM** — requests per rolling 60 s window (MiniMax Token Plan
//!   paid tier: 200; OpenRouter free models: 20; Aliyun Bailian).
//! - **TPM** — input+output tokens per rolling 60 s window (MiniMax:
//!   10,000,000 paid tier). Charged with an estimate before dispatch
//!   ([`crate::tokenize::estimate_typed_tokens`] + a `max_tokens`
//!   reserve), replaced by the upstream-reported usage after the
//!   response (ticket settle) — drift self-corrects.
//! - **Concurrency** — in-flight request cap (Zhipu GLM, DeepSeek).
//!   Held from dispatch until the response completes; released by
//!   dropping the ticket, so a client disconnect mid-stream cannot
//!   leak a slot.
//!
//! The windows are sliding-window logs, not fixed-minute counters: the
//! limiter guarantees "at most N in *any* 60-second span", which is
//! strictly stronger than whatever fixed/rolling minute window the
//! upstream enforces, so exact configured values are safe (operators
//! may still set lower numbers for margin against estimation drift).
//!
//! Over-limit behavior is cooperative, never an error: `acquire`
//! sleeps up to `max_wait` (default 2 s) for capacity; past that it
//! returns [`Throttled`] and the router treats the provider as a local
//! skip — move to the next chain entry, record the reason, and do NOT
//! mark a cooldown (pacing exhaustion is a transient local state, not
//! an upstream failure).
//!
//! Patterns follow the in-repo precedents: injected-capable tokio time
//! (`tokio::time::Instant` — `start_paused` tests advance it
//! deterministically), a `std::sync::Mutex` with poison recovery for
//! the tiny no-await critical sections (`auth::AuthRateLimiter`), and
//! shared state through `Arc` (`cooldown::CooldownCache`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use crate::config::{Config, RateLimitConfig};

/// Length of the rolling window behind `rpm`/`tpm`. Every upstream
/// limit this module models is a per-minute quota.
const WINDOW: Duration = Duration::from_secs(60);

/// All per-provider limiters, built once at startup from the config.
/// Providers without a `rate_limit:` block are absent from the map, so
/// their request path pays nothing but a hash lookup miss.
#[derive(Debug, Default, Clone)]
pub struct ProviderRateLimiters {
    map: HashMap<String, std::sync::Arc<ProviderLimiter>>,
}

impl ProviderRateLimiters {
    /// Build from the parsed config: one limiter per provider with a
    /// `rate_limit:` block.
    pub fn build(cfg: &Config) -> Self {
        let mut map = HashMap::new();
        for p in &cfg.providers {
            if let Some(rl) = p.rate_limit() {
                map.insert(
                    p.name().to_string(),
                    std::sync::Arc::new(ProviderLimiter::new_named(p.name(), rl)),
                );
            }
        }
        Self { map }
    }

    /// The limiter for `name`, or `None` when the provider has no
    /// `rate_limit:` block (dispatch must not gate on anything).
    pub fn get(&self, name: &str) -> Option<&std::sync::Arc<ProviderLimiter>> {
        self.map.get(name)
    }
}

/// One provider's gate: the rolling windows and (optionally) the
/// concurrency semaphore. Shared by every request that dispatches to
/// the provider through an `Arc`.
#[derive(Debug)]
pub struct ProviderLimiter {
    /// Provider name — used only in log messages (the one-shot
    /// "estimate exceeds the tpm budget" warning names it).
    name: String,
    rpm: Option<u32>,
    tpm: Option<u64>,
    permits: Option<std::sync::Arc<Semaphore>>,
    max_wait: Duration,
    state: Mutex<WindowState>,
}

#[derive(Debug, Default)]
struct WindowState {
    /// `(ticket id, arrival)` per admitted request — only tracked when
    /// `rpm` is configured.
    reqs: VecDeque<(u64, Instant)>,
    /// `(ticket id, arrival, charged tokens)` per admitted request —
    /// only tracked when `tpm` is configured.
    toks: VecDeque<TokEntry>,
    /// Running sum of `toks[].tokens` so window checks are O(1).
    tok_sum: u64,
    /// Monotonic ticket id — keys a window entry for `settle`/refund.
    next_id: u64,
    /// When the "estimate exceeds the whole tpm budget" warning was last
    /// emitted for this limiter. The condition is a persistent
    /// misconfiguration (the provider is skipped for *every* request
    /// until the limit is raised), so the warning repeats once per
    /// window instead of firing exactly once and disappearing into the
    /// log — otherwise the operator sees nothing but generic throttle
    /// lines.
    never_fits_warned_at: Option<Instant>,
}

#[derive(Debug)]
struct TokEntry {
    id: u64,
    at: Instant,
    tokens: u64,
}

/// Why `acquire` gave up: keeping the request within the limits would
/// have taken longer than the configured wait cap. `eta` is the wait
/// the window demanded on the window path, the wait budget consumed on
/// the semaphore path (a "when a slot frees" eta is not observable), or
/// zero on the never-fits path (no wait would help) — purely for the
/// WARN log; the router turns this into a local skip (no cooldown).
#[derive(Debug, Clone, Copy)]
pub struct Throttled {
    pub eta: Duration,
}

/// Permission to send one upstream request, plus the accounting it
/// still owes: the token-window entry to reconcile (`settle`) and the
/// concurrency slot (released when the ticket is dropped). Tickets are
/// owned by exactly one continuation — the provider call that follows.
#[derive(Debug)]
pub struct RateTicket {
    limiter: std::sync::Arc<ProviderLimiter>,
    id: u64,
    est_tokens: u64,
    /// Dropping the permit returns the concurrency slot — including on
    /// the client-disconnect path where nothing else runs.
    _permit: Option<OwnedSemaphorePermit>,
}

impl RateTicket {
    /// Replace the pre-flight estimate with the upstream-reported
    /// total (input+output [+ cache writes]) once the response is in.
    /// Consumes the ticket, releasing the concurrency slot.
    ///
    /// A zero total means the upstream surfaced no usage fields at all:
    /// the pre-flight estimate then stays charged (the entry keeps its
    /// current figure and slides out of the window unchanged) —
    /// refunding it would make the window optimistic about tokens that
    /// were very likely billed. A window entry that already aged out
    /// (stream longer than the window) is likewise left as-is.
    pub fn settle(self, actual_tokens: u64) {
        let actual = if actual_tokens == 0 {
            self.est_tokens
        } else {
            actual_tokens
        };
        let mut st = self
            .limiter
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = st.toks.iter_mut().find(|e| e.id == self.id) {
            let old = e.tokens;
            e.tokens = actual;
            // `e`'s borrow ends at the assignment above (NLL), so the
            // running sum is safe to touch here.
            st.tok_sum = st.tok_sum.saturating_sub(old).saturating_add(actual);
        }
        // `_permit` drops here with `self`.
    }
}

impl ProviderLimiter {
    /// A limiter without a provider name — for tests. The name only
    /// appears in log messages.
    pub fn new(cfg: &RateLimitConfig) -> Self {
        Self::new_named("", cfg)
    }

    /// A limiter for `name` (used in log messages).
    pub fn new_named(name: &str, cfg: &RateLimitConfig) -> Self {
        Self {
            name: name.to_string(),
            rpm: cfg.rpm,
            tpm: cfg.tpm,
            permits: cfg
                .max_concurrency
                .map(|n| std::sync::Arc::new(Semaphore::new(n as usize))),
            max_wait: cfg.max_wait(),
            state: Mutex::new(WindowState::default()),
        }
    }

    /// Whether the limiter's windows consume the token estimate at all.
    /// rpm-only and concurrency-only limiters never look at it, so the
    /// router can skip the request-text walk entirely on the hot path.
    pub(crate) fn needs_token_estimate(&self) -> bool {
        self.tpm.is_some()
    }

    /// One-shot warning for the "the request estimate exceeds the whole
    /// tpm budget" condition. Unlike ordinary pacing exhaustion this is
    /// a *configuration* problem: every request to this provider will
    /// be skipped, so the operator needs to see it clearly rather than
    /// as an endless stream of indistinguishable per-request throttle
    /// warnings. It repeats once per window (the condition persists for
    /// the process's lifetime) — a single warning can be missed.
    fn warn_never_fits_once(&self, est_tokens: u64) {
        let now = Instant::now();
        let should_warn = {
            let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let due = match st.never_fits_warned_at {
                None => true,
                Some(last) => now.saturating_duration_since(last) >= WINDOW,
            };
            if due {
                st.never_fits_warned_at = Some(now);
            }
            due
        };
        if should_warn {
            tracing::warn!(
                provider = %self.name,
                tpm = ?self.tpm,
                estimated_tokens = est_tokens,
                "rate_limit.tpm is below the estimated request size (input + max_tokens reserve); \
                 every request to this provider will be skipped until the limit is raised"
            );
        }
    }

    /// Current token-window charge. Test-only: unit tests outside this
    /// module (server.rs settle path) need to see the running `tok_sum`.
    #[cfg(test)]
    pub(crate) fn charged_tokens(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .tok_sum
    }

    /// Wait for (and charge) the capacity to send one request estimated
    /// at `est_tokens`. Total wait across the windows and the semaphore
    /// is capped at `max_wait`; past the cap, returns [`Throttled`] and
    /// leaves the windows untouched.
    ///
    /// Order matters: the windows admit first (charging the estimate),
    /// the semaphore second. A request that wins window capacity but
    /// times out on the semaphore has its window charge refunded, so
    /// queued-but-unsent requests never eat quota.
    pub async fn acquire(
        self: &std::sync::Arc<Self>,
        est_tokens: u64,
    ) -> Result<RateTicket, Throttled> {
        let deadline = Instant::now() + self.max_wait;
        loop {
            let now = Instant::now();
            let admission = {
                let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                self.try_admit(&mut st, est_tokens, now)
            };
            let id = match admission {
                Admission::Admit(id) => id,
                Admission::Never => {
                    // The estimate alone exceeds the whole tpm budget:
                    // sleeping cannot help, whatever max_wait says —
                    // give up immediately instead of holding the
                    // connection for the rest of the budget. eta 0 =
                    // "no wait would make it fit".
                    self.warn_never_fits_once(est_tokens);
                    return Err(Throttled {
                        eta: Duration::ZERO,
                    });
                }
                Admission::Wait(eta) => {
                    let wake = now + eta;
                    if wake > deadline {
                        return Err(Throttled { eta });
                    }
                    tokio::time::sleep_until(wake).await;
                    continue;
                }
            };
            let permit = match &self.permits {
                None => None,
                Some(sem) => {
                    match tokio::time::timeout_at(deadline, sem.clone().acquire_owned()).await {
                        Ok(permit) => Some(permit.expect("semaphore is never closed")),
                        Err(_) => {
                            // Semaphore timed out: undo the window charge
                            // so a queued-but-unsent request costs nothing.
                            let mut st =
                                self.state.lock().unwrap_or_else(PoisonError::into_inner);
                            refund(&mut st, id, est_tokens);
                            // The whole budget was consumed waiting on a
                            // concurrency slot; report the wait we paid —
                            // a ~0 remainder would hide from the log that
                            // the slot was the binding constraint.
                            return Err(Throttled {
                                eta: self.max_wait,
                            });
                        }
                    }
                }
            };
            return Ok(RateTicket {
                limiter: self.clone(),
                id,
                est_tokens,
                _permit: permit,
            });
        }
    }

    /// Prune expired entries and decide whether `est_tokens` fits the
    /// windows *now*. On success, charges the windows and returns the
    /// ticket id; otherwise returns how long until the most binding
    /// restriction lifts.
    fn try_admit(&self, st: &mut WindowState, est_tokens: u64, now: Instant) -> Admission {
        prune(st, now);
        let mut eta = Duration::ZERO;
        if let Some(rpm) = self.rpm {
            if st.reqs.len() >= rpm as usize {
                let oldest_at = st.reqs.front().expect("len >= rpm implies non-empty").1;
                eta = eta.max(oldest_at + WINDOW - now);
            }
        }
        if let Some(tpm) = self.tpm {
            if st.tok_sum + est_tokens > tpm {
                // `None` means even an empty window cannot fit the
                // estimate (est > tpm): no wait makes it fit, so the
                // only outcome is to throttle — return `Never` now
                // rather than an eta the deadline check would sleep on.
                match token_eta(&st.toks, st.tok_sum, est_tokens, tpm, now) {
                    Some(wait) => eta = eta.max(wait),
                    None => return Admission::Never,
                }
            }
        }
        if eta.is_zero() {
            let id = st.next_id;
            st.next_id += 1;
            if self.rpm.is_some() {
                st.reqs.push_back((id, now));
            }
            if self.tpm.is_some() {
                st.tok_sum += est_tokens;
                st.toks.push_back(TokEntry {
                    id,
                    at: now,
                    tokens: est_tokens,
                });
            }
            Admission::Admit(id)
        } else {
            Admission::Wait(eta)
        }
    }
}

enum Admission {
    /// Windows admit: the ticket id carrying the charges.
    Admit(u64),
    /// Wait at least this long, then retry.
    Wait(Duration),
    /// The estimate alone exceeds the whole tpm budget: no wait makes
    /// it fit, so `acquire` must give up immediately.
    Never,
}

/// When the currently-binding window entry leaves the 60 s window —
/// the earliest moment the estimate could fit. `None` when even an
/// empty window cannot fit the estimate (`est > tpm`): every entry
/// dropped, `sum` reaches 0 and `0 + est <= tpm` still fails.
fn token_eta(
    toks: &VecDeque<TokEntry>,
    tok_sum: u64,
    est: u64,
    tpm: u64,
    now: Instant,
) -> Option<Duration> {
    let mut sum = tok_sum;
    for e in toks.iter() {
        sum = sum.saturating_sub(e.tokens);
        if sum + est <= tpm {
            return Some((e.at + WINDOW).saturating_duration_since(now));
        }
    }
    None
}

/// Expired entries leave the windows.
fn prune(st: &mut WindowState, now: Instant) {
    while st
        .reqs
        .front()
        .is_some_and(|(_, at)| *at + WINDOW <= now)
    {
        st.reqs.pop_front();
    }
    while st.toks.front().is_some_and(|e| e.at + WINDOW <= now) {
        let e = st.toks.pop_front().expect("front checked above");
        st.tok_sum = st.tok_sum.saturating_sub(e.tokens);
    }
}

/// Remove the charges of ticket `id` from both windows (semaphore
/// timeout path). Entries may have aged out already — then there is
/// nothing to undo.
fn refund(st: &mut WindowState, id: u64, est_tokens: u64) {
    if let Some(pos) = st.reqs.iter().position(|(entry_id, _)| *entry_id == id) {
        st.reqs.remove(pos);
    }
    if let Some(pos) = st.toks.iter().position(|e| e.id == id) {
        st.toks.remove(pos);
        st.tok_sum = st.tok_sum.saturating_sub(est_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RateLimitConfig;

    fn limiter(rpm: Option<u32>, tpm: Option<u64>, mc: Option<u32>, max_wait_ms: Option<u64>) -> std::sync::Arc<ProviderLimiter> {
        std::sync::Arc::new(ProviderLimiter::new(&RateLimitConfig {
            rpm,
            tpm,
            max_concurrency: mc,
            max_wait_ms,
        }))
    }

    fn named_limiter(name: &str, tpm: u64) -> std::sync::Arc<ProviderLimiter> {
        std::sync::Arc::new(ProviderLimiter::new_named(
            name,
            &RateLimitConfig {
                rpm: None,
                tpm: Some(tpm),
                max_concurrency: None,
                max_wait_ms: None,
            },
        ))
    }

    #[test]
    fn never_fits_warns_once_per_window_and_names_the_provider() {
        // The "estimate exceeds the whole tpm budget" case is a
        // configuration problem, not transient pacing: the operator must
        // see it clearly instead of an endless stream of per-request
        // throttle warnings — and since the condition persists for the
        // process's lifetime, the warning repeats once per window
        // instead of firing exactly once and disappearing into the log.
        use crate::tracing_capture::{capture_tracing, CaptureWriter};

        let lim = named_limiter("mm", 100);
        let w = CaptureWriter::new();
        let ((), captured) = capture_tracing(&w, || {
            // `capture_tracing` installs a thread-local subscriber, so
            // the events must be produced on this thread — drive the
            // runtime from here rather than from an async test.
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime");
            rt.block_on(async {
                for _ in 0..3 {
                    let err = lim.acquire(500).await.expect_err("never fits");
                    assert_eq!(err.eta, Duration::ZERO);
                }
            });
        });
        assert_eq!(
            captured
                .matches("rate_limit.tpm is below the estimated request size")
                .count(),
            1,
            "one warning per window, not one per request: {captured:?}"
        );
        assert!(
            captured.contains("mm"),
            "the warning must name the provider: {captured:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn never_fits_warning_repeats_after_a_window() {
        // The warning is not fire-and-forget: a misconfiguration that
        // silently routes 100% of a provider's traffic to the fallback
        // chain must stay visible, so it repeats once the window has
        // passed.
        use crate::tracing_capture::TracingCapture;

        let lim = named_limiter("mm", 100);
        let cap = TracingCapture::install();
        let _ = lim.acquire(500).await.expect_err("never fits");
        // Same window: still one warning.
        let _ = lim.acquire(500).await.expect_err("never fits");
        let captured = cap.take();
        assert_eq!(
            captured
                .matches("rate_limit.tpm is below the estimated request size")
                .count(),
            1,
            "no repeat inside the same window: {captured:?}"
        );

        // Past the window, the warning fires again. The paused clock
        // jumps to the sleep's wake instant, so the limiter's window
        // arithmetic moves with it.
        tokio::time::sleep(WINDOW + Duration::from_secs(1)).await;
        let _ = lim.acquire(500).await.expect_err("never fits");
        let captured2 = cap.take();
        assert_eq!(
            captured2
                .matches("rate_limit.tpm is below the estimated request size")
                .count(),
            1,
            "the warning repeats after a window: {captured2:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn rpm_window_admits_the_limit_then_paces_until_the_oldest_expires() {
        // max_wait must exceed the 60 s window here: the point of the
        // test is the paced sleep actually happening (under the paused
        // clock it advances time to exactly the wake instant), not the
        // throttle-on-cap behavior covered by the max_wait tests below.
        let lim = limiter(Some(2), None, None, Some(120_000));
        let t0 = Instant::now();
        lim.acquire(0).await.expect("first admission free");
        lim.acquire(0).await.expect("second admission free");
        // Third request must wait for the first one to age out of the
        // 60 s window — under the paused clock, the sleep advances time
        // to exactly that point.
        lim.acquire(0).await.expect("third admitted after the window slides");
        let waited = Instant::now() - t0;
        assert!(
            waited >= WINDOW - Duration::from_millis(1),
            "third admission should have waited ~60s, waited {waited:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn window_fully_recovers_after_sixty_seconds() {
        let lim = limiter(Some(1), None, None, Some(120_000));
        lim.acquire(0).await.expect("admitted");
        // A full window later the budget is back — repeatedly.
        for _ in 0..3 {
            tokio::time::advance(WINDOW + Duration::from_secs(1)).await;
            lim.acquire(0).await.expect("budget recovered");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn tpm_window_paces_on_estimated_tokens_and_settle_frees_budget() {
        let lim = limiter(None, Some(1000), None, Some(120_000));
        let t0 = Instant::now();
        let ticket = lim.acquire(600).await.expect("600 fits an empty window");
        // The upstream reports fewer tokens than we reserved — settle
        // replaces the charge, so the next request fits immediately.
        ticket.settle(100);
        lim.acquire(600).await.expect("100 + 600 <= 1000 after settle");
        // Third request: 700 + 600 > 1000, must wait for the first
        // entry to expire.
        lim.acquire(600).await.expect("admitted once the window slides");
        assert!(
            Instant::now() - t0 >= WINDOW - Duration::from_millis(1),
            "third admission should have waited for the window"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn settle_above_estimate_tightens_the_window() {
        let lim = limiter(None, Some(1000), None, Some(120_000));
        let ticket = lim.acquire(200).await.expect("fits");
        ticket.settle(900); // upstream billed more than the estimate
        // 900 + 200 > 1000 — the replacement charge binds immediately.
        let t0 = Instant::now();
        lim.acquire(200).await.expect("waits for the settled entry to expire");
        assert!(Instant::now() - t0 >= WINDOW - Duration::from_millis(1));
    }

    #[tokio::test(start_paused = true)]
    async fn max_wait_zero_throttles_without_waiting() {
        let lim = limiter(Some(1), None, None, Some(0));
        lim.acquire(0).await.expect("first free");
        let t0 = Instant::now();
        let err = lim.acquire(0).await.expect_err("max_wait 0 = never wait");
        assert!(err.eta > Duration::ZERO);
        assert_eq!(Instant::now(), t0, "no time may pass when throttling");
    }

    #[tokio::test(start_paused = true)]
    async fn max_wait_bounds_the_sleep_and_throttles() {
        let lim = limiter(Some(1), None, None, Some(1500));
        lim.acquire(0).await.expect("first free");
        let err = lim
            .acquire(0)
            .await
            .expect_err("60s eta exceeds the 1.5s cap");
        assert!(err.eta >= Duration::from_secs(1), "eta names the real wait");
    }

    #[tokio::test]
    async fn estimate_larger_than_tpm_never_fits_and_throttles() {
        // Real time, deliberately: the point is that an unsatisfiable
        // estimate is refused *immediately* — a 120 s budget must not
        // be burned sleeping on a window no wait can satisfy (the old
        // behavior slept the full 60 s window per loop iteration).
        let lim = limiter(None, Some(100), None, Some(120_000));
        let t0 = std::time::Instant::now();
        let err = lim
            .acquire(500)
            .await
            .expect_err("500 tokens cannot fit a 100-TPM window");
        assert_eq!(err.eta, Duration::ZERO, "eta reports that no wait fits");
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "never-fits must not wait the max_wait budget"
        );
        // The failed request must not have charged anything.
        let st = lim.state.lock().unwrap();
        assert_eq!(st.tok_sum, 0, "throttled request leaves no charge");
    }

    #[tokio::test]
    async fn rpm_only_limiter_never_looks_at_the_estimate() {
        // The estimate exists for the TPM window only: an rpm-only
        // limiter must admit any estimate the rpm window has room for —
        // including one far above any plausible tpm figure (the router
        // skips the estimate walk for such limiters and charges 0).
        let lim = limiter(Some(5), None, None, None);
        lim.acquire(u64::MAX).await.expect("rpm window admits");
    }

    #[tokio::test]
    async fn semaphore_timeout_reports_the_consumed_wait_budget() {
        // The WARN log's eta must say the slot was the binding
        // constraint (the full budget burned), not a ~0 remainder.
        let lim = limiter(None, None, Some(1), Some(40));
        let held = lim.acquire(0).await.expect("occupies the only slot");
        let err = lim.acquire(0).await.expect_err("no slot free within 40ms");
        assert_eq!(err.eta, Duration::from_millis(40), "eta = consumed budget");
        drop(held);
    }

    #[tokio::test]
    async fn concurrency_gate_holds_until_the_ticket_drops() {
        // Real time: the interesting behavior (a pending acquire
        // resolving when the permit is returned) interacts with the
        // paused-clock auto-advance in ways that flake; a real 50 ms
        // wait is bounded and unambiguous.
        let lim = limiter(None, None, Some(1), Some(50));
        let t1 = lim.acquire(0).await.expect("first permit granted");
        let t2 = lim.clone().acquire(0).await;
        assert!(t2.is_err(), "second concurrent acquire must throttle");
        drop(t1); // slot released
        let t3 = lim.acquire(0).await.expect("slot free again after drop");
        drop(t3);
    }

    #[tokio::test]
    async fn concurrency_gate_admits_after_settle() {
        let lim = limiter(None, None, Some(1), Some(50));
        let t1 = lim.acquire(0).await.expect("granted");
        t1.settle(42); // consumes the ticket → permit returned
        lim.acquire(0).await.expect("admitted after settle");
    }

    #[tokio::test]
    async fn semaphore_timeout_refunds_the_window_charge() {
        // rpm 2 + concurrency 1 + short cap: hold the permit, then two
        // acquires charge rpm entries and throttle on the semaphore.
        // Those refunds must leave the rpm window with only the
        // original request.
        let lim = limiter(Some(3), None, Some(1), Some(30));
        let held = lim.acquire(0).await.expect("occupies the only slot");
        for _ in 0..2 {
            lim.acquire(0)
                .await
                .expect_err("no slot free within 30ms");
        }
        drop(held);
        // 3 - 2 refunded = 1 charged: two more acquires fit without
        // waiting (third would pace).
        lim.acquire(0).await.expect("refund 1");
        lim.acquire(0).await.expect("refund 2");
        let st = lim.state.lock().unwrap();
        assert_eq!(st.reqs.len(), 3, "held + two admitted; refunded never charged");
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_acquires_serialize_within_the_window() {
        // A 10-minute cap: five admit at once, five pace through one
        // window slide; none may give up.
        let lim = limiter(Some(5), None, None, Some(600_000));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let lim = lim.clone();
            handles.push(tokio::spawn(async move { lim.acquire(0).await.is_ok() }));
        }
        let results = futures_util::future::join_all(handles).await;
        let admitted = results
            .into_iter()
            .filter(|r| matches!(r, Ok(true)))
            .count();
        assert_eq!(
            admitted, 10,
            "all ten must eventually admit (paced), none may throttle"
        );
    }

    #[test]
    fn build_maps_only_configured_providers() {
        // A provider without `rate_limit:` must not appear in the map —
        // its request path stays ungated.
        let raw = r#"
providers:
  - name: limited
    type: anthropic
    api_key: sk-test
    rate_limit:
      rpm: 10
  - name: plain
    type: anthropic
    api_key: sk-test
models:
  - name: m
    primary: limited
    fallback_chain: [plain]
"#;
        let cfg = Config::parse(raw).expect("parse");
        let limits = ProviderRateLimiters::build(&cfg);
        assert!(limits.get("limited").is_some());
        assert!(limits.get("plain").is_none());
    }
}
