# Copilot 402 quota fallback plan

## Summary

GitHub Copilot can reject inference requests with HTTP `402 Payment Required` and a body such as `You have exceeded your monthly quota`. Copilot already preserves that response as `ProxyError::Upstream { status: 402, body }`, but the router only falls back when `ProxyError::is_cooldownable()` returns true. Because `402` is explicitly outside the current cooldownable set, both `Router::complete()` and `Router::stream()` return the upstream error immediately instead of recording the failed attempt, cooling Copilot down, and advancing to the next provider. Treat `402` as quota exhaustion globally at the router error-classification boundary, and give it the configured quota/rate-limit cooldown rather than the five-second transient-failure cooldown.

## Root cause

The cooldownable status list is not defined inline in `router.rs::route()` (there is no single `route()` function in the current code). Both routing entry points delegate the decision to `ProxyError::is_cooldownable()`:

- `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:173` in `Router::complete()`
- `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:309` in `Router::stream()`

The exact status set is defined at `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:84-90`:

> ```rust
> pub fn is_cooldownable(&self) -> bool {
>     match self {
>         ProxyError::Upstream { status, .. } => {
>             matches!(*status, 401 | 404 | 408 | 429) || *status >= 500
>         }
>         _ => false,
>     }
> }
> ```

`402` is absent. The existing unit test makes the current behavior explicit by listing `402` among non-cooldownable statuses at `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:215-229`:

> ```rust
> for s in [400u16, 402, 403, 409] {
>     check(s, false);
> }
> ```

Copilot preserves non-success HTTP statuses as `u16` in all inference paths; it does not remap or special-case `402`:

- Responses, non-streaming: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:719-727`
- Responses, streaming before body streaming starts: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:750-758`
- Chat Completions, non-streaming: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:832-840`
- Chat Completions, streaming before body streaming starts: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:882-890`

All four paths use the same preservation shape:

> ```rust
> return Err(ProxyError::Upstream {
>     status: status.as_u16(),
>     body: text,
> });
> ```

The same status-preserving behavior exists in the generic providers, so the proposed router-level classification applies consistently rather than requiring a Copilot-only remap:

- OpenAI Responses complete/stream: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/openai_responses.rs:143-150` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/openai_responses.rs:178-185`
- OpenAI Chat Completions complete/stream: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/openai_compat.rs:146-153` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/openai_compat.rs:192-199`
- Native Anthropic complete/stream: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/anthropic.rs:189-203` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/anthropic.rs:227-240`
- The provider abstraction itself adds no status handling: `/home/zhouqt/Dropbox/src/llmproxy/src/providers/mod.rs:28-43`.

Current `402` trace:

1. Copilot returns `ProxyError::Upstream { status: 402, body: "You have exceeded your monthly quota" }`, preserving `402` as `u16` (`copilot.rs` locations above).
2. `Router::complete()` checks `e.is_cooldownable()` at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:173`; it is false because of `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:87`.
3. The router then checks `is_model_unsupported()` at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:193`. The observed quota body matches none of the strings at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:39-46`, so this is also false.
4. The catch-all branch at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:217-220` returns the original error immediately. No `RouteAttempt` is added and no cooldown entry is written.
5. Streaming behaves equivalently via `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:309-346`.

## Proposed fix

### 1. Classify HTTP 402 as cooldownable

Change the canonical status set in `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:84-90`. This is the set consumed by both router paths; duplicating a second list in `router.rs` would allow complete and streaming behavior to drift.

> ```diff
> -matches!(*status, 401 | 404 | 408 | 429) || *status >= 500
> +matches!(*status, 401 | 402 | 404 | 408 | 429) || *status >= 500
> ```

Update the table-driven regression test at `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:215-229`:

> ```diff
> -for s in [401u16, 404, 408, 429, 500, 502, 503, 504] {
> +for s in [401u16, 402, 404, 408, 429, 500, 502, 503, 504] {
>      check(s, true);
>  }
> -for s in [400u16, 402, 403, 409] {
> +for s in [400u16, 403, 409] {
>      check(s, false);
>  }
> ```

After this change, the existing cooldownable branches at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:173-191` and `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:309-327` will record the attempt, call `CooldownCache::mark_cooldown()`, and continue through the provider chain.

### 2. Give quota exhaustion the configured cooldown period

The router currently gives only `429` the model-configured cooldown and gives every other cooldownable response five seconds at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:180-184` and `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:316-320`. A monthly quota cannot recover after five seconds, so treat `402` like `429` in both paths:

> ```diff
> -let ttl = if *status == 429 {
> +let ttl = if matches!(*status, 402 | 429) {
>      Duration::from_secs(model.cooldown_seconds)
>  } else {
>      Duration::from_secs(5)
>  };
> ```

Apply the same diff in both `Router::complete()` and `Router::stream()`. This makes the requested cooldown-cache test assert the configured period rather than the transient five-second period.

### 3. Keep 402 out of model-unsupported classification and preserve branch precedence

`is_model_unsupported()` first accepts any 4xx at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:24-30`, then looks for narrow model-related body strings at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:31-47`. The observed body, `You have exceeded your monthly quota`, contains none of `not supported`, `not_supported`, `not_found`, `not exist`, `not a valid`, `model_not_`, `"model"`, or the paired `supported api model`/`you passed`, so it is not currently misclassified.

No production change to the heuristic is needed. Add a focused regression assertion near `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:1465-1514`:

> ```rust
> let quota = ProxyError::Upstream {
>     status: 402,
>     body: "You have exceeded your monthly quota".into(),
> };
> assert!(!is_model_unsupported(&quota));
> ```

Once `402` is cooldownable, the cooldownable guard remains intentionally before the heuristic in both paths (`/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:173` before `:193`, and `:309` before `:328`). Thus even an unusual 402 body that mentions a model is classified as quota/cooldown, not as the heuristic's 60-second model-catalog mismatch.

### 4. Do not add 403 globally

Keep `403 Forbidden` non-cooldownable. It has broader meanings than quota exhaustion, including authorization/policy rejection. Copilot itself treats `403` as credential rejection at its token-exchange endpoint (`/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:434-443`) and as auth failure for model discovery (`/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:610-629`). Globally cooling and falling back on every inference-side 403 could hide a permission or policy configuration error and send the request to a provider with different access controls. Preserve `403` in the false table at `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:227-229`, and add routing tests proving a generic 403 still surfaces immediately.

If a future provider documents a quota-specific 403 envelope, handle that with an explicit provider/body classification backed by observed payloads rather than making all 403 responses cooldownable.

## Test plan

### Error-classification unit tests

In `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:215-229`:

1. Move `402` to the cooldownable table.
2. Leave `403` in the non-cooldownable table.
3. Retain coverage for 401/404/408/429/5xx to prevent regression of existing behavior.

### Router integration tests with wiremock

Add tests near the existing wire-level 429 fallback test at `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:206-277`:

1. **Non-streaming 402 fallback:** configure the primary wiremock to return `402` with `You have exceeded your monthly quota`, and the backup to return 200. Assert backup output is returned, one attempt is recorded as `primary:402`, and its body is preserved.
2. **Configured cooldown:** inspect `router.cooldown().active()` as the existing TTL test does at `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:858-871`. Assert the primary entry has status 402 and remaining duration is positive and no greater than the configured `cooldown_seconds` (with a small scheduler-tolerant lower bound, e.g. configured 60 seconds and remaining greater than 55 seconds). Issue a second request and verify the primary wiremock call count remains one, proving the entry is actively skipped.
3. **Empty-body 402:** return a 402 with an empty body and assert fallback/cooldown still occur; classification must depend on status, not body text.
4. **403 negative control:** primary returns ordinary `403 Forbidden`, backup has `.expect(0)`, and `Router::complete()` returns `ProxyError::Upstream { status: 403, .. }`; assert primary is absent from `cooldown().active()`.

The existing integration helper preserves real HTTP status as `u16` at `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:87-115`, so these tests exercise the actual wire response → `ProxyError::Upstream` → router classification flow.

### Copilot provider preservation tests

The Copilot test module already has access to its test-only API-base override and stored-token fixture at `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:930-958`. Add wiremock tests beside `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:1429-1485` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:1904-1937` to lock down all relevant endpoint shapes:

- Chat Completions `complete()` preserves `402` and quota body.
- Chat Completions `stream()` preserves `402` before constructing the SSE stream.
- Responses `complete()` preserves `402` and quota body.
- Responses `stream()` preserves `402` before constructing the SSE stream.

These provider tests do not themselves prove fallback; they prove Copilot supplies the router with the exact status/body needed. The integration tests above prove router fallback and cooldown through real HTTP frames.

### Streaming router tests

Streaming and non-streaming use separate router branches, so test both despite identical classification intent:

1. Add a streaming-capable wiremock provider helper in `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:75-125`, or use the real `OpenAiCompatProvider`, so a pre-stream HTTP 402 becomes `ProxyError::Upstream` before `ProviderOutput::Stream` is returned.
2. Primary returns 402, backup returns a successful SSE response. Assert `Router::stream()` selects backup, records `primary:402`, and puts primary in cooldown for approximately `model.cooldown_seconds`.
3. Add the streaming 403 negative control: error surfaces, backup is not called, and no cooldown entry is created.

The router's existing mock-based streaming fallback test at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:741-758` should also be parameterized or supplemented with status 402 for fast branch-level coverage.

## Edge cases

- **402 with empty body:** must still cool down and fall back because `is_cooldownable()` is status-based. `CooldownCache::mark_cooldown()` already accepts an empty reason and logs a bounded placeholder via `/home/zhouqt/Dropbox/src/llmproxy/src/cooldown.rs:43-69`.
- **402 before streaming starts:** fallback is possible. Copilot checks the HTTP response status before constructing the body stream at `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:750-764` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:882-893`; this is the case the fix covers.
- **402 mid-stream / after success headers:** HTTP status cannot change after headers have been accepted. If quota exhaustion is encoded as an SSE error event or the connection fails after bytes have flowed, router-level fallback is impossible without duplicating or interleaving client-visible content. `Router::stream()` documents this boundary at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:261-263` and `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:302-306`. Surface the stream error; do not retry another provider.
- **Native Anthropic passthrough:** it is not a separate router path. `AnthropicProvider` preserves any non-success status as `u16` in complete and stream (`/home/zhouqt/Dropbox/src/llmproxy/src/providers/anthropic.rs:189-203`, `:227-240`), so a 402 there will also trigger fallback after the global classification change.
- **OpenAI-compatible/Responses providers:** they likewise preserve 402. The change intentionally applies to all providers because HTTP 402 semantically signals payment required, but Copilot is the only currently known provider in this codebase to emit it for monthly quota.
- **Quota body containing model-like text:** cooldown classification executes before `is_model_unsupported()`, so after the fix it remains on the quota path even if an upstream changes its message wording.
- **All fallbacks fail:** 402 is recorded in `RouteAttempt`; if the chain exhausts, `AllProvidersFailed` retains the last actual upstream error and the failed-provider list according to `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:229-238` and `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:118-130`.

## Risks

- **Global semantics:** changing `ProxyError::is_cooldownable()` means a 402 from Copilot, OpenAI-compatible, OpenAI Responses, or native Anthropic passthrough will all trigger fallback. HTTP 402 is reserved for payment-required semantics, so this is normally desirable, but a nonstandard upstream could use it for a permanent account/configuration error. The configured cooldown limits repeated calls while still allowing later recovery; tests should make this global behavior explicit.
- **Cooldown duration:** applying `model.cooldown_seconds` to monthly exhaustion can still be shorter than the actual reset window (default five minutes versus potentially days). It is safer than five seconds and follows the existing operator-configurable rate-limit policy, but a future enhancement could parse a trustworthy reset header. Do not infer a monthly reset time from body text in this fix.
- **Hidden authorization failures:** adding 403 would create a larger false-positive surface and potentially hide policy failures. This plan explicitly does not add it.
- **Streaming boundary:** only response-status failures before stream construction can fall back. Attempting fallback after bytes are emitted would corrupt the downstream conversation and is out of scope.
- **Retry count interaction:** `Router::complete()` may retry a cooldownable provider according to `max_retries_per_provider` before advancing (`/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:164-170`). Existing semantics currently allow this even after marking cooldown inside the inner loop. This fix should not silently alter retry policy; tests should configure one retry when asserting exactly one Copilot request.

## Files to modify

- `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:84-90` — add `402` to the canonical cooldownable status set.
- `/home/zhouqt/Dropbox/src/llmproxy/src/error.rs:215-229` — update the cooldownability table test and retain 403 as a negative case.
- `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:180-184` — use configured cooldown for 402 in `Router::complete()`.
- `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:316-320` — apply the same 402 TTL policy in `Router::stream()`.
- `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:1465-1514` — add quota-body regression coverage for `is_model_unsupported()` and status-branch separation.
- `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:206-277` — add wire-level 402 fallback and 403 non-fallback cases.
- `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:782-874` — assert the 402 cooldown entry uses the configured duration; add/extend a real-HTTP streaming helper and tests near the router-level fallback cases.
- `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:1429-1485` and `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:1904-1937` — add Copilot wiremock regression tests showing 402 status/body preservation for Responses and Chat Completions, complete and stream.

## Followup: Opus review (2026-07-30)

Opus reviewed the implementation that shipped in PR #18 against the test plan and flagged
gaps. The findings fall into three buckets.

### Findings accepted as required followup

- **Streaming 402 fallback test, exercising `Router::stream()` end-to-end.** Add a new
  test beside `stream_falls_back_on_cooldownable_error` at
  `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:742-759`. Because
  `WiremockOpenAiProvider::stream` returns `unimplemented!()` at
  `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:119-125`, this test
  must use the in-memory `MockProvider` pattern already proven at
  `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:742-759`. Primary returns
  `ProxyError::Upstream { status: 402, body: "You have exceeded your monthly quota" }`
  from `stream()`, backup returns `ProviderOutput::Stream(...)`, then assert backup was
  selected, `attempts[0]` carries `status: 402`, and `cooldown().active()` records
  primary with status 402 and a remaining TTL no greater than `model.cooldown_seconds`
  (scheduler-tolerant lower bound).
- **Streaming 403 negative control.** Add a parallel in-memory test: primary returns
  `ProxyError::Upstream { status: 403, body: "Forbidden" }` from `stream()`, no
  fallback configured, `Router::stream()` returns the original 403 error, and primary
  is NOT in `cooldown().active()`. This locks down the "403 must surface immediately"
  behavior the plan calls out at `/home/zhouqt/Dropbox/src/llmproxy/docs/PLANS/copilot-402-fallback.md:123-127`.

### Findings accepted as low-priority polish

- **Add `expect(1)` on the primary wiremock** in
  `mock_llm_provider_falls_back_when_primary_returns_402_quota` at
  `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:287-294`, mirroring the
  TTL test at `/home/zhouqt/Dropbox/src/llmproxy/tests/integration_router.rs:1118-1126`.
  Today the assertion only verifies `attempts.len() == 1`; the wiremock expectation
  gives an independent proof that primary was hit exactly once.
- **Tighten the comments on the four new copilot tests** at
  `/home/zhouqt/Dropbox/src/llmproxy/src/providers/copilot.rs:1940-2070`. Reword them
  to focus on "why" — copilot must surface 402 before SSE construction so the router
  can fall back — rather than restating the test name.

### Findings explicitly rejected with rationale

- **Drop one of the `is_model_unsupported` + `is_cooldownable` double assertions** in
  the regression block at `/home/zhouqt/Dropbox/src/llmproxy/src/router.rs:1465-1514`.
  Kept: the plan documents at `/home/zhouqt/Dropbox/src/llmproxy/docs/PLANS/copilot-402-fallback.md:107-121`
  that cooldown classification executes *before* `is_model_unsupported()`; both
  assertions together document that ordering in the test itself, and removing one would
  weaken that evidence.
- **Extract a deduplication helper for the two 402 tests.** Skipped — the two tests
  differ in body text (`"You have exceeded your monthly quota"` vs empty body) and the
  duplication is small. Per CLAUDE.md "do not add new abstractions" guideline, a
  helper would obscure the wiremock setup for negligible gain.
- **Add a chain-exhausted 402 test** (every provider returns 402, assert
  `AllProvidersFailed`). Skipped — `AllProvidersFailed`'s `into_response` behavior is
  already covered by existing tests with 5xx chains, and adding a 402 variant is
  low-value coverage that exercises the same code path.
