# Plan: research Claude Code → llmproxy `web_search_20250305` forwarding

## Goal

Identify whether and how llmproxy should forward Anthropic's `web_search_20250305`
hosted-tool request from Claude Code to upstream providers, and what changes
the proxy still needs. No code changes yet — research only.

## Branch

This document was authored on `main` (commit `ee77020`).

## TL;DR

| Question | Answer |
|---|---|
| Does Claude Code do a pre-flight capability probe before using `web_search_20250305`? | **No.** It is decided by a local string match in `WebSearchTool.isEnabled()` (`firstParty` / `vertex` / `foundry`). No `/v1/models` call. |
| Does Anthropic's `/v1/models/{model_id}` expose a `web_search` capability field? | **No.** `ModelCapabilities` exposes `batch / citations / code_execution / context_management / effort / image_input / pdf_input / structured_outputs / thinking` — no `web_search`. |
| Does Claude Code send `web_search_20250305` in `tools[]` or somewhere else? | **Both paths exist.** The client ultimately concatenates `extraToolSchemas` after the regular `toolSchemas` (`claude.ts:1385-1396`). The wire shape is the same: a `tools[]` entry with `type: "web_search_20250305"`. |
| Does llmproxy today forward `web_search_20250305` correctly? | **No.** The request **never reaches the router** — `Tool::input_schema: Value` is required, Claude Code's hosted tool has no `input_schema`, serde returns `missing field` → `AppJson` returns 400. Three further gaps exist even if that is fixed. |
| Do we need to **intercept** the tool call and execute a search ourselves? | **No.** The hosting upstream (api.anthropic.com, GitHub Copilot, GLM) is responsible for execution. Intercepting requires a search-provider abstraction the project does not have. |
| Is the full forward + translate workflow a must? | **Yes, but only the forward + translate parts.** Not intercept. |

## 1. Client-side: what Claude Code actually does

### 1.1 No pre-flight probe

`@anthropic-ai/claude-code` v2.1.88 (extracted from the source map that shipped
in the npm package). Evidence:

- `src/utils/model/modelCapabilities.ts:46-51` — `isModelCapabilitiesEligible()`
  returns `false` unless `process.env.USER_TYPE === 'ant'` AND
  `getAPIProvider() === 'firstParty'` AND `isFirstPartyAnthropicBaseUrl()`.
  Normal users get `false` and the `/v1/models` listing is never sent.
- `src/utils/model/modelCapabilities.ts:93` — only call site: `anthropic.models.list({ betas })`.
- `src/main.tsx:419` — fires once at startup with `void refreshModelCapabilities()`,
  but the function no-ops outside Anthropic-internal builds.
- `src/utils/model/validateModel.ts:57-74` — `validateModel()` does a probe via
  `sideQuery({ max_tokens: 1, messages: [{role:'user', content:'Hi'}], querySource: 'model_validation' })`.
  This is a `POST /v1/messages` with one token, **not** `/v1/models`, and **not**
  specifically about web_search.

### 1.2 Web-search gating is purely local

`src/tools/WebSearchTool/WebSearchTool.ts:168-193` — `isEnabled()`:

- `firstParty` (default api.anthropic.com) → `true`
- `vertex` → `true` only when model name includes `claude-opus-4` / `claude-sonnet-4` / `claude-haiku-4`
- `foundry` → `true`
- else → `false`

No network call. The decision is local.

### 1.3 Web-search tool call flow

`src/tools/WebSearchTool/WebSearchTool.ts:268-291` — `call()`:

- `tools: []` (empty)
- `extraToolSchemas: [makeToolSchema()]` (line 284) — schema is
  `{ type: 'web_search_20250305', name: 'web_search', allowed_domains, blocked_domains, max_uses: 8 }`
- `toolChoice: useHaiku ? { type: 'tool', name: 'web_search' } : undefined`
- `querySource: 'web_search_tool'`
- single user message: `'Perform a web search for the query: ' + query` (lines 257-259)

`src/services/api/claude.ts:1385-1396` — server-side concatenation:
`const allTools = [...toolSchemas, ...extraToolSchemas]`. So `web_search_20250305`
ends up in the same `tools[]` array on the wire.

`src/tools/WebSearchTool/WebSearchTool.ts:311-387` — streaming consumer:

- on `content_block_start` with `type: "server_tool_use"`, track `currentToolUseId`
- on `input_json_delta`, accumulate `partial_json`, extract query by regex
  `/"query"\s*:\s*"((?:[^"\\]|\\.)*)"/`
- on `content_block_start` with `type: "web_search_tool_result"`, emit
  `onProgress({ type: 'search_results_received', resultCount, query })`

`src/tools/WebSearchTool/UI.tsx` — terminal UI only renders one line
`Did N search(es) in Xs`. **No citations panel** (unlike Claude Desktop).

**Wire-shape consequence for the proxy**: the inbound request from the client
looks like a regular `POST /v1/messages` with `tools: [{type: "web_search_20250305",
name: "web_search", max_uses: 8, ...}]` and a single user message. No special
processing path is needed downstream — just forward + translate.

## 2. Server-side: what Anthropic exposes

`anthropic-sdk-python 0.117.1` (local install):
`/home/zhouqt/.local/lib/python3.12/site-packages/anthropic/`.

### 2.1 `GET /v1/models/{model_id}` returns `ModelInfo` with `capabilities`

- `resources/models.py:44-87` — `Models.retrieve(model_id)` issues `GET /v1/models/{model_id}`.
- `types/model_info.py:13-39` — `ModelInfo.capabilities: Optional[ModelCapabilities]`.
- `types/capability_support.py` — `CapabilitySupport { supported: bool }`.

### 2.2 `ModelCapabilities` does NOT include `web_search`

`types/model_capabilities.py:12-40`:

```python
class ModelCapabilities(BaseModel):
    batch: CapabilitySupport
    citations: CapabilitySupport
    code_execution: CapabilitySupport
    context_management: ContextManagementCapability
    effort: EffortCapability
    image_input: CapabilitySupport
    pdf_input: CapabilitySupport
    structured_outputs: CapabilitySupport
    thinking: ThinkingCapability
```

`types/beta/beta_model_capabilities.py` is identical. The Anthropic protocol
deliberately does not expose web_search as a model-level capability — it
exposes it as a per-request tool declaration (`type: "web_search_20250305"`).

**Implication for the proxy**: even if we added a `/v1/models` capability-caching
layer, it would not tell us "this upstream supports web_search". The only way
to find out is to send the tool and observe the upstream's behavior.

### 2.3 `web_search_20250305` tool type

`types/web_search_tool_20250305_param.py` — `WebSearchTool20250305Param` carries
fields: `name`, `type`, `allowed_callers`, `allowed_domains`, `blocked_domains`,
`cache_control`, `defer_loading`, `max_uses`, `strict`, `user_location`. All
forwarded verbatim by the SDK.

## 3. LiteLLM's approach (reference, not what we should copy)

`litellm/integrations/websearch_interception/handler.py:56` —
`WebSearchInterceptionLogger`. Strategy:

1. **Pre-hook** (`async_pre_request_hook`, handler.py:390-467): detect
   `web_search_20250305` (or `name: web_search`, or `litellm_web_search`).
   Rewrite into `get_litellm_web_search_tool()` (a function-named tool with
   `{query: string}` schema). Also force-converts `tool_choice: {type: tool,
   name: "web_search"}` → `name: "litellm_web_search"` (handler.py:373-388).
2. **Stream → non-stream** (handler.py:293-296, 314-316, 463-465): the search
   interception needs a stable non-streamed response to parse `tool_use`, so
   `stream=True` is rewritten to `stream=False`, with a flag.
3. **Agentic loop** (`async_should_run_agentic_loop`, handler.py:469-577): when
   the upstream response contains a `tool_use` named `litellm_web_search`,
   execute `litellm.asearch(query, ...)` (Perplexity / Tavily / router
   search tools), inject `tool_result` text, re-issue the upstream.
4. **Post-hook** (`async_post_agentic_loop_response_hook`, handler.py:788-845):
   if the client originally sent a native `web_search_*` tool, prepend
   `web_search_tool_result` blocks to the response so Anthropic-native
   clients (Claude Desktop) can render citations panels.
5. **Short-circuit** (`try_short_circuit_search`, handler.py:91-229): when the
   request is a single search-only `/v1/messages` (Claude Code's pattern),
   skip the LLM entirely and return a synthetic Anthropic response.

For native-capable upstreams (Anthropic API, Bedrock, Vertex, Azure AI),
`BaseAnthropicMessagesConfig.handles_web_search_natively()` returns `True`
by default (`litellm/llms/base_llm/anthropic_messages/transformation.py:117-128`)
and the interception is skipped — the request is forwarded verbatim.

**Why llmproxy should NOT copy this**: llmproxy has no search-provider
abstraction, no API-key plumbing for Tavily/Perplexity, and no agentic-loop
framework. Adding it would be ~1000+ LoC of Rust plus a new config schema
and credential-management surface. The hosting upstream (Anthropic API,
GitHub Copilot, or any compliant provider) is the right place to execute the
search.

## 4. Gaps in current implementation

| # | Gap | File:line | Effect |
|---|---|---|---|
| A | `Tool::input_schema` is required (no `#[serde(default)]`) | `src/anthropic.rs:231-237` | `web_search_20250305` has no `input_schema` → serde `missing field` error → `AppJson` returns **400** at the extractor layer. The request never reaches the router. |
| B | `tools` converter wraps every tool as `kind: "function"` | `src/conversion/request.rs:84-95` | Even if A were fixed, Anthropic-native tools get rewritten into OpenAI function tools. The existing plan `docs/PLANS/invalid-tool-parameters.md:101-106` already calls this out but no code change has landed. |
| C | `ServerToolUse` / `WebSearchToolResult` blocks dropped in assistant → input[] | `src/conversion/responses.rs:284-285` (`{}` empty arm) | Multi-turn conversations lose the web_search call+result context. The existing test `response_with_unknown_output_item_is_ignored` at `src/conversion/responses.rs:1424-1445` confirms the current behavior is "silently ignore `web_search_call`". |
| C-stream | `server_tool_use: None` hardcoded in streaming usage | `src/conversion/responses_stream.rs:362` | Streaming path also loses the `usage.server_tool_use` counter. |
| D | `/v1/models` endpoint doesn't forward `capabilities` | `src/server.rs:212-243` | Even if a client cared about `max_input_tokens` (Claude Code doesn't for normal users), the proxy would drop it. Cosmetic only — not a blocker for web_search. |
| E | No "hosted tool" concept in `src/conversion/mod.rs` | n/a | The converter can't tell "this is a server tool, forward it verbatim" from "this is a client function tool". |

### 4.1 Gap A — `Tool.input_schema` required, hosted tools have none

`src/anthropic.rs:231-237`:

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,  // ← required, no #[serde(default)]
}
```

A real request has `tools: [{type: "web_search_20250305", name: "web_search",
max_uses: 8}]`. serde returns `missing field 'input_schema'` at the HTTP
extractor layer (`src/extractor.rs:60-68` `AppJson`) and the client gets
**400**. The request never reaches `messages_handler`.

**Review correction**: the original plan described this as "silently dropped".
serde's `missing field` error means the failure is loud and early — any
request containing a hosted Anthropic tool (web_search, web_fetch,
code_execution) without `input_schema` is rejected at the boundary.

### 4.2 Gap B — `tools` converter wraps everything as `function`

`src/conversion/request.rs:84-95`:

```rust
tools: req.tools.as_ref().map(|ts| {
    ts.iter()
        .map(|t| ChatTool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: t.name.clone(),
                description: t.description.clone().unwrap_or_default(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}),
```

`docs/PLANS/invalid-tool-parameters.md:101-106` documents the design intent
("Splits `tools` into `web_search_tools` vs `regular_tools`") but the code is
pure passthrough-to-ChatTool. No split.

### 4.3 Gap C — `ServerToolUse` blocks dropped in Responses translator

`src/conversion/responses.rs:284-285` is one of the empty `{}` arms in the
match that builds `ResponseInputItem`s from `ContentBlock`. The test
`response_with_unknown_output_item_is_ignored` at line 1424-1445 codifies the
"silently drop `web_search_call`" behavior as intentional.

### 4.4 Gap C-stream — usage counter

`src/conversion/responses_stream.rs:362` (`server_tool_use: None`) plus
`src/conversion/response.rs:74` and `src/conversion/stream.rs:131` (same
hardcoded `None`) mean we never forward `usage.server_tool_use` from the
upstream to the client. Anthropic's `Usage.server_tool_use` (e.g.
`{"web_search_requests": 1}`) is modeled as `Option<Value>` in
`src/anthropic.rs:384` but never populated.

## 5. Recommended changes (two-phase)

### 5.1 Phase 1 — Fix Tool struct (unblocks all Anthropic-passthrough upstreams)

Change `src/anthropic.rs:231-237` `Tool` to:

```rust
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Server-tool type discriminator (e.g. "web_search_20250305"). Anthropic
    /// server tools are not client-side functions — they must be forwarded
    /// upstream with their original type field.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<String>,
    /// Tool schema. Required for function tools but absent on server tools
    /// (web_search, web_fetch, code_execution). Default to null so serde
    /// doesn't 400 on missing field.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub input_schema: Value,
    /// Additional server-tool parameters (max_uses, allowed_domains, …).
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
}
```

Note: `input_schema` must be `#[serde(default, skip_serializing_if =
"Value::is_null")]`. The original plan kept it required, which would still 400.

`AnthropicProvider::build_body` (`src/providers/anthropic.rs:255-268`) uses
`serde_json::to_value(&req)` — once `Tool` has these changes, the hosted tool
appears structural-Value-equivalent in the upstream request. **No converter change needed**
for the Anthropic-passthrough path.

Tests:
- `messages_request_roundtrips_every_documented_field`: add a hosted tool
  entry **without** `input_schema` (Claude Code wire shape).
- Separate `extra` flatten roundtrip test.
- `complete_passes_through_every_field_unmodified`: add `tools` entry with
  `{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}`.

### 5.2 Phase 2 — Tool split for Chat Completions and Responses providers

Detect hosted tools. Add `is_web_search_tool(t: &Tool) -> bool`:

```rust
fn is_web_search_tool(t: &Tool) -> bool {
    // `{type: "web_search_20250305"}` — canonical Anthropic shape.
    // `{type: "web_search_preview"}` — OpenAI Responses API shape.
    // Must NOT match `{name: "web_search", type: "function"}` (a coincidental
    // user-defined function tool).
    if let Some(type_str) = &t.kind {
        return type_str.starts_with("web_search_")
            || type_str == "web_search"
            || type_str == "web_search_preview";
    }
    false
}
```

**Chat Completions** — `src/conversion/request.rs`:
- If `is_web_search_tool(t)`: strip from `tools[]`, inject `extra_body`
  entry `{"web_search_options": {}}`. This matches OpenAI's wire format
  (see §8.1).
- Default: existing `ChatTool { kind: "function", ... }`.

**Responses API** — `src/conversion/responses.rs`:
- If `is_web_search_tool(t)`: map to `ResponsesTool::WebSearch
  { type: "web_search_preview", search_context_size, user_location }`.
  See §8.2.
- Default: existing `ResponsesTool::Function { ... }`.

### 5.2.a User-visible behavior on different upstream families

The implementation has three observable behaviors depending on which
provider chain serves the request:

- **Anthropic-passthrough** (`src/providers/anthropic.rs`): the request
  body is forwarded verbatim via `serde_json::to_value(req)` →
  `reqwest.json()`. The upstream produces full `server_tool_use` /
  `web_search_tool_result` content blocks and `usage.server_tool_use`
  counter; these flow back through the response unchanged. Claude Code's
  "Did N searches" UI line works.

- **OpenAI Chat Completions** (`src/providers/openai_compat.rs`): the
  upstream receives `web_search_options: {}` (empty object — the
  converter strips Anthropic-specific fields like `max_uses` and
  `allowed_domains`; only the mere presence of the key triggers the
  upstream's search behavior). The response is a normal chat completion
  with the search result baked into the assistant's text. **Claude Code's
  "Did N searches" UI line will not appear** — the upstream emits
  `annotations[]` with `url_citation` entries on the response's text
  content, but the proxy drops them at `src/conversion/response.rs:41`
  and `src/conversion/stream.rs:167` (hardcoded `citations: None`). The
  raw chat completion also has no `tool_calls` to indicate a search was
  performed. When Claude Code forces `tool_choice: {type: tool, name:
  "web_search"}`, the converter remaps it to `"auto"` — the upstream
  still searches because the user prompt explicitly instructs it, but
  the guarantee degrades from "forced" to "probabilistic".

- **OpenAI Responses API** (`src/providers/openai_responses.rs`): the
  upstream receives `tools: [{type: "web_search_preview"}]` and emits
  a `web_search_call` output item (the proxy silently drops it — see
  `src/conversion/responses.rs` `OutputItem::WebSearchCall` arm) followed
  by a `message` item whose `output_text` carries `url_citation`
  annotations (the proxy drops these at
  `src/conversion/responses_stream.rs:399` hardcoded
  `citations: None` and `src/conversion/responses.rs:442` where
  `OutputText` is constructed without annotations). Search results
  reach the client only as plain text in the assistant message; no
  "Did N searches" UI line, no citations.

**Operational implication**: clients running Claude Code against
Anthropic-passthrough upstreams get the full experience. Clients
running against OpenAI-family upstreams get search answers but lose
the visual chrome. Plan doc tracks response-direction translation in
`docs/PLANS/web-search-followup.md` (PR5).

## 6. Out-of-scope (intentionally)

- **Search provider intercept.** LiteLLM's `websearch_interception/` module
  shows the cost: ~2000+ LoC of Python (or ~1000+ LoC of Rust) plus a new
  config schema and credential-management surface. llmproxy does not have a
  search-provider abstraction. Adding it is a separate project.
- **/v1/models capability synthesis.** Anthropic's `ModelCapabilities` doesn't
  expose `web_search`, so a capability table built on it would not help with
  routing. Claude Code doesn't consume `capabilities` for web_search anyway.
- **Downgrading clients to MCP on failure.** Claude Code's MCP fallback path
  is local to the client; the proxy does not need to signal it. A 502 from
  the upstream is sufficient — the client will retry and may fall back
  itself.

## 7. Sources

- `/home/zhouqt/.local/lib/python3.12/site-packages/anthropic/`
  - `resources/models.py:44-87` — `Models.retrieve`
  - `types/model_info.py:13-39` — `ModelInfo.capabilities`
  - `types/model_capabilities.py:12-40`
  - `types/capability_support.py`
  - `types/web_search_tool_20250305_param.py`
- `/home/zhouqt/Dropbox/src/llmproxy/` (this repo)
  - `src/anthropic.rs:96-247` (ContentBlock), `288-391` (MessagesResponse, Usage)
  - `src/conversion/request.rs:84-95` (tools converter)
  - `src/conversion/responses.rs:284-285` (empty arm), `1424-1445` (test)
  - `src/conversion/responses_stream.rs:362` (server_tool_use: None)
  - `src/conversion/response.rs:74`, `src/conversion/stream.rs:131`
  - `src/server.rs:212-243` (list_models_handler)
  - `src/providers/anthropic.rs:255-268` (build_body), `326-403` (passthrough test)
  - `docs/PLANS/invalid-tool-parameters.md:101-106` (existing TODO)
- `/home/zhouqt/src/litellm/`
  - `litellm/integrations/websearch_interception/handler.py:56-1603`
  - `litellm/integrations/websearch_interception/tools.py:14-292`
  - `litellm/integrations/websearch_interception/transformation.py:14-469`
  - `litellm/llms/anthropic/chat/transformation.py:1322-1347` (map_web_search_tool)
  - `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py:336-352, 922-955`
  - `litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py:186-188`
  - `litellm/llms/base_llm/anthropic_messages/transformation.py:117-128` (handles_web_search_natively default)
  - `litellm/llms/openai/chat/gpt_transformation.py:167` (web_search_options supported param)
  - `litellm/llms/openai/chat/gpt_5_transformation.py:155, 182` (search model vs non-search model)
  - `litellm/types/llms/openai.py:590-598` (WebSearchOptions type)
  - `litellm/completion_extras/litellm_responses_transformation/transformation.py:334-335, 949-975`
- Claude Code v2.1.88 source-map extraction (external GitHub repo, NOT in
  this repo):
  - `src/tools/WebSearchTool/WebSearchTool.ts:168-193` (isEnabled)
  - `src/tools/WebSearchTool/WebSearchTool.ts:268-291` (call)
  - `src/tools/WebSearchTool/WebSearchTool.ts:311-387` (stream consumer)
  - `src/tools/WebSearchTool/WebSearchTool.ts:401-434` (mapToolResultToToolResultBlockParam)
  - `src/services/api/claude.ts:1385-1396` (extraToolSchemas concat)
  - `src/utils/model/modelCapabilities.ts:46-51, 93` (eligibility, only call site)
  - `src/utils/model/validateModel.ts:57-74` (validateModel probe)
  - `src/main.tsx:419` (refreshModelCapabilities fire-and-forget)

## 8. OpenAI wire shapes (research result)

LiteLLM research at `/home/zhouqt/src/litellm/` vetted the exact wire format
for OpenAI Chat Completions and Responses API.

### 8.1 OpenAI Chat Completions — `web_search_options` in extra_body

OpenAI **does not accept** `{"type": "web_search"}` inside `tools[]`.
Web search is declared as a top-level parameter `web_search_options`.

**Request shape:**
```json
{
    "model": "gpt-4o",
    "messages": [...],
    "web_search_options": {
        "search_context_size": "medium",
        "user_location": {
            "type": "approximate",
            "approximate": { "city": "SF", "country": "US", "region": "CA" }
        }
    }
}
```

Type: `WebSearchOptions` at `litellm/types/llms/openai.py:590-598`.
`search_context_size`: `"low" | "medium" | "high"` (optional).

**Response shape:** No `tool_calls` with type `"web_search"`. Instead the
response text carries `annotations[]` with `{"type": "url_citation",
"url_citation": {"url": "...", "title": "..."}}`. LiteLLM detects web search
usage by checking for `url_citation` annotations.

**LiteLLM translation** (`litellm/llms/anthropic/experimental_pass_through/
adapters/transformation.py:336-352, 922-955`): Anthropic `web_search_*` tool is
**stripped** from `tools[]` and converted to `web_search_options: {}`
on kwargs (line 945).

**Implication for llmproxy:**
- `anthropic_to_openai_request()` in `src/conversion/request.rs` catches
  `is_web_search_tool(t)` in `tools[]` → strips the tool, injects
  `extra["web_search_options"] = value` into the request body.
- `tool_choice {type: "tool", name: "web_search"}` must be mapped to
  `auto` (OpenAI has no host tool named "web_search" in function-tool space).

### 8.2 OpenAI Responses API — `web_search_preview` in tools[]

OpenAI Responses **does accept** a tool entry with type `web_search_preview`.

**Request shape:**
```json
{
    "tools": [
        {
            "type": "web_search_preview",
            "search_context_size": "medium",
            "user_location": { "type": "approximate", "approximate": {...} }
        }
    ]
}
```

Minimal form: `{"type": "web_search_preview"}`. `"web_search"` (without
`_preview`) is also accepted as an alias.

**Response shape — non-streaming:**
```json
{
    "output": [
        {
            "type": "web_search_call",
            "id": "ws_ab12",
            "status": "completed"
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "Here are the results...",
                "annotations": [{
                    "type": "url_citation",
                    "start_index": 0, "end_index": 10,
                    "url": "https://...", "title": "..."
                }]
            }]
        }
    ]
}
```

The `web_search_call` output item does **not** contain search results directly.
Results appear as `url_citation` annotations on the subsequent `output_text`.

**SSE events (streaming):**
- `response.web_search_call.in_progress` — `{ output_index, item_id }`
- `response.web_search_call.searching` — `{ output_index, item_id }`
- `response.web_search_call.completed` — `{ output_index, item_id }`

Defined at `litellm/types/llms/openai.py:1638-1653`.

**LiteLLM translation** (`litellm/llms/anthropic/experimental_pass_through/
responses_adapters/transformation.py:186-188`): Anthropic `web_search_*` tool
→ `{"type": "web_search_preview"}` at line 188.

**Reverse direction** (`litellm/responses/litellm_completion_transformation/
transformation.py:1252-1279`): `web_search_preview` / `web_search` tools
→ `web_search_options` on Chat Completions body.

**Implication for llmproxy:**
- `anthropic_to_responses_request()` in `src/conversion/responses.rs` catches
  `is_web_search_tool(t)` → maps to a new `ResponsesTool::WebSearch` variant.
  The `type` field becomes `"web_search_preview"`, not `"web_search_20250305"`.
- `tool_choice {type: "tool", name: "web_search"}` must be mapped to
  `auto` (OpenAI Responses has no function tool named "web_search").

### 8.3 LiteLLM interception handler for Responses API

`litellm/integrations/websearch_interception/handler.py:300-318`
(`_convert_responses_tools`): converts `web_search_preview` to
`{"type": "function", "name": "litellm_web_search"}` when interception is
enabled (for providers that don't natively support web search).

`litellm/integrations/websearch_interception/tools.py:85-113`
(`get_litellm_web_search_tool_responses`): Responses-API flat function tool
shape.

### 8.4 Cost tracking

- OpenAI Chat: `litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py
  :315-358` (`response_object_includes_web_search_call`) — detects `url_citation`.
- OpenAI Responses: same file detects `web_search_call` in `output[]`.
- Anthropic: `litellm/llms/anthropic/cost_calculation.py:116-129`
  (`get_anthropic_web_search_requests_from_response`) — reads
  `usage.server_tool_use.web_search_requests`.

### 8.5 Summary table

| Aspect | Chat Completions | Responses API |
|---|---|---|
| Request declaration | `web_search_options` top-level param | `tools[]` with `{"type": "web_search_preview"}` |
| Config fields | `search_context_size`, `user_location` in the param object | Same fields at tool level |
| Response marker | `annotations[]` with `url_citation` on text blocks | `output[]` item `{"type": "web_search_call"}` |
| Streaming events | N/A | `in_progress` / `searching` / `completed` |
| Tool_choice handling | `{type: "tool", name: "web_search"}` → `auto` | Same |

## 9. Review results (added 2026-07-30)

The following review was performed by an Opus-level agent after examining the
plan and cross-referencing every cited source file. Corrections have been
applied inline throughout the document above; this section records the
findings for completeness.

### Q1. Plan assessment

Research quality is high (Claude Code v2.1.88 source-map forensics, SDK
field-level citations, LiteLLM comparison). The strategic direction is correct:
forward + translate only, no intercept.

**Fatal flaw in the original draft:** Gap A was misdiagnosed. `Tool.input_schema`
is required (`pub input_schema: Value` with no `#[serde(default)]`). Claude Code's
hosted tool has no `input_schema` → serde returns `missing field` → `AppJson`
returns 400. The fix in §5.1 originally kept `input_schema` required, which
would still 400. Corrected to `#[serde(default, skip_serializing_if = "Value::is_null")]`.

**Two under-specified items:**
1. §5.4 (streaming usage counter) — `ResponsesUsage` has no web_search
   counter; `OutputItem::Unknown` drops all web_search_call fields; "if the
   upstream exposes it" never holds → no-op. Moved out of Phase 1.
2. §5.2 originally used `target_is_anthropic_passthrough` which is misleading:
   the passthrough provider never calls the converter. Corrected to explicit
   provider branches.

### Q2. Alternative slicing

Slice by severity, not lumped into one P0:

- **Phase 1 (minimal PR)**: only `src/anthropic.rs` `Tool` struct change. The
  Anthropic-passthrough provider (`build_body` serde-roundtrips the request)
  starts working end-to-end. Add an extractor-level integration test: POST
  Claude Code raw body → 200, upstream receives Value-equivalent tools. This
  fixes the "entry 400" hard failure.
- **Phase 2**: ResponsesTool `WebSearch` variant + tool_choice remapping for
  Copilot/OpenAI. See §5.2 and §8.
- **C and C-stream (multi-turn history + usage counters)** demoted to P1:
  current behavior is "silent loss, no crash"; OpenAI wire shapes are now
  documented in §8 but the response-to-Anthropic conversion is not yet
  designed.

### Q3. Risks

- `kind: Option<String>` (serde): no breakage. Old request without `type` →
  `kind: None`, `skip_serializing_if` omits it. Roundtrip byte-identical
  (verified in isolation).
- `extra: HashMap flatten`: no conflict. Same pattern used on
  `MessagesRequest` top level. Cannot combine with `deny_unknown_fields`
  (project doesn't use it). Explicit `"input_schema": null` is elided on
  roundtrip (harmless).
- **Content extractor empty string** (§5.3 original): `filter_map(get("text"))`
  on `web_search_result` items that have only `title/url/encrypted_content`
  returns empty string. This section is now demoted to P1 and the code is
  marked for fix when implemented.
- **`is_web_search_tool` false positive**: the original `t.name == "web_search"
  && t.kind.is_some()` would match `{name: "web_search", type: "function"}`.
  The current §5.2 only matches on `kind` string prefix, which avoids this.
- **`srvtoolu_` prefix**: OpenAI has no documented constraint on `call_id`
  format. Real risk: a `FunctionCall {name: "web_search"}` in multi-turn
  history could conflict with the upstream's built-in web_search tool. Needs
  real upstream verification.

### Q4. Test coverage

- `messages_request_roundtrips_every_documented_field`: must use Claude Code's
  exact wire shape (no `input_schema`, no `description`, with `max_uses`).
  Adding `input_schema` to make serde happy would produce a green test that
  doesn't guard the actual production path.
- `complete_passes_through_every_field_unmodified`: compatible with existing
  `json!()` matchers. `body_partial_json` (wiremock) requires both matcher and
  fixture to be updated together.
- `response_with_unknown_output_item_is_ignored`: not touched in Phase 1/2.
- Missing must-add tests:
  (a) extractor/server-level integration test (the 400 happens at this layer,
      not in any unit test)
  (b) `is_web_search_tool` truth table: `kind=web_search_20250305` → true,
      `kind=web_search_preview` → true, `kind=function` → false, `kind=None` → false
  (c) `Tool` serde boundary: kind None / known / unknown future type, missing
      `input_schema` still parses
  (d) `ResponsesTool::WebSearch` serialization shape (Phase 2)
  (e) tool_choice remapping (Phase 2)
  (f) wiremock end-to-end: upstream returns `web_search_call` / `server_tool_use`

### Q5. Summary

Strategy approved. Original implementation plan rejected. Corrective edits
applied inline. Phase 1 (Tool struct fix + extractor test) is ready for code.
Phase 2 (OpenAI mapping) has wire-shape research complete but implementation
depends on test (d) and (e) being written first.

## 10. Second review (2026-07-30)

A second Opus-level review was performed after Phase 1+2 wire-shape research
landed. This review (a) re-reads the corrected plan against the actual
`src/` code, (b) verifies the OpenAI wire-format claims via independent
documentation (a Microsoft Foundry doc note plus a `inspect_ai` PR
confirming `web_search_preview` model support list), and (c) re-examines
the risk register. Verdict: strategy is sound, but three Phase 2 details
need tightening before code lands, and §9's demoted-P1 items need an
explicit user-facing behavior paragraph.

**Cross-cutting evidence of Gap A**: while running this review, an
`OpenAI-compatible` upstream-side check rejected every tool-using search
request with the literal error `tools[0]: missing field 'input_schema' at
line 1 column 470` — i.e. the harness's own tooling reproduces the exact
400 the plan diagnoses. The fix in §5.1 is necessary, not optional.

### Q1. Plan assessment

Phase 1 is the correct minimal first cut. The corrected Tool struct
(`#[serde(default, skip_serializing_if = "Value::is_null")] input_schema`,
`kind: Option<String>` with `rename = "type"`, `extra: HashMap` flatten) is
the smallest surface that unblocks the Anthropic-passthrough provider via
`build_body`'s serde roundtrip — verified by reading `src/anthropic.rs:231-237`
and `src/providers/anthropic.rs:255-268`. The three originally fatal flaws
(input_schema required, no kind discriminator, extras dropped) are all fixed
in the §5.1 snippet. The plan no longer keeps `input_schema` required (the
original first-draft bug, fixed during the first review).

Phase 2 has two under-specified items and one missing consideration:

1. **Field mapping for `user_location` and search hints is under-specified.**
   §5.2 lists the `ResponsesTool::WebSearch` variant with
   `search_context_size, user_location` fields, but the converter body that
   fills them from Anthropic's tool fields is not specified. Anthropic's
   hosted-tool schema carries `user_location` (per
   `types/web_search_tool_20250305_param.py`, confirmed in §2.3), which maps
   1:1 to OpenAI's. `allowed_domains`, `blocked_domains`, `max_uses` have no
   OpenAI equivalent — drop them, but document the loss explicitly in §5.2.
   For Chat Completions, the `extra["web_search_options"] = {}` injection
   leaves the OpenAI-side defaults (search_context_size = medium, no
   user_location) intact, which is acceptable.

2. **Copilot Responses endpoint is not documented to support
   `web_search_preview`.** Verified via search: Copilot's `/v1/responses`
   endpoint is implemented, but `web_search_preview` support on Copilot's
   backend is undocumented and uncertain. An `inspect_ai` PR
   (`UKGovernmentBEIS/inspect_ai#1957`) confirms that web_search_preview is
   currently limited to a small explicit model allowlist (`gpt-4o`,
   `gpt-4o-mini`) at the Responses API; sending it to unsupported models
   returns 400. Copilot's backend may or may not honor the same allowlist.
   The Phase 2 PR should either (a) only route web_search tools to providers
   that explicitly advertise support via a config flag, or (b) accept that
   a 400 from Copilot will surface to the client and rely on
   `is_model_unsupported` heuristic + Claude Code's local MCP fallback (per
   §6 last bullet). Pick (b) as the P0 default; add a follow-up config flag
   for (a) if breakage is observed in production. This is the biggest
   unknown in Phase 2.

3. **`convert_tool_choice` lives in two places**, both must be updated.
   `src/conversion/request.rs:360-370` (Chat) and
   `src/conversion/responses.rs:327-337` (Responses). The Phase 2 snippet
   says "Chat Completions / Responses" but only references one location
   indirectly. Add both file:line citations to §5.2; the Copilot provider
   delegates to both functions (`src/providers/copilot.rs:716, 747, 828,
   876`), so a partial fix leaves Copilot broken.

The demotion of §5.3 (multi-turn history) and §5.4 (usage counter) to P1
is reasonable: Claude Code's WebSearchTool issues a separate single-turn
side query (§1.3, `extraToolSchemas` + single user message) so the main
conversation rarely carries server_tool_use blocks. Silent loss in the
OpenAI family means citations disappear (verified by grepping
`src/conversion/responses_stream.rs` for `annotations: None` — there are
12 such hardcodes, plus `src/conversion/responses.rs:74` and
`src/conversion/stream.rs:131` for `server_tool_use: None`) and
`usage.server_tool_use` is not propagated. P1 classification is acceptable.
However the plan must add an explicit "Phase 2 user-visible behavior"
paragraph stating that OpenAI-family upstreams will return search results
baked into text but without `server_tool_use`/`web_search_tool_result`
blocks and without `url_citation` annotations — the user gets answers but
Claude Code's "Did N searches" UI line stays blank and the Citations panel
is empty. That paragraph is missing today; add it to §5.2.

### Q2. Alternative slicing

**Raw JSON passthrough on `/v1/messages`** is unnecessary. The plan's
struct-with-flatten approach already preserves arbitrary nested fields:
`MessagesRequest` has `extra: HashMap<String, Value>` flatten at
`src/anthropic.rs:58`, with a roundtrip test at line 770. Phase 1 extends
the same pattern to `Tool`. The only loss-vs-passthrough gap is at deeper
nesting inside `ContentBlock` variants (e.g., unknown fields inside a
`ToolResult.content`), which is out of scope for web_search and would
explode the change surface.

A raw-bytes alternative would still carry one real benefit: future-proof
against any new top-level Anthropic field landing before the proxy
updates. Recommend NOT doing it now; revisit only if a concrete gap is
observed in production.

**Provider-level web_search capability declaration** is feasible but
premature. A `providers[].capabilities: {web_search: forward|strip|error}`
flag would let operators opt in per upstream and would protect
non-supporting OpenAI-compat providers (DeepSeek, MiniMax, OpenRouter
free tiers) from receiving `web_search_options` they don't understand.
Most OpenAI-compat providers tolerate unknown JSON keys, so the practical
risk is low. Apply the diagnose-first principle from
`feedback_diagnose_before_fixing.md`: ship Phase 1+2, watch for upstream
4xx logs with `web_search` substrings, then add the flag in a follow-up
if needed. Don't block Phase 2 on it.

### Q3. Risks

The serde design has one real risk and several non-issues:

- `Tool::kind: Option<String>` with `#[serde(rename = "type")]` does NOT
  collide with any adjacent enum tag. Verified: `Tool` is a struct (not an
  internally-tagged enum), `ThinkingConfig` already uses the same
  `rename = "type"` pattern at `src/anthropic.rs:257`, and
  `CacheControlEphemeral` at line 271. There is no struct-vs-enum
  serde-tag collision risk. The only thing to watch is that `Tool` is
  consumed as `Vec<Tool>` inside `MessagesRequest.tools`; the surrounding
  `tools` field has no `tag` attribute. Safe.

- `extra: HashMap flatten` with
  `skip_serializing_if = "HashMap::is_empty"` works as expected on
  roundtrip. The flattened map captures unknown fields on deserialize, and
  `skip_serializing_if` skips emission when the map is empty. Verified
  pattern in `MessagesRequest.extra` (line 58) which has identical attrs.
  The only behavior change vs. today: function tools with stray unknown
  top-level fields (rare in practice) will now be forwarded to upstreams
  via the Anthropic-passthrough path. On Chat Completions / Responses the
  converter builds the request explicitly so extras don't leak. Acceptable.

- `input_schema: Value` with
  `#[serde(default, skip_serializing_if = "Value::is_null")]` on roundtrip:
  an explicit `"input_schema": null` from a non-conforming client
  deserializes as `Value::Null` and is elided on serialization. Byte-level
  not identical (Null inserted vs absent); structural equivalence holds.
  Note that the plan's §5.1 says "byte-identical" — this is technically
  wrong because (a) `serde_json` is configured without `preserve_order`
  (`Cargo.toml:15` shows `serde_json = "1"` with no features), so default
  serialization sorts keys alphabetically via `BTreeMap`, and (b) null
  elision changes the body. Tests should assert structural Value equality
  via `assert_eq!(serde_json::to_value(&parsed)?, serde_json::to_value(&req)?)`,
  not string equality.

- `is_web_search_tool`'s `type_str == "web_search"` branch is effectively
  dead code for matching hosted Anthropic tools. The proxy's inbound
  protocol is Anthropic Messages, and Anthropic's hosted-tool type is
  always the versioned `web_search_20250305` (per §2.3). The bare
  `"web_search"` name is the Azure OpenAI Responses-API stable alias for
  `web_search_preview` (Microsoft Foundry docs: "use web_search tool in
  Azure OpenAI Responses API; web_search_preview is supported but not
  recommended"). On the standard OpenAI Responses API, the canonical name
  is `web_search_preview`. The bare alias appears in Azure Responses but
  never on the inbound side of an Anthropic-protocol request. Keep the
  branch for defensive robustness but add a one-line comment explaining
  its rationale (matches Azure Responses, non-conforming clients, future
  OpenAI naming changes).

- Copilot tool_choice → `auto` after the web_search tool is stripped. When
  Claude Code sends `toolChoice: {type: 'tool', name: 'web_search'}` on a
  forced-search turn (§1.3 `useHaiku` path), the user prompt is single-
  message `'Perform a web search for the query: ' + query`. With Chat
  Completions and `web_search_options: {}`, the model still almost
  certainly searches because the prompt explicitly instructs it; the
  forced semantics degrade from "guaranteed" to "highly probable". For
  Responses API, `tool_choice: auto` similarly degrades. Verify
  empirically — if `auto` causes Copilot/DeepSeek to skip search on
  marginal queries, the P1 follow-up should consider forcing via a
  tool_choice override `{type: "web_search_preview"}` where supported.
  Document this semantics change explicitly in §5.2 so reviewers know it
  is intentional, not a regression.

- **New risk identified this review:** `web_search_options` in `extra` is
  sent to ALL Chat-Completions-family providers, not just OpenAI.
  DeepSeek, MiniMax, OpenRouter free-tier, and GLM may 400 on unknown
  parameters (most tolerate them, but strict OpenAI-compat clones do
  exist). Phase 2 sends `web_search_options` to every provider whose
  config maps to a Chat-Completions client. The plan should explicitly
  call this out in §5.2 ("Phase 2 sends web_search_options to every
  Chat-Completions provider; observe logs in production and add an
  opt-out flag if strict upstreams 400"). Follow-up issue, not P0 blocker.

### Q4. Test coverage

§5.1's three tests plus §9's (a)–(f) cover most of the surface but miss
three things:

1. **`is_web_search_tool` truth table is incomplete.** §9 (b) lists four
   cases but should also include:
   - `name="web_search", kind=None` (a coincidental user function tool with
     the same name) → must return false.
   - `name="web_search", kind="function"` (OpenAI-style function tool with
     this name) → must return false. This is the exact bug the original
     first-draft `is_web_search_tool` had.

2. **`convert_tool_choice` remapping is untested** — and this is the
   highest-risk gap. LiteLLM (handler.py:373-388) explicitly handles
   `{type: "tool", name: "web_search"}` remap; the plan must add the
   equivalent test. Required cases:
   - `tool_choice = Tool{name: "web_search"}` + tools contains web_search
     entry → remap to `"auto"` on Chat and Responses paths (assert
     both `convert_tool_choice` functions independently).
   - `tool_choice = Tool{name: "get_weather"}` + tools contains a
     `get_weather` function + a web_search hosted tool → stays as
     `{type: "function", name: "get_weather"}` (don't over-remap).
   - `tool_choice = Auto` + tools contains web_search → stays `"auto"`
     (idempotent, no double-rewrite).
   - `tool_choice = Tool{name: "web_search"}` + tools does NOT contain
     web_search → passthrough unchanged (don't mangle unrelated function
     choices; this edge case is unspecified in §5.2 — recommend
     "passthrough").

3. **Phase 2 wiremock end-to-end is one-sided.** §9 (f) covers response
   translation of `web_search_call`, but that's P1 (demoted). For Phase 2
   itself, the request-direction e2e is needed:
   - wiremock asserts that a request with Claude Code's exact tools[]
     (mixed: function + web_search) produces an OpenAI Chat body with
     `web_search_options: {}` AND tools[] containing only the function
     tool.
   - Same for Responses API path → tools[] contains `{type:
     "web_search_preview"}` AND the function tool, with `tool_choice`
     remapped to `"auto"`.
   - Same for mixed: function + web_search_20250305 + a hypothetical
     `web_fetch_20250910` hosted tool (forward-compatible: future server
     tools get the same treatment — note that §5.2 currently only matches
     `web_search_*`, so `web_fetch` would fall through and get wrapped as
     a function tool; either widen the predicate or document that
     web_fetch is not handled in Phase 2).

Additional scenarios worth one assertion each:
- A request with only a hosted tool (no functions) → tools[] is empty,
  web_search_options / web_search_preview still injected.
- A request with only function tools → no injection (regression guard).
- `MessagesRequest` roundtrip with the corrected Tool struct: assert
  `extra` preserves `max_uses`, `allowed_domains`, `blocked_domains`,
  `user_location` Value-equivalent (not string-equivalent — see Q3).

### Q5. Summary and next steps

Strategy is approved. Phase 1 is ready for code; Phase 2 needs the
following tightening before implementation begins:

1. **Specify `user_location` mapping** from Anthropic's hosted-tool fields
   to OpenAI's `user_location` and `search_context_size` defaults in §5.2.
   Document the loss of `allowed_domains` / `blocked_domains` / `max_uses`.

2. **Add an explicit "Phase 2 user-visible behavior" paragraph** to §5.2
   stating that OpenAI-family upstreams return search results baked into
   text, without `server_tool_use` / `web_search_tool_result` blocks and
   without `url_citation` annotations — Claude Code's "Did N searches"
   UI line will not appear and the Citations panel will be empty. The
   Anthropic-passthrough provider is fully featured (those blocks flow
   through byte-identical via `build_body`).

3. **Add `convert_tool_choice` updates** to both
   `src/conversion/request.rs:360-370` and `src/conversion/responses.rs:327-337`
   to the Phase 2 scope citations; add the four-case test matrix above.
   Without both, Copilot GPT-5 (which routes through both converters at
   `src/providers/copilot.rs:716, 747, 828, 876`) is half-broken.

4. **Correct §5.1 "byte-identical" wording** to "structural Value
   equivalence". `serde_json` is configured without `preserve_order`
   (`Cargo.toml:15`), so default serialization sorts keys via `BTreeMap`;
   plus `skip_serializing_if = "Value::is_null"` elides explicit nulls.
   Tests must compare `serde_json::to_value(...)` output, not raw
   strings.

5. **Add a defensive-code comment** on the `type_str == "web_search"`
   branch in `is_web_search_tool` explaining its rationale (matches
   Azure Responses-API stable alias; harmless defensive check).

6. **Add the per-provider `web_search` opt-out flag** as a follow-up
   issue (P2) rather than blocking Phase 2. Diagnose-first — ship and
   observe.

7. **Verify Copilot's Responses endpoint accepts `web_search_preview`**
   before merging Phase 2. If Copilot's backend rejects it (likely on
   non-`gpt-4o/mini` models per `inspect_ai#1957`), the Phase 2 PR needs
   either an explicit `web_search_preview` allowlist keyed on the
   resolved model or a fallback to Chat Completions + `web_search_options`
   for GPT-5 + web_search. This is the biggest unknown in Phase 2 and
   should be validated against a live Copilot account before code lands.

**Recommended PR sequencing:**

- **PR1 (Phase 1)**: Tool struct fix + the three §5.1 tests + the §9 (a)
  extractor-level integration test. Self-contained; lands immediately.

- **PR2 (Phase 2)**: tool split, `ResponsesTool::WebSearch` variant, both
  `convert_tool_choice` updates, all tests in §5.1 + §9 (b)–(e) + the
  request-direction e2e wiremock tests from Q4.3. Gated on item 7
  (Copilot Responses endpoint verification); if 7 fails, fall back to
  Chat Completions + `web_search_options` only.

- **PR3 (P1 follow-up)**: response-direction translation of
  `web_search_call` / SSE `response.web_search_call.*` events into
  Anthropic's `server_tool_use` + `web_search_tool_result` blocks; forward
  `usage.server_tool_use`; preserve multi-turn history replay. Ships
  before Claude Code's "Did N searches" UI is treated as correct.
---
## 11. Implementation review (2026-07-30)

Opus-level review of commits `6afe7ac` (PR1) and `720c406` (PR2+PR3),
cross-referenced against §5.1, §5.2, §8, §10 Q5 and the live tree on branch
`fix/hosted-tool-input-schema`. `cargo check --lib` passes on the committed
tree. Verdict: the implementation adheres to the plan closely and the
request-direction logic is correct across all four `tool_choice` cases; but
PR2+PR3 shipped with zero new unit tests (the §10 Q5 gating was not honored),
one undocumented semantic addition (`max_uses` -> `search_context_size`), and
two §10 Q5 items silently skipped.

### 1. Strict adherence to plan

- **Tool struct (§5.1 incl. §10 revision): exact match.** `src/anthropic.rs:233-252`
  — `kind: Option<String>` with `rename = "type"` + skip-if-none; `input_schema`
  `#[serde(default, skip_serializing_if = "Value::is_null")]`; `extra: HashMap`
  flatten + skip-if-empty. The §10 fatal-flaw correction (`input_schema` defaulted,
  not required) is implemented. Doc comments present.
- **`is_web_search_tool` (§5.2): verbatim copy** of the plan snippet at
  `src/conversion/util.rs:23-30`, all three conditions. Faithful.
- **Chat conversion (§5.2 / §8.1): matches.** Strip via `filter_map`
  (`request.rs:86-101`), inject `web_search_options: {}` into `extra`
  (`request.rs:133-140`).
- **Responses conversion (§5.2 / §8.2): matches.** `ResponsesTool::WebSearch` with
  `#[serde(rename = "web_search_preview")]` (`responses.rs:112-118`); the enum is
  internally tagged (`responses.rs:96`) so the wire shape is
  `{"type": "web_search_preview", ...}` as required. `user_location` mapped 1:1
  (`responses.rs:132`) per §10 Q5#1.
- **SSE events (§8.2): match.** Three variants
  `response.web_search_call.{in_progress,searching,completed}` with
  `{output_index, item_id}` (`responses.rs:313-329`), matching §8.2 lines 461-463.
- **`tool_choice` remap "both" (§10 Q5#3): done in BOTH** `request.rs:104-116` and
  `responses.rs:164-176`. Copilot routes through both, so coverage is correct.
- **`OutputItem::WebSearchCall` (§8.2 response): correct.** `responses.rs:203-207`,
  `rename = "web_search_call"`, placed before `#[serde(other)] Unknown`
  (`responses.rs:209`) so it deserializes specifically instead of falling into
  `Unknown`.

**Deviations:**

- **`max_uses` -> `search_context_size` heuristic (`responses.rs:122-131`) is NOT in
  the plan.** §10 Q5#1 explicitly said `max_uses` "has no OpenAI equivalent — drop
  it." The implementation instead buckets `max_uses` into low/medium/high
  (>=10 high, >=5 medium, else low). This conflates two different concepts
  (`max_uses` = number of searches; `search_context_size` = richness of returned
  context). Net effect is harmless because Claude Code always sends `max_uses: 8`
  -> `"medium"`, which is OpenAI's default anyway — but it is an undocumented
  semantic invention (see §5).
- **`is_web_search_tool`'s `== "web_search_preview"` branch (`util.rs:27`) is
  redundant:** `"web_search_preview".starts_with("web_search_")` is already true.
  The plan snippet carried the same redundancy, so the impl faithfully copied it;
  not a bug, just dead-ish code.

### 2. Extra issues introduced

- **`is_web_search_tool` false positives: none.** The predicate keys off `kind`
  only, so `{name: "web_search", type: "function"}` (`kind = "function"`) -> false,
  and `{name: "web_search", kind: None}` -> false. The exact first-draft bug
  (matching on `name`) is avoided.
- **`web_search_options` broadcast to ALL Chat providers: confirmed unconditional**
  (`request.rs:136-140`). DeepSeek / MiniMax / GLM / strict OpenAI-compat clones
  receive it. This is the headline production risk, but it is per-plan (§10 Q3
  "new risk", diagnose-first; followup G). Not a regression vs plan.
- **`web_search_preview` broadcast to ALL Responses providers incl. Copilot:
  confirmed unconditional** (`responses.rs:110-133`). Copilot support is
  undocumented (§10 Q1.2; followup H). The §10 Q5#7 "verify Copilot before code
  lands" gate was skipped. Biggest unknown; acknowledged in the followup but not
  yet validated.
- **`OutputItem::WebSearchCall` silently dropped** (`responses.rs:459`;
  `responses_stream.rs:148, 420`): consistent with the P1 demotion (§9 / §10). The
  drop is per-plan; however the §10 Q5#2 "Phase 2 user-visible behavior" paragraph
  that was supposed to document this degradation in §5.2 was never added (still
  followup K), so the regression is currently undocumented in the plan.
- **`tool_choice` edge — tools WITHOUT web_search but
  `tool_choice = {tool, web_search}`:** `has_web_search = false` -> falls through to
  `convert_tool_choice`, emitting function-form `{"name": "web_search"}` unchanged
  (`request.rs:111-115`, `responses.rs:172-175`). Correct per §10 Q4 case 4
  (passthrough; the upstream 400s but we don't mangle). Untested.
- **Duplicate SSE variants: resolved.** Three distinct variants + distinct renames;
  the match at `responses_stream.rs:256-258` handles all three as no-ops;
  `#[serde(other)]` at `responses.rs:344` catches any future unknown event;
  `cargo check` passes. No compile error in the committed tree.
- **NEW: Chat only-hosted-tool emits an empty `tools` array.** When the only tool
  is web_search, `filter_map` yields an empty `Vec`, so `tools = Some([])` and
  serializes as `"tools": []` (`openai.rs:49-50` skips only on `None`, not on
  empty). LiteLLM omits the field. A strict upstream could 400 on an empty tools
  array. Not in the followup.
- **NEW (dormant): `user_location` is cloned verbatim** (`responses.rs:132`)
  assuming Anthropic <-> OpenAI shape compatibility, which is unverified. Dormant
  because Claude Code's schema (§1.3) sends no `user_location`.

### 3. Test coverage

PR1 added tests; **PR2+PR3 added none.** Against §10 Q4:

- **`is_web_search_tool` truth table (6 cases): ABSENT.** `util.rs:130` has a tests
  module but it covers the strict-schema helper, not the predicate. Followup A
  (P0). Confirmed.
- **`convert_tool_choice` 4-case matrix: ABSENT on both paths.** Existing
  `tool_choice` tests (`request.rs:535-583`; `responses.rs:606, 907, 926, 1418`)
  cover auto/any/tool/none but never the web_search remap. Followup B (P0).
  Confirmed.
- **Request-direction wiremock e2e (Chat + Responses): ABSENT.** The only web_search
  e2e is the Anthropic-passthrough test (`providers/anthropic.rs:473`, PR1). No
  `openai_compat.rs` / `openai_responses.rs` coverage. Followup C (P0). Confirmed.
- **`Tool::extra` roundtrip: PARTIAL.** `max_uses` is implicitly covered —
  `messages_request_roundtrips_every_documented_field` (`anthropic.rs:551`) feeds
  `{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}` and
  `assert_subset` (`anthropic.rs:493-511`) verifies those keys survive. But
  `allowed_domains` / `blocked_domains` / `user_location` are not in the fixture.
  Followup E. Partial.
- **Also missing:** `ResponsesTool::WebSearch` serialization-shape test (§9 (d));
  `OutputItem::WebSearchCall` deserialization test; an SSE no-op test for the three
  new events.

Net: the §10 Q5 statement "Phase 2 ... implementation depends on test (d) and (e)
being written first" was **not honored** — PR2 landed its production logic before
its tests. This is the single largest quality gap and the reason PR4 (followup)
exists.

### 4. Items omitted

- **§10 Q5#2 "Phase 2 user-visible behavior" paragraph in §5.2: NOT added**
  (followup K). The plan still does not document that OpenAI-family upstreams lose
  `server_tool_use` / `web_search_tool_result` / `url_citation`.
- **§10 Q5#5 defensive comment on the bare `"web_search"` branch: NOT added**
  (`util.rs:26`; followup L).
- **§10 Q5#7 Copilot Responses validation before merge: NOT done;** deferred to
  followup H as post-hoc observation. The pre-merge gate was skipped.
- **§9(a) / §10 Q4(a) extractor-level integration test: only PARTIALLY satisfied.**
  PR1's test (`providers/anthropic.rs:473`) builds `MessagesRequest` via
  `serde_json::from_value`, exercising the serde parse, but does NOT drive the axum
  `AppJson` extractor where the original 400 actually occurred. The HTTP-boundary
  regression guard is still missing.
- **§10 Q5#4 "byte-identical" -> "structural Value equivalence" wording: NOT
  corrected** in the plan doc (§5.1 line 271 still says "byte-identical"). Practice
  is fine (tests use `assert_subset` structural comparison); doc wording stale.
- **§10 Q5#1 "document the loss of `allowed_domains` / `blocked_domains` /
  `max_uses`": not documented;** instead `max_uses` was mapped (see §1 deviation),
  contradicting the plan's "drop it".
- **§10 Q5#3 (both `convert_tool_choice`) and #1 (`user_location` mapping): DONE** —
  these two items fully landed.

### 5. Recommended immediate fixes

**Into a PR1-PR3 fixup (small + safe):**

1. Add a server/extractor-level integration test that POSTs Claude Code's raw body
   (hosted tool, no `input_schema`) through the axum router and asserts 200 +
   upstream sees the tool. Guards the actual `AppJson` 400 boundary that the
   provider-level test bypasses (§9(a)). High value, ~40 LoC.
2. Collapse an empty Chat `tools` Vec to `None` (`request.rs:85-101`, e.g.
   `.filter(|v| !v.is_empty())` after `collect`) so the only-hosted-tool case omits
   `tools` instead of sending `"tools": []`. NEW finding; avoids a strict-upstream
   400.
3. Decide the `max_uses` -> `search_context_size` heuristic (`responses.rs:122-131`):
   either document it in §5.2 as an intentional mapping or revert to the plan's
   "drop `max_uses`". Cheap; recommend documenting since the net effect equals
   OpenAI's default. NEW finding / plan deviation.

**Already tracked in `web-search-followup.md` (leave there):**

4. A / B / C (predicate truth table, `tool_choice` matrix, Chat + Responses e2e) ->
   PR4, P0. This is the highest-risk gap.
5. K / L (user-visible-behavior paragraph, defensive comment) -> doc polish.
6. H (Copilot `web_search_preview` validation) -> elevate: the skipped §10 Q5#7
   pre-merge gate should be run before PR2 is declared stable.

**Correction to the followup doc itself:**

7. Item F lists three `server_tool_use: None` sites; there are **FOUR**. Add
   `src/conversion/responses.rs:513` (non-streaming Responses path). The cited
   `responses_stream.rs:362` is now line 370 after the PR3 reline. (`response.rs:74`
   and `stream.rs:131` are correct.)

None of the above are P0 blockers for the already-merged logic except the test gap
(#4), which PR4 already owns.

---
## 12. Final implementation review (post-fixup 871a49e)

Read-only Opus-level review of fixup `871a49e` on branch
`fix/hosted-tool-input-schema`, checked against §11's recommendation list,
the new §5.2.a paragraph, and `web-search-followup.md` item F. Verification
basis: `cargo test --lib` (469 passed) and `cargo test --test
integration_router` (18 passed, including the new test) green on the
committed tree; every site claim below is grep-verified against the tree.

### 1. Correctness of 871a49e changes

**(a) Extractor integration test — purpose met, but the wiremock fixture is
dead and the test-internal comment overclaims.**

The test (`tests/integration_router.rs:786-852`) does guard the original
failure: pre-PR1, serde's `missing field 'input_schema'` surfaced as
`ProxyError::BadRequest` → 400 via `src/extractor.rs:60-68`, and
`assert_ne!(status, BAD_REQUEST)` would have failed. That is exactly the
layer PR1's provider-level test (`src/providers/anthropic.rs:473`) bypassed
(it parses `MessagesRequest` in-process), so this closes a real hole.

However:

- **It cannot distinguish an AppJson 400 from an upstream-produced 400.**
  `src/error.rs:79` maps `ProxyError::Upstream { status, .. }` to the same
  HTTP status 1:1, so a 400 originating upstream would fail the assertion
  even with a healthy AppJson. Today's fixture masks this: the wiremock
  `Mock` (path `/v1/chat/completions`, matcher
  `body_partial_json({web_search_options: {}, model: "gpt-4o"})`) **never
  matches**, because `build_axum_app` (line 681) routes through the
  in-memory `WiremockOpenAiProvider` (lines 76-126), which builds its own
  body — ignoring `req.tools` entirely — and POSTs to `/v1/messages` (line
  98), not `/v1/chat/completions`. Unmatched route → wiremock 404 → router
  surfaces the last `Upstream` error → client gets 404 ≠ 400 → the test
  passes.
- **NEW FINDING (test quality): the `body_partial_json` assertion is
  decorative.** The real `OpenAiCompatProvider` /
  `anthropic_to_openai_request` never execute in this test; the
  `ProviderConfig::OpenaiCompat` entry in `configs` is never instantiated
  (`Router::new` receives the providers map directly). There is no
  `.expect(1)` and no `verify()`, so wiremock silently ignores the
  unmatched mock. The test-internal comment at lines 796-799 ("Body is
  asserted to be a superset via body_partial_json; the OpenAI-compat
  converter injects `web_search_options` and strips the hosted tool") is
  factually false, and the matcher's `model: "gpt-4o"` even contradicts the
  fixture's `model: "claude-test"` (an empty `model_rewrite` would leave it
  unchanged) — confirming the mock was never designed to match. Fix: swap
  the wiremock fixture for an in-memory provider returning `Ok` and assert
  200 + response body, or drop the fixture and let followup C own the
  wire-shape assertions. §11 fix #1 asked for "assert 200 + upstream sees
  the tool" — only the extractor-boundary half is satisfied.

**(b) Empty `tools` → `None` (`src/conversion/request.rs:84-106`) —
correct, with two notes.**

Logic is right and all tests stay green. Edge cases: `None` → `None`;
hosted-only → `None` (the intended fix; matches LiteLLM); mixed →
function-only (unchanged); `Some([])` → `None` — a second behavior change
beyond the commit message's "only-hosted-tool" framing (a client-sent empty
array previously emitted `"tools": []`); benign and strictly more
compatible.

- **Inconsistency:** the Responses path (`src/conversion/responses.rs:92`)
  still uses `.map`, so a client-sent `tools: []` still emits
  `"tools": []` on the Responses API. The hosted-only case is unaffected
  there (a `WebSearch` entry is always pushed, keeping the array
  non-empty). Low priority.
- **NEW FINDING (untested edge, medium risk): `tool_choice: "auto"` now
  travels without `tools`.** On Claude Code's forced-search turn (§1.3
  `useHaiku` path: `tool_choice: {type: "tool", name: "web_search"}` plus
  a hosted-only tools array), the remap at `request.rs:113-116` emits
  `Some(json!("auto"))` while `tools` now collapses to `None`. Strict
  OpenAI endpoints reject `tool_choice` when no tools are enabled
  (documented validation: `Invalid parameter: 'tool_choice' is only
  supported when 'tools' are enabled`; whether `"auto"` alongside
  `web_search_options` triggers it needs live verification). Pre-fixup the
  same request sent `tools: []`, which a strict upstream may also have
  rejected — so this is not a regression, but it is an open edge. The
  natural completion: also collapse `tool_choice` to `None` when `tools`
  collapsed (`web_search_options` alone enables searching; `"auto"` is
  redundant). Belongs in followup B as a fifth case.

**(c) Heuristic doc comment (`src/conversion/responses.rs:122-132`) —
semantics accurate, cross-reference wrong.**

The `max_uses` (search-count cap) vs `search_context_size` (context-window
tier; OpenAI documents low/medium/high as a latency-vs-tokens tradeoff, so
"token budget" is fair) distinction is accurate. This satisfies §11 fix #3
(document rather than revert); net effect equals OpenAI's default (Claude
Code's `max_uses: 8` → `"medium"`).

- **NEW FINDING (doc bug): the comment ends "PR4 / web-search-followup.md
  item F may replace it with explicit `None`" — both references are
  wrong.** PR4 is the tests PR (A+B+C), and item F is about
  `usage.server_tool_use` forwarding, unrelated to `search_context_size`.
  No followup item actually tracks the drop-the-heuristic decision. Either
  point at §11 item 3 or add a dedicated item.

**(d) Defensive comment (`src/conversion/util.rs:25-30`) — accurate.**

Bare `web_search` is the Azure OpenAI Responses-API stable alias (Microsoft
Learn documents `web_search` as GA and `web_search_preview` as "supported
but not recommended"); Anthropic inbound traffic is always the versioned
`web_search_20250305` (§2.3); the branch is harmless. Satisfies §10 Q5#5
and followup L. (The redundant third condition `== "web_search_preview"`,
already noted in §11, remains — unchanged, cosmetic.)

### 2. §5.2.a accuracy

Checked paragraph-by-paragraph against the tree:

**Anthropic-passthrough paragraph — accurate except one claim.**
`serde_json::to_value(req)` at `src/providers/anthropic.rs:260` (modulo
model rewrite + stream flag — fair shorthand), response returned as raw
JSON (`ProviderOutput::Json`, line 206), SSE passed through. **Wrong:**
"Claude Code's 'Did N searches' UI line and the Citations panel work" —
§1.3 of this same document establishes that Claude Code v2.1.88 renders
**no** citations panel (terminal UI, one line; "unlike Claude Desktop").
The blocks do flow back unchanged, so the claim holds for Claude Desktop /
raw-API consumers but not for Claude Code.

**Chat Completions paragraph — accurate, three omissions.**
`web_search_options: {}` injection (`request.rs:139-143`) ✓; `citations:
None` at `response.rs:41` and `stream.rs:167` ✓ (plus an uncited third site
at `openai_compat.rs:730`). Precision note: `src/openai.rs` has **no**
`annotations` field on its chat message type at all, so upstream
annotations are dropped at deserialization — the cited `citations: None`
sites are where they would have been reconstructed. Omissions: (i) the Chat
path drops `max_uses` / `allowed_domains` / `blocked_domains` /
`user_location` wholesale (`web_search_options` is always `{}`) — §10 Q5#1
loss documentation was only done for the Responses path; (ii) the forced →
`auto` tool_choice degradation (§10 Q3 risk #5 explicitly asked for it in
§5.2); (iii) non-supporting Chat-family upstreams will likely *silently
ignore* `web_search_options` (an answer without searching, with no signal)
— §10 Q3 only warned about 400s.

**Responses API paragraph — request side accurate, the drop-site citation
is doubly wrong.** `ResponsesTool::WebSearch` injection
(`conversion/responses.rs:113-143`) ✓; `OutputItem::WebSearchCall` dropped
at `conversion/responses.rs:468` and early-returned in streaming at
`responses_stream.rs:145-148`; the three SSE events no-op at
`responses_stream.rs:256-258` ✓; "no citations panel, no search counter" ✓.
**Wrong:** "see `responses_stream.rs:468, 554, 755, 803, 813` hardcoded
`annotations: None`" — (1) the line numbers are stale (post-PR3 reline; the
twelve `annotations: None` occurrences today are at 482, 568, 769, 817,
827, 851, 921, 985, 1244, 1305, 1465, 1469); (2) **all twelve are inside
`mod tests` (which starts at `responses_stream.rs:427`) — they are test
fixtures, not the production drop.** The production drops are the `..` in
`ResponseOutputTextDone { output_index, text, .. }`
(`responses_stream.rs:186` — the annotations field *is* parsed, see
`src/responses.rs:219`, and discarded here) plus `citations: None` at
`conversion/responses.rs:442` (non-streaming) and `responses_stream.rs:399`
(block start).

The operational-implication paragraph and the PR5 pointer are accurate.

### 3. followup item F completeness

Exhaustive grep for `server_tool_use: None` across `src/`:

| Site | Kind |
|---|---|
| `src/conversion/responses_stream.rs:370` | production (Responses SSE) — listed in F ✓ |
| `src/conversion/responses.rs:522` | production (Responses non-streaming) — listed in F ✓ |
| `src/conversion/response.rs:74` | production (Chat non-streaming) — listed in F ✓ |
| `src/conversion/stream.rs:131` | production (Chat SSE) — listed in F ✓ |
| `src/anthropic.rs:736` | test fixture — correctly excluded by F ✓ |

The four-site list is exhaustive, so §11 fix #7 is honored. **However, F's
premise sentence is wrong for the Responses path**: "Today neither OpenAI
nor Anthropic-protocol upstream surfaces `web_search_requests` count via
this field." The OpenAI Responses API *does* surface
`usage.server_tool_use.web_search_requests`; the counter is lost because
`ResponsesUsage` (`src/responses.rs:158-169`) has neither that field nor a
flatten, so it is discarded at deserialization. F's "leave hardcoded `None`
until at least one upstream exposes the counter" therefore under-scopes the
work: add the field to `ResponsesUsage` and plumb it through at
`responses.rs:522` / `responses_stream.rs:370`. (Whether Chat Completions
`web_search_options` responses carry a `server_tool_use` usage block is
unverified here.)

Adjacent followup-doc staleness found while checking:

- **G (multi-turn)** cites only `conversion/responses.rs` with stale lines
  (284-285, 456-459; current: 285-286, 344-345, and the `WebSearchCall` arm
  at 468) and **omits the Chat-path drop sites entirely**:
  `conversion/request.rs:282-283` and `345-346` (both turn directions).
- **H** inherits §5.2.a's bad `annotations: None` citations
  (468/554/755/803/813) — same fixture-vs-production error.
- **No item** tracks the drop-the-heuristic decision that
  `responses.rs:130-132` claims is item F.

### 4. Remaining review gaps

Status of the §11 backlog (followup letters) on the committed tree:

| Item | Status | Evidence |
|---|---|---|
| A — `is_web_search_tool` truth table | **ABSENT** | `util.rs` tests (from line 137) still only cover `strictify_schema`; no predicate tests anywhere |
| B — `convert_tool_choice` four-case matrix | **ABSENT** | `request.rs:538`, `responses.rs:615, 916, 935, 1427` cover auto/any/tool/none only; no remap cases. Now more urgent: add a fifth case for hosted-only + forced choice (see 1b) |
| C — request-direction wiremock e2e (Chat + Responses) | **ABSENT** | no web_search tests in `openai_compat.rs` / `openai_responses.rs`; the new integration test does **not** cover this (its converter never runs) |
| D — forward `usage.server_tool_use` | unchanged | all four sites `None` (acceptable per F, but see the F premise correction) |
| E/G — multi-turn history replay | unchanged | arms still skip at `request.rs:282-283, 345-346`, `responses.rs:285-286, 344-345, 468` |
| §9(d)/§11 extras — `ResponsesTool::WebSearch` serialization shape, `WebSearchCall` deserialization, SSE no-op tests | **ABSENT** | — |
| §11 #1 — extractor-level test | **partial** | AppJson boundary guarded; "assert 200 + upstream sees the tool" not satisfied (see 1a) |
| §11 #2 — empty tools → None | done | `request.rs:105` |
| §11 #3 — heuristic decision | done (documented) | with a wrong cross-reference (1c) |
| §11 #6 / H — Copilot `web_search_preview` live verification | **NOT DONE** | the §10 Q5#7 pre-merge gate remains skipped |

**Land now:** PR4 (A+B+C) — the request-direction logic of PR2+PR3 still
has zero direct unit coverage; that was §11's single largest quality gap
and remains so. H (live Copilot check) before the branch is declared
stable. **Can wait:** D and G — graceful degradation, no crash paths, P1
per §9/§10.

### 5. Overall quality assessment

**Goal attainment.** The TL;DR goal ("forward + translate, no intercept")
is met for the P0 scope the plan finalized (§9/§10 demoted
response-direction translation to P1): the entry 400 is fixed and now
guarded at the layer where it actually fired; request-direction conversion
is correct across all three upstream families and at both
`convert_tool_choice` sites (Copilot's dual path at
`copilot.rs:716, 747, 828, 876` is covered). Response-direction translation
(citations, `server_tool_use` / `web_search_tool_result` blocks, usage
counters) remains lossy by design and is tracked.

**Risks, ranked:**

1. Copilot `/v1/responses` + `web_search_preview` remains unverified
   against a live account — possible client-visible 400s; the mitigation is
   `is_model_unsupported` + Claude Code's local MCP fallback (§6), per the
   plan's decision (b) in §10 Q1.2.
2. Zero direct unit coverage of the predicate / remap / wire shapes until
   PR4 lands.
3. NEW: strict-upstream 400 on `tool_choice: "auto"` + absent `tools` for
   forced-search turns (1b).
4. `web_search_options` broadcast unconditionally to all Chat-family
   providers — per-plan diagnose-first (§10 Q3; followup G).
5. Silent citation / counter loss on OpenAI-family upstreams — P1, now
   documented in §5.2.a.

**Verdict: push/merge-able.** No crash paths; failure modes degrade rather
than fail; tests green (469 lib / 18 integration) on the committed tree;
all four fixup items are net-positive. The new test's misleading comment +
dead fixture (1a) and the wrong cross-reference (1c) should be corrected in
or alongside PR4 — nits, not blockers.

### 6. Documentation completeness

§5.2.a edits needed:

- Replace the annotations citation with the production drop sites:
  `responses_stream.rs:186` (`text, ..` discards the parsed annotations —
  the field exists at `src/responses.rs:219`), `conversion/responses.rs:442`,
  and `responses_stream.rs:399`. The current five line numbers point into
  the test module and are stale.
- Delete "and the Citations panel work" (contradicts §1.3; Claude Code has
  no citations panel).
- Add: the Chat path drops `max_uses` / `allowed_domains` /
  `blocked_domains` / `user_location` (`web_search_options` is always `{}`).
- Add: a forced `tool_choice` degrades to `auto` on both OpenAI families
  (§10 Q3 risk #5 asked for this in §5.2).
- Optional: note that non-supporting Chat upstreams may silently ignore
  `web_search_options` (search-less answer, no signal).

Followup-doc edits needed:

- F: correct the premise (the Responses API exposes the counter;
  `ResponsesUsage` drops it at deserialization — the work is field +
  plumbing, not waiting).
- G: add the Chat-path sites (`request.rs:282-283, 345-346`); refresh the
  Responses line numbers (285-286, 344-345, 468).
- H: rewrite the annotations citations (same as §5.2.a above).
- Add an item tracking the drop-the-heuristic decision (currently
  mis-referenced as "item F" at `responses.rs:130-132`).
- B: add a fifth case (hosted-only + forced tool_choice → decide
  auto-vs-`None`; see 1b).

Leftover from §10 Q5#4: §5.1 still says "byte-identical" where it should
say "structural Value equivalence" — cosmetic, still uncorrected.
