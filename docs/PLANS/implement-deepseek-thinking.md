# 实施计划：修复 DeepSeek thinking 误判（strip-and-retry）

> 本文件为可直接执行的实施文档。**不动 router、不动 config、不新增模型名前缀判断**。所有修改收敛在 `AnthropicProvider::complete` 内，`stream` 沿用既有"流式一旦开流不重试"契约。

---

## 0. 关键事实速览

| 项 | 值 |
|---|---|
| 唯一的生产代码修改文件 | `src/providers/anthropic.rs` |
| 测试改动文件 | `src/providers/anthropic.rs`（`mod tests`）+ `tests/integration_router.rs` |
| 不动的文件（明确范围） | `src/router.rs`、`src/config.rs`、`src/error.rs`、`src/providers/mod.rs`、`src/conversion/stream.rs`、`config.example.yaml`、`README.md` |
| `ProviderConfig::Anthropic` 新增字段？ | **不加** |
| `has_thinking_error` 移走？ | 不移（保留为辅助），新增 `should_strip_and_retry(req, body)` 作 gate |
| `complete()` 重试上限 | `max_attempts = 2`（首次 + 一次 strip-retry），第二次仍 400 → 透传 raw body |
| `stream()` 重试？ | 不重试，保持"流式一旦开流不重试"契约，`thinking_not_supported_error` 原封不动 |
| 关键行号参考 | `anthropic.rs:193` 400+thinking 分支；`anthropic.rs:231` 同分支 stream 版；`anthropic.rs:251-253` `has_thinking_error`；`anthropic.rs:255-268` `build_body`；`router.rs:303-307` 流式不重试契约；`error.rs:85-92` `is_cooldownable` 只含 401/402/404/408/429/5xx |

> **校验步骤**：实施前必须 `grep -rn "ProviderConfig::Anthropic" src/ tests/`，得到 5 个构造点（`providers/mod.rs:99`、`providers/mod.rs:160`、`config.rs:151`、`config.rs:162`、`config.rs:557`），验证本次改动没有意外破坏任何一处。本计划不修改构造点形状（不删/不加字段），grep 通过即可。

---

## 1. 问题分析

### 1.1 触发场景

- 用户在 `config.yaml` 中为 `deepseek` provider 配置 **`type: anthropic`**（而非 `openai_compat`），使请求走 DeepSeek 的 Anthropic 兼容 Messages 端点 `https://api.deepseek.com/v1/messages`。
- 用户在 `config.yaml` 中假设 deepseek-v4-pro **支持 thinking**，加入 primary Claude 启用 thinking 的 fallback 链。
- 客户端首轮启用 thinking → 多轮后 history 中累积了 Anthropic 原生 `signature` 的 assistant thinking block。

### 1.2 错误链路

1. 客户端 → `AnthropicProvider::complete()`（`src/providers/anthropic.rs:174-207`）。
2. 代理把请求 verbatim 转发到 DeepSeek Anthropic 兼容端点。
3. DeepSeek 返回 400 + Anthropic 风格错误体：`{"error":{"message":"The \`content[].thinking\` in the thinking mode must be passed back to the API."}}`。
4. `has_thinking_error(body)`（`src/providers/anthropic.rs:251-253`）子串匹配两个固定标记 → 返回 `true`。
5. 当前实现 (`anthropic.rs:193`) → 调用 `thinking_not_supported_error()` → 客户端拿到 `400 thinking_not_supported` 友好错误（**但实际 deepseek-v4-pro 是支持 thinking 的**，误报）。

### 1.3 根因

`has_thinking_error()` 把 Anthropic 协议里**两类不同语义的错误折叠成一个判定**：

- **真·协议违规**：多轮对话中 assistant 历史 thinking block 缺 `signature` 或 `signature` 跨模型不匹配 → 客户端必须修复。
- **兼容端点的伪误报**：DeepSeek 的 Anthropic Messages 复刻对跨模型的 Anthropic 原生 signature 拒收（即拒绝 Anthropic 风格的 signature 同时报同款字面量错误）。

**区分两类所需的唯一额外信息是请求上下文**——`req.messages` 里是否存在 assistant + `type == "thinking"` block。当前实现抛掉了这个上下文；同样 body 的两种触发条件被强行归一化。

### 1.4 litellm 实证

参考 `litellm/llms/anthropic/common_utils.py:907-919` 的检测器 `is_anthropic_invalid_thinking_signature_error`，litellm 的字符串锚定是 **三连必须**：

```
"thinking" + "signature" + ("invalid" | "valid string")
```

DeepSeek 的实际报错 `content[].thinking must be passed back` **不含 `signature` 子串**，litellm 的自愈**当前不会触发**（commit `e59add11cd` 把检测收窄为 Bedrock/Vertex 专向）。因此"整条移植 litellm"在本场景命中不了。

但 `litellm/llms/deepseek/messages/transformation.py:113-130` 的 `DeepSeekAnthropicMessagesConfig.transform_anthropic_messages_request` 仅做 tools `custom` 清洗，thinking block 是按 Anthropic 原样透传的——确认 DeepSeek **不重签** Anthropic 原生 signature，必须由代理**剥离**后才能让深求返回 200。

### 1.5 为什么必须 strip-and-retry，不能纯透传

若改成"thinking 400 → 直接透传 raw body 为 `Upstream { status: 400 }`"：

- 客户端在响应里看到 400，但 deepseek-v4-pro 的 catalog 显示支持 thinking；
- 多轮对话中客户端**仍会**按 Anthropic 协议在下一轮把 assistant history 的 thinking block 带回来；
- 由于 client 是 thinking-aware（启用了 thinking），下一轮又会触发相同 400 → **死循环**。

正确路径：代理解一次 stripping，**把上游不兼容的字段从历史中抹掉**，让本轮拿到 200。这样客户端下一轮 history 中就不会再带回被剥掉的 thinking block，链路稳定。

### 1.6 已否决的替代方案（一句话）

- **新增 config 字段 `supports_thinking` / `thinking_supported`**：用 flag 掩盖根因 + 粒度粗到 provider 下所有模型 + 命名误导（"上游能否处理 thinking" 与事实不符）。已否决。
- **按模型名前缀（`deepseek-*` 跳过检测）**：硬编码进代码、新模型/新 provider 都要改代码、违反配置驱动原则。已否决。
- **router 层 thinking 能力预检**（`Provider::supports_thinking` trait 方法）：上游端点兼容声明不可静态判定，预检必失败。已否决。
- **纯透传 thinking 400**：如 §1.5 所述构成死循环。已否决。

---

## 2. 完整 Todo 列表

| Step | 文件 | 类型 | 内容 |
|---|---|---|---|
| 0（可选非阻塞） | `src/providers/anthropic.rs`（`complete()` 的 400 分支） | 行为不变 + 诊断 | 加 `tracing::warn!` 输出 `req_has_assistant_thinking` + 顶层 `thinking` 字段存在与否 + body 前 200 字符；生产灰度确认触发条件 |
| 1 | `src/providers/anthropic.rs`（`complete()` 实现） | 行为变更 | 新增 `should_strip_and_retry(req, body)` gate；新增 `max_attempts = 2` 循环 |
| 2 | `src/providers/anthropic.rs`（`complete()` 实现 + 同文件 tests 模块） | 行为变更 + 测试 | 写 `strip_thinking_blocks(body)` mutate 逻辑：过滤 messages[*].content 中 `thinking` / `redacted_thinking` block，整条 content 清空的消息跳过，置顶层 `body["thinking"] = Value::Null`；写 §4 单测 |
| 3 | `src/providers/anthropic.rs`（`stream()`） | 行为不变 | 显式注释：流式路径不重试保留友好错误（对应 `router.rs:303-307` 契约） |
| 4 | 无 | — | **无 config 改动**，`ProviderConfig::Anthropic` 不加任何字段 |
| 5 | `tests/integration_router.rs` | 行为变更 + 测试 | 新增 2 条 wiremock 集成测试覆盖端到端 |
| 6 | `src/providers/anthropic.rs`（tests 模块） | 改写/合并现有测试 | `complete_returns_friendly_error_on_thinking_mismatch` 与 `complete_friendly_error_uses_client_model_when_no_rewrite` 这两条因 complete 路径改成 strip-and-retry 需要改写；`stream_returns_friendly_error_on_thinking_mismatch` 保留不动 |

> 实施前先跑：`grep -rn "ProviderConfig::Anthropic" src/ tests/`。期望命中 5 处：(`src/providers/mod.rs:99`、`src/providers/mod.rs:160`、`src/config.rs:151`、`src/config.rs:162`、`src/config.rs:557`)。本计划**不改动其构造形状**，仅核对无新增误编译。

---

## 3. 代码变更细节

### 3.1 涉及锚点（精确行号）

| 位置 | 用途 |
|---|---|
| `src/providers/anthropic.rs:174-207` | `complete()` 实现（本计划主要修改面） |
| `src/providers/anthropic.rs:209-245` | `stream()` 实现（仅加注释，不重试） |
| `src/providers/anthropic.rs:248-253` | `has_thinking_error` 保留作辅助 |
| `src/providers/anthropic.rs:255-268` | `build_body` 保留不变，**不用改它** |
| `src/providers/anthropic.rs:783-864` | 现有 `complete_*_thinking_*` 测试，需要按 §4 改写/合并 |
| `src/providers/anthropic.rs:946-989` | 现有 `stream_returns_friendly_error_on_thinking_mismatch` 保留 |
| `src/providers/anthropic.rs:992-1023` | `thinking_mismatch_does_not_retry_when_thinking_absent` 已验证非-thinking 400 不受影响，保留 |
| `src/anthropic.rs:82-92` | `Message { role, content: MessageContent }`，`MessageContent::Text` / `Blocks` —— mutate 时要按此形状处理 |
| `src/anthropic.rs:94-200` | `ContentBlock` —— 仅 `Thinking` 与 `RedactedThinking` 两种变体需要被 strip |
| `src/router.rs:303-307` | 流式"一旦开流不重试"契约（保持不变，加注释引用） |
| `src/error.rs:85-92` | `is_cooldownable` 仍只覆盖 401/402/404/408/429/5xx，**400 透传后不进 cooldown，不触发 fallback**，这是当前已验证行为 |

### 3.2 完整修改：`complete()` 重写为带 strip 的一次重试循环

文件：`src/providers/anthropic.rs`
函数：`async fn complete()`（当前定义于行 174-207）

伪代码（替换现有 174-207 整段函数体，对外签名不变）：

```rust
async fn complete(
    &self,
    req: &MessagesRequest,
    model_rewrite: &HashMap<String, String>,
) -> Result<ProviderOutput> {
    let merged = self.merged_rewrite(model_rewrite);
    let mut body = build_body(req, &merged, false)?;
    let url = self.messages_url();
    let api_key = self.api_key.clone();

    let mut attempt: u32 = 0;
    let max_attempts: u32 = 2;
    loop {
        attempt += 1;
        let resp = self.http
            .post(&url)
            .bearer_auth(&api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if status.is_success() {
            let val: Value = serde_json::from_str(&text)?;
            return Ok(ProviderOutput::Json(val));
        }

        // Decide whether to strip thinking blocks and retry.
        let status_code = status.as_u16();
        let is_thinking_400 =
            status_code == 400 && has_thinking_error(&text);

        if is_thinking_400
            && attempt < max_attempts
            && should_strip_and_retry(req, &text)
        {
            // Mutate the OUTGOING body in place:
            //   - drop content blocks of type {thinking, redacted_thinking}
            //     from every assistant/user message;
            //   - skip entire messages whose content array becomes empty;
            //   - clear top-level thinking.
            strip_thinking_blocks(&mut body);
            continue;
        }

        // No retry path applies: surface the raw upstream body verbatim.
        return Err(ProxyError::Upstream {
            status: status_code,
            body: text,
        });
    }
}
```

精确逻辑说明：
- `attempt` 从 1 开始；循环条件为 `attempt < max_attempts` 才会进 strip 分支（保证**仅 strip 一次**）。
- 首次 400 + gate 命中 → mutate body → `continue` 再次进 `post()`。
- 第二次任何非 2xx（无论是否仍为 thinking 400）→ 走 `Upstream { status, body }` 透传，**不再重试**。
- 始终**不调用** `thinking_not_supported_error`（complete 路径里这个 helper 在本计划中彻底不再被调用）。这样上游的真实错误真正透传给客户端，不会被 friendly 文案再次掩盖。

### 3.3 新增辅助函数（放在 `has_thinking_error` 后面，`build_body` 前面）

```rust
/// Decide whether a 400 with a thinking-style body should trigger the
/// strip-and-retry path. The body alone cannot disambiguate "real
/// protocol violation" from "the upstream's compat layer dislikes the
/// cross-model signature"; the request context decides.
///
/// Both conditions must hold:
///   (a) upstream body matches the canonical thinking-error substring
///       pair (delegated to `has_thinking_error`), AND
///   (b) the incoming request already contains an assistant message
///       whose `content` includes a `Thinking` block (i.e. there is
///       something concrete to strip).
fn should_strip_and_retry(req: &MessagesRequest, body: &str) -> bool {
    if !has_thinking_error(body) {
        return false;
    }
    req.messages.iter().any(|m| {
        m.role == "assistant"
            && matches!(
                &m.content,
                MessageContent::Blocks(bs) if bs.iter().any(|b| matches!(
                    b,
                    ContentBlock::Thinking { .. }
                ))
            )
    })
}

/// Strip thinking-related content from the outgoing `body` JSON value
/// in place. After this call:
///   - every Thinking / RedactedThinking block has been dropped;
///   - if a message's `content` array is now empty (or contained only
///     thinking blocks), the message itself is removed;
///   - the top-level `thinking` key is set to `null`, regardless of
///     whether it existed.
fn strip_thinking_blocks(body: &mut Value) {
    if let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
        messages.retain_mut(|msg| {
            if let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                blocks.retain(|b| !matches!(
                    b.get("type").and_then(|t| t.as_str()),
                    Some("thinking") | Some("redacted_thinking")
                ));
                !blocks.is_empty() // keep only if there's still content
            } else {
                true
            }
        });
    }
    body["thinking"] = Value::Null;
}
```

要点：
- **顶层置 `null` 而不是移除 key**：对应 litellm `data.pop("thinking")` 的保守行为，避免破坏依赖 `"thinking" in body` 字段存在性的客户端 JSON 模式（且 Anthropic Messages 协议把 None 与 absent 都视为关 thinking —— 但本服务是给上游发，因此置空字符串/Null 语义明确即可）。这里采用 `null`：上游 DeepSeek 看到的语义最稳。
- **整条消息删 vs 留空内容**：按 litellm 一致做法整条删除；若日后 Anthropic Messages 协议变严要求"空 content 拒绝"，再切到"插入占位 text block"。
- 对 `MessageContent::Text(_)` 不动（首次请求、纯文本 history 不会触发 strip 分支，gate 已经保护）。

### 3.4 `stream()` —— 显式加注释，不重试

文件：`src/providers/anthropic.rs:209-245`
不动函数体，仅在 `has_thinking_error` 分支前补一行注释：

```rust
// Streaming keeps the original friendly-error path on purpose: once
// the SSE byte stream starts flowing, retrying would double-emit
// content to the client. See Router::stream comment at
// router.rs:303-307 for the system-wide "no retry once streaming"
// contract. The AnthropicProvider::stream path therefore DOES NOT
// apply strip-and-retry; a thinking-style 400 still surfaces as
// `thinking_not_supported` here.
if status.as_u16() == 400 && has_thinking_error(&text) {
    return Err(self.thinking_not_supported_error(...));
}
```

### 3.5 不动的文件——明确清单

| 文件 | 原因 |
|---|---|
| `src/router.rs` | `is_model_unsupported` 子串规则不变；`is_cooldownable` 不变；流式不重试契约不动 |
| `src/config.rs` | `ProviderConfig::Anthropic` 不加字段，`deny_unknown_fields` 风险无关 |
| `src/error.rs` | `is_cooldownable` 不动（400 thinking 透传后不会进 cooldown，这是已知行为；详见 §5 编码注意 R-D） |
| `src/providers/mod.rs` | `build()` 构造 `AnthropicProvider` 的参数签名不变 |
| `src/conversion/stream.rs:93` | 裸 `reasoning` key fallback 是独立项（R-G），不在本计划范围 |
| `config.example.yaml` | 私有配置差异不是仓库 schema 变更 |
| `README.md` | 同上 |

---

## 4. 完整测试设计与变更

### 4.1 单元测试 `src/providers/anthropic.rs::tests`

按顺序新增 / 改写：

| # | 测试名 | 类型 | 关键断言 |
|---|---|---|---|
| U1 | `has_thinking_error_matches_anthropic_error_message` | 保留原测试 | 字面量断言不变；追加 DeepSeek 精确字面量 case（`r#"{"error":{"message":"The \`content[].thinking\` in the thinking mode must be passed back to the API."}}"#`） |
| U2 | `complete_thinking_error_strips_and_retries_with_thinking_history` | 新增 | wiremock 第一响应 400 + thinking 字面量 → 第二响应 200 + 正常体；断言 wiremock `expect(2)` 命中两次；断言第二次请求体不含 `messages[*].thinking`、`redacted_thinking` 任一元素 |
| U3 | `complete_thinking_error_does_not_strip_when_no_thinking_history` | 新增 | `req.messages` 仅 user、无 assistant-thinking → wiremock `expect(1)`；返回 `ProxyError::Upstream { status: 400, body: <原 body> }`；assertion `parsed["error"]["type"] != "thinking_not_supported"`（带 raw 字面量，不进 friendly） |
| U4 | `complete_thinking_error_strip_removes_top_level_thinking_param` | 新增 | 顶层 `thinking={"type":"enabled","budget_tokens":2000}` + messages 中含 assistant-thinking block；wiremock `expect(2)`：first 400 thinking-body、second 200；断言**第二次**请求体的顶层 `thinking` 是 JSON null |
| U5 | `complete_thinking_error_strips_only_once_when_second_attempt_still_fails` | 新增 | wiremock `expect(2)`：两次都 400 thinking-body；返回 `ProxyError::Upstream { status: 400, body: <**第二次**的 body> }`；断言这次请求体的 messages 已被 strip、顶层 thinking 为 null |
| U6 | `complete_passes_through_unrelated_400_not_thinking` | 保留原测试 | 完全不动：`complete_passes_through_unrelated_400_not_thinking`（行 911-944）只断言 `model not found` 原文透传，与 strip 分支无关 |
| U7 | `thinking_mismatch_does_not_retry_when_thinking_absent` | 保留原测试 | 完全不动：行 992-1023。`plain text error` 不命中 `has_thinking_error`，应原样透传 |
| U8 | `stream_does_not_apply_strip_and_retry_on_thinking_error` | 新增 | 同 setup 作为 `complete_*` 的对偶；wiremock `expect(1)` 即 400 thinking-body；返回 `Upstream { status: 400, body: <原 body> }`。注意：这里走 stream() 但请求体里**也没**自带 assistant-thinking block，行为与 U3 一致；如果想测**带** history 时 stream 仍不重试，则再加一条 `stream_thinking_error_keeps_friendly_envelope_when_thinking_in_history`（用例与 `stream_returns_friendly_error_on_thinking_mismatch` 行 946-989 合并到这条，确保 friendly 文案、provider 字段、模型名齐全） |

> **改写已有测试**：
>
> 1. `complete_returns_friendly_error_on_thinking_mismatch`（行 783-864）—— **必须改写**。原测试断言 complete 路径在带 thinking 错误时返回 `thinking_not_supported` friendly envelope；新策略下，**带 assistant-thinking history 的请求**会优先 strip-and-retry 而不是 friendly。改写方向：
>    - 将其重命名为 `complete_thinking_error_keeps_friendly_envelope_when_no_thinking_history`，调用 `thinking_request(false)`（**不**带 assistant-thinking），断言仍走 friendly envelope（同当前的 status=400、`error.type == "thinking_not_supported"`、provider 字段齐全）。
>    - 若想保留"写入未启用 thinking 时也会被翻译"这条意图，可改写 U3 反向断言。
> 2. `complete_friendly_error_uses_client_model_when_no_rewrite`（行 866-908）—— **必须改写**。其 setup 是 `thinking_request(false)`（不自带 history），按新策略仍然命中 friendly 分支。**保留**该测试意图，只需保留 friendly envelope 断言（含 `client_model == upstream_model == "claude-model"`）。
> 3. `complete_returns_friendly_error_on_thinking_mismatch` **如是改写**为 U-3 / U-1 合并等价，与 U6/U7 共同保证 complete 路径在"无 history 时 = friendly、有 history 时 = strip-and-retry"两种行为全覆盖。

### 4.2 集成测试 `tests/integration_router.rs`

每个测试都构造一个真实的 `Router` 并通过 `Router::complete` 端到端跑（路由层 + wiremock 全链路），使用现成的 `wiremock_provider` 助手（`tests/integration_router.rs:68-74`）。

| # | 测试名 | 关键断言 |
|---|---|---|
| I1 | `http_end_to_end_anthropic_provider_strips_and_succeeds` | 单一 Anthropic provider（mock 其 Anthropic 端点 `/v1/messages`），mock 服务端返回 400+thinking-body 后挂上 200 响应；客户端请求带 assistant-thinking 历史；断言（a）响应 200 + 正常 assistant 文本；（b）wiremock 收到两次请求；（c）第二次请求体的 `messages[*]` 不含 `type==thinking`、`type==redacted_thinking` 任一块 |
| I2 | `http_end_to_end_anthropic_provider_strip_max_attempts_then_passthrough` | mock 端点永远返回 400+thinking-body；wiremock `expect(2)` 两次之后返回错误（实际应当用 `expect(2)` + 超额请求报错时 panic 来验证），断言客户端拿到**最后一次**响应（status=400 + raw body），且 router 不回退到 backup（即未产生 `RouteAttempt`）—— 由于 400 不属于 `is_cooldownable`，router 直接 return，对应实现于 `src/router.rs:218-222` |

**重要**：I1 与 I2 都是通过**单 provider 链**跑（`fallback_chain: []`），保证端到端断言只反映本计划改动；多 provider 链的 fallback 行为已在 `mock_llm_provider_*` 系列覆盖。

### 4.3 不动的测试

- `tests/integration_router.rs::mock_llm_provider_*` 系列 —— 这些走 `WiremockOpenAiProvider` 而非 `AnthropicProvider`，与本计划无关，全保留。
- `complete_passes_through_every_field_unmodified`、`complete_passes_through_web_search_20250305_hosted_tool`、`complete_preserves_thinking_signature_in_response` —— 全字段透传 + web_search 工具 + thinking signature roundtrip，与 strip 分支正交，保留。
- `build_body_*` / `merged_rewrite_*` 系列 —— `build_body` 函数签名未动，保留。
- `list_models_*` 系列 —— 与本计划无关，保留。
- `passthrough_sse_*` 系列 —— 与本计划无关，保留。
- `error.rs::tests` —— 不动，`is_cooldownable` 表不变。

---

## 5. 编码注意事项

- **R-A（mutate 边界）**：strip 后 messages[*].content 清空时，**整条消息删除**而非保留 `{"type":"text","text":""}` 占位。litellm 取整条删除。如果将来 Anthropic 协议改成拒绝空 content 消息，再切到占位策略；目前最小复现优先。
- **R-B（流式契约）**：`stream()` 不重试是**有意取舍**（对应 `router.rs:303-307` 的"流式一旦开流不重试"契约），不是疏漏。Step 3 的注释明确说明。
- **R-C（friendly 路径）**：`complete_thinking_error_keeps_friendly_envelope_when_no_thinking_history`（由 `complete_returns_friendly_error_on_thinking_mismatch` 改写）与 `complete_friendly_error_uses_client_model_when_no_rewrite` 是 complete 路径 friendly-envelope 的**唯一**两个正向测试。流式版本保留不动。
- **R-D（透传不进 cooldown）**：400 thinking 透传 raw body 后，`is_cooldownable` 返回 `false`（`error.rs:85-92` 仅命中 401/402/404/408/429/5xx），`is_model_unsupported` 也不匹配（`router.rs:25-49` 子串无 `not supported` 等）。router `complete()` 看到非 cooldownable 非 model-unsupported 错误时**直接 return**（`src/router.rs:218-222` 段）。所以 strip-and-retry 第二次仍 400 后，链路直接终止、客户端看到 raw 400。**这是预期行为**，不是 bug——它对应"我们已经试过了剥除重试，仍然失败，让客户端决定"。
- **R-E（gate 收紧）**：判定仍使用最严子串（`has_thinking_error` 原文不动），恰好同时命中 Anthropic 原生与 DeepSeek 的伪误报；但**只有**当 `req.messages` 中确实有 `assistant + Thinking` block 时才进入 strip。裸 400、`model not found`、generic 400 完全不走 strip。
- **R-F（日志）**：不要在 anthropic.rs 里新写 `summarize_for_log` 之类的摘要工具——项目记忆指出 `summarize_for_log` 已经在 copilot.rs 和 openai_responses.rs 重复实现，本计划内复用 `tracing` 直接输出即可，避免范围蔓延。可选 Step 0 仅打 `req_has_assistant_thinking` 布尔、`body_first_200_chars`、`provider`、`client_model`。
- **R-G（后续独立项，不在本计划范围）**：`src/conversion/stream.rs:93` 当前只认 `delta.reasoning_content`，不识 OpenAI 格式返回的裸 `reasoning` key。这是与本计划**正交**的兼容性补丁，建议作为下次独立任务处理：
  - 改动：`reasoning_content` 为空/null 时 fallback 到 `delta.get("reasoning")`，渲染逻辑不变。
  - 测试：在 `conversion/stream.rs::tests` 加 `bare_reasoning_key_renders_thinking_block`（`delta: { "reasoning": "..." }` 无 `reasoning_content` → `ContentBlockStart(Thinking)` + `ThinkingDelta`）。
  - **不在本次 PR 中合并**。

### 5.1 项目层硬性约定

- 单元测试一律用 `#[tokio::test]` + `wiremock` 本地拉起的 `MockServer`，参考 `complete_passes_through_every_field_unmodified`（`src/providers/anthropic.rs:325-470`）和已有的 `ResponseTemplate` + `body_partial_json` 配套用法。
- `ProviderConfig::Anthropic` 的字段顺序与 serde rename 命名沿用现状（`name, api_key, api_base, model_rewrite, use_proxy`），本计划不改变。
- 改动前先 `grep -rn "ProviderConfig::Anthropic" src/ tests/` 锁定 5 个构造点，逐一确认不被本次改动影响（详见 §0）。
- `deny_unknown_fields` 在 `Config` / `ServerConfig` / `ProxyConfig` / `ModelConfig` / `LoggingConfig` / `ProviderConfig` 都开着，本次不新增字段所以无需 `#[serde(default)]` 配套。

---

## 6. 上线 / 回滚提示（参考，非测试要求）

- 上线顺序：PR 含 anthropic.rs + 测试 → `cargo test --all` 全绿 → 灰度一台配置 DeepSeek (`type: anthropic`) 的代理跑真实流量 → 看日志是否还有 thinking 400 落到客户端 → 若仍有，则 Step 0 的 `req_has_assistant_thinking` 字段可直接证明 "客户端历史里压根没带 thinking block"，把信号带给用户。
- 回滚：单文件 revert `anthropic.rs` 即可，不影响 router/config/其他 provider 类型。

---

# 第三轮：代码审查记录（Sonnet agent，2026-07-31，实施后）

> 审查对象：实施后的 `git diff HEAD`（`src/providers/anthropic.rs` + `tests/integration_router.rs`）。
> 总体结论：**有条件通过**——生产代码与权威计划逐字对齐、无死循环/无越界改动；strip 的两个边界（空消息序列、thinking:null 注入）无防护也无测试覆盖，且新增测试里有假绿隐患。

## Q1 计划遵循度

- **符合**：`complete()` 重试循环、`should_strip_and_retry` 双条件 gate、`strip_thinking_blocks` mutate（过滤块/空消息删除/顶层置 null）、stream() 不重试、无 config 字段、无 router/error 改动——全部与权威计划逐项一致。删除 complete() friendly 分支的裁决**正确且彻底**（`complete()` 内零 `thinking_not_supported_error` 调用，helper 仍被 stream() 使用，无死代码）。
- **遗漏**：
  - D1：U1 测试"追加 DeepSeek 字面量"与原生断言**字节相同**，零信息量重复。
  - D2：可选 Step 0 的 tracing 诊断日志未实现（计划标为可选，非阻断）。
  - D3：`implement-deepseek-thinking.md` 自身 §3.2 vs §4.1 的矛盾未按权威计划 Step 5 修订，仍留在仓库。

## Q2 额外问题（实施中引入 / 计划继承）

| # | 严重度 | 位置 | 问题 | 状态 |
|---|---|---|---|---|
| P1 | 高 | `anthropic.rs:329` | strip 整条删除消息后 messages 可能变空或连续同角色 → 可能制造新的、更难懂的 400。计划 R-A 已接受 litellm 式整条删除，但无保护 | 记录，计划内已接受 |
| P2 | 高 | `complete()` 整体 | strip 非一次性自愈：跨模型历史每轮回传 thinking 块 → 每轮两次上游请求 + prompt cache 每轮 miss，会话越长代价越大 | 记录为已知代价 |
| P3 | 中 | `anthropic.rs:306-309` | **gate 只匹配 `Thinking`，strip 剥 `Thinking`+`RedactedThinking`，类型集合不一致**。历史只有 redacted_thinking 时 gate 不触发 → 不 strip 直接透传 400 | **建议修**（加变体，1 行） |
| P4 | 中 | `anthropic.rs:335` | `body["thinking"] = Value::Null` 会给本无 thinking 键的请求**注入 null**（客户端关 thinking 但历史带 thinking 块时）。litellm 是删键 `pop`，计划把删键改写成置 null 未实证 | **建议修**（改删键） |
| P5 | 中 | `complete()` strip 分支 | strip-retry 全程零日志，运维不可观测（违反 Diagnose before fixing 精神） | **建议修**（补 warn 日志） |
| P6 | 中 | `anthropic.rs:397-406` | U1 重复空断言制造"已覆盖 DeepSeek"假信号 | 记录 |
| P7 | 中 | `anthropic.rs:956`、`:1150`、`integration_router.rs:786` | strip 断言可被空 `messages` 数组假绿通过（fixture 全删也通过） | **建议修**（补长度/保留断言） |
| P8-P10 | 低 | — | `has_thinking_error` 双调用、`api_key.clone()` 不必要、rustfmt 未验证 | 记录 |
| P11 | 低 | — | 透传 raw 后上游措辞若含 "model"/"not supported" 会意外命中 `is_model_unsupported` 触发 fallback（当前字面量不命中） | 记录 |

## Q3 测试覆盖缺口

- **零覆盖**：`redacted_thinking` 剥离分支；整条消息删除分支（fixture 的 assistant 消息含 text 块，永远非空）；`should_strip_and_retry` 纯函数（role != assistant、assistant+Text、RedactedThinking）；strip 后第二次返回 429/5xx。
- **假绿**：三条 strip 断言未钉死"只剥 thinking、不误删"（补 `messages.len()` + 保留 text 块断言）；顶层 thinking 断言 `is_null()` 无法区分置 null 与删键。
- **缺用例**：非 thinking 400 + 有 thinking 历史（body gate 独立生效）；I2 未显式断言 `x-llmproxy-failed-providers` 头不存在。

## 待办建议（供下一轮实施）

1. **P3** gate 加 `ContentBlock::RedactedThinking`；**P4** 改删键 `body.as_object_mut()?.remove("thinking")`；**P7** 三处补强断言；**P5/D2** 加 `tracing::warn!`（trigger + 二次失败）。
2. 补纯函数单测：`strip_thinking_blocks_removes_message_that_becomes_empty`、`_removes_redacted_thinking`、`_leaves_string_content`、`_on_body_without_messages_key`；`should_strip_and_retry_requires_assistant_role`（含 redacted 期望 true，会红以证 P3）。
3. 补 T1（非 thinking 400 + thinking 历史）、T8（strip 后 500 透传 + router 层行为）、I2 显式断言 header。
4. **D3** 修订本文档 §3.2/§4.1 矛盾（删除 friendly 残留，对齐权威计划）。
5. **P2** 在文档中如实记录"每轮 strip 成本"为已知限制。
