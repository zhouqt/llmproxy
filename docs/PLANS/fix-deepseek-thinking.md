# 修复 DeepSeek 的 thinking 模式处理

## 问题

当请求启用 thinking 模式且 fallback chain 中有 `deepseek` provider（配置为 `type: anthropic`，使用 DeepSeek 的 Anthropic 兼容 Messages API）时，代理返回错误的错误信息：

```
400 Provider 'deepseek' does not support thinking/reasoning mode for model 'deepseek-v4-pro'.
```

但 DeepSeek **实际上支持 thinking**。这是一个误报。

## 根因

### has_thinking_error() 过于宽泛

`src/providers/anthropic.rs:251` 中的检查：

```rust
fn has_thinking_error(body: &str) -> bool {
    body.contains("content[].thinking") && body.contains("must be passed back")
}
```

该函数用于识别 Anthropic 上游返回的经典 thinking 错误：

```
{"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}
```

在原生 Anthropic API 中，这个错误意味着 "你用的模型不支持 thinking，但你在消息历史中传了 thinking block"。

**但 DeepSeek 的 Anthropic 兼容端点返回完全相同的错误体**，含义不同。DeepSeek 支持 thinking，但它可能要求多轮对话中的 thinking block 以特定格式传递。代理把 DeepSeek 的这个错误当作 "模型不支持 thinking" 来处理，从而产生了误报。

### 错误传播路径

1. 客户端发送 thinking-enabled 请求 → `AnthropicProvider::complete()`
2. 代理把请求 verbatim 转发到 `https://api.deepseek.com/v1/messages`
3. DeepSeek 返回 400 + 上述 thinking 错误
4. `has_thinking_error(body)` → true
5. 代理生成 `thinking_not_supported_error` 返回给客户端

## 核心争议

这个错误信息有两种可能的含义：

**A) DeepSeek 的 thinking 实现确实有兼容性问题**。多轮对话中，DeepSeek 的 Anthropic 端点可能要求 thinking block 以特定格式传送，而 Anthropic 原生 API 的格式它无法直接处理。这需要修改请求转换逻辑。

**B) DeepSeek 的 thinking 实现是完整的**。错误是因为其他原因触发的（如并发请求导致 block 状态异常），只需要跳过 `has_thinking_error` 检查就能暴露真实错误。

无论哪种情况，代理都不应该把 "不支持的 thinking" 作为最终结论。

## 修复方案

### Step 1: 为 AnthropicProvider 添加 supports_thinking 配置（初步修复）

**修改 `ProviderConfig::Anthropic`，添加可选的 `thinking_supported` 字段**。

用户在 `config.yaml` 中为 DeepSeek 设置 `thinking_supported: true` 后，代理不再将 DeepSeek 的 thinking 错误转换为 "thinking_not_supported"，而是把上游原始错误传递给客户端。

改动文件：
- `src/config.rs` — `ProviderConfig::Anthropic` 新增 `#[serde(default)] supports_thinking: bool`
- `src/providers/anthropic.rs` — `AnthropicProvider` 存储 `supports_thinking`，在 `complete()` 和 `stream()` 中加判断：如果 `supports_thinking` 则跳过 `has_thinking_error` 检查

### Step 2: 暴露真实错误，诊断 DeepSeek 实际需求

在 Step 1 的基础上，用户可以看到 DeepSeek 返回的真实错误。这将揭示：
- 是 DeepSeek 完全支持 thinking 只是格式要求不同
- 还是 DeepSeek 的 Anthropic 端点对多轮 thinking 有根本性的限制

基于诊断结果决定后续步骤。

### Step 3（如果需要）: 修正请求转换以兼容 DeepSeek 的 Anthropic 端点

如果 DeepSeek 的 Anthropic 端点对 thinking block 的格式有额外要求，需要修改 `build_body()` 或相关的序列化逻辑，确保发送给 DeepSeek 的请求体符合其规范。

### Step 4: 测试

- 单元测试：验证 `supports_thinking: true` 时 `has_thinking_error` 检查被跳过
- 集成测试：验证 DeepSeek 场景下 thinking error 正确透传

## 涉及文件

| 文件 | 改动 |
|---|---|
| `src/config.rs` | `ProviderConfig::Anthropic` 添加 `supports_thinking: bool` 字段 |
| `src/providers/anthropic.rs` | `AnthropicProvider` 存储 + 使用 `supports_thinking`，跳过 `has_thinking_error` |
| `docs/PLANS/fix-deepseek-thinking.md` | 本文件 |

---

# 评审意见（3 个 Opus agent，2026-07-31）

> 注：首次写入本文档时 Write 工具静默失败（返回成功但文件未落盘），三个 agent 在缺文档的情况下基于"方案简述"评审。后重新写入成功。下面合并三家意见，按评审要求的 5 个问题组织。

## 评审结论（先行）

**拒绝按当前方案实施。** 三份评审一致认为：方案用 config 标志掩盖了 `has_thinking_error` 判定语义的错误（把两类不同的错误折叠成 "不支持 thinking"），没有先诊断真问题（违反 CLAUDE.md "Diagnose before fixing" 原则），且跳闸后把可观测的友好错误换成更难诊断的原始错误。**大概率不需要 config flag。**

---

## Q1: 该方案是否最佳方案？

**否。** 三家 agent 均判定不是最佳方案。

### 核心缺陷（Agent 1 为主，Agent 2 呼应）

`has_thinking_error()`（`src/providers/anthropic.rs:251-253`）把 Anthropic 协议里**两类截然不同的错误**折叠成一个判断：

- **真·协议违规**：多轮对话中 assistant 的 `thinking` block 缺 `signature`，或 `signature` 与上一轮不匹配。Anthropic 原生确实返回这个 400。
- **兼容端点复用同一错误消息**：DeepSeek / 其他 Anthropic 兼容端点触发条件不同（例如首次请求就报）。

**区分这两类的信息是请求上下文（`req.messages` 里有没有 assistant + thinking block），不是上游 body 字符串。** 当前实现抛掉了请求上下文，这是判定不严谨的根源。给 provider 加 `supports_thinking: true` 跳过整个检测，等于把根因继续埋下去。

### 命名误导（Agent 1 重点，Agent 2 问题 4）

`supports_thinking` / `thinking_supported` 暗示"上游**能否**处理 thinking"。但 deepseek-v4-pro 确实支持 thinking——标志表达的是**假信息**。它实际表达的是"代理对这个 provider 的 thinking 错误**不翻译**"。正确命名应为 `pass_through_thinking_errors: bool` 或 `translate_thinking_errors: bool`（默认 true）。

### 前提矛盾（Agent 1 问题 10，已实测验证）

`config.example.yaml:150-157` 里 DeepSeek 配置为 **`type: openai_compat`**（走 `https://api.deepseek.com/v1` OpenAI 格式端点），**不是** anthropic。而 `has_thinking_error` 只存在于 `anthropic.rs`——openai_compat 路径走 `conversion/` 翻译，**根本不会触发这个检查**。

用户报的错误信息（"Provider 'deepseek' does not support thinking..."）来自 `anthropic.rs::thinking_not_supported_error`，说明用户实际 config.yaml 里 DeepSeek 是 `type: anthropic`（走 DeepSeek 的 Anthropic 兼容端点）。**计划必须先把用户当前 config 写清楚**，否则 scope 覆盖的是私有配置而非仓库默认配置。

---

## Q2: 更好的方法是什么？

### 推荐主路径：基于请求上下文改进错误检测（Agent 1 备选 B，Agent 2 采纳建议）

把 `has_thinking_error(body)` 改为 `is_thinking_signature_violation(body, req)`：

- 仅当 `req.messages` 中**已经存在** `role: "assistant"` 且包含 `type: "thinking"` 的 block 时，才把这种 body 翻译为 `thinking_not_supported`（真协议违规，保持友好错误）。
- **首次请求**（无 assistant 消息、或 assistant 消息无 thinking block）时报这个错 → **直接透传**原始 body，因为这是上游端点的兼容性 bug，不该被代理的友好文案掩盖。

实现成本：把 `req` 引用传进调用点（`complete()` / `stream()` 都有 `req`），加 5-10 行模式匹配。**无需 config flag，无需改 `ProviderConfig`。** 这一改完，DeepSeek 的误报自动消失。

### 端到端目标（Agent 1 问题 9）

"primary 用 thinking、DeepSeek 真的支持 thinking 时，fallback 链应该直接工作"——这个目标在 DeepSeek 端点兼容性细节明确之前无法验证，可能有三种情况：

- a) 端点完全兼容，只是错误措辞相同 → 修检测逻辑即可。
- b) 端点对 thinking 配置有特殊要求（语法/字段不同）→ 必须改 `build_body()`。
- c) 端点的 Anthropic Messages 复刻对 thinking 是阉割版 → 需要在 router 层永远跳过 DeepSeek 走 thinking 请求。

**方案没有"如何确定是哪种"的诊断步骤**——必须先做生产诊断（CLAUDE.md 明文要求），再决定修哪一层。

### 不推荐的备选

- **按模型名前缀（deepseek-* 跳过检测）**：把"哪些上游 thinking 兼容"硬编码进代码，违反配置驱动原则，新模型/新 provider 都要改代码（Agent 1 问题 5，拒绝）。
- **router 层 thinking 能力预检（`Provider::supports_thinking`）**：上游端点声称支持只是兼容性有差，预检无法静态判断，不必要（Agent 1 问题 7）。

---

## Q3: 方案会引入哪些额外问题？

按严重等级列出（Agent 2 为主）：

| # | 严重度 | 问题 | 涉及位置 |
|---|---|---|---|
| 1 | **高** | **标志粒度太粗**：per-provider 标志对 provider 下所有模型生效。DeepSeek 有 thinking 模型（v4-pro）和非 thinking 模型（deepseek-chat），一个标志对两者都错。若将来 deepseek-chat 走同一 provider，用户收到原始 400 而非友好提示 | `config.rs:95-106`、`config.example.yaml:154-157` |
| 2 | **高** | **跳闸后 400 不进 cooldown、不触发 fallback，链路直接终止**（实测验证）：`is_cooldownable()` 只含 401/402/404/408/429/5xx（`error.rs:85-92`）；`is_model_unsupported()` 的子串不匹配该 body（`router.rs:25-49`）；router 对非 cooldownable 非 model-unsupported 错误**直接 return**（`router.rs:218-222`、`347`）。注意：当前 friendly-error 路径同样终止链路，所以这不是新回归，但透传后运维可观测性更差 | `router.rs`、`error.rs` |
| 3 | **中** | **`deny_unknown_fields` 兼容性**：新字段必须 `#[serde(default)]`，否则旧 config.yaml 直接反序列化失败 | `config.rs:66`、`95-106` |
| 4 | **中** | **破坏运维契约**：`thinking_not_supported_error`（`anthropic.rs:87-122`）的文档明确承诺给运维清晰的 "reconfigure fallback chain" 信号；透传原始 400 正是文档警告要避免的 "silently degraded responses" | `anthropic.rs:87-122` |
| 5 | **中** | **多轮/并发场景漏掉**：历史里残留 thinking block、客户端切模型等场景下，一旦跳过友好错误，运维可见性丢失 | `anthropic.rs:251-253` |
| 6 | **低** | **命名不一致**：方案里 `supports_thinking` / `thinking_supported` 混用，未收敛；字段名需要 Rust 命名 + serde rename + 文档 + 测试 + example.yaml 五处一致 | 全文档 |
| 7 | **低** | **文档同步遗漏**：`config.example.yaml` 与 README "Provider types" 章节未列入改动清单 | `config.example.yaml`、`README.md` |

**额外发现**：CLAUDE.md "已知坑"指出 `summarize_for_log` 在 copilot.rs 和 openai_responses.rs 重复实现——若本次要给日志加摘要化，可顺手提到 `src/util.rs`（但避免范围蔓延）。

---

## Q4: 测试覆盖评估与改进

**现有计划 Step 4 测试设计不充分**（Agent 3 系统评估）：方案只写了 2 条测试，仅触及 8 类覆盖缺口中的 2 项。

### 关键缺口

1. 新 config 字段反序列化（deny_unknown_fields 下必备）：缺字段 / 显式 true / 显式 false / 未知子字段拒绝，4 条。
2. `complete()` 双向对照：flag=true 跳闸 + flag=false（默认）保留友好错误（回归保护）。
3. `stream()` 的同样两条。
4. flag=true 时**非 thinking 400** 仍走原透传路径（边界焊死）。
5. router 行为锁定：`is_model_unsupported(thinking 400) == false`、`is_cooldownable(400 + thinking body) == false`、端到端 `x-llmproxy-failed-providers` 头不存在 + backup `expect(0)`。
6. 现有 `ProviderConfig::Anthropic` 构造点连锁更新——实施前 `grep -rn "ProviderConfig::Anthropic" src/ tests/`（编译即验证）。
7. `has_thinking_error` 现有单测保留为契约锚点，追加 DeepSeek 精确字面量 case。

### 建议的测试清单（可直接粘贴进 Step 4）

单元测试 `src/providers/anthropic.rs::tests`：
- `complete_skips_thinking_check_when_supports_thinking_true` — 400 + canonical thinking body → `Upstream { status: 400, body }`，body 原文透传，不含 `thinking_not_supported` / `provider` / `client_model` / `upstream_model`
- `stream_passes_thinking_400_through_when_supports_thinking_true` — 同上走 stream()
- `complete_explicit_supports_thinking_false_equals_default` — 显式 false 与默认逐字节等价
- `stream_explicit_supports_thinking_false_equals_default`
- `complete_unrelated_400_still_passes_through_when_supports_thinking_true` / stream 对称版 — 400 + `model not found` 仍原文透传、不进 friendly 分支
- 保留 `has_thinking_error_matches_anthropic_error_message`，追加 DeepSeek 精确字面量断言

单元测试 `src/config.rs::tests`：
- `anthropic_supports_thinking_defaults_to_false_when_field_absent`
- `anthropic_supports_thinking_parses_explicit_true` / `_explicit_false`
- `anthropic_supports_thinking_rejects_unknown_subfield`

单元测试 `src/router.rs::tests` + `src/error.rs::tests`：
- `is_model_unsupported_does_not_match_canonical_thinking_400`
- `upstream_400_with_thinking_body_is_not_cooldownable`

集成测试 `tests/integration_router.rs`：
- `http_end_to_end_thinking_400_from_anthropic_provider_surfaces_as_400_no_fallback` — 状态 400、body 含 `content[].thinking` 原文、`x-llmproxy-failed-providers` 头不存在、backup `expect(0)`
- `http_end_to_end_thinking_400_default_provider_returns_friendly_envelope` — 默认路径回归保护（`error.type == "thinking_not_supported"`）

---

## Q5: 结论与待办

### 结论

| 评审维度 | 结论 |
|---|---|
| Q1 是否最佳方案 | **否** — 用 flag 掩盖根因，未诊断、未验证端到端目标可达性 |
| Q2 更好方案 | **先做生产诊断（加 tracing 日志），再基于请求上下文改进 `has_thinking_error` 判定；大概率不需要 config flag** |
| Q3 额外问题 | 3 高 3 中 2 低（见上表），最坏情况：所有 DeepSeek thinking 请求 → 400 → 跳过友好错误 → 不进 cooldown → 不进 fallback → 客户端收到 raw error，运维无提示 |
| Q4 测试覆盖 | **不充分** — 需补 15+ 条用例，见清单 |

### 修订后的实施顺序

1. **诊断**（不改行为）：在 `anthropic.rs::complete()` / `stream()` 的 thinking-400 分支加 tracing 日志，记录 (a) `req.messages` 是否含 assistant+thinking block (b) 顶层 thinking 配置 (c) body 前 200 字符。跑用户实际场景收集日志。
2. **判定**：日志确认 DeepSeek 报错时 `req.messages` 中是否有 thinking block——
   - 无 → 上游误报 → 改进 `has_thinking_error` 接受 `req` 引用（请求上下文判定）
   - 有 → 真协议违规 → 保持友好错误 + 在 `client_model` 上加日志
3. **修复**：按上一步结论，改 `has_thinking_error` 判定 / `build_body()` / `is_model_unsupported` 关键词表三者之一。
4. **测试**：按 Q4 清单补测试；实施前先 `grep -rn "ProviderConfig::Anthropic" src/ tests/`。
5. **文档**：同步 `config.example.yaml`（anthropic 类型注释）与 README Provider types 章节。

> 若坚持 config flag 方案（不推荐），至少必须：统一命名（`pass_through_thinking_errors`）、per-model 粒度、保留友好错误包装、加 `#[serde(default)]`、加诊断日志、补 fallback 成功路径测试。

---

# 第二轮评审意见（litellm 实证驱动，Sonnet agent，2026-07-31）

> 基于对 litellm 仓库的实证阅读，覆盖范围：
> `litellm/llms/base_llm/anthropic_messages/transformation.py`、
> `litellm/llms/custom_httpx/llm_http_handler.py:1880-1941`、
> `litellm/llms/anthropic/common_utils.py:907-958`、
> `litellm/llms/deepseek/messages/transformation.py`、
> `litellm/llms/deepseek/chat/transformation.py`。

## 修订后的结论

第一轮"先诊断、再决定改哪层、大概率不需要 config flag"的方向**继续成立**，但收敛到具体方案：**采用 litellm 的 strip-think-and-single-retry 模式 + 本项目自有的上下文感知判定条件**，全部在 `AnthropicProvider::complete` 内完成，**不动 router、不动 config**。

**关键事实修正**：litellm 的 `is_anthropic_invalid_thinking_signature_error`（`common_utils.py:907-919`）要求同时含 `thinking`+`signature`+(`invalid`|`valid string`)。**用户场景报错 `content[].thinking must be passed back` 不含 `signature`**，litellm 的自愈对当前场景**不会触发**（commit e59add11cd 有意窄化、只覆盖 Bedrock/Vertex）。所以"整条移植 litellm 逻辑"是一条窄到错过的路径——**借鉴其模式，检测器本地收敛**。

**为什么必须 strip-and-retry 而非纯透传**：litellm 的 `DeepSeekAnthropicMessagesConfig.transform_anthropic_messages_request`（`deepseek/messages/transformation.py:113-130`）仅透传 + 清洗 `custom` tools → DeepSeek 必然拒收跨模型的 Anthropic 原生 signature → 修复只能由代理做。纯透传 raw 400 给客户端，客户端下一轮又带上同样的 thinking block，**死循环**。

## 修订后的修复方案

| 步骤 | 文件 | 改动 |
|---|---|---|
| 0（可选非阻塞） | `anthropic.rs::complete` 加 tracing 日志 | 记录 `(req_has_assistant_thinking, top_level_thinking_kind, body_first_200_chars)`，确认触发条件 |
| 1 | `anthropic.rs::complete` | 加内层 `max_attempts = 2` 循环；首次 400 + thinking body + req 含 assistant thinking block → mutate 本地 body：过滤 `messages[*].content` 中 type ∈ {thinking, redacted_thinking} 的 block，跳过清空后的消息；`body["thinking"] = Value::Null`（对应 litellm 的 `data.pop("thinking")`）→ 重发 |
| 2 | `anthropic.rs::complete` | **不重试**的情况：消息无 assistant thinking block 的 thinking 400 → **透传** raw body 为 `Upstream { status: 400, body }`，不再调用 `thinking_not_supported_error` |
| 3 | `anthropic.rs::stream` | **不做** strip-and-retry（保留流式"一旦开流不重试"契约，`router.rs:303-307` 已明文）。thinking 400 仍走当前友好错误 |
| 4 | 无 | **无 config 改动**，`ProviderConfig::Anthropic` 不新增字段 |
| 5 | doc | 重写 Step 2/3/4，删除 `thinking_supported` 配置痕迹 |

### 检测器设计（精确化）

```rust
fn should_strip_and_retry(req: &MessagesRequest, body: &str) -> bool {
    if !(body.contains("content[].thinking") && body.contains("must be passed back")) {
        return false;                          // 与现有 has_thinking_error 同子串
    }
    req.messages.iter().any(|m| {
        m.role == "assistant"
            && matches!(&m.content, MessageContent::Blocks(bs)
                if bs.iter().any(|b| matches!(b, ContentBlock::Thinking { .. })))
    })
}
```

**为什么不会过度 strip**：两条件 AND。(a) 上游必须精确命中 thinking-style 错误子串（剥离 "model not found" 等无关 400）；(b) 请求里必须含历史 thinking block。裸 400、generic 400、model_not_found 完全不进入 strip 分支。

### fallback 链行为预期

- primary（Claude native）走 thinking 成功 → history 带 Claude signature，切到 DeepSeek fallback。
- DeepSeek 第一次 → 400 thinking-error → strip 重试 → DeepSeek 收到无 thinking history + 顶层无 thinking 配置的请求 → 200，返回非 thinking 响应。
- 客户端下一轮把这条非 thinking assistant 消息带回 → DeepSeek 不再 400。
- **最大风险**：(i) 客户端带顶层 `thinking={"type":"enabled","budget_tokens":N}`，DeepSeek Anthropic 端点拒 budget_tokens —— strip 时必须 `body["thinking"] = Value::Null`，否则重试仍 400；(ii) **死循环**——strip 后第二次仍 400 → 第二次后不再重试、直接透传 raw body 进 `Upstream { status: 400 }`。
- stream() 路径：无重试，**保留友好消息**（流式无法自愈，友好提示有价值）。

## 测试清单（取代第一轮清单）

`src/providers/anthropic.rs::tests`：

1. `complete_thinking_error_strips_and_retries_with_thinking_history` — wiremock hit 2：first 400 thinking-body、second 200；断言 second 请求体不含 `messages[*].thinking`、顶层无 `thinking` 字段。
2. `complete_thinking_error_does_not_strip_when_no_thinking_history` — messages 无 assistant thinking block → hit 1，返回 `Upstream { status: 400, body: <原 body> }`，**非** `thinking_not_supported`。
3. `complete_unrelated_400_passes_through_unchanged` — 保留回归。
4. `complete_thinking_error_strips_only_once_when_second_attempt_still_fails` — hit 2 都 400；返回 raw `Upstream`，第二次请求体是 stripped 版。
5. `complete_thinking_error_strip_removes_top_level_thinking_param` — 顶层 thinking + messages assistant thinking 都存在；first 400、second 200；断言 second body 顶层无 `thinking`。
6. `thinking_mismatch_does_not_retry_when_thinking_absent` — 保留回归（`anthropic.rs:993`）。
7. `stream_does_not_apply_strip_and_retry_on_thinking_error` — 流式 1 次即返回 400（无重试）。

集成测试 `tests/integration_router.rs`：
- `http_end_to_end_anthropic_provider_strips_and_succeeds` — 端到端 200，wiremock `expect(2)`。
- `http_end_to_end_anthropic_provider_strip_max_attempts_then_passthrough` — 400→400，status 400 + raw body + 无 `x-llmproxy-failed-providers` 头。

现有 `complete_returns_friendly_error_on_thinking_mismatch` 等（`anthropic.rs:783-864`）因 complete() 路径改为 strip-and-retry 而需要改写/合并；`stream_*` 版本保留（流式仍走友好）。

## 遗留风险（R-A ~ R-F）

- **R-A**：strip 跳过清空后消息 vs 保留占位 `{"type":"text","text":""}`——litellm 直接跳过空内容消息；建议先按 litellm 实现 + 最小复现验证（Messages API 拒绝空 content）。
- **R-B**：stream() 不重试是有意取舍（保持"流式一旦开流不重试"契约），需在 patch 注释说明。
- **R-C**：`thinking_not_supported_error` 的 complete 路径测试需改写/合并，stream 版本保留。
- **R-D**：若 Step 0 日志显示 DeepSeek 报错时 `req.messages` 无 assistant thinking block（与推断矛盾），则 gate 关闭，行为变回"纯透传"，strip 永不触发。
- **R-E**：判定用最严格子串（`content[].thinking` AND `must be passed back`）恰好匹配 DeepSeek + Anthropic 原生口径；其他兼容端点报不同语义错误时不会误触发 strip。
- **R-F**：不要新写日志摘要函数，复用现有 `summarize_for_log` 路径，避免范围蔓延。

## 对第一轮意见的修正

- 第一轮建议 (a) 模型名前缀 / (b) router 层预检：**保留驳回**。
- 第一轮建议 (c) "基于请求上下文改进 `has_thinking_error`"：**采纳为 detection gate**，但**不是裸判定**，而是 strip-and-retry 的前置条件。
- 第一轮建议"先做生产诊断"：**降级为可选 Step 0**（litellm 行为已确认 DeepSeek 拒收跨模型 signature；死循环已被 strip-and-retry 化解）。
- 第一轮建议"大概率不需要 config flag"：**确认仍不需要**。

---

# 补充：llmgateway 实证（2026-07-31，独立旁证）

> 研究 `/tmp/llmgateway/`（Elixir LLM 网关）后补充一条独立于 litellm 的旁证，仅记录与本计划相关的结论。

## llmgateway 的架构规避（背景，非本计划改动）

llmgateway 内部 canonical 格式是 **OpenAI chat/completions**，`Convert.to_provider/2` 只对 `provider_type == :anthropic` 转 Anthropic Messages 格式，**DeepSeek 一律以 OpenAI 格式通信**（`reasoning_effort` 出站、`reasoning_content`/`reasoning` 入站）。因此 llmgateway 从不经过 DeepSeek 的 Anthropic 兼容端点，`content[].thinking must be passed back` 400 对它不成立。这从第三方项目角度再次印证 litellm 的结论：DeepSeek 的 thinking 原生路径是 OpenAI 格式，Anthropic 兼容端点才是兼容层。**llmproxy 的 strip-and-retry 修复方向不变。**

## 值得补充的一条：流式响应兼容裸 `reasoning` key（R-G）

llmgateway commit `6069968`（"render bare reasoning key as thinking block"）专门处理了一个问题：**DeepSeek 等上游在流式 delta 里用裸 `reasoning` key 而非 OpenAI 标准的 `reasoning_content`**。其修复是 `reasoning = delta["reasoning_content"] || delta["reasoning"]`，两者任一存在都渲染成 Anthropic `thinking` block。

**llmproxy 现状**：`src/conversion/stream.rs:93` 只认 `choice.delta.reasoning_content`，**没有 fallback 到裸 `reasoning`**。若 DeepSeek 将来走 `openai_compat` 端点（conversion/ 路径，OpenAI 格式）且按 llmgateway 观测的方式返回裸 `reasoning`，llmproxy 会漏掉 thinking 渲染——thinking block 直接静默丢失。

**处置**：与本次 strip-and-retry 主计划**正交**（主计划只改 `AnthropicProvider`，不碰 conversion/）。作为独立小修复挂在此计划之后，不进本次改动范围：

| 文件 | 改动 |
|---|---|
| `src/conversion/stream.rs:93` | `reasoning_content` 为空时 fallback 到 `delta["reasoning"]`，渲染逻辑不变 |

测试：在 `src/conversion/stream.rs::tests` 追加一条 `bare_reasoning_key_renders_thinking_block`（`delta: {"reasoning": "..."}` 无 `reasoning_content` → 产出 `ContentBlockStart(Thinking)` + `ThinkingDelta`），与现有 `reasoning_only_chunk_emits_thinking_block` 对称。
