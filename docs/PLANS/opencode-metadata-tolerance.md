# Plan: OpenCode Zen metadata-line tolerance

**症状来源**:线上 operator 观察到 OpenCode Zen (OpenAI 兼容 Chat Completions 提供商) 在 SSE 流尾部追加一行非标准 metadata,代理每次都打 `DEBUG skipping malformed SSE line: ... (missing field \`id\`)`,多一行 debug 日志 + 一次丢弃。
**review 来源**:对当前 `ChatChunk` 反序列化策略的代码 review。
**前置**:`docs/PLANS/round1..5-gpt5-cleanup.md`(本计划不修改任何历史 plan,作为独立 plan 落在同目录)。

---

## Background

OpenCode Zen 在 SSE 流的 `[DONE]` 之前会额外推一条
`{"choices":[],"x-opencode-type":"inference-cost","cost":"...","normalizedUsage":{...}}`
(JSON 良构,但不包含 `id` / `model`,且 `choices` 为空)。当前 `ChatChunk`
把这三个字段声明为必填(`src/openai.rs:339,351,352`),反序列化失败 → 落到
`src/providers/openai_compat.rs:222-226` 的 `tracing::debug!("skipping malformed...")`
分支。这条 metadata 是合法 JSON,**不是**真正畸形的 payload(HTML 错误页、
被截断的流尾);它只是 ChatChunk schema 的一个超集 —— `extra` 字段携带了
上游私有的成本/usage 元信息,核心字段恰好被 proxy 当成必填。

每条 OpenCode Zen 请求都会多一行 debug 日志,长期累积对运维噪声有影响,
更重要的是:这条日志会让 future operator 误以为"上游发畸形 JSON",把真
正的问题信号淹没在噪声里。

---

## Root cause

`src/openai.rs:337-355` `ChatChunk` 反序列化策略过紧:

```rust
// src/openai.rs:338-352
pub struct ChatChunk {
    pub id: String,                              // L339 — required
    #[serde(default = "default_chunk_object")]
    pub object: String,                          // L345 — tolerated
    #[serde(default)]
    pub created: i64,                            // L349 — tolerated
    pub model: String,                           // L351 — required
    pub choices: Vec<ChunkChoice>,               // L352 — required
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}
```

三个字段与 OpenCode Zen metadata 行直接冲突:
- `id: String`(L339)— metadata 没有 `id`,反序列化失败。
- `model: String`(L351)— metadata 没有 `model`,反序列化失败。
- `choices: Vec<ChunkChoice>`(L352)— 实际是 `[]`,但被必填。当前能跑
  通纯属 fixture 写了 `"choices":[]`,本质上仍是 required(任何缺 `choices`
  字段的对象都会失败,只是空数组是值缺省语义绕过)。

`object` 和 `created` 已经是 `#[serde(default)]`,正是为了容忍 Copilot 的
同类省略(L341-L349 的注释也明确写了这一点)。`id` / `model` 没被同样对待
是历史不一致 —— Copilot 凑巧总是带这两个字段,所以早期观察没问题;OpenCode
Zen 不带,问题才暴露。

落点位置 `src/providers/openai_compat.rs:222-226` 的 `Err` 分支本身**是对的**
(真正畸形时仍需要它处理 HTML/截断),只是**不应**把良构的非-ChatChunk JSON
推到那里。

---

## The fix

### Step 1 — `src/openai.rs:338-355` 把 `ChatChunk` 改为结构宽容

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChunk {
    /// Some upstreams (e.g. OpenCode Zen) emit non-standard SSE
    /// metadata lines that lack `id`. Tolerate the missing field —
    /// it is never read by the proxy (translator only uses
    /// `choices` / `usage` / `object`).
    #[serde(default)]
    pub id: Option<String>,
    /// `object` is the OpenAI discriminator (`"chat.completion.chunk"`).
    /// Some upstreams omit it (e.g. GitHub Copilot's SSE chunks).
    #[serde(default = "default_chunk_object")]
    pub object: String,
    /// `created` is the OpenAI timestamp; Copilot's SSE chunks omit it.
    #[serde(default)]
    pub created: i64,
    /// Some upstreams (e.g. OpenCode Zen metadata) emit chunks without
    /// a model. Tolerate; never read by the proxy.
    #[serde(default)]
    pub model: Option<String>,
    /// OpenCode Zen metadata lines use `"choices": []`. Tolerate a
    /// missing field too, mirroring `ChatRequest`'s extra-bag pattern.
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
    /// Catch-all for upstream-private fields (e.g. OpenCode Zen's
    /// `x-opencode-type`, `cost`, `normalizedUsage`). Same pattern as
    /// `ChatRequest.extra` (`src/openai.rs:69-70`).
    #[serde(default, flatten)]
    pub extra: Value,
}
```

要点:
- `id` / `model` 从 `String` → `Option<String>`。全代码库 grep
  `chunk.id` / `chunk.model` / `c\.id` / `c\.model` 没有 reader
  (`grep` 输出零行),改 `Option` 安全。
- `choices` 加 `#[serde(default)]`,等价于 "缺省 = 空 vec"。
- `extra: Value` 跟随 `ChatRequest` 的既有模式(L69-70),保形所有未知字段,
  不引入新类型。
- `#[derive(Deserialize)]` 不动;`#[serde(flatten)]` 与现有 `Value` import
  (L6) 无冲突。

### Step 2 — `src/providers/openai_compat.rs:217-222` Ok 分支加 trace 日志

```rust
Ok(c) => {
    if c.extra.get("x-opencode-type").is_some() {
        tracing::trace!(
            extra = %c.extra,
            "absorbing upstream metadata line (not a ChatChunk)"
        );
    }
    if let Some(t) = self.translator.as_mut() {
        for ev in t.push_chunk(&c) {
            self.output_buffer.push_back(Self::encode(&ev));
        }
    }
}
```

要点:
- 走 `tracing::trace!` 而不是 `debug!`,默认不输出;RUST_LOG=trace 时给
  operator 一个可见的"哦原来是上游 metadata"标记,而不是"malformed SSE"
  这种误导信息。
- `extra.get("x-opencode-type").is_some()` 是结构化字段检查,字符串 probe
  (`payload.contains("x-opencode-type")`) 一旦 `extra` 已 flatten 就没必要。
- 不触发 `StreamEvent::Ping`(见 "What NOT to do")。

### Step 3 — `src/conversion/stream.rs::StreamTranslator::push_chunk` 不改

确认行为(`src/conversion/stream.rs:53-100`):
- L56-72:首 chunk 推 `MessageStart`,且 `started` flag 置 true,后续不重复。
- L74-76:`chunk.usage` 更新 `final_usage`;空 chunk 不影响。
- L78-97:`for choice in &chunk.choices` 迭代;`choices` 空 → 不进循环,
  返回的 `out` 只有 `MessageStart`(首 chunk)或 `vec![]`(后续)。这就是
  metadata 行期望的行为。**不需要改 `push_chunk`。**

只需新增 Step 4 的针对性测试覆盖这个路径。

### Step 4 — 测试

详见 "Test plan"。

---

## What NOT to do

1. **不要**为 metadata 行发 `StreamEvent::Ping`。`StreamEvent::Ping` 在
   `src/anthropic.rs:411` 声明,**全代码库只在两处出现**:
   `src/providers/openai_compat.rs:287`(event_name 字符串映射) +
   `src/providers/openai_compat.rs:678`(测试 fixture)。生产代码**从未**
   emit 过 `Ping`;引入"专门用于 OpenCode metadata"的 ping 是个没跑过
   真实 Claude Code 客户端的 client event,违背本仓库"先 wire 后代码"
   的原则。metadata 行的正确归宿是**静默吸收**(`push_chunk` 返回空
   事件,日志只走 trace),不是往流里塞一个客户端必须正确处理的事件。
2. **不要**把 `normalizedUsage` 合并进最终 `usage` chunk。`include_usage:
   true` 已经在请求里发出(`src/conversion/request.rs:46-48`),OpenCode
   的标准 `usage` chunk(若推)会照常走 `src/conversion/stream.rs:74-76` +
   `:115-129` 的 `prompt_tokens_details.cached_tokens →
   cache_read_input_tokens` 映射。metadata 行的 `normalizedUsage` 是
   upstream 私有信令,跨 provider 的 `reasoning_tokens` 透传决策是
   **另一个**议题(`src/openai.rs:223-230` 在非流式路径故意丢弃
   `reasoning_tokens`,本计划不动)。
3. **不要**用 `payload.contains("x-opencode-type")` 这种字符串 probe。
   `extra` flatten 之后,`c.extra.get("x-opencode-type")` 是结构化查询,
   字符串含检查是过度粗糙的(会受 JSON 字段重排/转义影响)。

---

## Blast radius

`OpenAiSseToAnthropic` 适配器被两个 provider 复用:

| Provider | 使用位置 | 受影响行为 |
|----------|---------|-----------|
| `src/providers/openai_compat.rs:216` | OpenCode Zen / 其他 OpenAI 兼容 | 主受益:metadata 行不再丢 |
| `src/providers/copilot.rs:676` | GitHub Copilot | 旁路:本次修改理论上经过 |

Copilot 兼容性分析:
- Copilot 的 SSE chunk 一直带 `id` 和 `model`,`Option<String>` 改造后仍是
  `Some(...)`,无观察差异。
- Copilot 的 SSE chunk 没有任何 `x-opencode-type` / `cost` / `normalizedUsage`
  这种私有字段,`extra: Value` 会落到 `Value::Null` 或 `Value::Object({})`,
  `get("x-opencode-type").is_some()` 返回 false,trace 日志不触发,行为完全
  等价于改造前。
- grep 全代码库 `ChatChunk` 引用,无代码读 `chunk.id` 或 `chunk.model`
  (`grep -rn ChatChunk src/` 输出仅在 `openai.rs` 自身的定义点 +
  `conversion/stream.rs` 的 `push_chunk(chunk: &ChatChunk)` 入参 +
  测试 fixture 构造点;后两者不访问字段,前者是定义本身)。
- 因此改 `id: String → Option<String>` 和 `model: String → Option<String>`
  对 Copilot **零影响**。

---

## Test plan

测试名按文件分组,每个都应是 fail-on-current-code,pass-on-fix(结构宽容测试除外)。

### `src/openai.rs` tests(新增,在 `mod tests` 末尾追加)

| 测试名 | 断言 |
|--------|------|
| `chat_chunk_accepts_opencode_metadata_line` | fixture 整条 metadata 行;`id.is_none()`,`model.is_none()`,`choices.is_empty()`,`usage.is_none()`;`extra["x-opencode-type"] == "inference-cost"`,`extra["cost"] == "0.00000000"`,`extra["normalizedUsage"]["inputTokens"] == 776` |
| `chat_chunk_accepts_missing_choices` | fixture `{"id":"c","model":"m"}`;`chunk.choices.is_empty()`(pin `#[serde(default)]` on `choices`) |
| `chat_chunk_accepts_minimal_openai_chunk` | fixture `{}`(空对象);所有字段默认值,`extra` 是 `Value::Object({})` 或 `Value::Null`(pin 全字段 `#[serde(default)]`) |
| `chat_chunk_preserves_extra_fields_alongside_known_fields` | fixture 同时含 `id`/`model`/`choices` + `x-opencode-type`;`Some(id)` 同时 `extra["x-opencode-type"] == ...`(pin flatten 与已知字段不互斥) |

### `src/providers/openai_compat.rs` tests(新增)

| 测试名 | 断言 |
|--------|------|
| `adapter_absorbs_opencode_metadata_without_events_or_logs` | SSE fixture:1 content chunk + metadata 行 + `[DONE]`;output 含 `MessageStart` + `text_delta` + `message_stop`,**不**含 metadata 产生的额外事件 |
| `metadata_line_does_not_disturb_standard_usage` | SSE fixture:content chunk + 标准 `usage` chunk + metadata 行 + `[DONE]`;`MessageDelta` 的 usage 字段反映标准 chunk(非 metadata 的 `normalizedUsage`) |
| `metadata_line_before_content_still_yields_valid_stream` | SSE fixture:metadata 行**先于** content chunk 推;output 含单个 `MessageStart`,且 `MessageStart` 在 metadata 行**之后**(确认 metadata 行不会触发提前 start) |
| `metadata_line_appears_in_trace_log_when_enabled` | 配 `tracing-subscriber` + `RUST_LOG=trace`,断言 `OpenCode` / `x-opencode-type` 字符串出现在 captured output(可选,视 tracing 测试基础设施是否现成) |

### `src/conversion/stream.rs` tests(新增)

| 测试名 | 断言 |
|--------|------|
| `empty_choices_and_no_usage_chunk_emits_nothing_after_start` | push 1 个 content chunk,然后 push `ChatChunk { choices: vec![], usage: None, id: None, model: None, .. }`;第二次 push 返回 `vec![]` |

### 回归测试(必须保持 green)

- `src/providers/openai_compat.rs::adapter_handles_fragmented_lines_malformed_data_and_eof`
- `src/providers/openai_compat.rs::stream_converts_openai_sse`
- `src/conversion/stream.rs::emits_start_then_text_then_final`
- `src/conversion/stream.rs::empty_and_anonymous_chunks_do_not_emit_events`
- 全部现有 `cargo test --lib` 测试(改动 `ChatChunk` 公共字段类型,理论上零回归,
  但需要 CI 兜底确认)

---

## Verification standard

每次 commit 必须通过的硬指标:

```bash
# 1. 类型 + clippy
cargo check
cargo clippy --all-targets -- -D warnings

# 2. 全部测试通过
cargo test --lib
# 预期净增 7-8 个测试;无回归

# 3. 新增测试 fail-on-current-code, pass-on-fix
cargo test --lib -- --nocapture chat_chunk_accepts_opencode_metadata_line
cargo test --lib -- --nocapture chat_chunk_accepts_missing_choices
cargo test --lib -- --nocapture chat_chunk_accepts_minimal_openai_chunk
cargo test --lib -- --nocapture chat_chunk_preserves_extra_fields_alongside_known_fields
cargo test --lib -- --nocapture adapter_absorbs_opencode_metadata_without_events_or_logs
cargo test --lib -- --nocapture metadata_line_does_not_disturb_standard_usage
cargo test --lib -- --nocapture metadata_line_before_content_still_yields_valid_stream
cargo test --lib -- --nocapture empty_choices_and_no_usage_chunk_emits_nothing_after_start

# 4. 覆盖率不下降
cargo llvm-cov --summary-only
# 行覆盖率 ≥ 历史 baseline

# 5. 现有 Copilot 路径不退化(grep 验证 + 测试通过)
cargo test --lib -- --nocapture copilot
cargo test --lib -- --nocapture stream_converts_openai_sse
```

**审查红线**:
- `ChatChunk.id` / `ChatChunk.model` 改 `Option` 后**不允许**留 `.unwrap()`
  在生产代码;任何 `.unwrap()` 都视为 review 红线。
- 不引入 `StreamEvent::Ping` 的 emit 路径(本计划明确禁止)。
- 不动 `src/anthropic.rs::StreamEvent` 定义。
- 不修改 `src/providers/openai_compat.rs:222-226` 的 `Err` 分支(它对
  真畸形 payload 是正确的)。
- commit message 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md` / `config.yaml` / `*.local.yaml` / `.env*`。
- 不修改任何 `docs/PLANS/round*.md`(历史 plan 不改)。
- 不动 `fix/copilot-gpt5-compat-v1`(archive)。

---

## Implementation order (3 个 commit)

1. **`feat(openai): make ChatChunk structurally tolerant of unknown SSE lines`**
   - Step 1:`src/openai.rs` 的 `ChatChunk` 改 `Option<String>` + `#[serde(default)]`
     + `extra: Value`。
   - Step 1 配套的 4 个 `src/openai.rs` 测试(`chat_chunk_*` × 4)。
   - Step 3 配套的 `src/conversion/stream.rs::empty_choices_and_no_usage_chunk_emits_nothing_after_start`。
   - 验证:全部 green,Step 1 单独可 revert。

2. **`feat(openai-compat): trace-log absorbed upstream metadata lines`**
   - Step 2:`src/providers/openai_compat.rs` Ok 分支加 trace 日志。
   - Step 2 配套的 3 个 `src/providers/openai_compat.rs` 测试
     (`adapter_absorbs_*`, `metadata_line_does_not_disturb_*`,
     `metadata_line_before_content_*`)。
   - 验证:全部 green;`RUST_LOG=info` 下不输出新增日志(默认 trace 不开);
     `RUST_LOG=trace` 下能看到 metadata 吸收标记。

3. **`docs(plans): opencode metadata tolerance plan`**
   - 本文件落地 `docs/PLANS/opencode-metadata-tolerance.md`。
   - 不需要额外代码改动。

如果 review agent 在 commit 1 之后插入额外 P0(例如发现 Copilot 上
`extra: Value` 序列化有问题),优先插入修复,后续 commit 顺延。

---

## Out of scope

本计划**不**修复:
- **`normalizedUsage` 合并进 `usage` chunk**:跨 provider 决策,影响
  Anthropic Usage 的语义(`reasoning_tokens` 是否要 surface),需要单独
  RFC。
- **`reasoning_tokens` 在 OpenAI 路径的透传**:已经在 `src/openai.rs:223-230`
  显式拒绝(`OpenAiChatResponse.reasoning_content` 没有 `#[serde]`,
  反序列化时被丢弃)。本计划不动。
- **OpenCode Zen 的非标准 `cost` 字段**:trace 日志已经能看到,要不要
  上报给 Anthropic 的 `server_tool_use` / 第三方成本 metric,是产品决策。
- **`StreamEvent::Ping` 的 emit 路径**:Claude Code 是否处理 Ping 是
  wire-contract 议题,需要实际抓包验证客户端行为后才能动。
- **其他 OpenAI 兼容提供商的私有 metadata 字段**:同一机制会自然吸收
  (因为 `extra: Value` 是 catch-all),但每个 provider 单独 trace 日志
  标记是更大的改动,本计划只标记 OpenCode 的 `x-opencode-type`。

---

## Risk assessment

| # | 风险 | 严重度 | 缓解 |
|---|------|-------|------|
| R1 | **过度宽容的 parse 屏蔽真畸形**:以前反序列化失败的对象(例如截断到中间、含未知必填字段的 partial chunk)现在也会 parse 成功,悄悄走 `push_chunk` 路径后返回空事件,无任何告警。 | 中 | Err 分支保留,真畸形(非 object/纯 HTML/截断 JSON)依然走 debug 日志;新增测试覆盖"良构但缺关键字段"vs"JSON 解析层失败"两类。本计划不引入对此的运行时告警(成本与信噪比不划算)。 |
| R2 | **`Option<String>` 改造的字段被未来代码 `unwrap()` 误用** | 低 | 当前无 reader(grep 零结果);commit 1 的红线条款明确"不留 unwrap";type system 会拦住后续误用(`Option::unwrap` 在 `clippy::unwrap_used` 开启下报 lint)。 |
| R3 | **Copilot 路径退化** | 低 | 见 "Blast radius";测试覆盖 + grep 验证双重保险。 |
| R4 | **`extra: Value` flatten 与已知 `serde(default)` 字段冲突** | 低 | serde 在 `flatten` 与已知字段同 key 时优先已知字段,其他字段进 `extra`,行为是定义良好的;测试 `chat_chunk_preserves_extra_fields_alongside_known_fields` 直接 pin。 |
| R5 | **trace 日志泄露上游私有信息** | 低 | `extra` 全量打印确实含 `cost` 等字段;但 trace 级别默认不输出,只在 operator 主动开 RUST_LOG=trace 时出现,且仅在 OpenCode Zen 路径触发,频率 = 1 行/请求。 |
| R6 | **回归 `extra: Value` 在序列化路径的行为** | 极低 | `ChatChunk` 是 `Deserialize` only(`src/openai.rs:337`),不参与序列化,不存在序列化兼容性问题。 |
| R7 | **`#[serde(default, flatten)] pub extra: Value` 与未来添加的字段冲突** | 极低 | serde 在 flatten + 已知字段时是 `extra` 只承接未知字段;新增已知字段只需 `#[serde(default)] pub xxx: ...`,无需动 `extra`。 |
| R8 | **本计划不修真正的高优先级问题** | 信息 | 本计划定位是 noise reduction + 运维可观测性,不属于 P0 正确性修复。如果发现 OpenCode Zen metadata 真正影响业务(例如 `normalizedUsage` 包含 Claude Code 依赖的 cost 估算),那是另一个 plan 的事。 |

总评:R1 是唯一需要持续关注的运行时风险,**且**只影响"运维告警密度"
(本计划本身就是降低告警密度,所以可接受);其他都是测试与代码 review
层面的常规风险,机制已经覆盖。