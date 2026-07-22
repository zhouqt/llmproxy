# Round 1: GPT-5.x Clean 分支补全计划

**分支**：`fix/copilot-gpt5-compat` (HEAD: `bc05a49`)
**review 来源**：opus review agent
**归档分支**：`fix/copilot-gpt5-compat-v1` (12 commit, 含本轮要补回的 4 个修复)

`fix/copilot-gpt5-compat` clean 分支 6 commit 相对于 v1 缺失了 4 个基于实际抓包
得出的修复——其中 3 个会让 Claude Code + GPT-5.x 真实场景 **坏掉或挂死**。本轮目标
是把这 4 个修复回填,加上一个独立的 panic 修复,再补 6 个缺失的测试。

---

## 必须修复 (P0 — 实测可触发生产故障)

### P0-1: 流式响应 `function_call` 不报 `stop_reason: tool_use`
**症状**：Claude Code 看到 `end_turn` 当作模型已说完,直接丢弃 tool_use,工具永远
不执行。非流式路径(`src/conversion/responses.rs:377`)是对的,只有流式错。

**根因**：`ResponsesStreamTranslator::finalize` (responses_stream.rs:174-219) 中
`final_stop_reason` 只看 `response.status` 一个字段,从不跟踪 `function_call` 输出项。

**修复**：参考 v1 commit `0289ac6`。
- 在 translator 上加 `has_tool_calls: bool`,遇到 `ResponseOutputItemAdded` 且
  `item.kind == function_call`(或 `kind: FunctionCall`)时置 true。
- `finalize()` 在 `completed`/`failed`/`incomplete` 分支里,如果
  `has_tool_calls`,优先返回 `tool_use`;否则保持现有 status 映射。

**验证**：
```rust
// 必须通过的测试
#[tokio::test]
async fn streaming_response_with_function_call_reports_tool_use_stop_reason() {
    // 给一段 Responses 流,含 response.output_item.added(type=function_call)
    // 不含任何 output_text.delta
    // 断言 translator.finalize() 产出的 message_delta.stop_reason == Some("tool_use")
}
```

### P0-2: Copilot 不发 `[DONE]` 时客户端挂死
**症状**：Copilot 真实场景经常在 `response.completed` 后 **不发** `[DONE]`,且
保持 TCP 连接。代理不 finalize,客户端永远等 `message_stop`。

**根因**：`ResponsesSseToAnthropic::process_lines` (openai_responses.rs:174-212)
只在 `[DONE]` 或 EOF 时调 `translator.finalize()`。EOF 测试通过纯粹是
wiremock 主动关 body,真实链路不会关。

**修复**：参考 v1 commit `4aa3d49`。
- `ResponsesStreamTranslator` 暴露 `is_terminal(&ev) -> bool`,在
  `ResponseCompleted` / `ResponseFailed` / `ResponseIncomplete` 上返回 true。
- `process_lines` 在 push_event 之后判断:若 `t.is_terminal(&ev)` 立即
  `t.finalize()` 把 `message_delta` + `message_stop` 排进 output_buffer,
  然后继续等上游字节流。**不要**因为 terminal 就关连接——上游可能还在 tail。

**验证**：
```rust
#[tokio::test]
async fn stream_finalizes_on_response_completed_without_done_sentinel() {
    // mock server 推: response.created → response.output_item.added →
    //   response.output_text.delta("hi") → response.completed
    // 然后 stream::pending() 模拟连接挂着
    // 在 200ms 超时下收集所有事件,断言收到 message_stop
}
```

### P0-3: `output_index` 不一致时双发 `content_block_stop` 丢失 tool_call
**症状**：Copilot 真实抓包(见 v1 `f5e0abe`)显示同一 `function_call` 的
`output_item.added` 用 `output_index=1`,但 `function_call_arguments.*` 用
`output_index=0`。当前 routing 按 `output_index` 分发,导致：
- block 1 opened (item.added)
- block 0 stopped (args.done)
- `finalize()` 再 stop block 1
- 客户端看到幻影双 stop,丢弃 tool_call。

**根因**：`ResponsesStreamTranslator` 完全按 `output_index` 路由 deltas/dones
(responses_stream.rs:132-151),没用上 `item_id`。

**修复**：参考 v1 commit `f5e0abe`。
- 给 translator 加 `fc_item_index: HashMap<String, u32>`,key 是
  `function_call` 类型的 `item_id`,value 是该 item 对应的 `output_index`(首次
  在 `output_item.added` 时建立)。
- `ResponseFunctionCallArgumentsDelta` / `Done` 的 `output_index` 必须先查
  `fc_item_index` 找到真实的 item_id,再走 `block_map`。**不要**直接用
  delta/done 自带的 `output_index`。
- `text_item` 类型照旧用 delta/done 自带的 `output_index`(text 不会出现
  index 漂移)。

**验证**：port v1 测试 `function_call_args_with_mismatched_output_index_*`。

### P0-4: `truncate_user` 在多字节字符上 panic
**症状**：用户 `metadata.user_id` 含中文/emoji(>64 字节但 <64 字符),`user[..64]`
按字节切,在多字节字符中间切 → Rust panic → 请求 500。

**根因**：`src/conversion/responses.rs:38-44`
```rust
if user.len() <= 64 { user.to_string() }
else { user[..64].to_string() }   // ← 字节切片,不在 char boundary
```

**修复**：改成按字符切。
```rust
pub(crate) fn truncate_user(user: &str) -> String {
    if user.chars().count() <= 64 { user.to_string() }
    else { user.chars().take(64).collect() }
}
```
- 性能可接受:`chars().count()` 是 O(n) 但 user_id 通常很短。
- 同步更新测试 `truncate_user_truncates_long_strings`,加一个多字节用例。

**验证**：
```rust
#[test]
fn truncate_user_handles_multibyte_user_id_without_panic() {
    let user = "用".repeat(30);  // 90 字节,30 字符
    let truncated = truncate_user(&user);
    assert_eq!(truncated.chars().count(), 30);
    assert!(truncated.is_char_boundary(truncated.len()));
}
```

### P0-5: `response.created` 带 `usage: null` 反序列化失败
**症状**：Copilot 在 `response.created` SSE 上发 `usage: null`,非流式完整响应也
可能带 `null`。当前 `#[serde(default)] pub usage: ResponsesUsage` 只覆盖缺字段,
不覆盖显式 null → `serde_json::from_str` 失败 → `ProxyError::Json` → 客户端 500。
流式路径:`ResponsesStreamEvent::ResponseCreated` 上的 usage 同问题,会让
"skipping malformed" 误报。

**根因**：`src/responses.rs:133` 字段类型不是 `Option`。

**修复**：参考 v1 commit `f9ff27b`。
- `pub usage: Option<ResponsesUsage>`(保持 `#[serde(default)]` 让缺省也能反序列化)。
- 所有访问 `response.usage` 的地方 (`responses.rs:388`, translator `final_usage`)
  改 `.as_ref().cloned().unwrap_or_default()`。
- 测试覆盖显式 null、缺省、正常值三种 case。

---

## 应该修复 (P1 — 设计稳健性)

### P1-1: snapshot-only reply 数据丢失
**症状**：如果上游(或 replay)只在 `output_item.added` / `output_text.done` 给出
完整文本,中间没有 `output_text.delta`,现有 `db0e67b` 的"空 start"会让客户端
收到空消息。

**修复**：在 `text_item` 块上跟踪 `emitted_any_delta: bool`:
- `output_text.done` 时若 `!emitted_any_delta`,把 done 事件的 `text` 字段作为单条
  `text_delta` 补发,然后再 stop。
- `output_item.added` 的 snapshot 不直接发,只存到 block.text 里等后续 delta
  (保持现状)。

### P1-2: gpt-5 谓词重复 + chat path 不一致
**症状**：`model.starts_with("gpt-5")` 散落在 `copilot.rs:37` 和
`conversion/responses.rs:110`,chat-completions 路径 (`conversion/request.rs`)
完全不应用 24h 升级。

**修复**：
- `src/util.rs` 或 `src/providers/gpt_family.rs` 新增
  `pub fn gpt5_family(model: &str) -> bool { model.starts_with("gpt-5") || model.starts_with("o") }`
  (o-series 共享同样的 24h 限制)。
- `copilot.rs` 和 `responses.rs` 改调这个函数。
- `conversion/request.rs` 的 Chat Completions 路径若模型命中 gpt5_family 也做
  同样的 retention 升级(目前 Chat 路径不传 cache_control 字段,所以这条可能直接
  跳过——先检查 request.rs,确实需要再加)。

### P1-3: `max_tokens` + `max_completion_tokens` 同时发出有兼容性风险
**症状**：OpenAI 严格校验在 o-series / gpt-5 上 reject `max_tokens`。当前
22e89fe 不分模型一律双发。

**修复**：gpt5_family → 只发 `max_completion_tokens`;其他 → 只发 `max_tokens`。
- 同步把 `truncate_user` 加注释引用同一个谓词。

### P1-4: `summarize_http_body` / `summarize_for_log` 重复
**症状**：~70 行 near-identical 代码,占位符还不一样(`<empty body>` vs
`<empty payload>`)。

**修复**：hoist 到 `src/util.rs` 单一 `pub fn summarize_for_log(input: &str, empty_placeholder: &str) -> String`。
- 调用方传 `"<empty body>"` 或 `"<empty payload>"`。
- 两个 caller 都改调 util 版本。

### P1-5: `*.done` 给未见过 output_index 创建幻影 block
**症状**：`output_text.done` / `function_call_arguments.done` 当前调用
`allocate_block`,对从未见过的 `output_index` 会创建 block 然后立刻 stop,
客户端看到无 start 的 stop。

**修复**：done handler 先 `if let Some(...) = block_map.get(...)` 再 allocate;
否则忽略(done 没有对应 start,本身就是异常路径,记 warn 即可)。

---

## 缺失测试 (P0-T1 ~ P0-T6)

每个测试都要 fail-on-current-code,pass-on-fix,作为 review 的硬指标:

| # | 测试名 | 文件 | pin 的 bug |
|---|--------|------|-----------|
| T1 | `streaming_function_call_response_reports_tool_use_stop_reason` | `src/conversion/responses_stream.rs` | P0-1 |
| T2 | `stream_finalizes_on_response_completed_without_done_sentinel` | `src/providers/copilot.rs` (或 openai_responses.rs) | P0-2 |
| T3 | `function_call_args_with_mismatched_output_index_routes_by_item_id` | `src/conversion/responses_stream.rs` | P0-3 |
| T4 | `truncate_user_handles_multibyte_user_id_without_panic` | `src/conversion/responses.rs` | P0-4 |
| T5 | `response_created_with_null_usage_decodes_and_finalizes_cleanly` | `src/responses.rs` + `src/conversion/responses_stream.rs` | P0-5 |
| T6 | `text_block_without_deltas_emits_done_text_as_fallback_delta` | `src/conversion/responses_stream.rs` | P1-1 |

---

## 验证标准 (Verification Standard)

每次提交必须通过的硬指标:

```bash
# 1. 类型检查 + clippy
cargo check
cargo clippy --all-targets -- -D warnings

# 2. 全部测试通过
cargo test --lib
cargo test --test '*'

# 3. 新增测试覆盖本轮所有 P0 修复
#    T1 ~ T5 必须存在且全绿;T6 至少存在(允许 fail-on-current 作 TDD 标记)
cargo test --lib -- --nocapture streaming_function_call_response_reports_tool_use_stop_reason
cargo test --lib -- --nocapture stream_finalizes_on_response_completed_without_done_sentinel
cargo test --lib -- --nocapture function_call_args_with_mismatched_output_index_routes_by_item_id
cargo test --lib -- --nocapture truncate_user_handles_multibyte_user_id_without_panic
cargo test --lib -- --nocapture response_created_with_null_usage_decodes_and_finalizes_cleanly

# 4. 覆盖率不下降
cargo llvm-cov --summary-only
# 行覆盖率 ≥ 98.34% (历史 baseline)
# 不新增未覆盖的 panic 分支(expect_variant! 统一消息机制保证)

# 5. 不退化 wiremock EOF 路径的现有测试
# (db0e67b 引入的 stream_responses_converts_sse_to_anthropic_for_gpt5 等)
```

**审查红线**:
- 任何新 mock 都必须基于真实 wire 抓包或公开 wire contract,**不允许**为通过测试
  伪造 fixture 数据。
- 修复必须保持 wire format 兼容(v1 抓包可作为参考)。
- 失败用例必须 fail-on-current-code,不允许"修复同时改测试绕过"。
- commit message 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md`(用户未确认前 untracked)。

---

## 实施顺序 (本轮建议 5 个 commit)

1. `fix(responses-stream): track tool_use stop_reason for function_call streams` (P0-1 + T1)
2. `fix(responses-stream): finalize inline on terminal SSE event, not only [DONE]` (P0-2 + T2)
3. `fix(responses-stream): route function-call args by item_id, not output_index` (P0-3 + T3)
4. `fix(conversion): safe-multibyte truncate_user + tolerate null usage` (P0-4 + P0-5 + T4 + T5)
5. `fix(responses-stream): emit snapshot text on done when no deltas arrived` (P1-1 + T6) — 可选

P1-2 ~ P1-5 留到 Round 2+。

---

## 后续轮次预览

- **Round 2**: P1-2 / P1-3 (chat path gpt-5 行为统一 + 双发 max_tokens 修复)
- **Round 3**: P1-4 (summarizer hoist)
- **Round 4**: P1-5 (幻影 block 防护)
- **Round 5**: 综合 review + 覆盖率回归 + 文档同步

如果 review agent 在 Round 2+ 找到新 P0,优先插入修复,后续 P1 顺延。