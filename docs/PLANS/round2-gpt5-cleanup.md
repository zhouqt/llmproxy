# Round 2: GPT-5.x Clean 分支 — 修复 Round 1 回归与设计加固

**分支**：`fix/copilot-gpt5-compat` (HEAD: `1754569`)
**review 来源**：opus review agent (Round 2)
**前置**：`docs/PLANS/round1-gpt5-cleanup.md` (Round 1 计划已实施)

Round 1 的 P0-2 实现被 opus 找到 **回归** —— 终端事件 inline finalize 后忘了
`self.finished = true; return;`,导致客户端虽然收到了 `message_stop`,但 adapter
继续从 reqwest 拉上游字节,直到 600s 超时后发出伪 `event: error`。本轮优先
修复这个回归,然后处理三个被推迟的 P1 设计项,加上一个新增的中游错误事件处理。

---

## 必须修复 (P0 — Round 1 回归)

### P0-A: 终端事件 finalize 后没置 `finished` → 600s 后伪 error
**症状**：Copilot 在 `response.completed` 后不发 `[DONE]` 也不关 TCP。
Round 1 修复 (`49f8d59`) 在 `openai_responses.rs:209-215` 调用了
`translator.finalize()` 把 `message_delta` + `message_stop` 排进 buffer,但
**没有**在 buffer 排完后 `self.finished = true; return;`。后果:
1. `poll_next` 排空 buffer,看到 `finished == false`,继续 poll reqwest 字节流。
2. Copilot 保持连接 → reqwest 永远不返回 → 600s 后 (proxy_client.rs:36-41 默认)
   reqwest 整请求超时 → `poll_next` 返回 `Err`。
3. `MappedStream` 把这个 `Err` 变成 `event: error` 推给客户端 —— 但实际
   响应早就正常结束了,这是 600 秒之后的伪错误。

**根因**：Round 1 plan 写的 "不要因为 terminal 就关连接" 被 haiku 实现误读
为"不要 `finished = true`"。v1 (commit `4aa3d49`) 的本意是"不要主动关上游
TCP" —— `self.finished = true` 关的是 adapter(停止 poll reqwest),底层连接
由 `self.inner` drop 时自然释放,跟 `finished` 无关。

**修复**：
```rust
// src/providers/openai_responses.rs:209-215 区域
if crate::conversion::responses_stream::is_terminal_event(&ev) {
    if let Some(mut t) = self.translator.take() {
        for ev in t.finalize() {
            self.output_buffer.push_back(Self::encode(&ev));
        }
    }
    self.finished = true;   // ← 缺失的一行
    return;                // ← 缺失的一行,跳出 process_lines
}
```

**为什么这是正确的**: `self.finished = true` 让 `poll_next` 在 buffer 排空后
直接返回 `Poll::Ready(None)`,adapter 被 drop → `self.inner` (reqwest
`bytes_stream`) 被 drop → 上游 TCP 连接释放。**完全不需要手动关连接**,
adapter 的生命周期自然管理。

**验证**:
- 现有 T2 (`stream_finalizes_on_response_completed_without_done_sentinel`)
  改强:不仅断言 `message_stop` 在 200ms 内出现,还要断言 adapter 在
  `message_stop` 后 **立刻** 返回 `None`(不是等超时)。
- 新增 T7: `adapter_terminates_immediately_after_terminal_event_no_hang`,
  跑 5s 不退出的 wiremock 流,断言 adapter 1s 内返回 `None`。

---

## 应该修复 (P1 — 设计稳健性)

### P1-3 (re-rank from Round 1): `max_tokens` XOR `max_completion_tokens`
**症状**:`src/conversion/request.rs:59-60` 在所有 Chat Completions 请求上
**同时**发 `max_tokens` 和 `max_completion_tokens`。OpenAI 对 o-series /
gpt-5 严格校验,带 `max_tokens` 会 reject 400。其他模型接受 `max_tokens`,
不识别 `max_completion_tokens`。双发在两个方向都有模型踩坑。

**修复**:引入 `gpt5_family()` 谓词(P1-2 同步做),在 `request.rs:59-60`:
- `gpt5_family(model)` → 只发 `max_completion_tokens`
- 其他 → 只发 `max_tokens`
- 测试覆盖两组互斥。

### P1-2: gpt-5 谓词统一 + chat path 24h 升级
**症状**:
- `model.starts_with("gpt-5")` 重复在 `copilot.rs:37` 和
  `conversion/responses.rs:110`。
- `conversion/request.rs:85-86` 转发 cache hints **没有** `in_memory`→`24h`
  升级,Chat Completions 路径给 Copilot gpt-5 发 24h 限定之外的 retention
  → 上游 400 或 cache miss。

**修复**:
- 新增 `src/util.rs` 单一谓词:
  ```rust
  pub fn gpt5_family(model: &str) -> bool {
      model.starts_with("gpt-5") || model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4")
  }
  ```
  (o-series 与 gpt-5 共享 24h 限制和 max_tokens 限制)
- `copilot.rs:37` 与 `conversion/responses.rs:110` 改调 `util::gpt5_family`。
- `conversion/request.rs` 添加同款 `in_memory`→`24h` 升级逻辑
  (参考 responses.rs:109-115)。

### P1-5: done handler 给未见过的 index 创建幻影 block
**症状**:`*.done` handler 调用 `allocate_block`:
- `responses_stream.rs:167` (text done)
- `responses_stream.rs:215` (fc args done)
- 加上 P0-3 留下的 `fc_item_index` fallback:args 事件的 item_id 未见过
  时 fall back 到 raw output_index,然后 `allocate_block`。

→ 对一个从未发出 `output_item.added` 的 index 凭空开块、立刻关块,客户端
看到无 start 的 stop。

**修复**:done handler 先查 `block_map.get()`:
```rust
let Some(&block_idx) = self.block_map.get(&key) else {
    tracing::warn!(?key, "done event for unseen block; ignoring");
    return;
};
let block = &mut self.blocks[block_idx];
// ... continue with the stop logic
```
适用 text done / fc args done 两处。fc args done 的 key 是
`(item_id, output_index)` 或 `item_id`(取决于 P0-3 的映射)。

**测试**:T8 `done_event_for_unseen_output_index_is_ignored`,
         T9 `fc_args_done_with_unknown_item_id_does_not_allocate_phantom`。

### P1-B: 中游 `error` SSE 事件被吞
**症状**:`src/responses.rs:217-292` 的 `ResponsesStreamEvent` 枚举没有
`error` 变体,任何上游 `{"type":"error",...}` 通过 `#[serde(other)]` 落到
`Unknown` → translator `push_event` 忽略 → adapter 不发任何事件给客户端。
如果流随后 stall,EOF finalize 会把空消息当成成功响应(`stop_reason: None`,
部分文本),客户端看到"截断但 success"。

**修复**:
- `ResponsesStreamEvent` 加 `Error { code: Option<String>, message: String,
  param: Option<String> }` 变体。`#[serde(rename = "error", tag = "type")]`
  或裸 `type: "error"`(看现有 tag 模式)。
- `ResponsesStreamTranslator::push_event` 处理 `Error`:emit
  `StreamEvent::Error { error: { type_: "upstream_error", message } }`,
  把 `final_stop_reason` 置为 None(让 EOF finalize 不发 message_delta),
  设 `finalized = true`(避免重复 emit)。
- adapter 收到 `Error` 立即 `finished = true; return;` (跟 terminal 同款)。
- **注意**:`StreamEvent::Error` 已在 anthropic.rs 存在,复用即可。

**测试**:T10 `midstream_error_event_surfaces_as_anthropic_error_and_terminates`。

---

## 测试缺失 (本轮新增)

| # | 测试名 | 文件 | pin 的项 |
|---|--------|------|---------|
| T7 | `adapter_terminates_immediately_after_terminal_event_no_hang` | `src/providers/openai_responses.rs` | P0-A |
| T8 | `done_event_for_unseen_output_index_is_ignored` | `src/conversion/responses_stream.rs` | P1-5 |
| T9 | `fc_args_done_with_unknown_item_id_does_not_allocate_phantom` | `src/conversion/responses_stream.rs` | P1-5 + P0-3 |
| T10 | `midstream_error_event_surfaces_as_anthropic_error_and_terminates` | `src/providers/openai_responses.rs` | P1-B |
| T11 | `chat_completions_emits_only_max_completion_tokens_for_gpt5` | `src/conversion/request.rs` | P1-3 |
| T12 | `chat_completions_emits_only_max_tokens_for_non_gpt5` | `src/conversion/request.rs` | P1-3 |
| T13 | `chat_completions_escalates_in_memory_to_24h_for_gpt5` | `src/conversion/request.rs` | P1-2 |

强化现有 T2: `stream_finalizes_on_response_completed_without_done_sentinel`
必须断言 adapter 在 `message_stop` 之后立刻返回 `None`(具体看代码,可能要在
外部用一个 `tokio::time::timeout` 框住第二次 `next()` 调用,超时则 fail)。

---

## 验证标准 (本轮)

```bash
cargo check                                          # 0 warnings (除已存在的 anthropic.rs:489)
cargo test --lib                                     # 全绿
cargo test --lib adapter_terminates_immediately_after_terminal_event_no_hang
cargo test --lib stream_finalizes_on_response_completed_without_done_sentinel
cargo test --lib done_event_for_unseen_output_index_is_ignored
cargo test --lib fc_args_done_with_unknown_item_id_does_not_allocate_phantom
cargo test --lib midstream_error_event_surfaces_as_anthropic_error_and_terminates
cargo test --lib chat_completions_emits_only_max_completion_tokens_for_gpt5
cargo test --lib chat_completions_emits_only_max_tokens_for_non_gpt5
cargo test --lib chat_completions_escalates_in_memory_to_24h_for_gpt5
cargo test --lib truncate_user_handles_multibyte_user_id_without_panic   # 回归
cargo test --lib streaming_function_call_response_reports_tool_use_stop_reason   # 回归
```

**审查红线** (继承 Round 1):
- 不允许为通过测试伪造 fixture;wiremock SSE 数据必须符合公开 Responses 协议。
- 不允许在生产代码里加 `unwrap()`。
- commit message 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md`。
- Round 1 的所有现有测试必须保持绿色。

---

## 实施顺序 (本轮 4 个 commit)

1. `fix(openai-responses): restore finished=true after terminal finalize` (P0-A + T7 + 强化 T2)
2. `feat(util): gpt5_family predicate + dedupe copilot/responses callers + chat 24h escalation` (P1-2 + T13)
3. `fix(conversion): emit max_tokens XOR max_completion_tokens per model family` (P1-3 + T11 + T12)
4. `fix(responses-stream): ignore done events for unseen blocks; handle upstream error SSE` (P1-5 + P1-B + T8 + T9 + T10)

---

## 后续轮次预览

- **Round 3**: P1-4 (summarizer hoist `summarize_http_body` / `summarize_for_log` 到 util)
- **Round 4**: P1-C (`/responses` 401-retry 测试覆盖);P1-1 多 part 边界 (文档化)
- **Round 5**: 综合 review,覆盖率回归,文档同步,清理任何残留 dead_code

Round 2 之后的 opus review 可能发现新的 P0,优先插入修复,后续 P1 顺延。