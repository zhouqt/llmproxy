# Round 5: GPT-5.x Clean 分支 — 收尾清理 + 合并准备

**分支**:`fix/copilot-gpt5-compat` (HEAD: `18d3aba`)
**review 来源**:opus review agent (Round 5 — final)
**前置**:`docs/PLANS/round1-gpt5-cleanup.md` ~ `round4-`

opus 给的 verdict:branch **功能上已可合并**,没有剩余的 P0/P1 正确性 bug。
但 R4 commit `6a70953` 把 `cooldown.rs` 的 `truncate_for_log` 换成
`summarize_for_log` 时 **没清理** 旧的常量和函数 + 三个 orphan unit test,
产生了 2 个 dead_code warning,**违反 R4 自己定的验证门槛**
(`"cargo check 只允许 anthropic.rs:489 pre-existing"`)。同时 R4 的
delta-side phantom block 防护 **漏了测试** T21/T22,出现回归时不会被发现。

本轮最后 3 个 commit 修这两件事,加一份整合性 summary 文档。

---

## 必须修复 (P2 — 违反自身验证标准)

### P2-1: cooldown.rs dead code 产生 2 个 warning
**症状**:
```
warning: constant `LOG_REASON_MAX_CHARS` is never used    (cooldown.rs:28)
warning: function `truncate_for_log` is never used        (cooldown.rs:33)
```

**根因**:R4 commit `6a70953` 把 `cooldown.rs:80` 的 `truncate_for_log(reason, 200)`
换成 `crate::util::summarize_for_log(reason, "<empty body>")`,但忘了:
- `LOG_REASON_MAX_CHARS: usize = 200;` (cooldown.rs:28) 现在没引用
- `truncate_for_log` 函数本身 (cooldown.rs:33-39) 现在只被 3 个 orphan
  unit test 引用
- 这 3 个 test (cooldown.rs:225-254) 测的是被废弃函数的实现细节,无价值

**修复**:
1. 删除 `src/cooldown.rs:28` (`const LOG_REASON_MAX_CHARS`)
2. 删除 `src/cooldown.rs:33-39` (`fn truncate_for_log`)
3. 删除 `src/cooldown.rs` 中的 3 个 `truncate_for_log_*` unit test
   (大约 `:224-254`),注释 `// no longer holds the upstream body — but \`truncate_for_log\` is`
   一并清理。
4. 验证 `cargo check --all-targets 2>&1 | grep warning` 现在只剩
   `unused variable: i` (anthropic.rs:489,pre-existing)。

**预计影响**:测试数 387 → 384(删 3 个 orphan test)。

### P2-2: R4 delta-side phantom block 防护缺测试
**症状**:R4 commit `0b05bef` 在 `responses_stream.rs:170-173` (text delta)
和 `:214-217` (fc-args delta) 加了 `block_map.get` 守卫,但只改了生产代码。
Plan 里的 T21 / T22 **从未被添加**:
- T21: `output_text_delta_with_unseen_output_index_emits_nothing`
- T22: `function_call_arguments_delta_with_unknown_item_id_emits_nothing`

后果:这两个守卫如果被未来 commit 误改回 `allocate_block`(破坏协议),
不会有 test 失败报警 —— 整个 R4 commit 的保护失效。

**修复**:在 `src/conversion/responses_stream.rs` 的 `#[cfg(test)] mod tests`
末尾加 2 个 test。每个 10-15 行:

```rust
#[test]
fn output_text_delta_with_unseen_output_index_emits_nothing() {
    let mut t = ResponsesStreamTranslator::new("msg_x", "gpt-5");
    // No output_item.added → block_map is empty.
    let out = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
        item_id: "msg_1".into(), output_index: 99, delta: "orphan".into(),
    });
    assert!(out.is_empty(), "must not allocate a phantom block; got {out:?}");
}

#[test]
fn function_call_arguments_delta_with_unknown_item_id_emits_nothing() {
    let mut t = ResponsesStreamTranslator::new("msg_x", "gpt-5");
    // Both fc_item_index miss AND output_index miss.
    let out = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
        item_id: "fc_unknown".into(), output_index: 42, delta: "{}".into(),
    });
    assert!(out.is_empty(), "must not allocate a phantom block; got {out:?}");
}
```
(具体字段以 `ResponsesStreamEvent` 枚举当前定义为准,如有差异照实际填。)

---

## 应该做 (P3 — 文档与可维护性)

### P3-1: 写 `docs/PLANS/gpt5-cleanup-summary.md`
**症状**:4 份 round plan 各描述各自的修复,但没人写过"从分支角度看,
整体修过哪些 bug、按什么顺序、当前状态"的整合文档。future maintainer
上手时只能 `git log` 一条条读。

**修复**:新增 `docs/PLANS/gpt5-cleanup-summary.md`:
- 时间线(R1-R4 各发现/修复了什么,opus review 提了什么建议但延后)
- 当前分支相对 `main` 的 26 commit 分类表(按 round 归组)
- 仍 open 的 follow-up items(N7 `BlockRegistry`、`8c06e5f` debug tracing、
  `response.failed` non-stream `Upstream` 错误码当前是 502 等)
- 与 v1 archive(`fix/copilot-gpt5-compat-v1`)的对比表(哪些 v1 commit
  被回填、哪些 v1 commit 被故意丢弃)

### P3-2: README.md 加 o-series routing 说明
**症状**:`README.md:291-296` 的 "GPT-5 routing" 段说 "the proxy
auto-routes any incoming model name starting with `gpt-5` to Copilot's
`/v1/responses` endpoint",但 R4 N1 修复后,**只有 gpt-5 才走 /responses;
o-series 仍然走 /chat/completions**(因为 Copilot /responses 不支持
o-series)。README 描述当前正确,但没说清楚 o-series 的处理。

**修复**:`README.md:291-296` 加一行:
> Note: only `gpt-5*` model names are routed to `/v1/responses`. Models
> from the `o-series` (`o1*`, `o3*`, `o4*`) continue to use
> `/v1/chat/completions`, since Copilot's Responses endpoint does not
> serve them. The proxy still applies the same request-shape
> adjustments (`max_completion_tokens` instead of `max_tokens`,
> `in_memory` → `24h` cache retention escalation) to o-series on the
> Chat Completions path.

### ~~P3-3: 更新 CLAUDE.md 关于 summarizer 的描述~~ **(不做)**
CLAUDE.md 是用户 untracked 的文件,本轮及之前所有 plan 都明示不提交
也不主动改。**保留 untracked 状态**;让用户自己决定是否需要更新。
本次 task 中不触碰该文件。

---

## 测试新增/删除

| 操作 | 测试名 | 文件 | 原因 |
|------|--------|------|------|
| 删 | `truncate_for_log_passes_through_short_strings` | `src/cooldown.rs` | P2-1 orphan |
| 删 | `truncate_for_log_truncates_long_strings_with_marker` | `src/cooldown.rs` | P2-1 orphan |
| 删 | `truncate_for_log_respects_utf8_char_boundaries` | `src/cooldown.rs` | P2-1 orphan |
| 加 | `output_text_delta_with_unseen_output_index_emits_nothing` (T21) | `src/conversion/responses_stream.rs` | P2-2 |
| 加 | `function_call_arguments_delta_with_unknown_item_id_emits_nothing` (T22) | `src/conversion/responses_stream.rs` | P2-2 |

预计净变化:测试数 387 → 386(+2 新 −3 旧)。

---

## 验证标准

```bash
cargo check --all-targets 2>&1 | grep -E "^warning|^error" | grep -v "unused variable: \`i\`"
# 必须空输出(只剩 anthropic.rs:489 的 pre-existing unused i)

cargo test --lib
# 386 passed; 0 failed

cargo test --lib -- output_text_delta_with_unseen_output_index_emits_nothing
cargo test --lib -- function_call_arguments_delta_with_unknown_item_id_emits_nothing
# 都必须 green

# 回归 spot-check(关键 P0 fix 不能退化):
cargo test --lib -- streaming_function_call_response_reports_tool_use_stop_reason
cargo test --lib -- adapter_terminates_immediately_after_terminal_event_no_hang
cargo test --lib -- truncate_user_handles_multibyte_user_id_without_panic
cargo test --lib -- fc_args_done_with_unknown_item_id_does_not_allocate_phantom
cargo test --lib -- unauthorized_responses_refreshes_and_retries_once
cargo test --lib -- error_event_envelope_uses_type_not_type_under_score
cargo test --lib -- endpoint_for_model_routes_o_series_to_chat_completions
```

**红线**:
- 不允许在生产代码加 `unwrap()`。
- 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md`(本轮 P3-3 主动放弃)。
- 不提交 `config.yaml`、`*.local.yaml`、`.env*`。
- 不修改任何 `docs/PLANS/round*.md`(它们是历史快照)。
- 不动 `fix/copilot-gpt5-compat-v1`(archive)。
- 387 个现有测试减去 3 个删、加 2 个新 = 386,必须保持全绿。

---

## 实施顺序 (本轮 3 个 commit)

1. `fix(cooldown): remove dead truncate_for_log + orphan tests after summarize_for_log swap` (P2-1)
2. `test(responses-stream): pin delta-side phantom-block guards` (P2-2 + T21 + T22)
3. `docs(plans): R1-R4 GPT-5 cleanup summary; README o-series note` (P3-1 + P3-2)

完成上述 3 个 commit 后,branch 进入合并候选状态。

---

## 合并后 follow-up items (不进本轮)

- **N7** `BlockRegistry` 抽象:把 `blocks` / `block_map` / `closed_blocks` /
  `deltas_seen` / `fc_item_index` 收成一个 struct。重构窗口应该是测试
  充分、修复沉淀后,不是合并前。
- **`8c06e5f` debug tracing**:v1 丢弃的 per-event debug 钩子。在 R3 Error
  wire-shape / R4 P2-2 这类 silent-drop 场景下,event-name-only debug
  log 仍然是最便宜的诊断手段。可以重做,挂在 `RUST_LOG=debug` 下。
- **`response.failed` non-stream 错误码**:目前固定 502 (`conversion/responses.rs:392`),
  实际 Copilot 可能给 4xx。后续从生产 trace 里收集 status 后决定。
- **`endpoint_for_model` 加 enum**:目前是 `&'static str` + 两个字符串字面量;
  Copilot 增加新端点时(比如 o-series 未来也上 /responses)改起来易漏。
  建议改成 `enum CopilotEndpoint { Responses, ChatCompletions }`,匹配
  失败时直接 panic in test。