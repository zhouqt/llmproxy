# Round 3: GPT-5.x Clean 分支 — 修 R2 残留缺陷 + 抽象共享代码

**分支**:`fix/copilot-gpt5-compat` (HEAD: `fe928b1`)
**review 来源**:opus review agent (Round 3)
**前置**:`docs/PLANS/round1-gpt5-cleanup.md`、`docs/PLANS/round2-gpt5-cleanup.md`

R2 的 `Error` SSE 实现(commit `6e0c49d`)被发现 **3 处缺陷**:
1. emit 的 `event: error` envelope 里用了 `"type_"` 而不是 `"type"`,Anthropic SDK pydantic 校验会拒;
2. `Error` match arm 写了两遍,产生 unreachable_patterns 警告;
3. `finalize()` 没设 `finalized`,Error 之后被再次 finalize 会重发 `message_delta` + `message_stop`。

同时 Round 1/2 推迟的 summarizer 抽取发现 **第三个调用点** —— `openai_compat.rs:225`
仍然把原始 payload 打到 debug 日志,R2 修过的"HTML 502 灌爆日志"问题在 Chat
Completions 路径还活着。

本轮同时清理几处遗留:陈旧文档、`deltas_seen` 多 part 行为固化、
`finalize()` 幂等、`/responses` 401-retry 测试覆盖。

---

## 必须修复 (P0 — R2 残留缺陷)

### P1-E: `event: error` envelope 用错字段名
**症状**:`responses_stream.rs:251` 和 :264 emit 的 JSON 是:
```json
{"type_": "upstream_error", "message": "..."}
```
但 Anthropic wire format 要求(参考 `server.rs:175-183` 的 `format_stream_error`):
```json
{"type": "error", "error": {"type": "upstream_error", "message": "..."}}
```
即外层 SSE envelope 的 `event: error` + 内层 data 是 `{"type":"error", ...}`,
**`error.type`**(不是 `error.type_`)。Claude Code SDK pydantic 校验失败会
把 error 当成无效消息忽略,导致用户看不到错误。

更糟糕:T10 测试 (`openai_responses.rs:1271`) 断言 `contains("\"type_:\"")`,
**把错误锁住了** —— 直接删测试、改代码不够,必须同步修测试断言。

**根因**:commit `6e0c49d` 写 Error 处理时用了 `StreamEvent::Error` 的字段名
(`type_`)作为 JSON key,没意识到 SDK 期望的是 wire-level `type`。

**修复**:
```rust
// responses_stream.rs:248-260 区域
ResponsesStreamEvent::Error { code, message, .. } => {
    let kind = "upstream_error";
    let payload = serde_json::json!({
        "type": "error",
        "error": {
            "type": kind,
            "message": message,
            // code 可选字段
        }
    });
    let encoded = format!("event: error\ndata: {}\n\n", payload);
    out.push_bytes(encoded);  // 或等价 push 到 output_buffer
    self.finalized = true;
    return out;
}
```
- 直接 emit wire 字节串,绕开 `StreamEvent::Error` 的内部表示差异。
- T10 改断言:`contains("\"type\":\"error\"")` 且 `contains("\"type\":\"upstream_error\"")`
  且 envelope 头是 `event: error`。

### P1-F: 重复 `Error` match arm → unreachable_patterns 警告
**症状**:`responses_stream.rs:248-260` 和 `:261-273` 字节级重复,是
commit `6e0c49d` 的 merge artifact。`cargo check --all-targets` 输出
`warning: unreachable_pattern`,build 不干净。

**修复**:删 `:261-273` 的重复 arm。顺便检查 `StreamEvent::Error` 自身的
struct 字段(`error` 子结构),确保 encode 路径单一。

### P1-G: `finalize()` 在 Error 后可以重发
**症状**:`ResponsesStreamTranslator::finalized` 字段(responses_stream.rs:56)
只被 Error arm 和 adapter 检查。`finalize()` 函数本身 (responses_stream.rs:279)
不检查 `finalized`,直接 emit `message_delta` + `message_stop` + 收尾
`content_block_stop`。如果 Error 已经 emit 了 `event: error`,然后 EOF finalize
再跑一遍 → 客户端看到 `event: error` 后又收到 `message_delta`+`message_stop`,
逻辑冲突。

**修复**:
```rust
pub fn finalize(&mut self) -> Vec<StreamEvent> {
    if self.finalized {
        return Vec::new();
    }
    self.finalized = true;
    // ... 原 finalize body 不变
}
```

---

## 应该修复 (P1)

### P1-4 (Round 1 deferred, expanded): summarizer 抽取 + 第三个调用点
**症状**:`summarize_http_body` 在 `copilot.rs:55-132`,`summarize_for_log` 在
`openai_responses.rs:222-296`,**加上** `openai_compat.rs:225` 把原始 payload
直接打到 debug log —— 第三处没修的"HTML 错误页灌爆日志"漏洞,OpenAI Chat
Completions 路径(Copilot 也走)同样会触发。

**修复**:
1. `src/util.rs` 加:
   ```rust
   pub fn summarize_for_log(input: &str, empty_placeholder: &str) -> String { ... }
   ```
   单一实现,接受占位符参数。
2. `copilot.rs:55-132` 的 `summarize_http_body` 改成 `crate::util::summarize_for_log(text, "<empty body>")` 的 thin wrapper(或直接 inline 调用,删 wrapper)。
3. `openai_responses.rs:222-296` 的 `summarize_for_log` 删除,调用点改 `crate::util::summarize_for_log(payload, "<empty payload>")`。
4. `openai_compat.rs:225` 改 `crate::util::summarize_for_log(payload, "<empty payload>")`,**这是本轮新修的洞**。
5. 在 `src/util.rs` 加单元测试覆盖 HTML、纯文本、空 body、多字节 4 个 case,
   给 hoisted 函数本身一份独立 coverage。

### P1-C: `/responses` 401-refresh-retry 测试
**症状**:`unauthorized_chat_refreshes_and_retries_once` 只覆盖
`/chat/completions` 路径;GPT-5 生产路径走 `/responses`,token 401 后的
refresh-and-retry 行为在该路径**没有测试**。

**修复**:在 `src/providers/copilot.rs` 的 `#[cfg(test)] mod tests` 加
`unauthorized_responses_refreshes_and_retries_once`,镜像 chat 路径测试:
- mock server: 第一次 `/responses` 返回 401,第二次返回 200 + 合法 body。
- 断言 provider 调用恰好 2 次,第二次成功。

### P1-F-ext: stale 文档 + openai.rs:34-36 注释
**症状**:`src/openai.rs:34-36` 的 doc comment 写 "We emit both so the
upstream picks whichever it recognizes" —— Round 2 改成 XOR 之后这段就过期了。

**修复**:删/重写为反映当前 XOR 行为。

### P1-D: `is_terminal_event` 文档
**症状**:R2 引入 P0-A 修复后,`response.completed` 后 adapter 不再 poll
reqwest。任何"completed 后还有数据"都被丢。这是有意的,但代码没注释。

**修复**:在 `responses_stream.rs:61-68`(`is_terminal_event` 定义)上方加
注释,说明"terminal → finalize + 关闭 adapter → 上游 tail 丢弃"是有意行为。

### P1-1 边界固化 (文档 + 测试)
**症状**:`deltas_seen` 按 block index key,多 part item 一 part 有 delta
就屏蔽其他 part 的 snapshot fallback。Copilot 实际只发单 part item,
可接受;但需要固化(测试)而不是口头保证。

**修复**:在 `responses_stream.rs` 加测试 `multipart_item_snapshot_fallback_invariant`,
明确 pin 当前行为:"多 part item 中只要任一 part 有 delta,该 item 内
其他 part 的 done snapshot 不再补发 delta"。

---

## 测试新增/调整

| # | 测试名 | 文件 | pin |
|---|--------|------|-----|
| T14 | `error_event_envelope_uses_type_not_type_under_score` | `src/providers/openai_responses.rs` | P1-E |
| T15 | `finalize_after_error_event_emits_nothing` | `src/conversion/responses_stream.rs` | P1-G |
| T16 | `finalize_called_twice_emits_nothing_second_time` | `src/conversion/responses_stream.rs` | P1-G |
| T17 | `unauthorized_responses_refreshes_and_retries_once` | `src/providers/copilot.rs` | P1-C |
| T18 | `summarize_for_log_in_util_handles_html_plain_empty_multibyte` | `src/util.rs` | P1-4 hoist |
| T19 | `openai_compat_malformed_sse_payload_is_summarized_in_debug_log` | `src/providers/openai_compat.rs` | P1-4 third site |
| T20 | `multipart_item_snapshot_fallback_invariant` | `src/conversion/responses_stream.rs` | P1-1 固化 |

**调整现有 T10** (`midstream_error_event_surfaces_as_anthropic_error_and_terminates`):
- 删 `contains("\"type_:\"")` 断言
- 改为 `contains("\"type\":\"error\"")` + `contains("\"type\":\"upstream_error\"")`
- 保留 adapter 在 error 后终止的断言

---

## 验证标准

```bash
cargo check --all-targets 2>&1 | grep -E "warning|error"  # 只允许 anthropic.rs:489 pre-existing
cargo test --lib                                            # 全绿
cargo test --lib -- error_event_envelope_uses_type_not_type_under_score
cargo test --lib -- finalize_after_error_event_emits_nothing
cargo test --lib -- finalize_called_twice_emits_nothing_second_time
cargo test --lib -- unauthorized_responses_refreshes_and_retries_once
cargo test --lib -- summarize_for_log_in_util_handles_html_plain_empty_multibyte
cargo test --lib -- openai_compat_malformed_sse_payload_is_summarized_in_debug_log
cargo test --lib -- multipart_item_snapshot_fallback_invariant
cargo test --lib -- midstream_error_event_surfaces_as_anthropic_error_and_terminates   # 改后
# 回归
cargo test --lib -- streaming_function_call_response_reports_tool_use_stop_reason
cargo test --lib -- adapter_terminates_immediately_after_terminal_event_no_hang
cargo test --lib -- truncate_user_handles_multibyte_user_id_without_panic
cargo test --lib -- fc_args_done_with_unknown_item_id_does_not_allocate_phantom
```

**红线 (继承 R1+R2)**:
- 不允许伪造 wire fixture;T14 必须 parse emit 出的 JSON,验证 schema 真符合 Anthropic error shape。
- 不允许在生产代码加 `unwrap()`。
- 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md`。
- 现有 379 个测试必须保持绿色。

---

## 实施顺序 (本轮 5 个 commit)

1. `fix(responses-stream): emit Anthropic-shape error envelope (type not type_)` (P1-E + T14 + 改 T10)
2. `fix(responses-stream): idempotent finalize + remove duplicate Error arm` (P1-F + P1-G + T15 + T16)
3. `feat(util): hoist summarize_for_log + fix third call site in openai_compat` (P1-4 + T18 + T19)
4. `test(copilot): cover /responses 401-refresh-retry path` (P1-C + T17)
5. `docs(refactor): stale comments + multipart item invariant pin` (P1-D + P1-1 + T20)

---

## 后续轮次预览

- **Round 4**: P2 — args delta phantom allocation (mirror done-side guard);
  `Error` 变体 deserialization 韧性(`message` 缺失时 fallback);`response.failed`
  上游错误信息透传(`StreamEvent::Error` 而不是 `end_turn`)。
- **Round 5**: 综合 review + 覆盖率回归(`cargo llvm-cov` ≥ 98.34%)+
  PR #5 description 更新(列出 16 commits 的关键修复)+ plan docs 交叉链接。