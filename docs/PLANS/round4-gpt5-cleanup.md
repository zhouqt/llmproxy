# Round 4: GPT-5.x Clean 分支 — 修复 N1 路由回归 + 收尾 delta 端幻影 block

**分支**:`fix/copilot-gpt5-compat` (HEAD: `6d97bf1`)
**review 来源**:opus review agent (Round 4)
**前置**:`docs/PLANS/round1-gpt5-cleanup.md`、`round2-`、`round3-`

R3 修复完了 `Error` SSE 的 wire-shape 缺陷,本轮 opus 找到 **本分支第一个
真回归** —— `gpt5_family` 把 o-series 也路由到 Copilot `/responses`,
但 Copilot 大概率只在 `/responses` 上服务 gpt-5.x 而非 o-series,
o-series rewrite 会撞 `unsupported_api_for_model` 400。同时 P2-1 / P2-2
升级到 P1:delta 端的幻影 block 是协议级违规,`Error` 反序列化韧性缺失
会让上游错误吞掉。

---

## 必须修复 (P1 — N1 是本分支唯一回归)

### N1: o-series 误路由到 `/responses`
**症状**:Commit `18f3b8d` 把 `copilot.rs:37` 的 `endpoint_for_model` 从
`model.starts_with("gpt-5")` 改为 `util::gpt5_family(model)`。后者把
`o1*`/`o3*`/`o4*` 也归入"GPT-5 家族"。但 **Copilot 在 `/responses` 上的
支持范围是 gpt-5.x only**,o-series(`o1`/`o3-mini`/`o4-mini`)只能走
`/chat/completions`。当用户把 `o3-mini` 配在 `model_rewrite`,代理会把它
送到 `/responses`,Copilot 回 `unsupported_api_for_model`,**且因为这是
"模型不支持"400**,路由器会把它当 `is_model_unsupported` 跳过该 provider,
且没有 fallback 链 → 客户端 400。

**验证证据**:`endpoint_for_model_classifies_by_prefix` (copilot.rs:1483-1506)
**没**有 o-series 用例 —— 这就是为什么 regression 溜过去了。

**根因**:把"请求塑形谓词"(max_completion_tokens / 24h 升级 — o-series 同样需要)
和"端点路由谓词"(Copilot `/responses` 仅服务 gpt-5.x — o-series 不该走)合并到
一个 `gpt5_family` 里。`request.rs` / `responses.rs` 的 3 处调用对,只有
`copilot.rs:37` 这一处不对。

**修复**:
```rust
// src/providers/copilot.rs:36-42
fn endpoint_for_model(model: &str) -> &'static str {
    // Copilot /responses 仅支持 gpt-5.x;o-series 走 /chat/completions
    // 同样要被 reject (Copilot 暂时不支持)。如果未来 Copilot 把 o-series
    // 也开进 /responses,改这一行 + 加 o-series endpoint 测试即可。
    // request.rs / responses.rs 继续用 util::gpt5_family 做 max_tokens /
    // 24h 升级 —— o-series 在那两个场景下确实需要相同处理。
    if model.starts_with("gpt-5") {
        "responses"
    } else {
        "chat_completions"
    }
}
```

**配套**:
- `endpoint_for_model_classifies_by_prefix` 加 o-series 断言:
  - `assert_eq!(endpoint_for_model("o3-mini"), "chat_completions");`
  - `assert_eq!(endpoint_for_model("o4-mini"), "chat_completions");`
  - `assert_eq!(endpoint_for_model("o1"), "chat_completions");`
- `util::gpt5_family` 保持现状(给 request.rs / responses.rs 用),
  补一行 doc comment 说明 **它不用于 endpoint routing**。

### P2-1 → P1: delta 端幻影 block
**症状**:`ResponseOutputTextDelta` (`responses_stream.rs:183`) 和
`ResponseFunctionCallArgumentsDelta` (`:224`) 都调 `allocate_block`。
对从未见过 `output_item.added` 的 `output_index`,**凭空开 block 发
content_block_delta**,然后 `finalize()` 又给它发 `content_block_stop`
—— Anthropic SDK 收到"无 start 的 delta / stop",协议校验失败抛
SDK 异常,整条流被截断。

R3 的 P1-5 修了 `*.done` 端(`:191-194`、`:242-245` 的 `block_map.get`
guard),但 delta 端没修。

**修复**:
```rust
// responses_stream.rs:183 (text delta) 和 :224 (fc args delta)
let Some(&block_idx) = self.block_map.get(&key) else {
    tracing::warn!(?key, "delta event for unseen block; ignoring");
    continue;
};
let block = &mut self.blocks[block_idx];
// ... 原 delta 应用逻辑
```
对 text delta,key 就是 `output_index`。对 fc args delta,key 优先用
`fc_item_index.get(item_id)`,fallback 用 raw `output_index`(若 fallback
也 miss → 同样 warn + skip,不要 `allocate_block`)。

**测试**:
- T21: `output_text_delta_with_unseen_output_index_emits_nothing`
- T22: `function_call_arguments_delta_with_unknown_item_id_emits_nothing`
  (覆盖 fc_item_index 查不到且 raw output_index 也查不到的双 miss 场景)

### P2-2 → P1: `Error` 反序列化韧性
**症状**:`ResponsesStreamEvent::Error { message: String }` (responses.rs:297)
的 `message` 字段必需、非可空。两种情况会触发 serde 失败 → 落入
`Err` 分支(`openai_responses.rs:226-228`)→ debug 日志记录"malformed"
→ **静默跳过**,后续 EOF finalize 让消息"成功 end_turn"完成 —— 上游
错误被完全吞掉,R3 的 Error SSE 修复等于失效:

1. 上游发 `{"type":"error","error":{"code":"x","message":"y"}}`
   (Assistants API / 部分 OpenAI 兼容栈的扁平+error 子对象格式)
2. 上游发 `{"type":"error","message":null}` 或省略 `message`
3. `#[serde(other)]` **不救** 内容字段失败,只救未知 tag。

**修复**:
```rust
// src/responses.rs:294-300
#[serde(rename = "error")]
Error {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    param: Option<String>,
    #[serde(default, flatten)]
    extra: Value,
},
```
translator 端 (`responses_stream.rs:267` 区域) 把 Error arm 改成
read `message` 字段:
- 若 `message.is_some()` → 用它
- 否则从 `extra["error"]["message"]` 提取
- 再否则 fallback `"upstream error"`
然后正常 emit `event: error` envelope。

**测试** T23: 三种 fixture 各跑一遍,断言 adapter 收到 Error 事件并
terminate(`finished = true`,无后续 `message_stop`)。
- `{"type":"error","message":"oops"}`
- `{"type":"error","error":{"message":"nested"}}`
- `{"type":"error"}` (空 body)

---

## 应该修复 (P2 — 设计稳健性)

### P2-3 (cleanup): 删 `raw_error_event` / `take_raw_error` 旁路
**症状**:R3 commit `7dcdcb3` 在 translator 上加了 `raw_error_event`
字段 + `take_raw_error` 方法,目的是"emit 字节串,绕开 `StreamEvent::Error`
encode"。但 `StreamEvent::Error` (anthropic.rs:401-402) 已经有
`#[serde(tag="type", rename_all="snake_case")]`,**本身就 emit
`{"type":"error",...}`**。旁路完全多余,且造成 N4 — envelope 在两处
构建 (`server.rs:176-184` vs `responses_stream.rs:269-280`),drift 风险。

**修复**:删 `raw_error_event` / `take_raw_error`,translator 直接
`push(StreamEvent::Error { error: json!({...}) })`,让 `encode()` 走
正常路径。改 T14 (现在断言 wire bytes) → 改断言 `output_buffer` 里
有正确的 `event: error\ndata: {"type":"error",...}` 字节串(用
encode 的实际产物,不是手写)。

### P2-3 (行为): `response.failed` 上游错误透传
**症状**:`response.failed` 事件在 `responses_stream.rs:255-266` 被映射
为 `end_turn`,`ResponsesResponse.extra` (responses.rs:135-136) 里
flatten 的 `error: {code, message}` 信息从未被读 → 客户端收到"空回复 +
end_turn"。

**修复**:translator 在 `response.failed` 分支:
- 从 event 的 `extra["error"]` 提取 `code` / `message`
- emit `StreamEvent::Error { error: { type_: "upstream_error", message } }`
- `finalized = true`(避免 finalize 再发 message_delta)
- adapter 收到后 `finished = true; return;`

非流式路径 (`conversion/responses.rs:377-386`):同样从 `extra.error`
提取,构造 `ProxyError::Upstream { status: 502, body: message }`。

**测试** T24: `response_failed_event_surfaces_error_details_not_end_turn`,
feed `response.failed` + extra.error fixture,断言 emit Error 事件而非
end_turn。

### N2: cooldown warn 用 summarize_for_log
**症状**:`cooldown.rs:80` 仍用 `truncate_for_log(reason, 200)`(只切长度
不剥 HTML)。GitHub 502 "Unicorn!" 页直接灌进 warn 日志。

**修复**:`truncate_for_log(reason, 200)` → `crate::util::summarize_for_log(reason, "<empty body>")`。

### 卫生:删重复段落 + 修陈旧行号引用
**症状**:R3 commit `29ccec6` 注释清理漏了 `responses_stream.rs:70-82`
的重复"intentional"段;test 注释里 7+ 处行号引用在后续 commit 后失准。

**修复**:
- 删 `responses_stream.rs:70-82` 的重复段,保留 `:65-69` 一份。
- 把 test 注释里"line 79"、"lines 118-121"之类改成"the assert at"
  短语,行号不引用。

---

## 测试新增/调整

| # | 测试名 | 文件 | pin |
|---|--------|------|-----|
| T21 | `output_text_delta_with_unseen_output_index_emits_nothing` | `src/conversion/responses_stream.rs` | P2-1 |
| T22 | `function_call_arguments_delta_with_unknown_item_id_emits_nothing` | `src/conversion/responses_stream.rs` | P2-1 |
| T23 | `error_event_with_missing_or_nested_message_still_terminates` | `src/conversion/responses_stream.rs` + `src/responses.rs` | P2-2 |
| T24 | `response_failed_event_surfaces_error_details_not_end_turn` | `src/conversion/responses_stream.rs` | P2-3 behavior |
| T25 | `endpoint_for_model_routes_o_series_to_chat_completions` | `src/providers/copilot.rs` | N1 |

**调整现有 T14** (`error_event_envelope_uses_type_not_type_under_score`):
P2-3 cleanup 后 envelope 改走 `StreamEvent::Error` 编码,测试断言改成
parse `event: error` 字节串的 data 行,验证 schema 等价 Anthropic SDK
期望(用 `serde_json::from_str::<serde_json::Value>` 解析 + 结构断言)。

---

## 验证标准

```bash
cargo check --all-targets 2>&1 | grep -E "warning|error"  # 只允许 anthropic.rs:489 pre-existing
cargo test --lib                                            # 全绿
cargo test --lib -- output_text_delta_with_unseen_output_index_emits_nothing
cargo test --lib -- function_call_arguments_delta_with_unknown_item_id_emits_nothing
cargo test --lib -- error_event_with_missing_or_nested_message_still_terminates
cargo test --lib -- response_failed_event_surfaces_error_details_not_end_turn
cargo test --lib -- endpoint_for_model_routes_o_series_to_chat_completions
cargo test --lib -- error_event_envelope_uses_type_not_type_under_score   # 调整后
cargo test --lib -- unauthorized_responses_refreshes_and_retries_once   # 回归
cargo test --lib -- finalize_after_error_event_emits_nothing             # 回归
cargo test --lib -- adapter_terminates_immediately_after_terminal_event_no_hang   # 回归
```

**红线**:
- **不允许**在生产代码加 `unwrap()`(T23 的 fallback 要显式 `Option` 处理)。
- **不允许**改 `gpt5_family` 的语义或测试基线(N1 修复只动 endpoint 路由,
  不动 util 函数,确保其他 3 处调用继续对 o-series 做正确塑形)。
- 不加 `Co-Authored-By`。
- 不提交 `CLAUDE.md`。
- 383 个现有测试必须保持绿色。

---

## 实施顺序 (本轮 5 个 commit)

1. `fix(copilot): route o-series to /chat/completions; keep gpt5_family for request shaping` (N1 + T25)
2. `fix(responses-stream): guard delta allocation against unseen blocks` (P2-1 + T21 + T22)
3. `fix(responses): tolerant Error decode for missing/nested message field` (P2-2 + T23)
4. `fix(responses-stream): remove raw_error_event bypass; surface response.failed error details` (P2-3 cleanup + P2-3 behavior + 调 T14 + T24)
5. `fix(cooldown): HTML-strip via summarize_for_log; clean stale line refs` (N2 + hygiene)

---

## 后续轮次预览 (Round 5)

- 综合 review
- `cargo llvm-cov` 覆盖率回归(≥ 98.34% line baseline)
- PR #5 description 更新 + plan docs 交叉链接
- 任何 R4 漏掉的 opus finding
- `BlockRegistry` 抽象(`N7`)— 如果时间允许;否则留作 follow-up
- 任何用户/运营在 R4 测试中发现的问题