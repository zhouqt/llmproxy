# DeepSeek thinking 修复：本轮不修的问题（deferred issues）

> 本文件记录"第三轮代码审查记录"（见 `implement-deepseek-thinking.md`）中明确**本轮不修**的问题，供后续轮次回头处理。本轮（P3/P4/P5/P7）已修复项不在本文件范围。

---

### P1 strip 后 messages 可能变空或连续同角色
- **严重度**：高
- **位置**：`src/providers/anthropic.rs:329`（`strip_thinking_blocks` 的整条消息删除分支）
- **问题**：strip 整条删除消息后，`messages` 数组可能变空或出现连续同角色消息，可能制造新的、更难懂的 400。
- **为什么不修**：计划 R-A 已接受 litellm 式整条删除，但无占位保护；当前无上游实测证明空/连续消息序列会被拒绝。
- **触发/修复条件**：若上游实测拒绝空/连续消息序列，需加 `{"type":"text","text":""}` 占位，或"清空则放弃 strip 直接透传"。

### P2 strip 非一次性自愈，每轮重复两次上游请求 + prompt cache 每轮 miss
- **严重度**：高
- **位置**：`src/providers/anthropic.rs`（`complete()` 整体）
- **问题**：跨模型历史每轮回传 thinking 块 → 每轮触发两次上游请求 + prompt cache 每轮 miss，会话越长代价越大。
- **为什么不修**：strip-and-retry 方案固有代价，属已知限制；上游（DeepSeek）当前行为可接受。
- **触发/修复条件**：若每轮双请求的成本不可接受，需 per-provider 记忆位预剥离（如按 provider 缓存"已确认不支持跨模型 signature"）。

### P6 U1 测试重复空断言制造假覆盖信号
- **严重度**：中
- **位置**：`src/providers/anthropic.rs:397-406`（`has_thinking_error_matches_anthropic_error_message`）
- **问题**：U1 追加的"DeepSeek 字面量"断言与 Anthropic 原生断言字节相同，零信息量重复，制造"已覆盖 DeepSeek"假信号。
- **为什么不修**：低优先级清理，不影响行为正确性。
- **触发/修复条件**：拿到真实 DeepSeek 抓包字面量时替换为真实字面量。

### P8 `has_thinking_error` 被调用两次
- **严重度**：低
- **位置**：`src/providers/anthropic.rs`（`is_thinking_400` 与 `should_strip_and_retry` 内各一次）
- **问题**：同一响应体对 `has_thinking_error` 做了两次子串扫描。
- **为什么不修**：纯可读性/微优化，无行为影响。
- **触发/修复条件**：下次路过 `complete()` 时顺手合并为一次调用并传参复用。

### P9 `complete()` 里 `api_key.clone()` 不必要
- **严重度**：低
- **位置**：`src/providers/anthropic.rs:182`（`complete()` 内 `let api_key = self.api_key.clone();`）
- **问题**：`api_key` 在循环内借用即可，无需 clone。
- **为什么不修**：微优化，属 clean-up 而非缺陷。
- **触发/修复条件**：任意一次对该文件的常规清理。

### P10 手工换行未经 rustfmt 验证
- **严重度**：低
- **位置**：`src/providers/anthropic.rs`（strip-retry 相关新增代码）
- **问题**：部分新增代码为手工换行，未在装有 rustfmt 的环境验证 `cargo fmt --check` 是否通过。
- **为什么不修**：环境缺 rustfmt，无法验证。
- **触发/修复条件**：在装有 rustfmt 的环境跑 `cargo fmt --check`，并按输出修正。

### P11 透传 raw body 后上游措辞可能意外命中 `is_model_unsupported`
- **严重度**：低
- **位置**：`src/providers/anthropic.rs`（`complete()` 透传 raw body 路径），涉及 `src/router.rs:25-49`
- **问题**：透传 raw body 后，若上游错误措辞含 "model" / "not supported" 会意外命中 `is_model_unsupported` 触发 fallback。
- **为什么不修**：当前 DeepSeek 字面量不命中，仅记录风险。
- **触发/修复条件**：出现"上游 error 文案含 `not supported`/`model` 但实为其他语义、被误判 fallback"的实测后再修。

### D1 U1 "DeepSeek 字面量"与 Anthropic 原生断言字节相同
- **严重度**：中
- **位置**：`src/providers/anthropic.rs:397-406`（`has_thinking_error_matches_anthropic_error_message`）
- **问题**：所谓"DeepSeek 字面量"与 Anthropic 原生断言字节相同，零信息量重复。
- **为什么不修**：与 P6 同源（同一段重复断言），一并低优先级处理。
- **触发/修复条件**：同 P6，拿到真实 DeepSeek 抓包字面量时替换。

### D2 完整 Step 0 诊断日志未实现
- **严重度**：中
- **位置**：`src/providers/anthropic.rs`（`complete()` 的 400 分支，计划 Step 0）
- **问题**：完整 Step 0 诊断日志（`req_has_assistant_thinking` + 顶层 `thinking` 字段存在与否 + body 前 200 字符）未实现。
- **为什么不修**：计划标为可选；本轮只做 P5 的 strip-trigger warn，完整诊断留待灰度需要。
- **触发/修复条件**：生产灰度时若仍见 thinking 400 落到客户端，需要确认"客户端历史里压根没带 thinking block"，此时补完整 Step 0 日志。

### D3 `implement-deepseek-thinking.md` 自身 §3.2 vs §4.1 矛盾未修订
- **严重度**：中
- **位置**：`docs/PLANS/implement-deepseek-thinking.md`（§3.2 vs §4.1）
- **问题**：文档 §3.2（置 null）与 §4.1（U4 断言顶层 thinking 为 JSON null）存在与删键语义不一致的残留表述。
- **为什么不修**：纯文档问题，不影响代码行为。
- **触发/修复条件**：按权威计划删除 friendly 残留表述、对齐删键语义时修订。

---

> 详见 [implement-deepseek-thinking.md](implement-deepseek-thinking.md) 第三轮审查记录。
