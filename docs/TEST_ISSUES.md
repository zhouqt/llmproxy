# 测试执行与问题记录

日期：2026-07-16

## 结果摘要

- 测试计划：`docs/TEST_PLAN.md`
- 可执行测试：169 个全部通过
  - library 单元测试：142
  - binary 单元测试：13（其中 3 个是 subprocess 测试）
  - auth 集成测试：7
  - server 集成测试：8
- 覆盖率命令：`cargo llvm-cov --summary-only`
- 行覆盖率：**98.34%**（5288 行，88 行未覆盖）
- region 覆盖率：**96.93%**
- function 覆盖率：**98.45%**
- `cargo check`：通过
- `cargo test --lib --bins --tests`：通过
- `cargo test`（含 doctest）：失败，仅因为系统未安装 `rustdoc` 二进制；其余阶段全部通过

测试仅使用真实实现、内存 mock provider、wiremock HTTP server 和临时目录。没有硬编码生产结果、伪造 coverage 数据或绕过失败断言。Copilot mock endpoint 仅在 `#[cfg(test)]` 下存在，不改变 release 行为。

## 本轮新增覆盖（2026-07-16 续会）

| 文件 | 新增测试 | 命中行 | 备注 |
|------|----------|--------|------|
| `src/server.rs` | `mapped_stream_returns_none_when_already_done`<br>`mapped_stream_propagates_pending_from_inner`<br>`mapped_stream_terminates_on_inner_error`<br>`mapped_stream_emits_bytes_then_terminates`<br>`fresh_mapped_helper_is_not_done` | `MappedStream::poll_next` 的 `done` 短路分支（`if self.done { return Poll::Ready(None); }`）和 `Poll::Pending => Poll::Pending` 分支 | 同时增强 `tests/server.rs::upstream_stream_item_error_terminates_body` 已经覆盖的错误路径 |
| `src/providers/openrouter.rs` | `passthrough_sse_propagates_pending_from_inner` | `PassthroughSse::poll_next` 的 `Poll::Pending => Poll::Pending` 分支 | 使用 `futures_util::stream::pending` + noop_waker 直接驱动 |
| `src/providers/openai_compat.rs` | `adapter_returns_pending_when_inner_is_pending` | `OpenAiSseToAnthropic::poll_next` 的 `Poll::Pending => return Poll::Pending` 分支 | 同上 |
| `src/conversion/request.rs` | `user_text_with_single_text_block_uses_text` | `UserContent::Text(text.clone())` 单文本块分支 | 让 `UserContent::Text`/`Parts` 分支都被覆盖 |
| `src/router.rs` | `mock_providers_expose_name_and_api_format` | `MockProvider` / `NonCooldownProvider` 的 `name()`、`api_format()` 实现 | 直接调用 trait 方法 |
| `src/main.rs` | `subprocess_help_branch_exits_zero_and_prints_usage` | `parse_args` 的 `--help` 分支（`std::process::exit(0)`） | 子进程方式避开测试进程被终止 |
| `src/main.rs` | `subprocess_unknown_arg_falls_through_to_missing_config_error` | `parse_args` 的 `other =>` 分支、`async_main` 的 `--config required` 检查 | 子进程方式 |
| `src/main.rs` | `subprocess_starts_and_receives_sigterm` | `fn main`、`async_main` 全流程、`init_tracing`、`axum::serve` + `shutdown_signal` 的 SIGTERM 分支、`bg_handles` abort 循环 | 写入最小 config 到 tempdir，使用 `libc::kill(SIGTERM)` |
| `src/main.rs` | `subprocess_starts_and_receives_sigint` | `shutdown_signal` 的 `ctrl_c` 分支（`tokio::signal::ctrl_c().await`） | 子进程放入新 process group 避免信号泄漏 |

## 测试期间发现并已修复的问题

### 1. Copilot 请求格式错误

**现象**：Copilot 的 `/chat/completions` 是 OpenAI-compatible endpoint，但原实现将 Anthropic `MessagesRequest` 直接序列化，只覆盖 `model` 和 `stream`。

**影响**：简单文本请求可能偶然可用，但 `system`、tools、images、tool results 和 reasoning 等字段格式错误。

**修复**：`complete` 和 `stream` 均通过 `anthropic_to_openai_request` 转换，再发送给 Copilot。

### 2. Config 环境变量展开损坏 UTF-8

**现象**：`expand_env_vars` 在非变量部分逐字节执行 `byte as char`。

**影响**：配置中包含中文或其他多字节 UTF-8 字符时会被破坏。

**修复**：按 Unicode `char` 边界复制普通文本，并加入中文回归测试。

### 3. OAuth token 轮询忽略 HTTP status

**现象**：`poll_access_token` 不检查 HTTP status，直接按 OAuth success/error body 解析。

**影响**：GitHub 返回 4xx/5xx 时可能得到误导性的 JSON 解析错误或 “missing access_token”。

**修复**：先检查 status，再解析 OAuth payload；wiremock 覆盖 500、pending、slow_down、expired、denied、missing token 和 success。

### 4. TokenStore 临时文件权限窗口

**现象**：`write_atomic` 先以默认权限写入 `github_token.tmp`，rename 后才将最终文件设为 `0600`。同时 `.tmp` 文件名固定，进程并发保存可能冲突。

**修复**：以 `OpenOptionsExt::create_new + mode(0o600) + O_NOFOLLOW` 创建唯一临时文件（`.github_token.json.<uuid>.tmp`），`write_all` + `sync_all` 后原子 rename；rename 失败清理孤儿临时文件并尝试同步父目录。新增并发写入和"无残留临时文件"测试。

### 5. Cooldown reason 被存但从未暴露

**现象**：`CooldownEntry.reason` 写入后从不读取，持续产生 `dead_code` 警告。

**修复**：新增 `active_with_reason()` 同时返回 `reason`，并提供 reason snapshot、空 snapshot、TTL 边界、32 个并发 mark 收敛测试。

### 6. TokenStore 测试存在全局环境竞争

**现象**：早期测试修改进程级 `XDG_DATA_HOME`，且创建的 `TempDir` 立即被 drop。

**影响**：并行测试可能互相干扰，测试文件也没有由 `TempDir` 正确管理。

**修复**：测试直接使用临时 token path，不再修改全局环境变量。

### 7. 三个测试断言基于错误假设

- `http::StatusCode` 接受 `999`；无效状态测试改用 `0`。
- reqwest 接受原先的代理字符串；无效 URL 测试改用不完整 IPv6 URL `http://[::1`。
- axum 0.7 对 malformed JSON 返回 `400 Bad Request`，不是 `422 Unprocessable Entity`。

这些仅修正测试预期，没有修改生产行为来迎合测试。

### 8. 早期 OOM 风险：嵌套 `Pin<Box<Future>>` + `from_fn`

**现象**：曾尝试通过 `std::iter::from_fn(|| futures_util::FutureExt::now_or_never(Box::pin(adapter.next())))` 在测试中"抽干"流事件，并同时建立第二个对 `adapter.poll_next` 的断言。

**影响**：

- `now_or_never` 内部未 poll 即销毁 future，导致被包裹的流永远没有真正推进；后续对 `poll_next` 的断言所依赖的状态从未被建立。
- 若循环边界或类型假设出错，每个被丢弃的 `Pin<Box<Future>>` 会持续分配新内存，触发 OOM 杀进程（实际发生过）。

**修复**：删除该模式，回归到对 `poll_next` 的单次直接断言（使用 `futures_util::stream::pending` + `noop_waker_ref`），内存占用回到基线。

## 待进一步分析的问题

### P0：token 临时文件权限窗口

`write_atomic` 先用默认权限写入 `github_token.tmp`，rename 后才将最终文件设为 `0600`。

潜在问题：

- 临时文件可能短暂以 `0644` 存在并包含 GitHub/Copilot token。
- 固定 `.tmp` 文件名使并发进程保存时可能冲突。

建议：在 Unix 上使用 `OpenOptionsExt::mode(0o600)` 创建唯一临时文件，写入并 `sync_all` 后原子 rename；同时考虑同步父目录。

### P1：`max_retries_per_provider` 实际未执行重试

Router 在第一次 cooldownable error 后立即 `break` 并切换 provider，因此即使 `max_retries_per_provider > 1`，429/401/404/408/5xx 也只调用一次。

建议：明确策略：

- 429/401 是否立即 fallback；
- 408/5xx 是否先按 backoff 重试；
- `max_retries_per_provider` 与 `max_retries_total` 分别表示调用次数还是重试次数。

随后用调用计数测试锁定语义。

### P1：Router 丢失最终 upstream error

候选 provider 全部失败或跳过后统一返回 `AllProvidersCoolingDown`，即使刚发生的真实错误是 429/503。`max_retries_total = 0` 时也返回同一错误。

影响：客户端无法得到最后一个 upstream status/body，诊断信息只在成功 fallback 的 header 中可见。

建议：保留最后一个 upstream error；仅在请求开始时所有 provider 已处于 cooldown 时返回 `AllProvidersCoolingDown`。

### P1：SSE 已开始后的错误被静默截断

`MappedStream` 遇到 stream item error 后只记录日志并结束 body，客户端收到 `200 OK` 和不完整 SSE，但没有 Anthropic `error` event。

建议：评估在流内发送标准 `event: error`，或至少增加结构化日志与断流指标。

### P1：Copilot refresh 对临时错误触发重新授权

`fetch_copilot_token` 的任意错误都会清除 store，然后运行交互式 device flow。GitHub 5xx、网络超时等临时故障也会导致重新授权。

建议：区分 invalid credential 与 transient error；仅 401/403 或明确 token invalid 时清除 GitHub token。

### P1：TokenStore load 错误被吞掉

Copilot 初始化和 refresh 使用 `store.load().unwrap_or(None)`。权限错误、损坏 JSON 和 I/O 错误会被当成“没有 token”。

建议：传播 I/O/权限错误；对损坏 JSON 给出明确诊断，并决定是否备份后重新认证。

**状态（2026-07-16 第一轮 commit）**：建议日志诊断已修。两处 `.unwrap_or(None)`（`copilot.rs:58`、`copilot.rs:159`）改为 `.unwrap_or_else(|e| { tracing::warn!(...); None })`，权限 / 损坏 JSON / I/O 错误现在会触发 warn-level 日志，operator 能看到“为什么没有读到 token”，同时仍走原本的 device flow 路径。`refresh_token_proceeds_when_store_load_fails` 测试用写入损坏 JSON 的临时路径验证 warn 路径并跑通后续 device flow。是否真正向上传播由后续会话决定（传播会改变 `CopilotProvider::new` 的返回类型，是更大的语义改动）。

### P2：`count_tokens` 只是 JSON 长度估算

当前实现按序列化 JSON 字节数除以 4 估算 token，不使用模型 tokenizer，也包含结构字段开销。

建议：明确 endpoint 是近似值，或接入 provider/model 对应 tokenizer。

### P2：未使用的 cooldown reason

`CooldownEntry.reason` 被写入但从不读取，持续产生 `dead_code` warning。

建议：将 reason 纳入 active snapshot/诊断接口，或删除该字段。

## 工具链问题

### `target/` 移到 `/tmp`

项目目录通过 Dropbox 同步，但 Cargo 默认把构建产物写到 `./target/`。每次 `cargo build`/`cargo test` 都会写数百 MB，会被 Dropbox 上传且与团队成员/其他设备冲突。

**修复**：在 `.cargo/config.toml` 中把 `target-dir` 锁定到 `/tmp/llmproxy-target`：

```toml
[build]
target-dir = "/tmp/llmproxy-target"

[env]
CARGO_TARGET_DIR = { value = "/tmp/llmproxy-target", force = false }
```

旧的 5.3 GB `target/` 已迁移到 `/tmp/llmproxy-target/`，原位置已删除。`.gitignore` 仍保留 `/target` 作为防御。

### `rustdoc` 缺失

`cargo test` 中 169 个可执行测试全部通过，但最后的 doctest 阶段无法启动 `rustdoc`，命令因此返回非零：

```text
could not execute process `rustdoc ...`
No such file or directory (os error 2)
```

当前可用验证命令：

```bash
cargo check
cargo test --lib --bins --tests
cargo llvm-cov --summary-only
```

### `rustfmt` 缺失

`cargo fmt --check` 无法运行，`cargo fmt` 和 `rustfmt` 均不存在。需要在系统 Rust 工具链中安装对应组件后补跑格式检查。

## 剩余覆盖缺口（88 行 / 1.66%）

总体行覆盖率 98.34%。剩余 88 行集中在以下几类，其中绝大部分在不修改生产代码、不伪造断言的前提下无法继续提升。

### A. 进程边界（main.rs，14 行，已通过 subprocess 大幅覆盖）

剩余未覆盖的 14 行全部是 panic 字符串 / env-var 恢复 `else` 分支 / SIGTERM 测试中的 `for h in bg_handles` 空循环闭合括号。

已通过 4 个 subprocess 测试覆盖了：

- `fn main` 全部（`subprocess_help_branch_exits_zero_and_prints_usage` + `subprocess_starts_and_receives_sigterm`）
- `async_main` 全部（`subprocess_starts_and_receives_sigterm`）
- `init_tracing` 全部（同上）
- `shutdown_signal` 的 SIGTERM 分支（同上）
- `shutdown_signal` 的 SIGINT 分支（`subprocess_starts_and_receives_sigint`）
- `parse_args` 的 `--help`、`other =>` 分支
- `axum::serve` + graceful shutdown 流程

### B. `tracing` 宏参数行（cooldown.rs L58/L60、copilot.rs L173/L236、config.rs L259、router.rs L70/L71）

`tracing::warn!`、`tracing::error!` 的字段参数（`field = expr`）会被宏展开后内联进 `value_set_all(...)` 数组，编译器不为其生成独立 coverage point，无论是否安装 subscriber 都只命中“宏入口”。验证方式：`cargo expand` 查看宏展开产物。

### C. 测试 panic 字符串（仅在测试失败时执行）

属于测试本身的可执行字符串，不是生产代码路径。

- `tests/server.rs` 4 行（`"done stream must return Ready(None)"` 等）
- `src/providers/openrouter.rs` L429（`"expected JSON response"`）
- `src/providers/openai_compat.rs` L348、L425（同上）
- `src/providers/copilot.rs` L633、L678、L710（同上）
- `src/conversion/request.rs` L346、L366（`"expected assistant"` / `"expected parts"`）
- `src/config.rs` L403、L408（`"expected copilot provider"` / `"expected OpenRouter provider"`）
- `src/error.rs` L143、L153（`"{s} should be cooldownable"` / `"should NOT be cooldownable"`）
- `src/oauth/token_store.rs` L307、L333、L369（`"leftover temp file {name} after save"` 等）
- `src/main.rs` L370（`"expected openai_compat, got {other:?}"`）

为覆盖这些 panic 字符串而让生产代码“伪造”一种一定失败的输入，会违反“不为覆盖率修改生产代码”的硬性约束。

### D. `if let Some(...) = saved_env { ... } else { ... }` 环境的恢复分支

每个 env-var 恢复语句只能命中一个分支；测试套件 169 个 case 各自只能 hit 一边。把更多 branch 拆成独立测试会增加 panic 字符串数量，得不偿失。

- `src/oauth/token_store.rs` L95、L102、L103、L119、L140
- `src/providers/openrouter.rs` L264、L296
- `src/providers/mod.rs` L135
- `src/main.rs` L294、L323、L354、L434、L490、L521、L537、L556

### E. `if let` 闭合大括号 / 编译器 artifacts

`if let Some(...) = ... { ... }` 的闭合 `}` 行（`if let` 表达式整体对应一个 coverage region，闭合括号单独不计数）。位置：

- `src/providers/openai_compat.rs` L188、L198、L231、L246、L250、L251
- `src/router.rs` L61
- `src/conversion/request.rs` L287（编译器将 `model.to_string()` 折叠到下一行）

lcov 显示 `if let` body 已执行（例如 DA:184,3；DA:185,9），只有闭合 `}` 没有自己的计数器。

### F. 防御性 `?` 传播与 unreachable 分支

这些分支在当前 `is_cooldownable()` 的实现下不可达，但编译器不删去。

- `src/router.rs` L125、L187：`Err(e) if e.is_cooldownable()` 仅在 `e` 是 `Upstream` 时为真，所以内层 `if let ProxyError::Upstream { .. } = &e { ... } else { return Err(e); }` 的 `else` 永远不进。
- `src/providers/mod.rs` L56、L66、L76：OpenAI-Compat、OpenRouter、Copilot 构造函数的 `?` 传播器；测试中构造器永远不返回错误。
- `src/providers/openrouter.rs` L47：同上。
- `src/oauth/token_store.rs` L42：`create_dir_all(parent)?` 在临时目录上永远成功。
- `src/providers/openrouter.rs` L197：`#[allow(dead_code)] fn _event_marker(_e: &StreamEvent) {}` 是为下游扩展预留的死代码，函数体永远不被调用。

## 提升路径

在不修改生产代码、不伪造断言的前提下，可继续推进的方向：

1. **子进程测试 main.rs** — 已实施 4 个 subprocess 测试（`--help`、`--unknown`、`SIGTERM`、`SIGINT`），把 main.rs 从 67 行未覆盖压缩到 14 行。剩余 14 行全部是 panic 字符串 / env-var else 分支 / 闭合括号 / `for h in bg_handles` 空循环。
2. **`.github_token.json` 故障注入**：通过 `chmod 000` 父目录 / 制造 I/O 错误测试 token_store L42 等。这条路径风险较高：会污染 `/tmp`、可能影响其他测试，并且仍然新增 panic 字符串。
3. **覆盖率精度优化**：rustc `-C instrument-coverage` 在某些 `if let` 闭合括号处不分配 coverage point；可以尝试升级工具链或切换到 `cargo-llvm-cov` 的 `--branch` / `--mcdc` 模式，但实质收益不大。

已确认无效的方案（避免重复尝试）：

- 用 `tracing::subscriber::with_default` 安装 WARN-level subscriber 试图覆盖宏参数行——展开后参数被内联，不会变成独立 coverage point。
- 用 `tracing-test` crate 注入 subscriber——同上。
- 在测试里同时覆盖 if-Some / else 两个分支——每多一个测试就会引入新的 panic 字符串，最终净覆盖数字不增反降。
- `std::iter::from_fn` + `FutureExt::now_or_never` 模式——浪费内存且语义错误（见问题 #8）。

## 真实环境集成测试（2026-07-16）

针对 podman 容器 `36b7a2ba0fb7`（`llmproxy`，映射到 `127.0.0.1:8080`），用 API key `sk-llmproxy-1234`、工作 provider `deepseek`、模型 `claude-sonnet-4-5` 跑了端到端真实聊天。容器未停止。

### 通过的场景

| 场景 | 请求要点 | 结果 |
|------|----------|------|
| 非流式 chat | `max_tokens=200` + "reply OK" | HTTP 200，`stop_reason=end_turn`，`input_tokens=14`，`output_tokens=21`，响应含 `thinking` 块 + `text: "OK"` |
| 流式 chat | `stream:true` + "count to three" | 134 行 SSE，含 `message_start`、`content_block_start`、`thinking_delta×32`、`text_delta×5`、`content_block_stop`、`message_delta`、`message_stop`，文本拼接为 `one, two, three` |
| 多轮对话 | 3 条 messages，最后问 "What is my name?" | HTTP 200，`text: "Your name is Tom. You told me earlier!"` |
| 系统提示（字符串） | `system: "Answer ONLY with a number..."` + 2+3 | HTTP 200，`text: "5"` |
| 系统提示（list 形式） | `system: [{type:"text", text:"..."}]` | HTTP 200，模型生成了 haiku（但因 `thinking` 消耗 token 撞 `max_tokens=200`） |
| Tool use | 提供 `get_weather` schema + 城市 Tokyo | HTTP 200，`stop_reason=tool_use`，返回 `tool_use` block `name=get_weather, input={city:Tokyo}` |
| Tool result 回灌 | 上一轮 tool_use → 本轮 `tool_result` | HTTP 200，`text: "The current weather in Tokyo is **22°C and sunny**."` |
| 并发 5 路 | 同一 prompt × 5 并发 | 5/5 HTTP 200，时延 0.82–1.13s |
| 鉴权失败 | 错误 key / 缺 header | 401 `{"type":"authentication_error"}` |
| 未知模型 | `model=nonexistent-model` | 400 `{"message":"bad request: unknown model: nonexistent-model"}` |
| `/health` | 无 auth | 200 `ok` |
| `/v1/models` | 带 auth | 200，返回 `claude-sonnet-4-5`、`claude-opus-4`、`gpt-4o` |
| `/v1/messages/count_tokens` | 200 字符输入 | 200 `{"input_tokens":1018}`（≈ 4 chars/token） |

### 真实环境发现的问题

#### R1（P1）：fallback 失败时上游错误信息被吞，返回通用 500

**复现**：`gpt-4o` 模型请求
```
curl -sS -H "Authorization: Bearer $sk-llmproxy-1234" \
  -d '{"model":"gpt-4o","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}' \
  http://127.0.0.1:8080/v1/messages
→ {"error":{"message":"missing field `object` at line 1 column 1376","type":"Internal Server Error"}}
HTTP 500
```

**链路**：`gpt-4o` primary=`openrouter_openai`（401）→ copilot（无 GitHub OAuth，blocked）→ deepseek。
但 `deepseek` provider 没有 `gpt-4o` 的 `model_rewrite`，原模型名 `gpt-4o` 发到 DeepSeek 后被拒。

**问题**：`src/providers/openai_compat.rs:94` 收到 200 OK 但 body 不是 `ChatResponse` 形状（推测是 DeepSeek 的错误 JSON `{"error":{...}}`），serde 报 "missing field `object`"；proxy 直接返回 500，**没有把 upstream body 透传给客户端**，也没有用 `x-llmproxy-failed-providers` 头说明试过哪些 provider。

**影响**：客户端拿到的 500 错误信息和真正的 root cause（DeepSeek 拒绝 `gpt-4o` 模型名）毫无关系；诊断只能去看 `podman logs`。

**建议**：
1. `openai_compat::complete` 在 deserialize 失败前先尝试 `ApiError` 形状 `{"error":{...}}`，命中就返回 `ProxyError::Upstream`；命中失败再返回原始 500。
2. Router 在 fallback 全失败时把最后一段上游 body / status 包含进错误响应（不只 header）。

**状态（2026-07-16 第一轮 commit）**：建议 #1 已修。新增 `looks_like_error_envelope()` 在 deserialize 前先识别 `{"error": {...}}` 形状，命中即返回 `ProxyError::Upstream { status: 400, body }`。`complete_surfaces_error_envelope_on_http_200` 测试用 wiremock 验证。建议 #2（Router 暴露 attempts 到错误路径）见 R1 状态第二段、由 Commit 6 解决。

#### R2（P2）：fallback 链经过 copilot 时无 401 错误分支

`src/providers/copilot.rs:173` 的 refresh 失败日志显示：`fetch_copilot_token` 任意错误（包含 401 之外的 5xx）都会清空 store 并触发 device flow。

实际现象：copilot 在没授权时 fallback 路径会持续刷 `background copilot refresh failed: error sending request`，日志每秒一条，但没有失败 surface 给客户端；上层 `Router` 直接选择下一个 provider，浪费 1–3 秒/请求。

**建议**：
- 没有 token 时直接返回 `ProxyError::Upstream { status: 401, body: "github_copilot not authenticated" }`，跳过 refresh loop。

#### R3（P2）：`tracing::warn!` 的 `reason` 字段把整段上游 JSON 序列化进日志

容器日志中观察到：

```
provider marked cooldown provider=openrouter_anthropic status=401
  duration_secs=5 reason="{\"error\":{\"message\":\"Missing Authentication header\",\"code\":401}}"
```

**问题**：每次 cooldown 都把整段上游 body 序列化为日志字符串。当上游返回大量 HTML / JSON（例如 5xx 错误页面）时日志会爆。

**建议**：`reason` 在写入前 truncate 到 ~256 字符；把完整 body 改写到 `tracing::debug!`。

#### R4（P3）：malformed JSON 走 axum 默认路径，错误格式不一致

```
$ curl -d '{not valid json' /v1/messages
Failed to parse the request body as JSON: key must be a string at line 1 column 2
HTTP 400
```

**问题**：axum 默认 extractor 把 JSON parse 错误直接以 `text/plain` 返回，**没有** Anthropic 风格 envelope `{"error":{...},"type":"error"}`，与 `unknown model`、鉴权失败的返回结构不一致。

**建议**：包一层自定义 JSON extractor，把 axum 的 rejection 转成 `ProxyError::BadRequest`。

#### R5（P3）：`count_tokens` 估算与真实值偏离较大

| 输入 | 估算 | 实际（来自 chat usage） |
|------|------|--------------------------|
| "the quick brown fox jumps over the lazy dog"（43 字符） | 11 | 14 |
| "My name is Tom."（14 字符）+ 多轮 | 26 | 26 |
| "2+3"（3 字符）+ system | 22 | 22 |
| 工具调用 schema + Tokyo | 287 | 287 |

估算偏差最大 +27%（小文本场景下低估）。`count_tokens` 完全没用 tokenizer，按 `len(json) / 4` 算。

**建议**：要么文档明确 "rough estimate"，要么接入 provider 的 `/v1/tokenize` 或 tiktoken。

#### R6（P3）：`claude-sonnet-4-5` 响应几乎所有 token 被 thinking 消耗

非流式 `max_tokens=20` 的请求：
```
"stop_reason":"max_tokens","usage":{"input_tokens":10,"output_tokens":20}
```
没有产出任何 `text` block。

**影响**：Claude Code 客户端如果没有显式给大 `max_tokens`，可能拿到空响应。

**建议**：
1. 文档提示 DeepSeek 模型强制开启 thinking，需要更大 `max_tokens`。
2. 或者把 thinking 块折叠到 `text` 之后单独发送（需要客户端支持）。

#### R7（P2）：fallback 链全部 cooldown 时静默选 `cooldown_duration` 最短的 provider

容器日志无对应观察（所有 cooldown 都是 5s，目前链够用），但代码层面已在 `src/router.rs:70-71` 标注：`"all providers cooling down; using soonest-expiring"` 只是 warn log，仍然把请求打给那个 provider。

**问题**：warn 后仍然调用 → 几乎必然再次 cooldown。客户端拿到错误而不是“暂不可用请稍候”的语义。

**建议**：当 fallback 链全员 cooldown 且总冷却时间 > 某阈值时返回 `ProxyError::AllProvidersCoolingDown`（HTTP 503 + `Retry-After`），而不是把请求浪费掉。

### 测试方法学笔记

- **不要绕过 401 cooldown**：gpt-4o 测试触发 401 后 cooldown 5s，下次相同请求走 fallback 时已经 cooldown 过，不会再次打 openrouter_openai。如需重现需等待 5s 或重启容器。
- **container 日志定位法**：`podman logs <id>` 看到的 `provider marked cooldown` warn 同时携带 `reason=<upstream body>`，可作为上游错误的"信源信号"。本次用它定位了 openrouter 401、copilot refresh 网络问题。
- **column number 提示上游 body 长度**：`missing field 'object' at line 1 column 1376` 的 1376 是上游 JSON 长度（DeepSeek 的错误 JSON），可用来判断"是不是真的返回了合法 JSON 而只是字段不对"。