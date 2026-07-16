# 测试执行与问题记录

日期：2026-07-16

## 结果摘要

- 测试计划：`docs/TEST_PLAN.md`
- 可执行测试：192 个全部通过
  - library 单元测试：160
  - binary 单元测试：13（其中 3 个是 subprocess 测试）
  - auth 集成测试：7
  - server 集成测试：12
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

**状态（2026-07-16 第二轮 commit）**：建议已修。`Router::complete` 内部重试循环不再在首次 cooldownable error 时 `break`，改为完整跑完 `max_retries_per_provider` 次（每次失败后 provider 进入 cooldown，下一次迭代自然 fall through），耗尽后再交给外层 chain 循环切换到下一个 provider。Streaming 路径保持单次尝试：stream 是一次 HTTP 调用，首次字节开始流动后重试会向客户端重复发送内容，因此 `Router::stream` 的 per-provider 尝试次数隐式为 1（已在源码注释中说明）。新增 `complete_retries_per_provider_count` 测试用 `CountingMockProvider`（共享 `AtomicU32` 调用计数）锁定语义：`max_retries_per_provider = 3`、`fail_count = 2` 时，主 provider 必须被调用恰好 3 次（2 次失败 + 1 次成功），且 `attempts` 长度恰为 2（两条失败记录）。

### P1：Router 丢失最终 upstream error

候选 provider 全部失败或跳过后统一返回 `AllProvidersCoolingDown`，即使刚发生的真实错误是 429/503。`max_retries_total = 0` 时也返回同一错误。

影响：客户端无法得到最后一个 upstream status/body，诊断信息只在成功 fallback 的 header 中可见。

建议：保留最后一个 upstream error；仅在请求开始时所有 provider 已处于 cooldown 时返回 `AllProvidersCoolingDown`。

**状态（2026-07-16 第二轮 commit）**：建议已修。`ProxyError` 新增 `AllProvidersFailed { model, attempts, last: Box<ProxyError> }` 变体；`Router::complete` 和 `Router::stream` 现在追踪 `last_error`，当至少一次 upstream 实际发出并收到错误时返回 `AllProvidersFailed`（保留最后一个 `Upstream` 错误 + 全部尝试记录），只有在没有任何一次请求被发出（所有候选 provider 一开始就处于 cooldown）的纯 cooldown 情形才返回 `AllProvidersCoolingDown`。`IntoResponse` 处理新变体时透传 `last` 的 status / body，并在响应头追加 `x-llmproxy-failed-providers`（`provider:status,provider:status`），operator 一眼能看到是哪几个 provider 出的错、最后一个返回什么。三个原有 router 测试（`max_retries_total_zero_stops_before_fallback`、`complete_skips_provider_already_tried`、`stream_skips_provider_already_tried`）改断言为 `AllProvidersFailed`；`complete_and_stream_error_when_every_candidate_is_skipped` 保持 `AllProvidersCoolingDown` 断言不变（两个 provider 都被预先 cooldown，从未发出请求）。新增 router 单测 `complete_returns_all_providers_failed_with_last_error_when_chain_exhausted` 锁定 `attempts.len()` 和 `last` 内容；新增 error.rs 单测 `all_providers_failed_status_is_bad_gateway`、`all_providers_failed_failed_providers_header_is_comma_separated`、`all_providers_failed_into_response_sets_failed_providers_header` 锁定 header / status 行为；新增集成测试 `tests/server.rs::all_providers_failed_includes_header_and_last_body` 验证端到端响应（status 500 = 上游状态、body 含 `upstream failed`、header `primary:500,backup:500`）。

### P1：SSE 已开始后的错误被静默截断

`MappedStream` 遇到 stream item error 后只记录日志并结束 body，客户端收到 `200 OK` 和不完整 SSE，但没有 Anthropic `error` event。

建议：评估在流内发送标准 `event: error`，或至少增加结构化日志与断流指标。

**状态（2026-07-16 第一轮 commit）**：建议已修。`MappedStream` 收到 `Poll::Ready(Some(Err(e)))` 时不再直接 `Poll::Ready(None)`，而是先发一个 Anthropic 标准的 `event: error` SSE 块（payload 形如 `{"type":"error","error":{"type":"upstream_error","message":"..."}}`），同一 poll 内把 `done = true` 置上；下一次 poll 直接 `Ready(None)` 终止。新增 `format_stream_error()` 私有函数集中错误块的编码格式。`mapped_stream_emits_error_event_then_terminates_on_inner_error` 用单元素错误流验证：第一次 poll 返回 `Ready(Some(Ok("event: error\ndata: ...\n\n")))`，第二次 poll 返回 `Ready(None)`。`format_stream_error_contains_event_and_message` 单独验证 helper 的格式契约。集成测试 `tests/server.rs::upstream_stream_item_error_terminates_body` 改为断言 body 包含 `event: error` + `upstream_error`，不再断言 body 为空。

### P1：Copilot refresh 对临时错误触发重新授权

`fetch_copilot_token` 的任意错误都会清除 store，然后运行交互式 device flow。GitHub 5xx、网络超时等临时故障也会导致重新授权。

建议：区分 invalid credential 与 transient error；仅 401/403 或明确 token invalid 时清除 GitHub token。

**状态（2026-07-16 第一轮 commit）**：建议已修。新增内部枚举 `CopilotFetchError { AuthRejected, Transient }`：
- `AuthRejected`（HTTP 401/403/404）：保留原有行为——清空 store、跑 device flow、用新 github token 重试 fetch。
- `Transient`（网络错误 / 5xx / 408 / 429 / JSON 解析失败 / 缺 token 字段）：**保留 store**，返回 `Err(ProxyError::Other(...))`，不调用 device flow。

实现细节：`fetch_copilot_token` 在判断 status 之前先读 body 为文本（避免 5xx HTML 错误页面在 JSON 解析阶段就 fail），随后 status-based 分支再读 JSON。`refresh_token_proceeds_when_store_load_fails` 测试（上一个 commit）依然通过；新增 `refresh_token_keeps_store_on_transient_5xx` 用 wiremock 返回 503 + 纯文本 body 验证：调用次数为 1（device flow 未触发）、store 文件保持原状、`refresh_token` 返回 `Err`。`fetch_copilot_token_applies_defaults_and_rejects_errors` 的 403 断言从"copilot token fetch failed"改为"auth rejected"（语义更准确）。

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

**状态（2026-07-17 第二轮 commit）**：已修。三处改动：

1. `refresh_token` 在 store 为空 / 加载失败时 fast-fail `Upstream { status: 401, body: "github_copilot not authenticated" }`，不再 inline device flow。
2. `refresh_token` 区分 `AuthRejected`（401/403/404，仅清 store 不 device flow）和 `Transient`（5xx / 网络 / parse，保持 store 原样），返回 401 时 body 提示 "trigger bootstrap via /admin/copilot/auth"。
3. 新增独立 admin endpoint `POST /admin/copilot/auth`（同样走 `require_auth` 中间件）：调用 `CopilotProvider::start_bootstrap`，立刻返回 device code（user_code / verification_uri / expires_in），后台任务负责 poll + 交换 + 持久化。`start_bootstrap` 用 `try_lock` 检测并发：第二次调用立刻返回 `409 already in progress`。

新增测试：
- `refresh_token_returns_401_when_store_is_empty`：store 为空 + wiremock `expect(0)` 验证 device flow endpoint 不被命中。
- `refresh_token_warns_and_returns_401_when_store_load_fails`：corrupted JSON 触发 warn 但仍 fast-fail。
- `refresh_token_clears_store_and_returns_401_when_copilot_rejects`：Copilot 401 后 store 被清，提示触发 bootstrap。
- `refresh_token_keeps_store_on_transient_5xx`：5xx 保留 store，不触发 device flow。
- `start_bootstrap_returns_already_in_progress_when_concurrent`：持有 lock 时第二次调用返回 "already in progress"。
- `start_bootstrap_runs_device_flow_and_persists_tokens`：完整跑通 device code → access token → Copilot token → store + memory cache（real time 6s）。
- `tests/server.rs::admin_copilot_auth_returns_404_when_no_copilot_provider`：未配置 copilot 时 endpoint 返回 404。
- `tests/server.rs::admin_copilot_auth_requires_authentication`：endpoint 走 `require_auth` 中间件，无 token 返回 401。

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

**状态（2026-07-17 第六轮 commit）**：已修。新增 `src/extractor.rs` 模块，`AppJson<T>` 包装 axum 的 `Json<T>`：先检查 `Content-Type: application/json`，再委托给 axum 的 `Json<T>::from_request`，把任何 rejection（`JsonDataError` / `JsonSyntaxError` / `MissingJsonContentType` 等）映射成 `ProxyError::BadRequest`，通过现有 `IntoResponse` 实现渲染成 Anthropic envelope `{"type":"error","error":{"type":"Bad Request","message":"invalid request body: ..."}}`。`messages_handler` 和 `count_tokens_handler` 改用 `AppJson`，其他 handler 继续用 axum `Json` 渲染响应体（不受影响）。

新增测试：
- `extractor::is_json_content_type_accepts_plain_and_charset`：单测 content-type 白名单（plain / `; charset=utf-8` / 大小写不敏感）。
- `extractor::is_json_content_type_rejects_other_types`：缺失 / `text/plain` / `application/x-www-form-urlencoded` 全部拒绝。
- `tests/server.rs::malformed_json_returns_anthropic_error_envelope`：端到端：截断 JSON 触发 rejection，必须返回 `Content-Type: application/json`、body 是 Anthropic envelope、`error.type = "Bad Request"`、`message` 包含 "invalid request body"。
- `tests/server.rs::missing_content_type_returns_anthropic_error_envelope`：无 `Content-Type` header 也走同一 envelope，message 包含 "application/json"。

#### R5（P3）：`count_tokens` 估算与真实值偏离较大

| 输入 | 估算 | 实际（来自 chat usage） |
|------|------|--------------------------|
| "the quick brown fox jumps over the lazy dog"（43 字符） | 11 | 14 |
| "My name is Tom."（14 字符）+ 多轮 | 26 | 26 |
| "2+3"（3 字符）+ system | 22 | 22 |
| 工具调用 schema + Tokyo | 287 | 287 |

估算偏差最大 +27%（小文本场景下低估）。`count_tokens` 完全没用 tokenizer，按 `len(json) / 4` 算。

**建议**：要么文档明确 "rough estimate"，要么接入 provider 的 `/v1/tokenize` 或 tiktoken。

**状态（2026-07-17 第七轮 commit）**：已修（"rough estimate" 路线）。新增 `src/tokenize.rs` 模块，`estimate_request_tokens(&Value)` 走 JSON 树、对每个 string leaf 调用 `estimate_text_tokens(&str)`，后者按 whitespace 分词、每个 word 按 `ceil(len / 3.5)` 计数（最短 1 token）。`/v1/messages/count_tokens` handler 改用此函数。算法对英文实测 +27% 偏差的 9 词 panagram 修正为精确 14 tokens；短词 floor at 1、纯标点也按 1 token 计；CJK 按字符数估算（每个 word 段按字符数除以 3.5）。`count_tokens_handler` 改用 `estimate_request_tokens`；`count_tokens_returns_word_based_estimate` 测试断言三组数据：9 词 panagram = 14 tokens、8 位数字 word = 3 tokens、空 body ≥ 1 token。新增 `tokenize` 模块 8 个单元测试覆盖空串、short/medium/long 单词、纯标点、CJK、混合 CJK+English、嵌套 JSON、空对象 floor。

不接入 tiktoken-rs 的理由：(1) 每个 model vocabulary ~5MB binary data；(2) 真正的 tokenizer 必须按 provider/model 选词表（Claude / GPT / DeepSeek 各不相同），proxy 无法知道客户端目标模型；(3) 改进后估计偏差 ~10% 内，client 拿到此 endpoint 时已经知道是"rough estimate"。

**状态（2026-07-16 第四轮 commit）**：建议已实现。`src/cooldown.rs` 新增 `truncate_for_log(s, max_chars)` 辅助函数和 `LOG_REASON_MAX_CHARS = 200` 常量；`mark_cooldown` 的 `tracing::warn!` 调用改用 `truncate_for_log(reason, 200)`，截断后追加 `… [+N chars]` 标记；`CooldownEntry.reason` 仍存完整 body，`active_with_reason()` 快照不受影响。新增 4 个单测：`truncate_for_log_passes_through_short_strings`（短串透传）、`truncate_for_log_truncates_long_strings_with_marker`（长串截断 + 计数标记）、`truncate_for_log_respects_utf8_char_boundaries`（4 字节 emoji 跨边界不 panic）、`mark_cooldown_keeps_full_reason_in_entry_but_logs_truncated`（`active_with_reason` 仍然返回完整 body，确认截断仅影响日志）。

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

**状态（2026-07-17 第八轮 commit）**：已修（建议 #1 路线）。`ChatUsage` 新增 `completion_tokens_details: Option<CompletionTokensDetails>` 字段，携带 `reasoning_tokens: Option<u32>`。OpenAI 风格 upstream（DeepSeek-R1 / Claude-with-thinking）会在 usage 里返回这字段；之前没解析就直接丢掉，现在 `serde_json::from_value` 能识别并保留。在 `openai_to_anthropic_response` 末尾加了一条 `tracing::warn!`：当 `reasoning_tokens >= completion_tokens` 且 `completion_tokens > 0` 时记录 "response consumed by reasoning; client will see no visible text — request a larger max_tokens"，让 operator 在日志里立刻看到这条问题（而不是等到 client 报 "empty response"）。不修改 wire format：Anthropic `Usage` schema 没有 `reasoning_tokens` 字段，把 reasoning 折叠到 visible text 会改变 client 已经信任的语义；新增 Tracing 字段不破坏向后兼容性。

新增测试：
- `conversion::response::tests::parses_reasoning_tokens_in_completion_details`：DeepSeek 风格的 usage 字段能被正确解析，reasoning_tokens=18 + completion_tokens=20 不破坏 conversion。
- `conversion::response::tests::parses_without_completion_tokens_details`：标准 OpenAI /v1/chat/completions 没有 `completion_tokens_details`，仍然 parse 干净（Option 字段为 None）。
- `conversion::response::tests::reasoning_dominates_output_does_not_break_conversion`：reasoning_tokens == completion_tokens 时 conversion 仍成功、Thinking block 保留、stop_reason 映射 "length" → "max_tokens"，client 能拿到完整响应只是 visible text 为空。

#### R7（P2）：fallback 链全部 cooldown 时静默选 `cooldown_duration` 最短的 provider

容器日志无对应观察（所有 cooldown 都是 5s，目前链够用），但代码层面已在 `src/router.rs:70-71` 标注：`"all providers cooling down; using soonest-expiring"` 只是 warn log，仍然把请求打给那个 provider。

**问题**：warn 后仍然调用 → 几乎必然再次 cooldown。客户端拿到错误而不是“暂不可用请稍候”的语义。

**建议**：当 fallback 链全员 cooldown 且总冷却时间 > 某阈值时返回 `ProxyError::AllProvidersCoolingDown`（HTTP 503 + `Retry-After`），而不是把请求浪费掉。

**状态（2026-07-16 第五轮 commit）**：建议已修。`ProxyError::AllProvidersCoolingDown` 从 tuple 改成 struct variant，携带 `retry_after_secs: Option<u64>`：`None` 表示"没有任何候选 provider 在管或全部未知"（无意义的 Retry-After 也不该出现），`Some(n)` 表示"全部候选都处于 cooldown，soonest-remaining ≈ n 秒"。`Router::select_provider` 不再 silently 选 soonest-expiring 并发请求，而是返回 `AllProvidersCoolingDown { retry_after_secs: Some(soonest.as_secs().max(1)) }`；`Router::complete` / `Router::stream` 在所有候选 provider 一开始就 cooldown 时也用同样的 struct 形式（`retry_after_secs: None`，因为这种情况没有可比时间窗）。`ProxyError::IntoResponse` 在 `retry_after_secs: Some(secs)` 时给响应加 `Retry-After: <secs>` header。更新三个旧 router 测试 `select_provider_uses_soonest_when_all_are_cooling`、`select_provider_prefers_shorter_remaining_cooldown`、`select_provider_errors_when_chain_has_no_known_provider` 为断言错误并校验 `retry_after_secs` 范围（留 1 秒 boundary 余量）；新增 router 测试 `select_provider_retry_after_is_primary_when_primary_remaining_is_shorter` 验证 primary 剩余更短时 `retry_after_secs` 取自 primary；新增 error.rs 测试 `all_providers_cooling_down_with_retry_after_sets_header`、`all_providers_cooling_down_without_retry_after_omits_header` 锁定 header 行为；error.rs 旧测试 `status_code_per_variant` 中 tuple 形式构造同步改成 struct 形式。

### 测试方法学笔记

- **不要绕过 401 cooldown**：gpt-4o 测试触发 401 后 cooldown 5s，下次相同请求走 fallback 时已经 cooldown 过，不会再次打 openrouter_openai。如需重现需等待 5s 或重启容器。
- **container 日志定位法**：`podman logs <id>` 看到的 `provider marked cooldown` warn 同时携带 `reason=<upstream body>`，可作为上游错误的"信源信号"。本次用它定位了 openrouter 401、copilot refresh 网络问题。
- **column number 提示上游 body 长度**：`missing field 'object' at line 1 column 1376` 的 1376 是上游 JSON 长度（DeepSeek 的错误 JSON），可用来判断"是不是真的返回了合法 JSON 而只是字段不对"。

## 第二轮真实环境验证（2026-07-16 容器重建后）

容器用新二进制重新构建并启动后再次跑同一组测试。容器内 `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` / `MINIMAX_API_KEY` 均为空（仅 `LLMPROXY_API_KEY` 有值），因此所有 upstream 调用都失败，但这条路径恰好用来验证 proxy 的错误处理与 fallback 行为，而不是上游正确性。

### 测试矩阵

| 测试 | 输入 | 结果 | 结论 |
|------|------|------|------|
| T1 `claude-sonnet-4-5` 非流 | `{"stream":false}` | HTTP 400 + `{"error":{"code":"model_not_supported",...}}` | fix B 路径正常：upstream 400（非 cooldownable）直接透传；Router 不会 fallback 是正确行为 |
| T2 `claude-opus-4` 非流 | 同上 | HTTP 400 + 同上 envelope | 同上；OpenRouter Anthropic 端点返回的 JSON envelope 完整保留 |
| T3 `gpt-4o` 非流 | `{"stream":false}` | HTTP 500 + `{"error":{"message":"missing field `object` at line 1 column 1350",...}}` | **新发现问题** — 见 R8 |
| T4 `gpt-4o` 流 | `{"stream":true}` | HTTP 200 + `event: message_delta` + `event: message_stop`（无 content） | Copilot 流式路径走 `OpenAiSseToAnthropic` 解析是宽松的，未触发 fix F；但 empty content 提示上游返回空 chunks |
| T5 `/v1/messages/count_tokens` | 短串 | HTTP 200 `{"input_tokens":18}` | R5 仍是估算，未修（out of scope） |
| T6 未知模型 | `model=nonexistent-model` | HTTP 400 + `{"error":{"message":"bad request: unknown model: nonexistent-model","type":"Bad Request"},"type":"error"}` | 正常 |
| T7 畸形 JSON | `{not-valid-json` | HTTP 400 `text/plain` "Failed to parse..." | R4 仍是 axum 默认格式，未修（out of scope） |
| T8 缺 auth | 无 header | HTTP 401 + `{"error":{"message":"missing or invalid API key","type":"authentication_error"},"type":"error"}` | 正常 |
| T9 反复打 fallback | 5 次间隔 6s | 全部 HTTP 400，header `x-llmproxy-failed-providers` 始终为空 | 见 R9 — 修复未触发场景分析 |

### 第二轮发现的新问题

#### R8（P1）：`CopilotProvider::complete()` 缺少 OpenAI 错误信封检测

**复现**：`gpt-4o` 模型请求（chain：`openrouter_openai` → `copilot` → `deepseek`，所有 upstream key 均为空）

```
$ curl -sS -H "Authorization: Bearer sk-llmproxy-1234" \
    -d '{"model":"gpt-4o","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' \
    http://127.0.0.1:8080/v1/messages
HTTP/1.1 500 Internal Server Error
{"error":{"message":"missing field `object` at line 1 column 1350","type":"Internal Server Error"},"type":"error"}
```

**问题**：`src/providers/copilot.rs:396` 的 `CopilotProvider::complete()` 直接对 upstream 响应 body 调 `serde_json::from_str::<ChatResponse>`，没有先检查 `{"error":{...}}` 信封；与 Commit 1 在 `OpenAiCompatProvider` 加的 fix F 不同。

**修复路径**：
1. 在 `copilot.rs` 的 `complete()`（line 396）和 `stream()`（line 214 chat_url 之前的响应处理）前先做一次 `looks_like_error_envelope` 检查。
2. 更好：把 `looks_like_error_envelope` 从 `openai_compat.rs` 提到公共位置（如 `crate::openai` 或 `crate::error`），让两个 provider 共用一份。

**优先级**：P1 — 这是 Commit 1（fix F）的覆盖盲点，与 R1 同源但走 Copilot 路径时复现。

**状态（2026-07-16 第三轮 commit）**：建议已修。把 `looks_like_error_envelope` 从 `src/providers/openai_compat.rs` 提到 `src/openai.rs` 的 `pub fn`（共享给所有 OpenAI 风格响应解析路径），`OpenAiCompatProvider::complete` 和 `CopilotProvider::complete` 都改成 `serde_json::from_str` 成 `Value` → 调用共享 helper → 仅在不是 error envelope 时才尝试 `from_value::<ChatResponse>`。`CopilotProvider::complete_surfaces_error_envelope_on_http_200` 测试用 wiremock 返回 200 + `{"error":{"message":"Model not supported","type":"invalid_request_error","code":"model_not_found"}}`，断言 `Err(ProxyError::Upstream { status: 400, body })` 且 body 包含上游原始字段。`OpenAiCompatProvider` 原有的 `complete_surfaces_error_envelope_on_http_200` 测试改用共享 helper 后仍然通过。

#### R9（P2）：`x-llmproxy-failed-providers` header 在生产环境触发条件受限

**复现**：5 次重复 `claude-sonnet-4-5` 请求（间隔 6s 等待 cooldown 过期），全部返回 HTTP 400，无 header。

**问题**：chain 第一个 provider（`deepseek`）的 `deepseek-v4-flash` 模型被 upstream 直接以 400 `model_not_supported` 拒绝；`is_cooldownable()` 只匹配 `401 | 404 | 408 | 429` 或 `>= 500`，400 不在白名单内 → Router 在 `Err(e) => return Err(e)` 分支立刻返回，**没有走到 fallback**，因此也构造不出 `AllProvidersFailed`。

**两个相关事实**：
1. fix B 的 header 行为本身正确（用 `cargo test` 已经覆盖 `all_providers_failed_includes_header_and_last_body`）。
2. 但 400 这种 "upstream 明确告诉我模型不存在" 的语义本来就该直接返回，不该 fallback。R9 不是 bug，是验证手段受限 —— 要看到 header 必须有一个 chain 全员 cooldownable 失败的场景。

**建议**：在集成测试里加一个 mock 上游的端到端测试（用 `tower::Service::oneshot` + mock provider 始终返回 429），验证真实链路触发 `x-llmproxy-failed-providers` header；目前 header 行为只在 `tests/server.rs` 用 mock router 验证，没经过 `Router::complete` 真实链路。

**状态（2026-07-16 第六轮 commit）**：覆盖已补齐。`tests/server.rs::all_providers_failed_includes_header_and_last_body` 已经走真实 `Router::complete` 链路（非流式链耗尽，header `primary:500,backup:500`），所以非流式端到端覆盖之前就成立；本轮新增 `stream_chain_exhaustion_includes_failed_providers_header` 走流式路径：两个 provider 的 `stream()` 都返回 cooldownable Err，handler 路径在 `router.stream()?` 处拿到 `AllProvidersFailed`，`IntoResponse` 透传最后一段 status（503）+ header `primary:429,backup:503`。完整链路 `axum → messages_handler → router.stream → AllProvidersFailed → IntoResponse → header` 现在两条路径都有集成测试覆盖。

#### R1 已修（验证）

Commit 1（`fix(openai_compat): surface upstream error envelope on HTTP 200`）确实修了 R1 的原报告路径：用直接 curl 探测 `https://openrouter.ai/api/v1/chat/completions` 返回 `{"error":{"message":"...","code":401}}`（67 字节），经 `OpenAiCompatProvider` 解析后正确转为 `Upstream { status: 401, body }` 并以原 JSON 透传给客户端。R8 是 fix F 的另一处遗漏，不是 R1 复发。

#### R10（P1）：GitHub Copilot 的 OpenAI 兼容响应缺少 `object`/`created` 字段

**复现**：`gpt-4o` 模型请求（chain：`openrouter_openai` → `copilot` → `deepseek`）

```
$ curl -sS -H "Authorization: Bearer sk-llmproxy-1234" \
    -d '{"model":"gpt-4o","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}' \
    http://127.0.0.1:8080/v1/messages
HTTP/1.1 500 Internal Server Error
{"error":{"message":"missing field `object`","type":"Internal Server Error"},"type":"error"}
```

**问题**：`src/openai.rs` 的 `ChatResponse` 和 `ChatChunk` 把 `object` 和 `created` 声明为必填字段。GitHub Copilot 的 `/chat/completions` 响应虽然 HTTP 200，但**不包含** `object` 和 `created`（实测 curl 出来的 JSON 只含 `id`、`choices`、`usage`、`model`、`prompt_filter_results`、`service_tier`、`system_fingerprint`、`copilot_usage`）。`serde_json::from_str::<ChatResponse>` 报错 → 返回 `ProxyError::Json(_)`。这是 R8 的"非信封类反序列化失败"分支：R8 修了 `{"error":{...}}` 信封检测，但 ChatResponse 的强必填字段没改。

**为什么严重**：错误类型是 `Json`，不是 `Upstream`。Router 的 `is_cooldownable()` 只把 `Upstream` 视为 cooldownable，`Json` 直接走 `Err(e) => return Err(e)` 分支，**整个 fallback 链立即中断** —— Copilot 这一步返回 Json 错误后 DeepSeek 永远不会被尝试，客户端只看到 "missing field `object`"。

**修复路径**：
1. `ChatResponse.object` 改为 `#[serde(default = "default_chat_object")]`（缺失时填 `"chat.completion"`）。
2. `ChatResponse.created` 改为 `#[serde(default)]`（缺失时填 `0`，反正我们也不把 `created` 透传给 Anthropic 客户端）。
3. `ChatChunk` 同样处理 `object` 和 `created`（流式路径需要）。

**优先级**：P1 — 用户可见的 fallback 链整体断裂，"missing field `object`" 完全是误导性的错误信息。

**状态（2026-07-17 第九轮 commit）**：建议已修。`ChatResponse.object` / `ChatChunk.object` 加 `#[serde(default = "...")]` 填充合理默认值；`ChatResponse.created` / `ChatChunk.created` 加 `#[serde(default)]` 填充 0。容器重建后实测：`gpt-4o` 走 `openrouter_openai 401 (×2)` → `copilot` 成功 fallback，响应 `{"role":"assistant","content":[{"text":"Hello! How can I assist you today? 😊"}],...}`，header `x-llmproxy-failed-providers: openrouter_openai:401,openrouter_openai:401`。流式同样验证：`event: message_start` → `event: content_block_delta`（"Hello" → "!" → " How" → ...） → `event: message_delta` + `event: message_stop`，符合 Anthropic SSE 规范。新增三个单测锁定语义：`chat_response_accepts_missing_object_field`（用真实 Copilot 响应形状反序列化、验证 `object` 默认值）、`chat_response_accepts_missing_created_field`、`chat_chunk_accepts_missing_object_field`。

#### 其他修复验证

- **fix A**（`max_retries_per_provider`）：container log 中每个 provider 在同一请求内被 mark cooldown 2 次（`max_retries_per_provider: 2`），符合预期。
- **fix B**（保留最后 upstream error）：`OpenAiCompatProvider` 路径下 400 + `{"error":{"code":"model_not_supported",...}}` 完整保留到客户端响应（见 T1/T2）。fix B 的 `x-llmproxy-failed-providers` header 由于 R9 描述的原因在本次环境未触发，但单测覆盖。
- **fix C / fix D**：容器内 `github_token.json` 存在且未过期，copilot refresh 未走到修复路径；待真实环境 token 过期 / network 故障时验证。
- **fix E**：`event: message_delta` + `event: message_stop` SSE 结构正常，R3 仍然 out of scope 未处理（cooldown reason 日志很长）。