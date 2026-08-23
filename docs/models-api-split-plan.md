# `/v1/models` 拆分计划:路由胜者列表 + 全量 provider 模型列表

## 背景

`/v1/models`(`src/server.rs` 的 `list_models_handler`)当前把静态 config 条目和所有
provider 的 `list_models()` 结果合并后按 `id` 去重(后出现者胜)。存在三个问题:

1. **去重胜者取决于 `HashMap` 迭代顺序**。`Router.providers` 是
   `HashMap<String, SharedProvider>`(`src/router.rs:61`),每次进程启动迭代顺序不同。
   同一模型 id 被多个 provider 暴露时,返回的 `owned_by` 是随机一个 provider,
   既不是真正会服务该模型的 provider(primary 优先),且跨重启不稳定。
2. **`owned_by` 不是配置里的 provider 名**:`anthropic.rs` 硬编码 `"anthropic"`;
   `openai_compat.rs` / `openai_responses.rs` 兜底为类型标签;copilot 用上游 vendor。
   与 `/admin/status`、`x-llmproxy-failed-providers` 使用的 provider 名不一致。
3. **语义混淆**:该端点无法同时表达"客户端实际会用哪些模型"与"上游各 provider
   分别提供哪些模型"。

用户确认原意图是"列出所有 provider 的全部模型",但现在需要把两种语义拆开:
现有 `/v1/models` 改为正确反映**当前路由实际使用的模型**(含真实来源 provider),
另建新端点列出**所有 provider 的全量模型**。

---

## 需求 1:修正现有 `/v1/models`(路由胜者视角)

### 1.1 聚合时注入来源 provider 名

`list_models_handler` 中,对每个 `provider.list_models()` 返回的条目覆写:

- **不变量:handler 层保证聚合后每个条目的 `owned_by` 都等于该 provider 在
  config 中的键名(`router.providers()` 的 key)**。provider 层代码允许省略
  该字段,兜底责任在 handler。
- 新增可选字段 `upstream_owned_by` ← provider 原始归属(vendor/organization),
  仅在原始条目带 `owned_by` 时保留。

静态 config 条目维持 `owned_by: "llmproxy"`。

### 1.2 按路由优先级去重

新增函数替代 `dedup_models_last_wins`:

```rust
fn dedup_models_by_routing_priority(entries: &mut Vec<serde_json::Value>,
                                    priority: &HashMap<(String, String), usize>)
// key = (上游模型 id, provider 名)
```

**priority 表以"上游 id"(即 `list_models()` 条目的 `id` 字段)为 key**,而不是
`config.models[*].name`(那是客户端名)。构建方式:

1. priority 只从 `config.models`(声明顺序的 `Vec`,天然确定,与 HashMap 无关)构建;
   对每条 chain 的 `(client_name, provider)` 对,用该 provider 的
   `merged_rewrite(&HashMap::new())` 翻译:`upstream_id = rewrite.get(client_name)
   .unwrap_or(client_name)`;
2. primary 记 `0`,`fallback_chain` 依次 `1, 2, ...`;同一 `(upstream_id, provider)`
   出现在多条 ModelConfig 时取最小值;
3. **应用 `can_serve_model(client_name)` 过滤**——注意参数是**客户端名**,不是
   翻译后的上游 id:`can_serve_model` 检查的是 `model_rewrite.contains_key(model)`
   (`openai_compat.rs` / `copilot.rs` 等),key 是客户端名,与路由器自身的
   分发检查(`router.rs` 中 `can_serve_model(&req.model)`)一致。若误传上游 id,
   非空 rewrite 的 provider 会被错误排除。该过滤作用于**去重的输入集**:
   不能服务该模型的 provider 的条目直接从 `/v1/models` 剔除(priority 表本身
   不受影响)——其条目仍保留在 `/admin/models`,以反映路由真实行为。
4. **静态 config 条目(`owned_by: "llmproxy"`)视为最低优先(`usize::MAX`)**:
   它们没有 provider 名,查 priority 表必然落空;规则为——同 id 存在有限
   priority 的 discovered 条目时 discovered 胜出,否则静态条目保留。

去重规则:同 id 多条目时保留 priority 最小(即实际路由最先使用)的条目;
跨多个 ModelConfig 的同分冲突以 `config.models` 声明顺序靠前者为次级裁决。
不在任何 chain 中的 id 按 `(provider 名, id)` 稳定排序后取第一个,保证确定性。
无 `id` 或 `id` 为空的条目沿用现有行为跳过并告警。

### 1.3 行为变化说明

- 返回的每个条目的 `owned_by` 都能对上 `/admin/status` 中的 provider 名。
- 客户端视角不变(仍是 OpenAI 风格 `{object: "list", data: [...]}`),
  仅字段取值更准确,新增字段为可选,向后兼容。
- **破坏性变更(内部)**:id 冲突时胜者从"随机 last-wins"变为"chain primary"。
  现有测试 `tests/server.rs` 的
  `list_models_aggregates_static_and_provider_discovered_models` 中 `gpt-4o`
  的预期 `owned_by` 将从 `"openai"`(discovered last-wins)改为 primary 的名字,
  测试需同步更新并在注释中说明这是有意的行为变更。

## 需求 2:新增 `/admin/models`(全量 provider 模型列表)

### 2.1 路由与形状

- `GET /admin/models`,注册在 `src/server.rs` 的路由表中,与 `/admin/status`
  同级(代理自省类端点,非 OpenAI 兼容面)。
- 鉴权:沿用 `/admin/*` 现有中间件行为(与 `/admin/status` 保持一致)。
- 响应形状(按 provider 分组,不做任何去重):

```json
{
  "object": "list",
  "providers": [
    {
      "provider": "deepseek",
      "source": "static|discovered|static+discovered",
      "cache_state": "populated|cold|auth_missing|fetch_failed",
      "models": [
        {"id": "...", "display_name": "...", "created": 0, "upstream_owned_by": "..."}
      ]
    }
  ]
}
```

- **静态来源来自 provider 级配置的 `model_rewrite` 表**(`ModelConfig` 上没有
  `model_rewrite` 字段,它在各 `ProviderConfig` 变体上):表非空 = 显式
  allow-list,逐键产出一条 `source: "static"` 条目(id 为 rewrite 的**值**,
  即上游名);表为空 = 无静态来源,全部条目为 discovered。
- **同一 provider 分组内的 static 与 discovered 合并规则**:按 `id`(字符串
  相等)在同 provider 内合并——先放入 static 条目;discovered 条目 id 已存在时,
  `source` 升级为 `"static+discovered"`,元数据(`created`/`display_name`/
  `upstream_owned_by`)以 discovered(上游目录)为准;否则插入为
  `"discovered"`。static 内部重复值(多个 key 映射到同一上游名)只保留一条,
  保留哪条的元数据无差别,但为确定性起见按**客户端 key 字典序最小者**胜出
  (`model_rewrite` 是 `HashMap`,迭代顺序随机,不能用声明序)。
  输出按 `id` 排序。
- 发现条目来自 `list_models()`。**`list_models()` 返回 `None` 时该 provider
  仍出现在结果中但 `models` 为空数组**。`cache_state` 的获取机制:
  - `Provider::list_models()` 只返回 `Option<Vec<Value>>`,`None` 不区分原因,
    因此非 Copilot provider 一律报 `"fetch_failed"`(不区分网络错误 / 非 2xx /
    JSON 解析失败);
  - Copilot 通过已有的 `AppState.copilot: Option<Arc<CopilotProvider>>` 句柄
    获取细粒度状态:在 `CopilotState` 上新增"最近一次失败原因"字段(在
    `cache_models_with_token` 的各错误分支内记录),并提供
    `pub fn cache_state(&self)` 方法;handler 优先用该句柄。枚举值对齐
    `CopilotState` 的真实信号——注意 `from_store` 构造时就从磁盘缓存加载,
    所以只有三种态:**`populated`**(内存有值)、**`auth_missing`**(内存被
    Auth 失败清空、磁盘缓存仍在)、**`cold`**(内存与磁盘均无)。不设
    `"never_fetched"`(构造后它与 `cold` 无法区分)。列表可能滞后一个后台
    刷新周期(实现时从 `spawn_background` 的刷新间隔取准确数值),非实时数据。
- 该端点**刻意不去重**:同一上游 id 可能同时出现在多个 provider 的列表里,
  需要唯一 id 视图的消费方(如容量统计)应在自己一层去重。大目录 provider
  (如 OpenRouter 数百个模型)会导致响应体较大——运维自省端点接受此权衡,
  不做分页。
- 单个 provider 查询失败不影响整体:始终返回 200(与 `/admin/status` 的降级
  语义一致),失败的以空 `models` + `cache_state` 表达,绝不 500。
- 元数据端点,仅做透传查询:**不传播 `provider_ignore`**(那是
  `/v1/messages` 请求体注入),OpenRouter 后端的 `/v1/models` 与消息端点的
  provider 路由无关。

### 2.2 实现

- handler:`all_models_handler(State<AppState>)`,复用各 provider 的
  `list_models()`。遍历顺序用 `state.config.providers`(`Vec`,声明序确定,
  与 `/admin/status` 的 `provider_status()` 模式一致),按名字到
  `state.router.providers()` 里查对应 `SharedProvider` 调 `list_models()`;
  `model_rewrite` 直接从 `ProviderConfig` 变体提取(必要时加访问器)。
- **并发 + 超时**:各 provider 的 `list_models()` 用
  `futures::future::join_all` 并发执行,每个调用包一层
  `tokio::time::timeout`(建议 10s 或可配置的 metadata 超时)。共享 reqwest
  client 默认超时 600s(`proxy_client.rs`),若不加超时,单个挂起的上游会阻塞
  端点长达 10 分钟;超时的 provider 按 `cache_state: "fetch_failed"`、空
  `models` 处理并继续。现有 `/v1/models` handler 的顺序迭代也顺带改为同样
  的并发+超时模式。
- 各 provider 内部的 `list_models()` 归一化逻辑保持不变;
  `owned_by` 兜底标签(`openai_compat` 等)在此端点不再有意义,
  由外层 `provider` 字段取代,内部归一化可去掉该兜底(见 2.3)。

### 2.3 provider 层小改

`anthropic.rs` / `openai_compat.rs` / `openai_responses.rs` 的 `list_models()`
归一化中,把硬编码的类型标签兜底改为透传上游值:上游缺失时整体省略该字段,
**由 handler 层把每个条目的 `owned_by` 补成 provider 配置名(不变量见 1.1;
§2.1 响应里的 `provider` 是分组包装字段,与条目级 `owned_by` 是两回事)**。
具体到各文件:

- `anthropic.rs`:目前**完全不读**上游响应的 `owned_by`(无条件硬编码
  `"anthropic"`),需改为像 `openai_compat.rs` 一样读
  `entry.get("owned_by")`,结果存入 `upstream_owned_by`,缺失则省略;
- `openai_compat.rs` / `openai_responses.rs`:已正确读上游 `owned_by`,
  只需删掉 `unwrap_or("openai_compat")` / `unwrap_or("openai_responses")`
  兜底字符串,缺失时省略字段;
- `copilot.rs`:输出中不再写 `owned_by`,`vendor` 移至 `upstream_owned_by`,
  条目级 `owned_by` 由 handler 填。

---

## 测试计划(维持 >97% 区域覆盖)

单元测试(`src/server.rs` 内):

- **删除旧函数 `dedup_models_last_wins` 及其三个直测**
  (`dedup_keeps_last_occurrence_for_duplicate_id` /
  `dedup_filters_out_empty_id_entries` / `dedup_preserves_unique_entries`,
  约 `src/server.rs:863-906`)——其覆盖被下列新测试取代。
- `dedup_models_by_routing_priority`:primary 胜 fallback;经
  `merged_rewrite` 翻译后的上游 id 正确匹配;`can_serve_model(client_name)`
  过滤正确(含非空 rewrite 下"客户端名是 key、上游 id 是 value"的用例);
  静态条目(`llmproxy`)对同 id discovered 条目必败、无 discovered 时保留;
  跨 ModelConfig 同分时 `config.models` 声明顺序裁决;未登记 id 稳定排序;
  无/空 `id` 条目跳过不 panic。
- handler 测试(fake provider,仿照现有 `tests/server.rs:887`):
  - 两 provider 暴露同一 id → `/v1/models` 胜者跟随 chain 顺序且确定;
  - `/v1/models` 条目 `owned_by` = provider 配置名,`upstream_owned_by` 保留上游值;
  - `/admin/models` 返回两个 provider 各自的全量列表(不去重)、空 `models`
    + `cache_state` 表示失败,整体仍为 200;
  - 挂起的 fake provider(不响应)→ 端点在超时内返回,该 provider 记为
    `fetch_failed`,其余 provider 数据完整(并发 + 超时路径覆盖)。

集成测试(`tests/server.rs`):

- 更新 `list_models_aggregates_static_and_provider_discovered_models`
  以匹配新的 `owned_by` 取值。
- 新增 `/admin/models` 路由存在性 + 鉴权 + 响应形状断言。

provider 单测更新:

- `anthropic.rs:1744`、`openai_compat.rs:1556`、`openai_responses.rs:1313`
  及各错误分支测试(行号实现时需复核):断言不再写入类型标签兜底,
  `owned_by` 缺省由上游值决定或整体省略。
- `copilot.rs` 的 `list_models` 相关测试:断言 vendor 移至
  `upstream_owned_by`,`owned_by` 不再由 provider 层填写。

## 实施顺序

1. provider 层 `owned_by` 兜底移除 + 单测更新(含 anthropic 读上游
   `owned_by`、copilot vendor 移位)。
2. Copilot 失败原因追踪:`CopilotState` 新增最近失败原因字段 +
   `CopilotProvider::cache_state()`。
3. server 层:priority 构建与新去重函数 + `/v1/models` 改造
   (priority 表在 `config.models` 上同步构建,与并发改造无关)。
4. 新增 `/admin/models` handler + 路由 + 鉴权对齐;两个 handler 的
   `list_models()` 采集统一改为并发 `join_all` + 每调用超时。
5. 全量测试:`cargo test --lib --bins --tests`,再跑
   `cargo llvm-cov --lib --bins --tests` 确认覆盖率不回退。
