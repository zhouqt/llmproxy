use std::sync::Arc;

use crate::config::Config;
use crate::cooldown::CooldownCache;
use crate::providers::copilot::CopilotProvider;
use crate::router::Router;
use crate::usage::UsageStats;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub router: Arc<Router>,
    pub cooldown: CooldownCache,
    pub http: reqwest::Client,
    /// Reference to the GitHub Copilot provider if one is configured. Used
    /// by the admin endpoint to trigger OAuth bootstrap on demand.
    /// `None` when no Copilot provider is configured.
    pub copilot: Option<Arc<CopilotProvider>>,
    /// Bounded ring buffer of per-request token usage records. Populated
    /// by `MappedStream` (streaming) and `messages_handler` (non-streaming);
    /// surfaced through `/admin/usage`. Capacity comes from
    /// `Config::usage_capacity` (default `usage::DEFAULT_USAGE_CAPACITY`).
    pub usage: UsageStats,
}
