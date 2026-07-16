use std::sync::Arc;

use crate::config::Config;
use crate::cooldown::CooldownCache;
use crate::providers::copilot::CopilotProvider;
use crate::router::Router;

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
}
