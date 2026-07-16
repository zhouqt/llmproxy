use std::sync::Arc;

use crate::config::Config;
use crate::cooldown::CooldownCache;
use crate::router::Router;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub router: Arc<Router>,
    pub cooldown: CooldownCache,
    pub http: reqwest::Client,
}
