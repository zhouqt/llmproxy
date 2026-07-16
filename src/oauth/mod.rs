pub mod device_flow;
pub mod token_store;

#[allow(unused_imports)]
pub use device_flow::{poll_access_token, request_device_code};
#[allow(unused_imports)]
pub use token_store::{StoredTokens, TokenStore};

pub const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub const GITHUB_SCOPES: &str = "read:user";
pub const GITHUB_BASE_URL: &str = "https://github.com";
