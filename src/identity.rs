//! The outbound identity of a single provider.
//!
//! Two levels, and only two:
//!
//! - The **global default** User-Agent (top-level `user_agent:` in the
//!   config) is attached by the two shared reqwest clients, so it reaches
//!   every provider. See `resolve_global_user_agent`.
//! - A **provider may override** that User-Agent, and may add a
//!   `referer` and custom `headers`, for itself only. The provider block
//!   in `providers:` *is* the scope — there is no separate routing table.
//!
//! Everything here is **best-effort**. A bad *value* is dropped with a
//! warning and the proxy keeps serving; only a bad *structure* (an
//! unknown key, a wrong YAML type) is a startup error, and that is
//! decided by serde, not by this module.
//!
//! Two rules the rest of the repo depends on:
//!
//! - **Field omission is correctness.** An unset option sends nothing;
//!   an empty value omits the header rather than sending it empty.
//! - **The credential keeps reqwest's "sensitive" marking.** The identity
//!   headers are applied with `RequestBuilder::headers` (which *replaces*
//!   per name) *after* the provider's own `.bearer_auth()` call, and the
//!   reserved-name list forbids the identity block from touching
//!   `authorization`, so the credential is neither dropped nor
//!   de-classified.

use std::collections::BTreeMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use reqwest::RequestBuilder;

use crate::config::default_user_agent;

/// Header names the proxy sets itself and an operator must not be able to
/// clobber: authentication, framing, or the Anthropic protocol identity.
///
/// Compared against the **parsed** `HeaderName`, which is always
/// lowercase — so `User-Agent` and `USER-AGENT` are caught too.
pub(crate) const RESERVED_HEADERS: &[&str] = &[
    "user-agent",
    "authorization",
    "proxy-authorization",
    "content-type",
    "content-length",
    "host",
    "accept",
    "transfer-encoding",
    "connection",
    "x-api-key",
    "anthropic-version",
    "x-app",
];

/// The header a provider's `referer:` is sent as.
///
/// `HTTP-Referer` (canonical form `http-referer`) is what an app
/// registry credits traffic on — for OpenRouter it is the *only* header
/// that creates an app attribution; without it the request shows up as
/// "Unknown" no matter what the User-Agent says.
const REFERER_HEADER: &str = "http-referer";

/// Value sentinel: one UUID generated once per provider, stable for the
/// lifetime of the proxy process.
const VALUE_AUTO: &str = "auto";
/// Value sentinel: a fresh UUID per outbound request.
const VALUE_AUTO_REQUEST: &str = "auto-request";

/// A configured header value, still unresolved for the per-request form.
#[derive(Debug, Clone)]
enum OutboundValue {
    Literal(HeaderValue),
    /// The provider's stable session id.
    Session,
    /// A fresh id per outbound request.
    PerRequest,
}

/// The outbound identity of one provider.
///
/// `Default` is the empty identity: no User-Agent override (the global
/// default applies), no referer, no custom headers. That is what a
/// provider built directly in a test gets.
#[derive(Debug, Clone, Default)]
pub struct OutboundIdentity {
    /// The provider's User-Agent override. `None` ⇒ nothing is attached
    /// and the shared client's default (the global UA) applies.
    user_agent: Option<HeaderValue>,
    /// The headers to send, in emission order. The referer comes first so
    /// that an explicit `HTTP-Referer` in `headers:` replaces it.
    entries: Vec<(HeaderName, OutboundValue)>,
    /// The stable id behind the `"auto"` sentinel.
    session_id: String,
}

impl OutboundIdentity {
    /// Materialise a provider's identity from its config fields.
    ///
    /// **Infallible**: every value-level problem degrades to a
    /// warn-and-drop. The warnings name the field and the reason and
    /// **never the value** — the config accepts `${VAR}`, so a value can
    /// be an env-provided credential, and runtime logs are not covered by
    /// `tools/scan-secrets.sh`.
    pub fn from_parts(
        provider: &str,
        user_agent: Option<&str>,
        referer: Option<&str>,
        headers: &BTreeMap<String, String>,
    ) -> Self {
        let mut entries: Vec<(HeaderName, OutboundValue)> = Vec::new();

        if let Some(raw) = referer {
            match header_value(raw) {
                Some(value) => entries.push((
                    HeaderName::from_static(REFERER_HEADER),
                    OutboundValue::Literal(value),
                )),
                None => tracing::warn!(
                    provider = %provider,
                    "provider.referer: not a legal http header value; the header is dropped"
                ),
            }
        }

        for (name, value) in headers {
            let Ok(header_name) = HeaderName::from_bytes(name.as_bytes()) else {
                tracing::warn!(
                    provider = %provider,
                    header = %name,
                    "not a legal http header name; the header is dropped"
                );
                continue;
            };
            if RESERVED_HEADERS.contains(&header_name.as_str()) {
                tracing::warn!(
                    provider = %provider,
                    header = %header_name,
                    "header name is reserved by the proxy; the header is dropped"
                );
                continue;
            }
            if value.is_empty() {
                // Omission, not an empty header: field omission is correctness.
                tracing::warn!(
                    provider = %provider,
                    header = %header_name,
                    "header value is empty; the header is omitted"
                );
                continue;
            }
            let resolved = match value.as_str() {
                VALUE_AUTO => OutboundValue::Session,
                VALUE_AUTO_REQUEST => OutboundValue::PerRequest,
                _ => match HeaderValue::from_str(value) {
                    Ok(v) => OutboundValue::Literal(v),
                    Err(_) => {
                        tracing::warn!(
                            provider = %provider,
                            header = %header_name,
                            "value is not a legal http header value; the header is dropped"
                        );
                        continue;
                    }
                },
            };
            entries.push((header_name, resolved));
        }

        let user_agent = user_agent.and_then(|raw| {
            match header_value(raw) {
                Some(value) => Some(value),
                None => {
                    tracing::warn!(
                        provider = %provider,
                        "provider.user_agent: not a legal http header value; \
                         the global default User-Agent is used"
                    );
                    None
                }
            }
        });

        Self {
            user_agent,
            entries,
            session_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Attach this identity to a request builder.
    ///
    /// Call this **last**, right before `.send()`: `RequestBuilder::headers`
    /// replaces per name, so the map can never clobber a provider-owned
    /// header (the reserved-name list forbids those names) and the
    /// credential keeps reqwest's sensitive marking.
    ///
    /// A `User-Agent` override set here wins over the shared client's
    /// default: reqwest fills the client-level default only into *vacant*
    /// header slots, and this one is occupied.
    pub fn apply(&self, rb: RequestBuilder) -> RequestBuilder {
        rb.headers(self.header_map())
    }

    /// The header map this identity contributes.
    ///
    /// The `"auto-request"` id is resolved **here**, at request time — a
    /// per-request id cannot exist at materialisation time.
    pub fn header_map(&self) -> HeaderMap {
        let mut map = HeaderMap::new();
        if let Some(ua) = &self.user_agent {
            map.insert(USER_AGENT, ua.clone());
        }
        for (name, value) in &self.entries {
            let value = match value {
                OutboundValue::Literal(v) => v.clone(),
                OutboundValue::Session => HeaderValue::from_str(&self.session_id)
                    .expect("a uuid is a legal http header value"),
                OutboundValue::PerRequest => HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                    .expect("a uuid is a legal http header value"),
            };
            map.insert(name.clone(), value);
        }
        map
    }
}

/// The global default User-Agent: the first usable of the configured
/// value and the built-in Claude Code default.
///
/// "Usable" means non-empty after trimming **and** a legal
/// `HeaderValue`. An illegal value must be caught here, before the
/// client is built: `ClientBuilder::user_agent()` records a builder
/// error and `build()` then returns `Err`, which would abort startup.
pub fn resolve_global_user_agent(configured: &str) -> String {
    let trimmed = configured.trim();
    if trimmed.is_empty() {
        tracing::warn!(
            "user_agent: the configured value is empty; the built-in default is used"
        );
        return default_user_agent();
    }
    if HeaderValue::from_str(trimmed).is_err() {
        tracing::warn!(
            "user_agent: the configured value is not a legal http header value; \
             the built-in default is used"
        );
        return default_user_agent();
    }
    trimmed.to_string()
}

/// A header value from a config string, or `None` when it is empty or
/// not a legal `HeaderValue`.
///
/// Note that **non-ASCII is valid** — `HeaderValue::from_str("Métro")` is
/// `Ok`. Only control characters (byte < 32 except tab, and DEL) are
/// rejected.
fn header_value(raw: &str) -> Option<HeaderValue> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    HeaderValue::from_str(trimmed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

    /// wiremock's `header()` matcher splits values on commas, which
    /// breaks a User-Agent like `claude-cli/2.1.224 (external, ...)`.
    /// Compare the raw value verbatim instead.
    struct HeaderVerbatim(&'static str, &'static str);

    impl Match for HeaderVerbatim {
        fn matches(&self, request: &Request) -> bool {
            request.headers.get(self.0).and_then(|v| v.to_str().ok()) == Some(self.1)
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn the_global_user_agent_prefers_the_configured_value() {
        assert_eq!(
            resolve_global_user_agent("omp/18.14.0"),
            "omp/18.14.0".to_string()
        );
    }

    #[test]
    fn an_empty_or_illegal_global_user_agent_falls_back_to_the_default() {
        assert_eq!(
            resolve_global_user_agent("   "),
            default_user_agent(),
            "an empty value must not become an empty User-Agent"
        );
        assert_eq!(
            resolve_global_user_agent("omp/1.0\nX: 1"),
            default_user_agent(),
            "an illegal value would make ClientBuilder::build() fail"
        );
    }

    #[test]
    fn a_user_agent_override_is_attached() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(HeaderVerbatim("user-agent", "omp/18.14.0"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let identity = OutboundIdentity::from_parts(
            "p",
            Some("omp/18.14.0"),
            None,
            &BTreeMap::new(),
        );
        identity
            .apply(reqwest::Client::new().get(&server.uri()))
            .send()
            .await
            .unwrap();
    }

    #[test]
    fn without_an_override_the_global_default_is_left_in_place() {
        // Nothing is attached, so the shared client's default User-Agent
        // still reaches the wire.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(HeaderVerbatim("user-agent", "global/1.0"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let client = crate::proxy_client::build_client(
            &crate::config::ProxyConfig::default(),
            "global/1.0",
        )
        .unwrap();
        let identity = OutboundIdentity::from_parts("p", None, None, &BTreeMap::new());
        identity.apply(client.get(&server.uri())).send().await.unwrap();
    }

    #[test]
    fn an_illegal_user_agent_override_is_dropped() {
        let identity = OutboundIdentity::from_parts(
            "p",
            Some("omp/1.0\nX: 1"),
            None,
            &BTreeMap::new(),
        );
        assert!(identity.user_agent.is_none());
    }

    #[test]
    fn the_referer_becomes_a_http_referer_header() {
        let identity = OutboundIdentity::from_parts(
            "p",
            None,
            Some("https://omp.sh/"),
            &BTreeMap::new(),
        );
        let map = identity.header_map();
        assert_eq!(map.get("http-referer").unwrap(), "https://omp.sh/");
    }

    #[test]
    fn an_explicit_referer_header_replaces_the_referer_field() {
        let identity = OutboundIdentity::from_parts(
            "p",
            None,
            Some("https://omp.sh/"),
            &headers(&[("HTTP-Referer", "https://other.example/")]),
        );
        let map = identity.header_map();
        assert_eq!(map.get("http-referer").unwrap(), "https://other.example/");
    }

    #[test]
    fn an_illegal_referer_is_dropped() {
        let identity = OutboundIdentity::from_parts("p", None, Some("bad\nvalue"), &BTreeMap::new());
        assert!(identity.header_map().is_empty());
    }

    #[test]
    fn illegal_and_reserved_headers_are_dropped() {
        let identity = OutboundIdentity::from_parts(
            "p",
            None,
            None,
            &headers(&[
                ("bad header", "x"),
                ("User-Agent", "evil/1.0"),
                ("x-bad-value", "bad\nvalue"),
                ("x-empty", ""),
            ]),
        );
        assert!(identity.header_map().is_empty());
    }

    #[test]
    fn the_session_id_is_stable_and_the_request_id_is_not() {
        let identity = OutboundIdentity::from_parts(
            "p",
            None,
            None,
            &headers(&[
                ("x-session", VALUE_AUTO),
                ("x-request", VALUE_AUTO_REQUEST),
            ]),
        );
        let first = identity.header_map();
        let second = identity.header_map();
        assert_eq!(
            first.get("x-session").unwrap(),
            second.get("x-session").unwrap()
        );
        assert_ne!(
            first.get("x-request").unwrap(),
            second.get("x-request").unwrap()
        );
    }

    #[test]
    fn the_default_identity_is_empty() {
        let identity = OutboundIdentity::default();
        assert!(identity.user_agent.is_none());
        assert!(identity.header_map().is_empty());
    }
}
