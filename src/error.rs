use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

use crate::conversion::util::SchemaError;
use crate::router::RouteStep;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    #[error("all providers cooling down for model {model}")]
    AllProvidersCoolingDown {
        model: String,
        /// Every step taken before giving up — each entry is a
        /// `RouteStep::LocalSkip` (cooldown or model_rewrite miss; no
        /// upstream call fired for this chain) or an upstream `Failed`.
        /// Populated by `Router::complete`/`stream` at the terminal arm
        /// so the `fail_reason=` log field, the `/admin/usage` row and
        /// the response header can all be derived from it (plan §Phase 1
        /// §空切片语义: a chain of length > 0 always pushes at least one
        /// step before reaching this arm; an empty chain leaves this
        /// field empty and the terminal label defaults to `cooldown`).
        attempts: Vec<RouteStep>,
        /// `Retry-After` hint (seconds) for the caller. `None` means
        /// every candidate provider is on cooldown but no further
        /// information is available (e.g. the chain has no known
        /// providers); `Some(0)` is reserved for "retry immediately"
        /// and is not currently produced.
        retry_after_secs: Option<u64>,
    },

    /// Every candidate provider either failed an upstream call or was
    /// on cooldown; `attempts` records the upstream attempts that
    /// actually fired (used to populate the `x-llmproxy-failed-providers`
    /// header), and `last` is the most recent upstream error so the
    /// caller can see the real cause instead of a generic "all cooling
    /// down" message.
    #[error("all providers failed for model {model}: last error: {last}")]
    AllProvidersFailed {
        model: String,
        attempts: Vec<RouteStep>,
        #[source]
        last: Box<ProxyError>,
    },

    #[error("upstream error: status={status}, body={body}")]
    Upstream { status: u16, body: String },

    /// Explicitly marks an error whose message is upstream text.
    ///
    /// `Other` carries both purely client-side errors (e.g. "copilot bootstrap
    /// already in progress") and errors that carry an upstream hint (a Copilot
    /// token refresh failure, a GitHub device-flow response body); collapsing
    /// the two would cost usability. So every error that really does carry
    /// upstream text uses this variant, which `without_upstream_details()` /
    /// `client_message(false)` genericise; a client-side `Other` stays readable.
    ///
    /// Unlike `Upstream` it takes no part in the cooldown decision and carries
    /// no status meaning: it says what the upstream *said*, not what it answered.
    #[error("{message}")]
    UpstreamText { message: String },

    #[error("authentication failed")]
    Unauthorized,

    #[error("bad request: {0}")]
    BadRequest(String),

    /// Router-side all-unmappable terminal: every chain entry was
    /// excluded by `model_rewrite`, so no upstream call could fire.
    /// Carries `attempts` so the header/log/UsageRecord paths can
    /// share the same shape as `AllProvidersFailed` — see plan
    /// §Phase 1 commit 4 (`BadRequest` split, round-6 BLOCKER fix).
    #[error("{message}")]
    RouterBadRequest {
        message: String,
        attempts: Vec<RouteStep>,
    },

    /// PR-13: a schema that cannot be strictified for OpenAI strict mode
    /// (external URI `$ref`, circular `$ref`) is a client-side request
    /// problem — surface it as a 400 rather than forwarding a schema the
    /// upstream will reject.
    #[error(transparent)]
    Schema(#[from] SchemaError),

    #[error("internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Yaml(#[from] serde_saphyr::Error),

    /// Any other error: the catch-all.
    ///
    /// Note: **do not** put an error that carries upstream text in here -- an
    /// error that says what the upstream said uses `UpstreamText`, so it is
    /// genericised when `expose_upstream_details=false`. What is left in `Other`
    /// is purely client-side / proxy-side text (local file-path errors, ...).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ProxyError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Unauthorized => StatusCode::UNAUTHORIZED,
            ProxyError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ProxyError::RouterBadRequest { .. } => StatusCode::BAD_REQUEST,
            ProxyError::ProviderNotFound(_) => StatusCode::NOT_FOUND,
            ProxyError::AllProvidersCoolingDown { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ProxyError::AllProvidersFailed { .. } => StatusCode::BAD_GATEWAY,
            ProxyError::Upstream { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            // 500 like `Other` / `Internal`: the error says what the upstream
            // said, not what status it answered with.
            ProxyError::UpstreamText { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Schema(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_cooldownable(&self) -> bool {
        match self {
            ProxyError::Upstream { status, .. } => {
                matches!(*status, 401 | 402 | 404 | 408 | 429) || *status >= 500
            }
            // `UpstreamText` carries no status and takes no part in cooldown.
            _ => false,
        }
    }

    /// Per-provider failure records for the `x-llmproxy-failed-providers`
    /// header. Returns `None` for errors that didn't actually go through
    /// the router (e.g. malformed JSON, bad request shape).
    pub fn failed_providers_header(&self) -> Option<String> {
        match self {
            ProxyError::AllProvidersFailed { attempts, .. } => {
                Some(format_attempts_header(attempts))
            }
            _ => None,
        }
    }

    /// The generic body a client sees when the deployment does not
    /// expose upstream details.
    ///
    /// Shaped like an Anthropic error object so the client parses it
    /// with the same code path it uses for every other error.
    pub const GENERIC_UPSTREAM_BODY: &'static str =
        r#"{"type":"error","error":{"type":"upstream_error","message":"upstream request failed"}}"#;

    /// The generic message used in an in-stream `event: error` when the
    /// deployment does not expose upstream details.
    pub const GENERIC_UPSTREAM_MESSAGE: &'static str = "upstream request failed";

    /// Strip the upstream body from this error.
    ///
    /// Used when `server.expose_upstream_details` is `false`: the
    /// client gets the status code and a generic message, but not the
    /// upstream's own error text (which can carry account, quota and
    /// backend details). The logs and the `/admin/usage` rows keep the
    /// full detail — they are read by the operator, not by the client.
    ///
    /// `AllProvidersFailed` collapses into the plain upstream error, so
    /// the `x-llmproxy-failed-providers` header (which names the
    /// failing backends) disappears with it.
    ///
    /// `UpstreamText` is the marker for "this really does carry upstream text"
    /// (see its definition); `Http` (whose Display holds the upstream URL) and
    /// `Internal` (which may hold upstream response parsing details) are
    /// genericised too. A client-side `Other` (e.g. "copilot bootstrap already
    /// in progress") stays as it is: it is about the client/proxy, not the upstream.
    pub fn without_upstream_details(self) -> Self {
        match self {
            ProxyError::Upstream { status, .. } => ProxyError::Upstream {
                status,
                body: Self::GENERIC_UPSTREAM_BODY.to_string(),
            },
            ProxyError::AllProvidersFailed { last, .. } => {
                // Keep the status the client would have seen today (the
                // last upstream error's status), drop everything else.
                let status = last.status_code().as_u16();
                ProxyError::Upstream {
                    status,
                    body: Self::GENERIC_UPSTREAM_BODY.to_string(),
                }
            }
            // Carries upstream text: replace it with the generic wording.
            ProxyError::UpstreamText { .. } => ProxyError::UpstreamText {
                message: Self::GENERIC_UPSTREAM_MESSAGE.to_string(),
            },
            // A reqwest error's Display holds the upstream URL.
            ProxyError::Http(_) => ProxyError::UpstreamText {
                message: Self::GENERIC_UPSTREAM_MESSAGE.to_string(),
            },
            // An internal error may hold upstream response parsing details.
            ProxyError::Internal(_) => ProxyError::UpstreamText {
                message: Self::GENERIC_UPSTREAM_MESSAGE.to_string(),
            },
            other => other,
        }
    }

    /// The message a client may see for this error.
    ///
    /// With upstream details exposed this is the error's own `Display`
    /// (which embeds the upstream body for upstream errors); without
    /// them it is the generic wording.
    pub fn client_message(&self, expose_upstream_details: bool) -> String {
        if expose_upstream_details {
            return self.to_string();
        }
        match self {
            // Only the variants that carry upstream text (or the upstream URL) are
            // replaced; errors about the request shape are backend-agnostic and readable.
            ProxyError::Upstream { .. }
            | ProxyError::AllProvidersFailed { .. }
            | ProxyError::UpstreamText { .. }
            | ProxyError::Http(_)
            | ProxyError::Internal(_) => Self::GENERIC_UPSTREAM_MESSAGE.to_string(),
            other => other.to_string(),
        }
    }
}

fn format_attempts_header(attempts: &[RouteStep]) -> String {
    crate::router::format_attempts_header(attempts, None).unwrap_or_default()
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();

        if let ProxyError::AllProvidersFailed { attempts, last, .. } = self {
            // Defer to the wrapped upstream error for status & body, but
            // tack the per-provider failure summary onto the response so
            // operators can see which providers actually returned errors
            // (the upstream body only shows the *last* attempt's status).
            let mut resp = last.into_response();
            if !attempts.is_empty() {
                if let Ok(v) = format_attempts_header(&attempts).parse() {
                    resp.headers_mut()
                        .insert("x-llmproxy-failed-providers", v);
                }
            }
            return resp;
        }

        if let ProxyError::Upstream { body, .. } = &self {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) {
                return (status, Json(parsed)).into_response();
            }
            return (status, body.clone()).into_response();
        }

        let body = json!({
            "type": "error",
            "error": {
                "type": status.canonical_reason().unwrap_or("error"),
                "message": self.to_string(),
            }
        });

        // Surface `Retry-After` for the all-cooldown case so clients can
        // back off instead of retrying into another cooldown window. The
        // header value is the soonest-expiring cooldown remaining (in
        // whole seconds, rounded up); we deliberately don't include it
        // for the "no known providers" sub-case (retry_after_secs = None)
        // because there's nothing meaningful to wait for.
        let mut resp = (status, Json(body)).into_response();
        if let ProxyError::AllProvidersCoolingDown {
            retry_after_secs: Some(secs),
            ..
        } = &self
        {
            if let Ok(v) = secs.to_string().parse() {
                resp.headers_mut().insert("retry-after", v);
            }
        }
        resp
    }
}

pub type Result<T> = std::result::Result<T, ProxyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_code_per_variant() {
        assert_eq!(
            ProxyError::Unauthorized.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ProxyError::BadRequest("x".into()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ProxyError::ProviderNotFound("x".into()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ProxyError::AllProvidersCoolingDown {
                model: "x".into(),
                attempts: vec![],
                retry_after_secs: None,
            }
            .status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            ProxyError::Internal("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            ProxyError::Config("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        // PR-13: an un-strictifiable schema (external/circular $ref) is a
        // client request problem → 400.
        assert_eq!(
            ProxyError::Schema(crate::conversion::util::SchemaError {
                message: "strictify_schema: external $ref at $ not supported".into(),
            })
            .status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ProxyError::Upstream {
                status: 503,
                body: "x".into()
            }
            .status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn cooldownable_status_codes() {
        fn check(status: u16, expected: bool) {
            let actual = ProxyError::Upstream {
                status,
                body: "x".into(),
            }
            .is_cooldownable();
            assert_eq!(actual, expected, "status {status} expected cooldownable={expected}");
        }
        for s in [401u16, 402, 404, 408, 429, 500, 502, 503, 504] {
            check(s, true);
        }
        for s in [400u16, 403, 409] {
            check(s, false);
        }
        // Non-Upstream errors are never cooldownable.
        assert!(!ProxyError::Internal("x".into()).is_cooldownable());
        assert!(!ProxyError::BadRequest("x".into()).is_cooldownable());
        assert!(!ProxyError::Unauthorized.is_cooldownable());
    }

    #[test]
    fn upstream_json_body_passes_through() {
        let body = serde_json::json!({"error": {"type": "rate_limit", "message": "slow down"}});
        let err = ProxyError::Upstream {
            status: 429,
            body: body.to_string(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "application/json"
        );
    }

    #[test]
    fn upstream_plain_body_passes_through() {
        let err = ProxyError::Upstream {
            status: 502,
            body: "not json".into(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn invalid_upstream_status_falls_back_to_502() {
        // Zero cannot be represented as an HTTP status code.
        let err = ProxyError::Upstream {
            status: 0,
            body: "x".into(),
        };
        // We don't crash — fall back to BAD_GATEWAY.
        assert_eq!(err.status_code(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn all_providers_failed_status_is_bad_gateway() {
        let err = ProxyError::AllProvidersFailed {
            model: "m".into(),
            attempts: vec![RouteStep::Failed {
                provider: "primary".into(),
                status: 503,
                body: "x".into(),
            }],
            last: Box::new(ProxyError::Upstream {
                status: 503,
                body: "down".into(),
            }),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn all_providers_failed_failed_providers_header_is_comma_separated() {
        let err = ProxyError::AllProvidersFailed {
            model: "m".into(),
            attempts: vec![
                RouteStep::Failed {
                    provider: "primary".into(),
                    status: 429,
                    body: "x".into(),
                },
                RouteStep::Failed {
                    provider: "backup".into(),
                    status: 503,
                    body: "y".into(),
                },
            ],
            last: Box::new(ProxyError::Upstream {
                status: 503,
                body: "down".into(),
            }),
        };
        assert_eq!(
            err.failed_providers_header().as_deref(),
            Some("primary:429,backup:503")
        );
    }

    #[test]
    fn all_providers_cooling_down_with_retry_after_sets_header() {
        let err = ProxyError::AllProvidersCoolingDown {
            model: "m".into(),
            attempts: vec![],
            retry_after_secs: Some(7),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers()["retry-after"], "7");
    }

    #[test]
    fn all_providers_cooling_down_without_retry_after_omits_header() {
        let err = ProxyError::AllProvidersCoolingDown {
            model: "m".into(),
            attempts: vec![],
            retry_after_secs: None,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            resp.headers().get("retry-after").is_none(),
            "retry-after must be absent when we have no cooldown info"
        );
    }

    #[test]
    fn all_providers_failed_into_response_sets_failed_providers_header() {
        // The HTTP response must carry both the upstream body (so the
        // caller sees the *last* provider's real error) and the
        // per-provider failure summary as a header — without the
        // header, operators only see one provider's status and can't
        // tell that the chain was exhausted.
        let err = ProxyError::AllProvidersFailed {
            model: "m".into(),
            attempts: vec![
                RouteStep::Failed {
                    provider: "primary".into(),
                    status: 500,
                    body: "primary down".into(),
                },
                RouteStep::Failed {
                    provider: "backup".into(),
                    status: 502,
                    body: "backup down".into(),
                },
            ],
            last: Box::new(ProxyError::Upstream {
                status: 502,
                body: "backup down".into(),
            }),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()["x-llmproxy-failed-providers"],
            "primary:500,backup:502"
        );
    }

    #[tokio::test]
    async fn without_upstream_details_replaces_the_upstream_body() {
        let err = ProxyError::Upstream {
            status: 429,
            body: "{\"quota\": \"exhausted for account foo\"}".into(),
        }
        .without_upstream_details();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = http_body_util::BodyExt::collect(resp.into_body())
            .await
            .unwrap()
            .to_bytes();
        let body = String::from_utf8_lossy(&bytes);
        assert!(!body.contains("exhausted for account"), "{body}");
        assert!(body.contains("upstream request failed"), "{body}");
    }

    #[test]
    fn without_upstream_details_drops_the_failed_providers_header() {
        let err = ProxyError::AllProvidersFailed {
            model: "m".into(),
            attempts: vec![RouteStep::Failed {
                provider: "primary".into(),
                status: 500,
                body: "primary down".into(),
            }],
            last: Box::new(ProxyError::Upstream {
                status: 500,
                body: "primary down".into(),
            }),
        }
        .without_upstream_details();

        let resp = err.into_response();
        assert!(
            resp.headers().get("x-llmproxy-failed-providers").is_none(),
            "the topology header must not survive the sanitization"
        );
    }

    #[test]
    fn client_message_hides_only_the_upstream_variants() {
        let upstream = ProxyError::Upstream {
            status: 500,
            body: "upstream said no".into(),
        };
        assert_eq!(
            upstream.client_message(true),
            upstream.to_string(),
            "with details exposed the message is the error's own Display"
        );
        assert_eq!(
            upstream.client_message(false),
            ProxyError::GENERIC_UPSTREAM_MESSAGE
        );

        // A client-side error says nothing about the backend, so it
        // stays informative.
        let bad = ProxyError::BadRequest("model 'x' is not configured".into());
        assert_eq!(bad.client_message(false), bad.to_string());
    }

    /// A2: an error carrying upstream text (`UpstreamText`) becomes the generic
    /// wording when the switch is off; `Http` (Display holds the upstream URL)
    /// and `Internal` are genericised as well; a client-side `Other` stays readable.
    #[test]
    fn client_message_hides_upstream_carrying_variants() {
        let text = ProxyError::UpstreamText {
            message: "copilot /models returned 503: quota exhausted".into(),
        };
        assert_eq!(
            text.client_message(false),
            ProxyError::GENERIC_UPSTREAM_MESSAGE,
            "upstream text must be replaced by the generic wording"
        );
        assert!(
            text.client_message(true).contains("quota exhausted"),
            "verbatim when the switch is on"
        );

        let internal = ProxyError::Internal("failed to parse provider response: x".into());
        assert_eq!(
            internal.client_message(false),
            ProxyError::GENERIC_UPSTREAM_MESSAGE
        );

        // A client-side Other holds no upstream information and stays readable.
        let client_side = ProxyError::Other(anyhow::anyhow!("copilot bootstrap already in progress"));
        assert!(client_side
            .client_message(false)
            .contains("bootstrap already in progress"));
    }

    /// A2: `without_upstream_details()` genericises the variants that carry
    /// upstream text while keeping the status semantics.
    #[tokio::test]
    async fn without_upstream_details_sanitizes_upstream_carrying_variants() {
        let text = ProxyError::UpstreamText {
            message: "copilot /models returned 503: quota exhausted".into(),
        }
        .without_upstream_details();
        assert_eq!(text.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!text.to_string().contains("quota exhausted"), "{text}");

        // A failed send's Display holds the upstream URL.
        let http_err = reqwest::Client::new()
            .get("http://127.0.0.1:1/models")
            .timeout(std::time::Duration::from_millis(50))
            .send()
            .await
            .expect_err("connecting to a closed port must fail");
        assert!(
            http_err.to_string().contains("127.0.0.1"),
            "precondition: a reqwest error's Display holds a URL, got: {http_err}"
        );
        let sanitized = ProxyError::Http(http_err).without_upstream_details();
        assert!(!sanitized.to_string().contains("127.0.0.1"), "{sanitized}");

        // A client-side Other is unaffected.
        let client_side =
            ProxyError::Other(anyhow::anyhow!("copilot bootstrap already in progress"));
        assert!(client_side
            .without_upstream_details()
            .to_string()
            .contains("bootstrap already in progress"));
    }

    /// Non-`AllProvidersFailed` errors must not produce the failed-providers
    /// header. The `_ => None` arm of `failed_providers_header()` is the
    /// gate; without this test it stays uncovered.
    #[test]
    fn failed_providers_header_is_none_for_non_all_failed_variants() {
        let err = ProxyError::Upstream {
            status: 502,
            body: "bad gateway".into(),
        };
        assert!(err.failed_providers_header().is_none());

        let err = ProxyError::BadRequest("nope".into());
        assert!(err.failed_providers_header().is_none());

        let err = ProxyError::ProviderNotFound("m".into());
        assert!(err.failed_providers_header().is_none());
    }
}
