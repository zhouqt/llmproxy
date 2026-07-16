use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

use crate::router::RouteAttempt;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    #[error("all providers cooling down for model {model}")]
    AllProvidersCoolingDown {
        model: String,
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
        attempts: Vec<RouteAttempt>,
        #[source]
        last: Box<ProxyError>,
    },

    #[error("upstream error: status={status}, body={body}")]
    Upstream { status: u16, body: String },

    #[error("authentication failed")]
    Unauthorized,

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ProxyError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Unauthorized => StatusCode::UNAUTHORIZED,
            ProxyError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ProxyError::ProviderNotFound(_) => StatusCode::NOT_FOUND,
            ProxyError::AllProvidersCoolingDown { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ProxyError::AllProvidersFailed { .. } => StatusCode::BAD_GATEWAY,
            ProxyError::Upstream { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_cooldownable(&self) -> bool {
        match self {
            ProxyError::Upstream { status, .. } => {
                matches!(*status, 401 | 404 | 408 | 429) || *status >= 500
            }
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
}

fn format_attempts_header(attempts: &[RouteAttempt]) -> String {
    attempts
        .iter()
        .map(|a| format!("{}:{}", a.provider, a.status))
        .collect::<Vec<_>>()
        .join(",")
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
        for s in [401u16, 404, 408, 429, 500, 502, 503, 504] {
            check(s, true);
        }
        for s in [400u16, 402, 403, 409] {
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
            attempts: vec![RouteAttempt {
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
                RouteAttempt {
                    provider: "primary".into(),
                    status: 429,
                    body: "x".into(),
                },
                RouteAttempt {
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
                RouteAttempt {
                    provider: "primary".into(),
                    status: 500,
                    body: "primary down".into(),
                },
                RouteAttempt {
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
}
