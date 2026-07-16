use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    #[error("all providers cooling down for model {0}")]
    AllProvidersCoolingDown(String),

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
            ProxyError::AllProvidersCoolingDown(_) => StatusCode::SERVICE_UNAVAILABLE,
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
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();

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
        (status, Json(body)).into_response()
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
            ProxyError::AllProvidersCoolingDown("x".into()).status_code(),
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
}
