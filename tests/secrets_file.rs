//! Integration test for the `secrets:` file provider — the llmproxy
//! half of the TPM-sealed `.env` plan
//! (`plans/feat-vault-secret-backend.md` §5.2 / §5.5).
//!
//! The unit tests in `src/config.rs` drive `Config::parse` with an
//! in-memory string; these drive `Config::load` with a real file on
//! disk, which is how the container deployment actually reaches the
//! proxy: the entrypoint unseals the TPM sealed object into
//! `/run/secrets/llmproxy.env` and the proxy reads that path.
//!
//! Every test uses its own environment-variable name — the process
//! environment is global and the test harness runs tests in parallel.

use std::fs;
use std::path::Path;

use secrecy::ExposeSecret;

use llmproxy::config::{Config, ProviderConfig};
use llmproxy::secrets::SecretsProvider;

/// Write `content` to a file inside `dir` and return its path.
fn write(dir: &Path, name: &str, content: &str) -> String {
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    path.display().to_string()
}

/// A config whose provider key comes from `${<var>}`.
fn config_yaml(var: &str, secrets_block: &str) -> String {
    format!(
        r#"
{secrets_block}
providers:
  - name: compat
    type: openai_compat
    api_key: "${{{var}}}"
    api_base: "https://example.test/v1"
models:
  - name: m
    primary: compat
"#
    )
}

fn api_key_of(cfg: &Config) -> String {
    match &cfg.providers[0] {
        ProviderConfig::OpenaiCompat { api_key, .. } => api_key.expose_secret().clone(),
        other => panic!("expected an openai_compat provider, got {other:?}"),
    }
}

#[test]
fn loads_api_key_from_the_secrets_file() {
    let dir = tempfile::tempdir().unwrap();
    let var = "LLMPROXY_IT_FILE";
    let secrets = write(dir.path(), "llmproxy.env", &format!("{var}=from-file\n"));
    let config = write(
        dir.path(),
        "config.yaml",
        &config_yaml(
            var,
            &format!(
                "secrets:\n  provider: file\n  file:\n    path: {secrets}\n    required: true"
            ),
        ),
    );

    let cfg = Config::load(&config).unwrap();
    assert_eq!(api_key_of(&cfg), "from-file");
    assert_eq!(cfg.secrets.provider, SecretsProvider::File);
}

#[test]
fn secrets_file_wins_over_the_process_environment() {
    let dir = tempfile::tempdir().unwrap();
    let var = "LLMPROXY_IT_ENVWINS";
    let secrets = write(dir.path(), "llmproxy.env", &format!("{var}=from-file\n"));
    let config = write(
        dir.path(),
        "config.yaml",
        &config_yaml(
            var,
            &format!("secrets:\n  provider: file\n  file:\n    path: {secrets}"),
        ),
    );
    std::env::set_var(var, "from-env");

    let cfg = Config::load(&config).unwrap();
    assert_eq!(api_key_of(&cfg), "from-file");
}

#[test]
fn missing_required_file_fails_the_load() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("absent.env").display().to_string();
    let config = write(
        dir.path(),
        "config.yaml",
        &config_yaml(
            "LLMPROXY_IT_REQUIRED",
            &format!(
                "secrets:\n  provider: file\n  file:\n    path: {missing}\n    required: true"
            ),
        ),
    );

    let err = Config::load(&config).unwrap_err();
    assert!(
        format!("{err}").contains("read secrets file"),
        "unexpected error: {err}"
    );
}

#[test]
fn missing_optional_file_leaves_the_config_usable() {
    let dir = tempfile::tempdir().unwrap();
    let var = "LLMPROXY_IT_OPTIONAL";
    let missing = dir.path().join("absent.env").display().to_string();
    let config = write(
        dir.path(),
        "config.yaml",
        &config_yaml(
            var,
            &format!(
                "secrets:\n  provider: file\n  file:\n    path: {missing}\n    required: false"
            ),
        ),
    );
    std::env::set_var(var, "from-env");

    let cfg = Config::load(&config).unwrap();
    assert_eq!(api_key_of(&cfg), "from-env");
}

#[test]
fn a_config_without_a_secrets_block_still_works() {
    let dir = tempfile::tempdir().unwrap();
    let var = "LLMPROXY_IT_NOBLOCK";
    let config = write(dir.path(), "config.yaml", &config_yaml(var, ""));
    std::env::set_var(var, "from-env");

    let cfg = Config::load(&config).unwrap();
    assert_eq!(api_key_of(&cfg), "from-env");
    assert_eq!(cfg.secrets.provider, SecretsProvider::None);
}
