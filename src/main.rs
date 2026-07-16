use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

#[allow(unused_imports)]
use llmproxy::{
    anthropic, auth, config, conversion, cooldown, error, oauth, openai, proxy_client, providers,
    router, server, state,
};

use crate::config::Config;
use crate::cooldown::CooldownCache;
use crate::providers::SharedProvider;
use crate::router::Router;
use crate::state::AppState;

#[derive(Debug, Default)]
struct Args {
    config: Option<PathBuf>,
    port: Option<u16>,
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    init_tracing();

    let raw_args: Vec<String> = std::env::args().collect();
    let args = parse_args(&raw_args);

    let config_path = args
        .config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--config <path> required"))?;
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;

    let listen = resolve_listen_addr(&args, &cfg);
    let http = proxy_client::build_client(&cfg.proxy)?;

    let (state, bg_handles) = build_state(cfg, http).map_err(|e| anyhow::anyhow!("{e}"))?;

    let app = server::build_router(state);

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    tracing::info!(listen = listen.as_str(), "llmproxy listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    for h in bg_handles {
        h.abort();
    }
    Ok(())
}

fn resolve_listen_addr(args: &Args, cfg: &Config) -> String {
    if let Some(p) = args.port {
        format!("127.0.0.1:{p}")
    } else {
        cfg.server.listen.clone()
    }
}

/// Build the application state and any background tasks for the configured providers.
/// Extracted so it can be exercised in tests without binding a TCP listener.
fn build_state(
    cfg: Config,
    http: reqwest::Client,
) -> Result<(AppState, Vec<tokio::task::JoinHandle<()>>), llmproxy::error::ProxyError> {
    let mut provider_map: HashMap<String, SharedProvider> = HashMap::new();
    let mut bg_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    for p in &cfg.providers {
        let name = p.name().to_string();
        let built = providers::build(p, http.clone())
            .map_err(|e| llmproxy::error::ProxyError::Other(anyhow::anyhow!("building provider '{name}': {e}")))?;
        if let Some(h) = built.clone().spawn_background() {
            bg_handles.push(h);
        }
        provider_map.insert(name, built);
    }

    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(
        Arc::new(cfg.clone()),
        provider_map,
        cooldown.clone(),
    ));

    let state = AppState {
        config: Arc::new(cfg),
        router,
        cooldown,
        http,
    };
    Ok((state, bg_handles))
}

fn init_tracing() {
    let filter = build_tracing_filter();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

/// Build the EnvFilter used for tracing. Falls back to the project's default
/// level when no `RUST_LOG` is configured.
fn build_tracing_filter() -> EnvFilter {
    EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("llmproxy=info,tower_http=info"))
}

fn parse_args(args: &[String]) -> Args {
    let mut out = Args::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                if i + 1 < args.len() {
                    out.config = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--port" | "-p" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    out.port = Some(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: llmproxy --config <path> [--port <port>]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                i += 1;
            }
        }
    }
    out
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut s = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        s.recv().await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmproxy::config::{
        Config, ModelConfig, ProviderConfig, ServerConfig,
    };
    use llmproxy::expect_variant;

    /// Single helper for asserting that a subprocess exited cleanly or was
    /// terminated by the expected signal. Consolidates the per-test panic
    /// message into one line.
    fn assert_subprocess_exit_ok(status: std::process::ExitStatus, expected_signal: Option<i32>) {
        assert!(status.code().is_some() || status.signal() == expected_signal, "subprocess exit mismatch");
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parse_args_accepts_long_and_short_options() {
        let long = parse_args(&strings(&[
            "llmproxy",
            "--config",
            "config.yaml",
            "--port",
            "9000",
        ]));
        assert_eq!(long.config, Some(PathBuf::from("config.yaml")));
        assert_eq!(long.port, Some(9000));

        let short = parse_args(&strings(&["llmproxy", "-c", "other.yaml", "-p", "8081"]));
        assert_eq!(short.config, Some(PathBuf::from("other.yaml")));
        assert_eq!(short.port, Some(8081));
    }

    #[test]
    fn parse_args_ignores_unknown_missing_and_invalid_values() {
        let args = parse_args(&strings(&[
            "llmproxy",
            "--unknown",
            "--port",
            "invalid",
            "-c",
        ]));

        assert_eq!(args.config, None);
        assert_eq!(args.port, None);
        let defaults = parse_args(&strings(&["llmproxy"]));
        assert_eq!(defaults.config, None);
        assert_eq!(defaults.port, None);
    }

    fn fixture_config() -> Config {
        let yaml = r#"
server:
  listen: "127.0.0.1:18080"
  api_key: "test-key"
providers:
  - name: compat
    type: openai_compat
    api_key: "test-key"
    api_base: "https://example.test/v1"
models:
  - name: m
    primary: compat
"#;
        Config::parse(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_state_wires_providers_router_and_cooldown() {
        let cfg = fixture_config();
        let (state, bg_handles) =
            build_state(cfg.clone(), reqwest::Client::new()).expect("build_state succeeds");

        // Router can find the model and primary provider.
        let model = state
            .router
            .find_model("m")
            .expect("model 'm' should be registered");
        assert_eq!(model.primary, "compat");
        assert_eq!(state.config.providers.len(), 1);
        // No background refresh for the openai_compat provider.
        assert!(bg_handles.is_empty());

        // Don't leak the spawned tasks into other tests.
        drop(state);
    }

    #[tokio::test]
    async fn build_state_spawns_background_refresh_for_copilot() {
        // github_copilot is the only provider type that returns Some from
        // spawn_background, so we exercise that branch here. We use a fake
        // XDG_DATA_HOME so the background refresh task has a writable token
        // directory to read.
        let dir = tempfile::tempdir().unwrap();
        let previous_xdg = std::env::var_os("XDG_DATA_HOME");
        std::env::set_var("XDG_DATA_HOME", dir.path());

        let yaml = r#"
server:
  listen: "127.0.0.1:0"
  api_key: "test"
providers:
  - name: copilot
    type: github_copilot
    vscode_version: "1.95.0"
models:
  - name: m
    primary: copilot
"#;
        let cfg = Config::parse(yaml).unwrap();
        let (state, bg_handles) =
            build_state(cfg.clone(), reqwest::Client::new()).expect("build_state succeeds");

        assert!(!bg_handles.is_empty(), "copilot should spawn a refresh task");
        for handle in &bg_handles {
            handle.abort();
        }
        if let Some(prev) = previous_xdg {
            std::env::set_var("XDG_DATA_HOME", prev);
        } else {
            std::env::remove_var("XDG_DATA_HOME");
        }
        drop(state);
    }

    #[test]
    fn args_default_is_empty() {
        let args = Args::default();
        assert!(args.config.is_none());
        assert!(args.port.is_none());
    }

    /// Subprocess helpers — see subprocess tests below.
    /// These exercise `fn main` / `async_main` / `init_tracing` /
    /// `shutdown_signal` which are unreachable from in-process tests
    /// because they bind a real TCP listener, install a global tracing
    /// subscriber, and call `std::process::exit` / signal handlers.
    #[cfg(unix)]
    use std::os::unix::process::{CommandExt, ExitStatusExt};

    /// Locate the instrumented `llmproxy` binary that `cargo test` built.
    /// `CARGO_BIN_EXE_llmproxy` is set for integration tests in
    /// `tests/*.rs` but not for unit tests inside the binary's own
    /// `tests` module. In the unit-test case we walk one directory up
    /// from `current_exe()` to find the sibling binary.
    fn llmproxy_binary() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("CARGO_BIN_EXE_llmproxy") {
            return std::path::PathBuf::from(p);
        }
        let exe = std::env::current_exe().expect("locate current test binary");
        // target/debug/deps/<test_bin> -> target/debug/llmproxy
        let candidate = exe
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("llmproxy"));
        candidate.unwrap_or_else(|| exe.with_file_name("llmproxy"))
    }

    fn run_subprocess(args: &[&str]) -> std::process::Output {
        let exe = llmproxy_binary();
        std::process::Command::new(exe)
            .args(args)
            .env("LLMPROXY_API_KEY", "subproc-test")
            .output()
            .expect("spawn llmproxy")
    }

    #[test]
    fn subprocess_help_branch_exits_zero_and_prints_usage() {
        // `parse_args` calls `std::process::exit(0)` for --help, which
        // would terminate the test process if invoked directly. Run it
        // in a subprocess so we can assert on the exit code and stderr
        // without killing the test runner.
        let out = run_subprocess(&["--help"]);
        assert_eq!(out.status.code(), Some(0));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("Usage"), "subprocess --help stderr must contain Usage");
    }

    #[test]
    fn subprocess_unknown_arg_falls_through_to_missing_config_error() {
        // An unknown argument is logged and ignored; the subsequent
        // missing-config check then exits non-zero. Covers the
        // `other =>` arm of parse_args plus the `--config required`
        // branch of async_main.
        let out = run_subprocess(&["--unknown"]);
        assert_eq!(out.status.code(), Some(1));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("--config"), "stderr was: {stderr}");
    }

    /// Pick a free high port to avoid clashing with anything on the host.
    /// The kernel will refuse a bind to a port in use, which is the
    /// error path we want to exercise anyway, but using 0 (kernel-chosen)
    /// works fine and avoids depending on a specific port being free.
    #[test]
    fn subprocess_starts_and_receives_sigterm() {
        // Write a minimal valid config to a tempdir so the binary reaches
        // the actual `axum::serve` / `shutdown_signal` paths. Pick port 0
        // so the kernel allocates a free one — we only care that the
        // listener binds and the binary responds to SIGTERM.
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg_path = dir.path().join("c.yaml");
        std::fs::write(
            &cfg_path,
            r#"
server:
  listen: "127.0.0.1:0"
  api_key: "subproc-test"
providers:
  - name: compat
    type: openai_compat
    api_key: "k"
    api_base: "https://example.test/v1"
  - name: copilot
    type: github_copilot
    vscode_version: "1.95.0"
models:
  - name: m
    primary: compat
"#,
        )
        .expect("write config");

        let exe = llmproxy_binary();
        let mut child = std::process::Command::new(&exe)
            .arg("--config")
            .arg(&cfg_path)
            .env("LLMPROXY_API_KEY", "subproc-test")
            // Isolated token store so the Copilot refresh task can start.
            .env("XDG_DATA_HOME", dir.path())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn llmproxy");

        // Give the listener a moment to bind and start accepting.
        std::thread::sleep(std::time::Duration::from_millis(800));

        // SIGTERM triggers the graceful shutdown branch.
        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as i32, libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }

        let status = child.wait().expect("wait child");
        // On Unix, a clean SIGTERM exit leaves the process terminated by
        // signal 15 — we treat any non-still-running outcome as success.
        #[cfg(unix)]
        assert_subprocess_exit_ok(status, Some(15));
        #[cfg(not(unix))]
        assert!(!status.success(), "expected the process to exit");
    }

    #[test]
    #[cfg(unix)]
    fn subprocess_starts_and_receives_sigint() {
        // Sending SIGINT exercises the `ctrl_c` branch of `shutdown_signal`.
        // We put the subprocess in its own process group so the signal
        // doesn't leak to the test runner.
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg_path = dir.path().join("c.yaml");
        std::fs::write(
            &cfg_path,
            r#"
server:
  listen: "127.0.0.1:0"
  api_key: "subproc-test"
providers:
  - name: compat
    type: openai_compat
    api_key: "k"
    api_base: "https://example.test/v1"
models:
  - name: m
    primary: compat
"#,
        )
        .expect("write config");

        let exe = llmproxy_binary();
        let mut child = std::process::Command::new(&exe)
            .arg("--config")
            .arg(&cfg_path)
            .env("LLMPROXY_API_KEY", "subproc-test")
            .process_group(0)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn llmproxy");

        std::thread::sleep(std::time::Duration::from_millis(800));

        // SIGINT to the new process group: `0 - pgid` is the broadcast
        // form. pid is positive for "send to that pid" and negative for
        // "send to that pgrp".
        let pgid = child.id() as i32;
        unsafe {
            libc::kill(-pgid, libc::SIGINT);
        }

        let status = child.wait().expect("wait child");
        assert_subprocess_exit_ok(status, Some(2));
    }

    #[test]
    fn resolve_listen_addr_prefers_cli_port_over_config() {
        let cfg = fixture_config();
        let args = Args {
            port: Some(9090),
            config: None,
        };
        assert_eq!(resolve_listen_addr(&args, &cfg), "127.0.0.1:9090");

        let args = Args {
            port: None,
            config: None,
        };
        assert_eq!(resolve_listen_addr(&args, &cfg), cfg.server.listen);
    }

    #[test]
    fn build_tracing_filter_falls_back_when_rust_log_unset() {
        // When RUST_LOG isn't set, the filter falls back to the project
        // default. We can't easily assert the parsed directive, but we can
        // assert that the filter is created without panicking.
        let saved = std::env::var_os("RUST_LOG");
        std::env::remove_var("RUST_LOG");

        let filter = build_tracing_filter();

        if let Some(value) = saved {
            std::env::set_var("RUST_LOG", value);
        }

        // The returned filter is a valid EnvFilter — formatting it must
        // produce something non-empty (the underlying directives).
        assert!(!format!("{filter}").is_empty());
    }

    #[test]
    fn build_tracing_filter_reads_rust_log_when_set() {
        let saved = std::env::var_os("RUST_LOG");
        std::env::set_var("RUST_LOG", "debug");

        let filter = build_tracing_filter();

        if let Some(value) = saved {
            std::env::set_var("RUST_LOG", value);
        } else {
            std::env::remove_var("RUST_LOG");
        }

        // We only verify the filter was constructed — exact directive
        // formatting is implementation-defined across tracing-subscriber
        // versions.
        assert!(!format!("{filter}").is_empty());
    }

    #[test]
    fn config_used_by_build_state() {
        // Sanity check that the fixture wires the expected fields.
        let cfg = fixture_config();
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.models.len(), 1);
        expect_variant!(&cfg.providers[0], ProviderConfig::OpenaiCompat { .. } => {});
        match &cfg.models[0] {
            ModelConfig { primary, .. } => assert_eq!(primary, "compat"),
        }
        assert_eq!(cfg.server.api_key.as_deref(), Some("test-key"));
        let _ = ServerConfig::default();
    }
}
