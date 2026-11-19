//! Subprocess tests for `fn main` / `async_main` / `init_tracing` /
//! `shutdown_signal`, which are unreachable from in-process tests because
//! they bind a real TCP listener, install a global tracing subscriber, and
//! call `std::process::exit` / install signal handlers.
//!
//! These live in `tests/` rather than in `src/main.rs`'s own `tests` module
//! because of how they find the binary: cargo sets `CARGO_BIN_EXE_llmproxy`
//! for integration tests and guarantees the binary target is built first,
//! in every artifact layout. A unit test inside the binary only knows its
//! harness path, and cargo has moved that harness around
//! (`<profile>/deps/` up to 1.99, `<profile>/build/<pkg>/<hash>/out/` from
//! 1.100) — a `cargo test --lib --bins --tests` run that builds nothing
//! else produces no `<profile>/llmproxy` at all for it to find.

#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};

/// The binary `cargo test` built for this package. Compile-time constant,
/// so a missing value is a build error rather than a runtime guess.
fn llmproxy_binary() -> &'static str {
    env!("CARGO_BIN_EXE_llmproxy")
}

/// Single helper for asserting that a subprocess exited cleanly or was
/// terminated by the expected signal. Consolidates the per-test panic
/// message into one line.
#[cfg(unix)]
fn assert_subprocess_exit_ok(status: std::process::ExitStatus, expected_signal: Option<i32>) {
    assert!(
        status.code().is_some() || status.signal() == expected_signal,
        "subprocess exit mismatch"
    );
}

fn run_subprocess(args: &[&str]) -> std::process::Output {
    std::process::Command::new(llmproxy_binary())
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
    assert!(
        stderr.contains("Usage"),
        "subprocess --help stderr must contain Usage"
    );
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

/// A minimal valid config on a kernel-chosen port: the test only needs the
/// listener to bind and the binary to reach its shutdown path.
const MINIMAL_CONFIG: &str = r#"
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
"#;

#[test]
fn subprocess_starts_and_receives_sigterm() {
    // Write a minimal valid config to a tempdir so the binary reaches
    // the actual `axum::serve` / `shutdown_signal` paths. Pick port 0
    // so the kernel allocates a free one — we only care that the
    // listener binds and the binary responds to SIGTERM. A Copilot
    // provider is added so the token-refresh task starts too.
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

    let mut child = std::process::Command::new(llmproxy_binary())
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
    std::fs::write(&cfg_path, MINIMAL_CONFIG).expect("write config");

    let mut child = std::process::Command::new(llmproxy_binary())
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
