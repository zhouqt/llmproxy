//! Sandbox canary -- a compile-time isolation self-check.
//!
//! It probes two things; a breach of either one fails the build:
//! 1. **Egress**: can the build reach the network? (Third-party code that runs at build time can exfiltrate if it can.)
//! 2. **Credentials**: does the environment hold token-shaped variables that must not be there?
//!    (A container with `--network=none` blocks egress, **but not environment variables** -- hence this separate check.)
//!
//! Note: only the **presence of a variable name** is checked here; values are never printed.
//!
//! There is no "relaxed mode": a build environment with network access
//! simply does not pass. There used to be a `SANDBOX_CANARY_RELAXED=1`
//! switch whose only effect was to let a build that **should fail** pass
//! quietly. The switch itself was a hole, so it is gone.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The probe targets. Reaching any of them counts as "not isolated".
const PROBES: &[(&str, u16)] = &[
    ("static.crates.io", 443),
    ("index.crates.io", 443),
    ("crates.io", 443),
    ("1.1.1.1", 443),
];

/// Environment variable names that must not exist at build time. Names only, never values.
const FORBIDDEN_ENV: &[&str] = &[
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_PAT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CI_JOB_TOKEN",
    "NPM_TOKEN",
    "NODE_AUTH_TOKEN",
    "CARGO_REGISTRY_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
];

/// A variable whose name contains this fragment is a violation.
///
/// A proxy variable can re-route compile-time egress to an uncontrolled exit --
/// exactly the breach this canary looks for, just from the other side. So the
/// rule is "the name contains PROXY", not a list of the known variables.
const PROXY_MARKER: &str = "PROXY";

/// The one exception: `NO_PROXY` / `no_proxy`.
///
/// `NO_PROXY` is an **exclusion list** ("hosts that bypass the proxy"), not
/// an exit: it sends no traffic anywhere; it keeps traffic direct. Not a credential.
///
/// And `run_in_container()` in `tools/cargo-in-container.sh` injects
/// injects `-e NO_PROXY=...` / `-e no_proxy=...` unconditionally (the test
/// suite talks to mock servers on 127.0.0.1 and reqwest sends loopback
/// requests to the proxy too), so a bare "name contains PROXY" rule would
/// always report a false positive; a check that is always red is skipped.
///
/// Matching is case-insensitive on the name.
const NO_PROXY_ALLOWED: &str = "NO_PROXY";

fn main() {
    let mut problems: Vec<String> = Vec::new();

    for (host, port) in PROBES {
        let reachable = (*host, *port)
            .to_socket_addrs()
            .map(|addrs| {
                addrs
                    .into_iter()
                    .any(|a| TcpStream::connect_timeout(&a, Duration::from_secs(3)).is_ok())
            })
            .unwrap_or(false);
        if reachable {
            problems.push(format!("network reachable: {host}:{port}"));
        }
    }

    for name in FORBIDDEN_ENV {
        if std::env::var_os(name).is_some() {
            problems.push(format!("proxy variable present: {name}"));
        }
    }

    // Names only, never values: a value is never printed.
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy().to_uppercase();
        // `NO_PROXY` is an exclusion list, not an exit -- see NO_PROXY_ALLOWED.
        if name == NO_PROXY_ALLOWED {
            continue;
        }
        if name.contains(PROXY_MARKER) {
            problems.push(format!("proxy variable present: {name}"));
        }
    }

    if problems.is_empty() {
        println!("sandbox-canary: isolation confirmed (no egress / no credential variables)");
        return;
    }

    for p in &problems {
        println!("sandbox-canary: {p}");
    }

    panic!(
        "sandbox-canary: the sandbox has been broken -- {}. The build environment must not have these capabilities.",
        problems.join("; ")
    );
}
