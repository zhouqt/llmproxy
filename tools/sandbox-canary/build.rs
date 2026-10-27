//! Sandbox canary -- a compile-time isolation self-check.
//!
//! It probes two things; a breach of either one fails the build:
//! 1. **Egress**: can the build reach the network? (Third-party code that runs at build time can exfiltrate if it can.)
//! 2. **Credentials**: does the environment hold token-shaped variables that must not be there?
//!    (A container with `--network=none` blocks egress, **but not environment variables** -- hence this separate check.)
//!
//! Note: only the **presence of a variable name** is checked here; values are never printed.
//!
//! Relaxed mode: with `SANDBOX_CANARY_RELAXED=1` the canary only reports and does not fail (for a manual run in a networked environment).

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
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CI_JOB_TOKEN",
    "NPM_TOKEN",
    "CARGO_REGISTRY_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
];

fn main() {
    let relaxed = std::env::var_os("SANDBOX_CANARY_RELAXED").is_some();
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
            problems.push(format!("sensitive environment variable present: {name}"));
        }
    }

    if problems.is_empty() {
        println!("sandbox-canary: isolation confirmed (no egress / no credential variables)");
        return;
    }

    for p in &problems {
        println!("sandbox-canary: {p}");
    }

    if relaxed {
        println!("sandbox-canary: SANDBOX_CANARY_RELAXED is set, this run does not fail");
        return;
    }

    panic!(
        "sandbox-canary: the sandbox has been broken -- {}. The build environment must not have these capabilities.",
        problems.join("; ")
    );
}
