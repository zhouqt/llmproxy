//! Sandbox canary —— 编译期隔离自检。
//!
//! 探测两件事,任何一件破口都让构建失败:
//!   1. **出网**:能否连到外网?(编译期第三方代码若能出网,就能外泄)
//!   2. **凭据**:环境里是否有本不该出现的 token 类变量?
//!      (容器 `--network=none` 挡得住出网,**挡不住环境变量** —— 所以这一项独立检查)
//!
//! 注意:这里只检查**变量名是否存在**,绝不打印变量值。
//!
//! 宽松模式:设置 `SANDBOX_CANARY_RELAXED=1` 时只报告不失败(供本地有网环境手工试跑)。

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// 探测目标。任一可达即视为"未隔离"。
const PROBES: &[(&str, u16)] = &[
    ("static.crates.io", 443),
    ("index.crates.io", 443),
    ("crates.io", 443),
    ("1.1.1.1", 443),
];

/// 编译期不该存在的环境变量名。只看名字,不看值。
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
            problems.push(format!("出网可达: {host}:{port}"));
        }
    }

    for name in FORBIDDEN_ENV {
        if std::env::var_os(name).is_some() {
            problems.push(format!("存在敏感环境变量: {name}"));
        }
    }

    if problems.is_empty() {
        println!("sandbox-canary: 隔离确认(无出网 / 无凭据变量)");
        return;
    }

    for p in &problems {
        println!("sandbox-canary: {p}");
    }

    if relaxed {
        println!("sandbox-canary: SANDBOX_CANARY_RELAXED 已设置,本次不失败");
        return;
    }

    panic!(
        "sandbox-canary: 隔离已被打破 —— {}。编译环境里不该有这些能力。",
        problems.join("; ")
    );
}
