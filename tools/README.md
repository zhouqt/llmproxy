# tools/

Everything that helps **build, check or develop** this project, and that
runs on a developer machine or in CI. Nothing here is part of the running
service — the runtime scripts live in [`../deploy/`](../deploy/).

| Script | What it does |
|---|---|
| `cargo-in-container.sh` | Runs `cargo` (build, test, audit, anything) inside the pinned `rust:1-alpine` image, with the repository mounted read-only. For hosts that have no Rust toolchain of their own. |
| `build-image.sh` | Thin wrapper around `podman build -f Dockerfile` for hosts that cannot reach `docker.io` directly. It holds no proxy value on purpose — export `HTTPS_PROXY` yourself. |
| `scan-image.sh` | Scans the built image with the pinned scanner (Trivy/Grype). |
| `sandbox-canary/` | A probe crate whose `build.rs` fails the build if the compile environment can reach the network or holds credential-shaped environment variables. |
| `check-base-digests.py` | Parses the `FROM` lines of the `Dockerfile` and checks that every base image is pinned by digest. `--resolve` / `--set` move a pin; `--selftest` proves the check goes red. |
| `check-tool-versions.py` | Compares the tool versions pinned in CI with the upstream releases, and can update them. |
| `lock-drift.py` | Rejects a dependency that moved **backwards**. `--locked` cannot catch this: a downgraded lock file is a perfectly consistent lock file. `--selftest` proves the check goes red. |
| `policy-selftest.sh` | Runs every policy gate against the unmodified tree and against deliberately broken copies, so a gate that never goes red cannot pass unnoticed. |
| `scan-secrets.sh` | Audits the working tree and the git history for credential-shaped strings. |
| `install-hooks.sh` | Installs the git hooks below into `.git/hooks/` (symlinks, per clone). `git config core.hooksPath tools/hooks` is the alternative. |
| `hooks/pre-commit` | The pre-commit hook: runs `scan-secrets.sh`. |
