# Supply chain

What this repository does to keep the bytes it builds trustworthy, what it
deliberately does *not* do, and how the pinned inputs are kept current.

The short version: **the controls that stop something are in the
Dockerfile and in `Cargo.lock`; the controls that tell you something is
wrong are in CI.** Mixing those two up is how supply-chain work usually
turns into a green checkmark that means nothing.

## 1. What is actually enforced

| Control | Where | What it really does | What it does not do |
|---|---|---|---|
| `Cargo.lock` sha256 | cargo | Every crate's bytes are pinned; a tampered tarball fails the checksum | Say anything about whether the crate is *malicious* |
| `--locked` | every cargo invocation | The lock file agrees with the manifest | Detect a **downgrade** — a downgraded lock file is a perfectly consistent lock file. `tools/lock-drift.py` is what catches that |
| `--offline` | builds | Cargo does not download anything | **Sandbox anything.** `build.rs` still runs, with the network and the environment. It is a cache hint, not a security control |
| `RUN --network=none` | Dockerfile | The compile step has no egress at the kernel level | Hide environment variables. Hence the credential half of the canary |
| `cargo-auditable` | Dockerfile | The dependency list is embedded in the binary, so Trivy/Grype can see the Rust crates | Prove the build was clean |
| `cargo-deny` | CI (`deny.toml`) | Advisory / license / source policy | Prevent a malicious crate from doing anything. It is a linter with a human behind it |
| `cargo-vet` | CI (`supply-chain/`) | Every dependency is **explicitly exempted** (per version) or audited — the ledger is complete | Be a barrier you can skip. It is a ledger: a new dependency is simply not in it, and the build fails. It does **not** claim any dependency was manually audited: `supply-chain/audits.toml` is empty and every current dependency sits under a version-pinned exemption |
| `lock-drift.py` | CI (pull requests **and** pushes to main) | Rejects a dependency that moved **backwards** | Catch a malicious bump forward — that is `cargo deny`'s job |
| `cosign sign` (keyless) | `docker.yaml` | The published image has a signature tied to this repository's workflow | Sign the source. The image is the artifact, so the image is what gets signed |

The split, stated once: **Dockerfile + `Cargo.lock` are the walls. CI is
the alarm.** A green CI run does not mean the build is safe; it means the
policy in `deny.toml` and the ledger in `supply-chain/` were satisfied.

## 2. The compile step is sandboxed, and that is proven in both directions

The three-stage Dockerfile splits the build so that third-party code that
runs at compile time (`build.rs` of ~30 crates) has **no network**:

```
tools  (network)  -- install cargo-auditable / cargo-cyclonedx
fetch  (network)  -- cargo fetch --locked   (downloads + checksums, compiles nothing)
builder (NO network) -- cargo auditable build --release --offline --locked
runtime          -- alpine + the binary + the SBOM
```

The `fetch` stage is the reason the split is possible: `cargo fetch`
downloads and checksum-verifies without compiling anything, so no
third-party `build.rs` runs while the network is up.

`tools/sandbox-canary/` is the self-check. Its `build.rs` probes whether
the compile environment can reach the network and whether credential-shaped
environment variables are present, and **fails the build** if it finds
either. CI runs it twice:

1. with the network reachable it **must fail** (it detects a breach)
2. with `--network none` it **must pass** (the isolation holds)

The negative test asserts the canary's own failure marker in the build log
(the panic message `隔离已被打破` from `tools/sandbox-canary/build.rs`),
not merely a non-zero exit status: a build that fails for an unrelated
reason (a typo in the canary crate) would otherwise make the positive
test meaningless. The marker is deliberately *not* the crate name —
cargo prints `Compiling sandbox-canary` and the `build.rs` path for any
failure of that crate, so grepping for the name would match a compile
error in the canary itself. There is no "relaxed" switch — a build
environment that can reach the network simply does not pass.

A canary that only ever passes proves nothing, which is why the negative
test exists.

### What counts as a credential-shaped environment variable

The canary looks at variable **names only**; it never prints a value.

| Variable | Verdict | Why |
|---|---|---|
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` (and any other name containing `PROXY`) | **violation** | A proxy can redirect the compile step's egress to an uncontrolled exit — the same breach the canary exists to catch, in another direction. The rule is "name contains `PROXY`", not an enumeration, so a new proxy variable is caught without a code change. |
| `NO_PROXY` / `no_proxy` | **allowed** | It is the *bypass list* ("do not use a proxy for these hosts"), not an exit and not a credential. `tools/rust-build.sh`'s `run_in_container()` injects `-e NO_PROXY=…` / `-e no_proxy=…` unconditionally, because the test suite talks to mock servers on `127.0.0.1` and reqwest would otherwise send those requests to the proxy. Without this exception the canary is guaranteed to fail locally — and a check that is always red is a check nobody runs. |
| `GITHUB_TOKEN`, `AWS_*`, `*_API_KEY`, `SSH_AUTH_SOCK`, … | **violation** | Credential-shaped; `--network=none` blocks egress but does **not** hide the environment. |

## 3. Keeping the pinned inputs current

There are four kinds of pinned input, and they are updated by different
things.

### 3.1 Rust dependencies — Dependabot (weekly, Monday)

`.github/dependabot.yml`, `cargo` ecosystem. Opens one grouped PR for all
crate bumps. The PR is not auto-mergeable on purpose: a group merges a
major bump (axum 0.7 → 0.8) with the routine patch bumps, and a major
bump is a migration, not noise.

Two things will make that PR red, on purpose:

- `lock-drift.py` rejects a **downgrade** (label the PR `supply-chain-ok`
  to override, deliberately and visibly).
- `cargo vet` fails if a bumped crate no longer matches its version-pinned
  exemption. The fix is to look at the new version and extend the ledger —
  `cargo vet` will tell you which crate it is.

### 3.2 Base images — the pin moves for a CVE, not for a rebuild

`docker.io/library/rust:1-alpine@sha256:...` and
`docker.io/library/alpine:3@sha256:...` are pinned by digest.

The obvious way to keep them current is to move the pin whenever the tag
is rebuilt. **That is deliberately not what happens here.** A tag is
rebuilt constantly — `rust:1-alpine` far more often than it has a
security fix in it — so "the tag moved" produces a pull request per
rebuild. That is noise, and noise in the channel that would have
carried a real security update is worse than no channel at all.

So the policy is:

| Event | What happens |
|---|---|
| upstream rebuilds the tag | **nothing.** The pin stays |
| the pinned image has a HIGH/CRITICAL CVE with a fix, and the current build of the tag does not | the pin moves, in a pull request |
| the pinned image has a CVE and the current build of the tag still has it | an issue is filed: upstream has not fixed it, moving the pin would not help |

There is deliberately **no `docker` ecosystem in `dependabot.yml`** for
the same reason. The `base-image-cve` job of `refresh-pins.yml` owns
this: it scans the pinned digest, and only then looks at what the tag
resolves to.

`tools/check-base-digests.py` is the hand that moves the pin:

```sh
tools/check-base-digests.py --list      # what is pinned
tools/check-base-digests.py --resolve   # what the tags point at now
tools/check-base-digests.py             # report drift (exit 1 = the tag moved)
tools/check-base-digests.py --set IMAGE@DIGEST   # move a pin, deliberately
```

There is no "follow the tag" mode. The only way to move a pin is to say
which image goes to which digest.

CI runs the drift check on every build, but it only **warns**: a moved
tag is a fact, not a defect, and a red build over a rebuild nobody
asked for trains everyone to ignore red builds.

The Dockerfile keeps **literal** `FROM` lines so the check can see which
images are pinned — a build ARG would hide them.

### 3.3 RustSec advisories — the weekly cron

`.github/workflows/security-audit.yml` runs `cargo-audit` on a schedule
(Monday 03:17 UTC). A schedule is required here: the advisory database
changes with **no change in this repository**, so a push-triggered check
would never see a new CVE. On a scheduled run it files an issue instead of
failing (a cron run that goes red is a cron run people learn to ignore).

### 3.4 The tools pinned in CI — monthly, automated

Some versions are not in a manifest Dependabot can read:

| Pin | Where | How to bump |
|---|---|---|
| `cargo-deny 0.20.2` | `ci.yml` | `cargo install --locked --version <new> cargo-deny`; the new version's `deny.toml` schema may differ |
| `cargo-vet 0.10.3` | `ci.yml` | `cargo install --locked --version <new> cargo-vet`; then `cargo vet fmt` and re-commit `supply-chain/` |
| `cargo-cyclonedx 0.5.9` | `ci.yml`, `Dockerfile` | both places |
| `cargo-auditable 0.7.7` | `Dockerfile` | |
| `cargo-llvm-cov 0.9.1` | `ci.yml` | the coverage bar (`--fail-under-regions 97`) may need re-checking after a bump |

`.github/workflows/refresh-pins.yml` runs **monthly** and opens a pull
request when crates.io has a newer version of any of them.
`tools/check-tool-versions.py` does the lookup and the rewrite:

```sh
tools/check-tool-versions.py            # report; exit 1 if outdated
tools/check-tool-versions.py --update   # rewrite the pins in place
```

The version stays exact — no caret, no range — because pinning is the
point. The PR is still a decision: these tools run with full permissions,
and `cargo-deny`'s config schema and `cargo-vet`'s ledger format have both
changed between minor versions. A tool that cannot read its own config is
a silently useless check, so the PR says so in its body.

The workflow **never merges**. It also cannot run until it is merged to
the default branch: GitHub only runs scheduled workflows from the default
branch.

### 3.5 CVEs in the base image — monthly, reported

RustSec advisories (the weekly `cargo-audit` cron) cover the Rust
**crates**. They say nothing about the **Alpine `apk` packages** in the
base image — those live in a different advisory database and need a
container scanner.

The `base-image-cve` job in `refresh-pins.yml` runs Trivy against the
pinned **digests** (not the tags: the digest is the exact image the build
pulls) and files an issue when it finds a HIGH or CRITICAL vulnerability
that has a fix available. It does not fail the run — a scheduled run that
goes red and cannot be actioned is noise, and the issue is the alarm.

When there *is* a finding, the job goes one step further and asks
whether upstream has rebuilt the tag without it. If so, the pin moves
(in a pull request). If not, the issue says so, because moving the pin
would not help: the fix has to come from upstream, or the base image
version has to change.

**What this does not cover: the Rust toolchain** inside
`rust:1-alpine`. It is installed by rustup as tarballs, not as apk
packages, so no container scanner sees it — Trivy reports the apk
packages and nothing about `rustc`. The toolchain's own advisories are a
separate feed that nothing here reads. That gap is real and is not
papered over; the honest summary is that the toolchain moves only when
somebody moves the pin on purpose.

The scanner itself is pinned by commit SHA like every other action, and
Dependabot's `github-actions` ecosystem keeps that pin current.

What a run actually downloads, measured:

| | size | |
|---|---|---|
| `trivy` binary | 50 MiB | every job |
| `trivy-db` (the advisory database) | 136 MB | cached by the action, so a warm run updates it incrementally |
| `trivy-java-db` (Maven index) | **990 MB** | skipped: `TRIVY_SKIP_JAVA_DB_UPDATE=true`, Alpine has no JARs |
| the images being scanned | 336 MiB (`rust:1-alpine`) + 3.7 MiB (`alpine:3`) | Trivy pulls them itself; the image pull, not the database, is the bulk of it |

## 4. Why the build host uses a proxy

`rustbuilder` cannot reach `docker.io`: the TLS connection is reset. It
goes through a proxy instead:

```sh
tools/build-image.sh          # sets HTTPS_PROXY and calls podman build
```

The proxy changes the *route*, not the source — the image still comes from
`docker.io`, with the digest the Dockerfile pins, so the build host and CI
build from the same bytes. A mirror registry would have been a second
source of truth for "the current `rust:1-alpine`"; the two were measured
to agree, but there is no reason to depend on that.

CI needs nothing: GitHub Actions reaches `docker.io` directly.

## 5. The binary is reproducible. The image is not.

The `builder` stage compiles with

```
RUSTFLAGS="--remap-path-prefix=/app=. --remap-path-prefix=/usr/local/cargo=deps"
```

so the build path never reaches the binary. Measured: the same source built
with `WORKDIR /app` and with `WORKDIR /build` produced **byte-identical**
binaries (`sha256:65bd786c…` in both cases, 8,163,832 bytes), and the
binary contains no `/app` or `/usr/local/cargo` strings at all.

That is the property that matters for verification: a binary can be rebuilt
from the source and compared, so the signed image can be checked against a
build you made yourself.

The **image** as a whole is not bit-reproducible, and it is not worth
pretending otherwise: `cargo-cyclonedx` stamps the SBOM with the build
timestamp and a random `serialNumber`, and the image layers carry their own
timestamps. Two builds of the same commit give the same binary inside two
different image digests. Compare the binary, not the image.

## 6. Signing and verification

`docker.yaml` signs the pushed image with cosign, keyless, using GitHub's
OIDC provider. No signing key exists anywhere to leak — the certificate is
minted for that one run and expires with it. The certificate identity is
pinned to this repository's workflow path, so a signature minted by
another workflow does not verify against it.

The deploy side must verify before running the image:

```sh
cosign verify \
  --certificate-identity-regexp \
    "^https://github.com/<owner>/<repo>/.github/workflows/docker.yaml@.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/<owner>/<repo>@sha256:<digest>
```

The identity is pinned to **this repository's workflow path**, not to
`<owner>/<repo>/.*` — a signature minted by a different workflow in the
same repository would not verify against it.

cosign v3 moved `--certificate-identity*` out of `cosign sign` (they are
verify-side flags), so the workflow **signs and then verifies** the
signature immediately, against exactly the identity the deploy side will
use. That is where the constraint is enforced; without that step it would
exist only in a comment. cosign is pinned to an exact version
(`v3.1.4`) — the installer would otherwise take "whatever is latest", and
a release that changed the bundle format would change what this
repository signs.

### 6.1 The deploy side checks it

`deploy/verify-image.sh` is the deploy side's half:

```sh
deploy/verify-image.sh [IMAGE_REF] [LOCAL_TAG]
# default: ghcr.io/zhouqt/llmproxy:latest  ->  localhost/llmproxy:latest
```

It resolves the tag to the immutable digest (the registry's
`Docker-Content-Digest`, cross-checked against the sha256 of the manifest
body), verifies the signature **for that digest**, and only then pulls and
tags it. Verifying a tag and then running the tag would leave a window in
which the tag moves between the check and the run; the digest is what gets
verified and what gets run.

cosign runs from its official image, pinned by digest
(`ghcr.io/sigstore/cosign/cosign@sha256:0f061548…`, v3.1.4), through
podman — so nothing has to be installed on the deploy host, and the
verifier cannot move under us. Set `COSIGN=/path/to/cosign` to use a
binary instead. The package is public, so no credential is stored on the
deploy host either.

The script exits non-zero when the signature does not verify. A deployment
script must stop there: running the image anyway would defeat the point.

That refusal is not hypothetical. Run against a tag whose image was
pushed before this repository signed anything (the `latest` tag that
predates the signing step), the script resolves the digest, finds no
signature for it, and stops:

```
verify-image: zhouqt/llmproxy:latest -> sha256:dce84f0e…
verify-image: verifying the signature
Error: no signatures found
verify-image: the signature of ghcr.io/zhouqt/llmproxy@sha256:dce84f0e… did NOT verify; refusing to pull it
```

An unsigned image is not "probably fine" — it is exactly the thing this
check exists to catch. Deploy a tag that CI signed, and nothing else.

### 6.2 The scanner runs the same way

`tools/scan-image.sh <image-ref>` runs the scanner from its official
image, pinned by digest
(`ghcr.io/anchore/grype@sha256:af9f30d1…`, v0.111.0), with the
vulnerability database in a cache directory (`DB_CACHE`) and the proxy
passed through when the host sets one. `--fail-on <severity>` makes it
exit non-zero on a finding; without it, it only reports.

What it found on the images this repository pins by digest:

| image | scanner | HIGH | CRITICAL | other |
|---|---|---|---|---|
| `alpine:3@sha256:be265606…` | grype (deploy host) | 0 | 0 | 3 × Medium (`busybox` CVE-2025-60876) |
| `alpine:3@sha256:be265606…` | Trivy (CI) | 0 | 0 | — |
| `rust:1-alpine@sha256:b1d5dd5a…` | grype (deploy host) | **5** | 0 | 9 × Medium (`binutils`), 3 × Medium (`busybox`) |
| `rust:1-alpine@sha256:b1d5dd5a…` | Trivy (CI) | 0 | 0 | — |

Two things to read out of that table.

**The runtime image is clean.** `alpine:3` — the image the shipped
binary actually runs in — has no HIGH and no CRITICAL finding.

**The two scanners disagree about the build image.** Grype's database
knows five High findings in `binutils` 2.45.1-r1 inside the pinned
`rust:1-alpine` (CVE-2025-69650, CVE-2025-69649, CVE-2026-3442,
CVE-2026-3441, CVE-2026-6846); Trivy's database reports none of them.
`binutils` is a build tool — it is in the image that compiles the
crate, not in the image that runs it — so none of these ship. But a
scanner disagreement is not something to paper over: it means a clean
Trivy report is not evidence that an image is clean, only evidence that
Trivy has not heard of the finding yet. Both scanners run over the
pinned digests for that reason.

**The pin stays where it is.** A pin moves for exactly one reason: the
pinned image has a HIGH/CRITICAL finding that the *current build of the
same tag* does not have. That comparison was made for
`rust:1-alpine`: the newest build of the tag
(`sha256:e15c563b…`, scanned with the same grype build) reports the
**same five High `binutils` findings** — the newest build does not fix
them either. So there is nothing to gain by moving the pin, and it
stays.

### 6.3 Where the heavy work runs

The scans need gigabytes: the vulnerability database, the image layers,
and the extracted layer cache all have to fit at the same time. The
build host has ~19 GB in total and ran out of room; the deploy host has
61 GB.

So the work moved to the deploy host, in a container built from the
*pinned* `rust:1-alpine` digest — the same compiler CI uses, running
against the git-synced source tree:

```bash
ssh loki 'cd /home/zhouqt/llmproxy-build && sh tools/rust-build.sh test --lib --bins --tests'
```

`tools/rust-build.sh` mounts the repository **read-only** and puts the
registry cache and the target directory on the host, so a failed build
cannot rewrite `Cargo.lock` under a tree that gets pushed, and the
downloads are not repeated per container. Nothing about the build
changes by moving it: the source of truth is still this repository, and
the image is still the digest the Dockerfile pins.


## 7. The gates are tested in both directions

A gate that has never been seen failing is indistinguishable from a gate
that is silently broken. `cargo deny check` exits 0 on a policy with no
teeth; `cargo vet` exits 0 on a ledger that matches nothing; a scanner
that reports nothing looks exactly like a clean image.

So the checks are exercised the way the sandbox canary is: in **both**
directions.

`tools/policy-selftest.sh` injects one violation per case, in a scratch
copy of the repository (the real tree is never touched), and asserts that
the tool refuses it:

| case | violation | must happen |
|---|---|---|
| control | none | cargo-deny and cargo-vet pass on the unmodified tree |
| licences | the allow-list is narrowed to one licence | cargo-deny refuses the graph |
| sources | crates.io is removed from the allowed registries | cargo-deny refuses the graph |
| vet | one dependency is moved to an unvetted version | cargo-vet refuses it |
| drift | one dependency is moved to an older version | lock-drift refuses it |

The control row is what makes the rest meaningful: if the tools do not
pass on the unmodified tree, the red cases would prove nothing.

CI runs it as the `policy-selftest` job, and the job fails if any case
was **skipped** — a self-test that quietly stops covering a case is
worse than no self-test.

The container scanner is tested the same way: CI scans a deliberately old
Alpine release and asserts that the scanner reports findings on it. A
clean result on the pinned images only means something if the scanner is
known to go red on something.

```sh
tools/policy-selftest.sh        # the cargo gates, both directions
tools/scan-image.sh <image-ref> # a scanner, pinned by digest
```

`tools/scan-image.sh` runs the scanner from its official image pinned by
digest (`ghcr.io/anchore/grype@sha256:af9f30d1…`), so nothing is
installed on the host and the scanner cannot move under us. The proxy is
passed through, because the build host reaches docker.io only through one.

