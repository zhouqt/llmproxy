# syntax=docker/dockerfile:1
#
# The image is built in four stages. The point of the split is **to give
# third-party code that runs at build time no network and no credentials**,
# and to make the binary in the image **readable to an external scanner**.
#
# Why the split (the measurements behind it are in
# plans/harden-cargo-deps.md §2):
#   - `cargo build --offline` is **not a sandbox** -- build.rs still reaches
#     the network and still reads every environment variable.
#   - A container with `--network=none` is real kernel-level isolation
#     (TCP/DNS are cut), but it **does not stop environment variables**.
#   -> So the network is allowed for exactly one step, "fetch the
#      dependencies / install the tools". The compile step must be
#      `RUN --network=none`, and the compile environment must hold no
#      credentials at all.
#
# Base images: straight from Docker Hub (docker.io). GitHub Actions can
# reach docker.io directly, and the released image should come from the
# same source -- putting an ECR mirror in between adds a conversion and a
# trust point. Pinned by digest (a mutable tag is an input that is not
# pinned).
#
# Some build hosts cannot reach docker.io: the TLS connection may be
# reset. They go through a **proxy**:
#   HTTPS_PROXY=<proxy> podman build …
# or use tools/build-image.sh (which sets the same thing). Going through a
# proxy still pulls the same docker.io -- not a mirror registry -- so the
# build inputs are identical to CI's (measured: for the same tag, the two
# registries serve byte-identical manifests and the same digest).
#
# The FROM lines must be **literal**. tools/check-base-digests.py parses
# them to spot a tag that drifted; written as ARG(${VAR}) it cannot parse
# them, and then nobody knows which image is actually pinned.
#
# NOTE: a moved tag does **not** change this sha256 by itself. A pinned
# digest moves for exactly one reason: the pinned image has a CVE that a
# fresh build of the same tag has already fixed -- see the base-image-cve
# job in .github/workflows/refresh-pins.yml.
# To move it by hand:
#   tools/check-base-digests.py --resolve && tools/check-base-digests.py --set <image>@<digest>
#
#   rust:1-alpine  → sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87
#   alpine:3       → sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a

# ---- Stage 0: tools (network) ---------------------------------------------
# cargo-auditable and cargo-cyclonedx are crates.io products too, and they
# run with full privileges -- the same kind of risk as the dependencies
# they audit. There are only two mitigations: a pinned version and a
# `--locked` install (which verifies the bytes against the crate's own
# Cargo.lock). Upgrading either version is an **explicit decision**.
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS tools
RUN cargo install --locked --version 0.7.7 cargo-auditable \
 && cargo install --locked --version 0.5.9 cargo-cyclonedx

# ---- Stage 1: fetch the dependencies (network) ------------------------------
# `cargo fetch` only downloads and verifies the sha256s from Cargo.lock;
# it **does not compile** -- so no third-party build.rs runs in this stage.
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS fetch
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
# cargo refuses to parse a manifest with no targets, so the fetch stage
# needs a placeholder source file. It is thrown away with the stage.
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs
RUN cargo fetch --locked

# ---- Stage 2: compile (no network while compiling) -------------------------
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS builder
RUN apk add --no-cache pkgconf openssl-dev

WORKDIR /app
# Move the downloaded registry from the fetch stage (including the index
# cache -- cargo needs it to verify checksums).
COPY --from=fetch /usr/local/cargo /usr/local/cargo
# Move only the two tools' binaries, not their dependencies -- the tools
# themselves are not part of the build output.
COPY --from=tools /usr/local/cargo/bin/cargo-auditable /usr/local/cargo/bin/
COPY --from=tools /usr/local/cargo/bin/cargo-cyclonedx /usr/local/cargo/bin/
COPY Cargo.toml Cargo.lock ./

# Reproducibility: strip the build-time paths out of the binary. /app is
# this Dockerfile's fixed WORKDIR and /usr/local/cargo is where the
# registry sources live -- both are "where this was compiled" information.
# With them gone, the same lock file produces the same binary hash on any
# path.
ENV RUSTFLAGS="--remap-path-prefix=/app=. --remap-path-prefix=/usr/local/cargo=deps"

# Dependency layer: compile every dependency with an empty main first;
# the builder caches this layer. `--offline` guarantees that nothing is
# fetched from the network again, and `--network=none` guarantees that
# this step has no way out.
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs
RUN --network=none cargo build --release --offline --locked

# Sandbox self-check: confirm that this layer really is isolated. A hole
# here fails the build immediately.
COPY tools/sandbox-canary/ tools/sandbox-canary/
RUN --network=none cargo build --release --offline --locked --manifest-path tools/sandbox-canary/Cargo.toml

# The real source: only llmproxy itself has to be recompiled; the
# dependency layer reuses the cache above.
COPY src/ src/
# cargo-auditable instead of cargo build: it embeds the dependency
# manifest in the binary, so an image scanner (Trivy/Grype) can report the
# versions and CVEs of crates like ring and rustls -- otherwise a stripped
# binary is a black box to the scanner.
RUN --network=none cargo auditable build --release --offline --locked \
    && cp target/release/llmproxy /llmproxy

# SBOM: CycloneDX, shipped with the image (not kept in the repository --
# it would go stale with every upgrade and turn into a source of noise).
#
# cargo-cyclonedx has no --output-file, and `--override-filename` does not
# allow a path separator, so the file is written to the current directory
# (/app) and the tool appends the extension itself -> the result is
# sbom.json, renamed below to sbom.cdx.json to match the path in the
# runtime image.
RUN --network=none cargo cyclonedx --format json --override-filename sbom \
    && mv sbom.json /app/sbom.cdx.json

# ---- Stage 3: runtime ------------------------------------------------------
FROM docker.io/library/alpine:3@sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a
# tpm2-tools provides tpm2_load/tpm2_unseal; tpm2-tss-tcti-device is the
# device TCTI the tools need to talk to /dev/tpmrm0. The two are
# separate packages on Alpine -- installing only the first fails at
# runtime with `Could not load tcti, got: "(null)"`.
RUN apk add --no-cache ca-certificates \
        tpm2-tools tpm2-tss-tcti-device

COPY --from=builder /llmproxy /usr/local/bin/llmproxy
COPY --from=builder /app/sbom.cdx.json /usr/local/share/llmproxy/sbom.cdx.json
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

EXPOSE 8080
# The entrypoint unseals the TPM-sealed secrets into /run/secrets and
# then execs the command below -- see deploy/entrypoint.sh.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["llmproxy", "--config", "/etc/llmproxy/config.yaml"]

# ---------------------------------------------------------------------------
# Refreshing a base image digest:
#   tools/check-base-digests.py            # check that the digest still matches the tag
#   tools/check-base-digests.py --registry public.ecr.aws   # for hosts that can reach ECR
# When the digest drifts, Dependabot (docker ecosystem) opens a PR to change this line.
# ---------------------------------------------------------------------------
