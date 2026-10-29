#!/bin/sh
# Build or test the crate inside the pinned Rust container, on a host that
# has no Rust toolchain of its own.
#
# Why this exists
# ---------------
# Compiling needs room: the target directory, the registry cache and the
# toolchain together are several gigabytes. The build host (rustbuilder)
# has 19 GB in total and ran out of it -- a debug build of the whole
# crate plus a container scanner's database does not fit next to each
# other. The deploy host (loki) has 61 GB, already has the source tree
# (it is a git clone of this repository), and already has the pinned
# base image in its local storage.
#
# So the same work runs there, in the *same container image* the
# Dockerfile pins: the compiler is the one CI uses, and the digest is
# the one CI builds from. This is a place to run the build, not a
# different build -- the source of truth is still this repository.
#
# Usage:
#   tools/rust-build.sh fetch --locked
#   tools/rust-build.sh build --release --offline --locked
#   tools/rust-build.sh test --lib --bins --tests
#   tools/rust-build.sh -- <command> ...      # anything else
#
# Environment:
#   RUST_IMAGE       the image to run (default: the digest the Dockerfile
#                    pins -- the same compiler CI uses). An override
#                    must keep the digest pin: a value without an
#                    "@sha256:" suffix is refused.
#   CARGO_HOME_DIR   where the registry cache lives on the host
#                    (default: /var/tmp/llmproxy-cargo). Keeping it
#                    across runs means the dependencies are downloaded
#                    once, not once per container.
#   CARGO_TARGET_DIR where the build output lives on the host
#                    (default: /var/tmp/llmproxy-target). This is the
#                    same path the repository's .cargo/config.toml uses,
#                    so the container and the host agree on where the
#                    output goes.
#   HTTPS_PROXY      passed through to the container when set.
#
# The repository is mounted **read-only**. Cargo must not be able to
# rewrite Cargo.lock under the tree that gets pushed; `--locked` is the
# real guard, and a read-only mount makes the failure mode loud instead
# of silent.

set -eu

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# The digest the Dockerfile pins. Kept literal for the same reason the
# Dockerfile keeps it literal: a tag that moves is an input that moved.
RUST_IMAGE=${RUST_IMAGE:-docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87}

CARGO_HOME_DIR=${CARGO_HOME_DIR:-/var/tmp/llmproxy-cargo}
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/var/tmp/llmproxy-target}

PODMAN=${PODMAN:-podman}

die() {
    printf 'rust-build: %s\n' "$1" >&2
    exit 1
}

# An override has to keep the digest pin, the same rule the cosign image
# (deploy/verify-image.sh) and the scanner (tools/scan-image.sh) follow:
# a floating tag here would let the compiler itself be swapped between
# two runs of this script, which is a bigger hole than the one the
# script closes. The default above already carries a digest, so it is
# not rejected by this check.
case "$RUST_IMAGE" in
    *@sha256:*) ;;
    *) die "RUST_IMAGE must be pinned by digest (an @sha256: suffix is required): $RUST_IMAGE" ;;
esac

command -v "$PODMAN" >/dev/null 2>&1 || die "podman is not installed (PODMAN=$PODMAN)"

[ $# -ge 1 ] || die "usage: rust-build.sh <cargo-subcommand> [args...] | -- <command>"

# The registry cache holds downloaded crates and the target directory
# holds the compiled output; both are private to the user running the
# build. umask has to come before the mkdir, or the directories are
# created group/world-readable and the 0700 applies to nothing.
umask 077
mkdir -p "$CARGO_HOME_DIR" "$CARGO_TARGET_DIR"

# The cargo command runs through the rustup shim in the image, which
# reads RUSTUP_HOME. CARGO_HOME is pointed at the mounted cache
# instead, so the toolchain stays where the image put it and only the
# registry cache and the installed binaries are shared with the host.
#
# The proxy is passed through only when it is set: an unset variable
# would otherwise become an empty string inside the container, and an
# empty proxy is a proxy that cannot be reached.
#
# The value is passed by NAME (`-e HTTPS_PROXY`), never as
# `-e HTTPS_PROXY=value`: the latter puts the proxy URL into podman's
# argv, where every process on the host can read it from
# /proc/<pid>/cmdline. With the bare name, podman copies the value from
# its own environment.
#
# NO_PROXY covers the loopback. The test suite talks to mock servers on
# 127.0.0.1, and reqwest honours the proxy variables for those requests
# too -- the proxy answers them with an empty body, and every test that
# expects JSON fails with "EOF while parsing a value". Excluding the
# loopback is what makes the suite pass with a proxy in the environment.
#
# The helper prints only the literal words `-e HTTPS_PROXY` (never a
# value), so word-splitting its output below is safe.
proxy_opts() {
    if [ -n "${HTTPS_PROXY:-}" ]; then
        printf -- '-e\nHTTPS_PROXY\n'
    fi
    if [ -n "${HTTP_PROXY:-}" ]; then
        printf -- '-e\nHTTP_PROXY\n'
    fi
}

run_in_container() {
    # The podman argv is built from the right: the command goes last,
    # after the image name, and each fixed option is prepended in front
    # of it. Prepending is what keeps every option BEFORE the image --
    # podman stops parsing options at the image, and anything after it
    # becomes the container's command.
    set -- "$RUST_IMAGE" "$@"
    set -- -w /app "$@"
    set -- -e no_proxy=localhost,127.0.0.1,::1 "$@"
    set -- -e NO_PROXY=localhost,127.0.0.1,::1 "$@"
    set -- -e CARGO_TARGET_DIR=/tmp/llmproxy-target "$@"
    set -- -e CARGO_HOME=/cargo-home "$@"
    set -- -v "$CARGO_TARGET_DIR:/tmp/llmproxy-target" "$@"
    set -- -v "$CARGO_HOME_DIR:/cargo-home" "$@"
    set -- -v "$REPO_ROOT:/app:ro" "$@"
    # shellcheck disable=SC2046 # proxy_opts emits fixed literal words
    set -- $(proxy_opts) "$@"
    set -- run --rm "$@"

    "$PODMAN" "$@"
}

# `--` means "run this, not cargo".
if [ "$1" = "--" ]; then
    shift
    [ $# -ge 1 ] || die "nothing to run after --"
    run_in_container "$@"
    exit $?
fi

run_in_container cargo "$@"
