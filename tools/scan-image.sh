#!/bin/sh
# Scan a container image for vulnerabilities, using a scanner that is
# pinned by digest and runs from its official image.
#
# Why this exists
# ---------------
# Two different things need scanning, and neither is covered by RustSec:
#
#   - the Alpine packages in the base images (a container scanner's job)
#   - the Rust crates inside our own image (visible only because
#     cargo-auditable embeds the dependency list in the binary)
#
# The scanner runs from its official image, pinned by digest, so nothing
# is installed on the host and the scanner itself cannot move under us.
#
# Usage:
#     tools/scan-image.sh <image-ref> [--fail-on <severity>]
#
#     image-ref   e.g. docker.io/library/alpine:3@sha256:...
#     --fail-on   exit non-zero when a finding at this severity or
#                 higher is found (default: none -- report only)
#
# Environment:
#     HTTPS_PROXY / HTTP_PROXY  passed through to the scanner; the
#                 build host reaches docker.io only through a proxy.
#     PODMAN      path to podman (default: podman)
#     SCANNER    the scanner image, pinned by digest
#     DB_CACHE   where the vulnerability database is cached (default: a
#                temporary directory, removed on exit)

set -eu

PODMAN=${PODMAN:-podman}
SCANNER=${SCANNER:-ghcr.io/anchore/grype@sha256:af9f30d1de4ecfbbcd2b674371e8df787879e16832af8ab81551b0bf1ec0ad5b} # v0.111.0

# The scanner is the tool that decides whether an image is acceptable,
# so an override has to keep the digest pin: a floating tag would let
# the scanner itself be swapped between two runs.
case "$SCANNER" in
    *@sha256:*) ;;
    *)
        printf 'scan-image: SCANNER must be pinned by digest (an @sha256: suffix is required): %s\n' \
            "$SCANNER" >&2
        exit 2
        ;;
esac

usage() {
    sed -n '2,/^set -eu/p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

[ $# -ge 1 ] || usage
IMAGE_REF=$1
shift

FAIL_ON=""
FAIL_ON_SET=false
while [ $# -gt 0 ]; do
    case "$1" in
        --fail-on)
            [ $# -ge 2 ] || {
                printf 'scan-image: --fail-on needs a severity\n' >&2
                exit 2
            }
            FAIL_ON=$2
            FAIL_ON_SET=true
            shift 2
            ;;
        *) printf 'scan-image: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# An empty severity is not "no threshold" -- it is a threshold nobody
# wrote. "--fail-on ''" used to be silently ignored, so the scan ran
# and reported findings while the caller believed it had asked for a
# gate. Refuse it.
if [ "$FAIL_ON_SET" = true ] && [ -z "$FAIL_ON" ]; then
    printf 'scan-image: --fail-on needs a severity (e.g. HIGH,CRITICAL)\n' >&2
    exit 2
fi

command -v "$PODMAN" >/dev/null 2>&1 || {
    printf 'scan-image: podman is not installed (PODMAN=%s)\n' "$PODMAN" >&2
    exit 2
}

# The vulnerability database is a few hundred megabytes. It is cached
# between runs so a second scan does not download it again.
if [ -z "${DB_CACHE:-}" ]; then
    DB_CACHE=$(mktemp -d)
    trap 'rm -rf "$DB_CACHE"' EXIT
fi
mkdir -p "$DB_CACHE"

printf 'scan-image: scanning %s with %s\n' "$IMAGE_REF" "$SCANNER"

set -- run --rm

# The proxy is only passed through when it is set; an unset variable
# would otherwise become an empty string in the container.
#
# The value is passed by NAME, not by name=value: `-e HTTPS_PROXY=...`
# puts the proxy URL into podman's argv, where every other process on
# the host can read it out of /proc/<pid>/cmdline. With the bare name,
# podman copies the value out of its own environment.
#
# Every option has to come BEFORE the image name: podman stops parsing
# options at the image, so an option after it is passed to the scanner
# instead (which does not understand it).
if [ -n "${HTTPS_PROXY:-}" ]; then
    set -- "$@" -e HTTPS_PROXY
fi
if [ -n "${HTTP_PROXY:-}" ]; then
    set -- "$@" -e HTTP_PROXY
fi

set -- "$@" \
    -e GRYPE_DB_CACHE_DIR=/cache \
    -v "$DB_CACHE:/cache"

if [ -n "$FAIL_ON" ]; then
    set -- "$@" --fail-on "$FAIL_ON"
fi

exec "$PODMAN" "$@" "$SCANNER" "$IMAGE_REF"
