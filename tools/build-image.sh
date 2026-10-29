#!/bin/sh
# Build the container image on a host that cannot reach docker.io directly.
#
# The build host (rustbuilder) has its TLS connection to docker.io reset,
# so the pull goes through a proxy. The proxy only changes the *route*:
# the image still comes from docker.io, with the digest the Dockerfile
# pins -- the same bytes CI builds from. That is deliberately different
# from using a mirror registry, which would be a second source of truth
# for what "the current rust:1-alpine" means.
#
# Usage:
#   tools/build-image.sh                 # build llmproxy:local
#   tools/build-image.sh --no-cache      # extra args go to podman build
#
# The proxy is NOT set by this script. If the host needs one, export
# HTTPS_PROXY (and/or HTTP_PROXY) before calling it; `podman build`
# passes the variables it finds in its environment through to the RUN
# steps. The script deliberately holds no proxy value: a hardcoded
# address is wrong on every host but one, and a value written here ends
# up in git history and in `ps` output.
set -eu

TAG="${IMAGE_TAG:-llmproxy:local}"
cd "$(dirname "$0")/.."

exec podman build -f Dockerfile -t "${TAG}" "$@" .
