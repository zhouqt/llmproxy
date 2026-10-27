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
# Override the proxy with HTTPS_PROXY if the default is not right:
#   HTTPS_PROXY=http://host:port tools/build-image.sh
set -eu

: "${HTTPS_PROXY:=http://192.168.11.1:12345}"
export HTTPS_PROXY
export HTTP_PROXY="${HTTPS_PROXY}"

TAG="${IMAGE_TAG:-llmproxy:local}"
cd "$(dirname "$0")/.."

exec podman build -f Dockerfile -t "${TAG}" "$@" .
