#!/bin/sh
# Pull a published llmproxy image, verify its signature, and tag it for
# the local deployment.
#
# Why this exists
# ---------------
# The image is signed in CI with cosign (keyless, GitHub's OIDC provider),
# and the certificate identity is pinned to this repository's workflow
# path. Until something on the deploy side runs `cosign verify`, that
# signature is decoration: the image is signed, and nobody checks.
#
# This script is that "something". It resolves the tag to the immutable
# digest, verifies the signature for THAT digest, and only then pulls and
# tags it. Verifying a tag and then running the tag would leave a window
# in which the tag moves between the check and the run -- so the tag is
# resolved first and the digest is what gets verified and what gets run.
#
# Usage:
#     deploy/verify-image.sh [IMAGE_REF] [LOCAL_TAG]
#
#     IMAGE_REF   what to pull   (default ghcr.io/zhouqt/llmproxy:latest)
#     LOCAL_TAG   what to tag it (default localhost/llmproxy:latest)
#
# Environment:
#     COSIGN   path to a cosign binary. When it is not set, cosign runs
#              from its official image, pinned by digest, through
#              podman -- so nothing has to be installed on the host.
#     PODMAN   path to podman       (default: podman)
#
# Exit codes: 0 = verified and tagged, non-zero = refused. A deployment
# script must stop on a non-zero exit; running the image anyway would
# defeat the point.

set -eu

die() {
    printf 'verify-image: %s\n' "$1" >&2
    exit 1
}

IMAGE_REF=${1:-ghcr.io/zhouqt/llmproxy:latest}
LOCAL_TAG=${2:-localhost/llmproxy:latest}
PODMAN=${PODMAN:-podman}

# The repository the image belongs to. The identity regexp is scoped to
# this repository's workflow path, so a signature minted by another
# workflow -- or by a fork that somehow got id-token: write -- does not
# verify against it.
#
# REPO goes into a REGEXP, so it has to be a plain name: a value like
# ".*" would widen the identity to "any repository", which is exactly
# the check this script exists to make. Anything outside the ordinary
# GitHub owner/repository charset is refused, and the dots are escaped
# so that "foo.bar" matches "foo.bar" and not "fooXbar".
REPO=${REPO:-zhouqt/llmproxy}
case "$REPO" in
    '') die "REPO must not be empty" ;;
    *[!A-Za-z0-9._/-]*) die "REPO must contain only letters, digits, '.', '_', '-' and '/'" ;;
esac
REPO_RE=$(printf '%s' "$REPO" | sed 's/[.[\*^$()+?{|]/\\&/g')
IDENTITY_RE="^https://github\.com/${REPO_RE}/\.github/workflows/docker\.yaml@.*"
ISSUER="https://token.actions.githubusercontent.com"

# cosign's official image, pinned by digest. The tag is written next to
# it for humans; the digest is what is pulled.
#
# An override has to keep the digest pin: a floating tag here would let
# the verifier itself be replaced between two runs of this script, which
# is a bigger hole than the one the script closes.
COSIGN_IMAGE=${COSIGN_IMAGE:-ghcr.io/sigstore/cosign/cosign@sha256:0f06154836d2de34f887a5c273506d52ca43edaa1d9e53e6510fa78a48f69c2d} # v3.1.4
case "$COSIGN_IMAGE" in
    *@sha256:*) ;;
    *) die "COSIGN_IMAGE must be pinned by digest (an @sha256: suffix is required): $COSIGN_IMAGE" ;;
esac

# The manifest types the registry may answer with. Without the index
# types it answers with the per-architecture manifest, whose digest is
# NOT the digest CI signed.
ACCEPT="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"

command -v "$PODMAN" >/dev/null 2>&1 || die "podman is not installed (PODMAN=$PODMAN)"
command -v curl >/dev/null 2>&1 || die "curl is not installed"

# Split the reference into repository and tag. The registry host is
# ghcr.io; the repository path is everything after it.
case "$IMAGE_REF" in
    ghcr.io/*) repo_path=${IMAGE_REF#ghcr.io/} ;;
    *) die "unsupported registry in $IMAGE_REF (only ghcr.io is used)" ;;
esac
case "$repo_path" in
    *:*) tag=${repo_path##*:}; repo_path=${repo_path%:*} ;;
    *) die "$IMAGE_REF has no tag" ;;
esac

printf 'verify-image: resolving %s:%s to a digest\n' "$repo_path" "$tag"

# A pull token for the package. The package is public, so this is an
# anonymous token -- no credential is stored on the deploy host.
token=$(curl -sSf "https://ghcr.io/token?scope=repository:${repo_path}:pull&service=ghcr.io" \
    | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
[ -n "$token" ] || die "no pull token from ghcr.io for ${repo_path}"

# The registry states the digest of what the tag points at in a header.
# The body is hashed as well and the two must agree: if they do not,
# something mangled the response and the honest answer is "cannot check".
#
# The pull token is passed to curl through a config file, not through
# the command line: an argument is visible to every other process on the
# host (`ps`, /proc/<pid>/cmdline) for as long as curl runs. The config
# file lives in a private directory and is removed on exit.
headers=$(mktemp)
body=$(mktemp)
curlcfg=$(mktemp)
trap 'rm -f "$headers" "$body" "$curlcfg"' EXIT
chmod 600 "$curlcfg"
printf 'header = "Authorization: Bearer %s"\n' "$token" > "$curlcfg"
curl -sSf -D "$headers" -o "$body" \
    --config "$curlcfg" \
    -H "Accept: ${ACCEPT}" \
    "https://ghcr.io/v2/${repo_path}/manifests/${tag}" \
    || die "could not fetch the manifest for ${repo_path}:${tag}"

stated=$(tr -d '\r' < "$headers" | awk 'tolower($1) == "docker-content-digest:" { print $2 }')
computed="sha256:$(sha256sum "$body" | cut -d' ' -f1)"
[ -n "$stated" ] || die "the registry did not state a digest for ${repo_path}:${tag}"
if [ "$stated" != "$computed" ]; then
    die "the registry stated $stated but the manifest hashes to $computed; the response was altered in transit"
fi
digest=$stated
printf 'verify-image: %s:%s -> %s\n' "$repo_path" "$tag" "$digest"

printf 'verify-image: verifying the signature\n'
if [ -n "${COSIGN:-}" ]; then
    # A cosign binary on the host, when there is one.
    "$COSIGN" verify \
        --certificate-identity-regexp "$IDENTITY_RE" \
        --certificate-oidc-issuer "$ISSUER" \
        "ghcr.io/${repo_path}@${digest}" \
        || die "the signature of ghcr.io/${repo_path}@${digest} did NOT verify; refusing to pull it"
else
    # Otherwise cosign runs from its official image, pinned by digest.
    # Nothing is installed on the host, and the verifier is as pinned as
    # everything else in this repository.
    "$PODMAN" run --rm \
        --network host \
        "$COSIGN_IMAGE" verify \
        --certificate-identity-regexp "$IDENTITY_RE" \
        --certificate-oidc-issuer "$ISSUER" \
        "ghcr.io/${repo_path}@${digest}" \
        || die "the signature of ghcr.io/${repo_path}@${digest} did NOT verify; refusing to pull it"
fi

# Only now is the image fetched. It is pulled BY digest, so the bytes
# that arrive are the bytes that were verified.
printf 'verify-image: pulling the verified image\n'
"$PODMAN" pull "ghcr.io/${repo_path}@${digest}" >/dev/null
"$PODMAN" tag "ghcr.io/${repo_path}@${digest}" "$LOCAL_TAG"

printf 'verify-image: ok: %s is the verified %s\n' "$LOCAL_TAG" "$digest"
printf 'verify-image: deploy with %s\n' "$LOCAL_TAG"
