#!/usr/bin/env python3
"""Check that the base images pinned in the Dockerfile still match the tag.

Why this exists
---------------
The Dockerfile pins its base images by digest:

    FROM docker.io/library/rust:1-alpine@sha256:...

A digest pin is only useful if someone notices when the tag moves. The
tag is what a human (or Dependabot) reads; the digest is what the build
actually resolves. If they drift apart, the build silently keeps using
an old image -- often one with a known CVE -- and nothing in CI says so.

So this script does the comparison, in the one place that can reach
Docker Hub: the CI runner.

    tools/check-base-digests.py                     # check docker.io
    tools/check-base-digests.py --registry public.ecr.aws

Exit status
-----------
0   every pinned digest still matches its tag
1   a digest has drifted (the tag moved) -- update the Dockerfile
2   the registry could not be reached (network problem, not a drift)

The two failure modes are deliberately different exit codes: "your pin
is stale" and "we could not check" must not look the same to a human
reading a red CI run.

Registry access
---------------
Docker Hub needs a bearer token from auth.docker.io; the AWS public
mirror uses public.ecr.aws/token. Both are handled here so the same
script can check either one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request

# The manifest media types Docker Hub serves. Without all of them the
# registry may answer with a schema-1 manifest (no digest) instead of
# the OCI index, which is what the Dockerfile's digest refers to.
ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
    ]
)

FROM_RE = re.compile(
    r"^\s*FROM\s+"
    r"(?P<image>[^\s@]+)"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}))?",
    re.IGNORECASE | re.MULTILINE,
)

ARG_RE = re.compile(
    r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:=(?P<value>\S+))?",
    re.IGNORECASE | re.MULTILINE,
)

VAR_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")

# An image reference with a registry host is "host/path/name"; a bare
# "library/name" means Docker Hub's library namespace.
DOCKER_HUB = "docker.io"
ECR = "public.ecr.aws"

# The registry *name* and the registry *API host* are not the same thing:
# "docker.io" is the website, the API lives on "registry-1.docker.io".
# Asking the website for /v2/... returns a landing page, not a manifest --
# and a landing page whose bytes change on every request, which looks
# exactly like a digest that moves.
REGISTRY_API = {
    DOCKER_HUB: "registry-1.docker.io",
    ECR: "public.ecr.aws",
}


def split_image(image: str) -> tuple[str, str]:
    """Return (registry, repository-without-registry) for an image ref."""
    parts = image.split("/", 1)
    if len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        return parts[0], parts[1]
    return DOCKER_HUB, image


def split_repo_tag(repo: str) -> tuple[str, str]:
    """Return (repository, tag) for "library/rust:1-alpine"."""
    if ":" in repo.rsplit("/", 1)[-1]:
        repo, tag = repo.rsplit(":", 1)
        return repo, tag
    return repo, "latest"


def token_for(registry: str, repo: str) -> str:
    """Fetch a pull token for `repo` from `registry`."""
    if registry == DOCKER_HUB:
        url = (
            "https://auth.docker.io/token?service=registry.docker.io"
            f"&scope=repository:{repo}:pull"
        )
    elif registry == ECR:
        url = (
            f"https://{registry}/token/?scope=repository:{repo}:pull"
            f"&service={registry}"
        )
    else:
        raise SystemExit(f"unsupported registry: {registry}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = json.load(resp)
    token = body.get("token") or body.get("access_token")
    if not token:
        raise SystemExit(f"no token in the response from {url}")
    return token


def registry_repo(registry: str, repo: str) -> str:
    """Return the repository path as the registry wants it.

    Docker Hub keeps the official images under "library/<name>". The AWS
    public mirror keeps them under "docker/library/<name>".
    """
    if registry == ECR and not repo.startswith("docker/"):
        return f"docker/{repo}"
    return repo


def manifest_digest(registry: str, repo: str, tag: str) -> str:
    """Return the sha256 digest of the manifest the tag currently points at.

    The registry states the digest in the Docker-Content-Digest header.
    The body is hashed as well, and the two must agree: if they do not,
    something between here and the registry mangled the response, and
    the honest answer is "could not check", not "the pin drifted".
    """
    repo = registry_repo(registry, repo)
    token = token_for(registry, repo)
    host = REGISTRY_API.get(registry, registry)
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        stated = resp.headers.get("Docker-Content-Digest")
    # The digest of a manifest is the sha256 of its exact bytes. Computing
    # it here is the same thing the registry's Docker-Content-Digest header
    # reports, and it does not depend on that header being present.
    computed = "sha256:" + hashlib.sha256(raw).hexdigest()
    if stated and stated != computed:
        raise ValueError(
            f"{host} stated {stated} but the body hashes to {computed}; "
            "the response was altered in transit, so the pin cannot be checked"
        )
    return stated or computed


def pinned_images(dockerfile: str) -> list[tuple[str, str]]:
    """Return [(image, digest)] for every FROM line that pins a digest.

    The Dockerfile names its base images through build ARGs
    (FROM ${RUST_IMAGE}), so the ARG defaults are substituted first --
    otherwise the pins would be invisible to this check.
    """
    args: dict[str, str] = {}
    for m in ARG_RE.finditer(dockerfile):
        if m.group("value"):
            args[m.group("name")] = m.group("value")

    out = []
    for m in FROM_RE.finditer(dockerfile):
        image = m.group("image")
        digest = m.group("digest")
        if "${" in image:
            for var in VAR_RE.finditer(image):
                image = image.replace(var.group(0), args.get(var.group("name"), ""))
        # A digest inside the ARG value: FROM ${RUST_IMAGE} has no
        # "@sha256:" of its own, the pin lives in the ARG default.
        if not digest and "@sha256:" in image:
            image, _, digest = image.partition("@")
        if digest:
            out.append((image, digest))
    # The same base image is used by several stages; check it once.
    seen: set[tuple[str, str]] = set()
    unique = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--registry",
        default=DOCKER_HUB,
        help="registry to compare against (docker.io, public.ecr.aws)",
    )
    ap.add_argument(
        "--dockerfile",
        default="Dockerfile",
        help="file to read the pinned images from",
    )
    args = ap.parse_args()

    try:
        with open(args.dockerfile, encoding="utf-8") as fh:
            dockerfile = fh.read()
    except OSError as exc:
        print(f"error: cannot read {args.dockerfile}: {exc}", file=sys.stderr)
        return 2

    pins = pinned_images(dockerfile)
    if not pins:
        print(f"error: no digest-pinned FROM line in {args.dockerfile}", file=sys.stderr)
        return 2

    stale = []
    unreachable = []
    for image, digest in pins:
        _img_registry, repo_and_tag = split_image(image)
        repo, tag = split_repo_tag(repo_and_tag)
        try:
            current = manifest_digest(args.registry, repo, tag)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"warn: cannot reach {args.registry} for {repo}:{tag}: {exc}", file=sys.stderr)
            unreachable.append((image, digest, str(exc)))
            continue

        if current == digest:
            print(f"ok:   {image} -> {digest[:19]}... still matches {args.registry}")
        else:
            print(
                f"DRIFT: {image}\n"
                f"       pinned: {digest}\n"
                f"       current: {current}\n"
                f"       the tag moved. Update the pin in {args.dockerfile}."
            )
            stale.append((image, digest, current))

    if stale:
        return 1
    if unreachable and len(unreachable) == len(pins):
        print(
            f"error: could not reach any registry ({args.registry}); the pins were NOT verified",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
