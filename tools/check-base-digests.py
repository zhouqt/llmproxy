#!/usr/bin/env python3
"""Manage the base images pinned in the Dockerfile.

The policy
----------
The Dockerfile pins its base images by digest:

    FROM docker.io/library/rust:1-alpine@sha256:...

A tag moves every time upstream rebuilds the image, and `rust:1-alpine`
is rebuilt constantly. A pin that follows the tag would move constantly
too, and a pull request per rebuild is a pull request nobody reads.

So the pin does **not** follow the tag. It moves for exactly one reason:
the pinned image has a vulnerability that a newer build has fixed. The
CVE scan in `.github/workflows/refresh-pins.yml` is what decides; this
script is the hand that moves the pin once that decision is made.

    tools/check-base-digests.py                      # report drift
    tools/check-base-digests.py --list               # what is pinned
    tools/check-base-digests.py --resolve            # what the tags point at now
    tools/check-base-digests.py --set IMAGE@DIGEST   # move a pin, deliberately

Exit status
-----------
0   done
1   a digest has drifted (the tag moved). On its own this is not an
    error: a moved tag is only worth acting on when it carries a fix.
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


def rewrite_pins(dockerfile: str, stale: list[tuple[str, str, str]]) -> int:
    """Point the pins at the digests the tags currently resolve to.

    Only the digest is replaced: the image name, the tag and everything
    around it stay exactly as they were, so the diff of a refresh PR is
    one line per base image.

    The digest string is replaced wherever it appears, not just on the
    FROM line. The Dockerfile's header lists the same digests in a
    comment, and a comment that disagrees with the FROM line is a lie
    that survives until somebody reads it.
    """
    with open(dockerfile, encoding="utf-8") as fh:
        text = fh.read()
    changed = 0
    for image, old, new in stale:
        if old in text:
            text = text.replace(old, new)
            changed += 1
        else:
            print(
                f"warn: the pinned digest for {image} does not appear in "
                f"{dockerfile}; it was not updated",
                file=sys.stderr,
            )
    if changed:
        with open(dockerfile, "w", encoding="utf-8") as fh:
            fh.write(text)
    return changed


def apply_sets(dockerfile: str, pins: list[tuple[str, str]], specs: list[str]) -> int:
    """Move the given pins to the given digests.

    This is the deliberate half of the job: unlike `--update`, which
    follows the tag, this moves a specific image to a specific digest --
    the one a CVE scan has cleared.
    """
    by_image = {image: digest for image, digest in pins}
    stale: list[tuple[str, str, str]] = []
    for spec in specs:
        image, _, digest = spec.partition("@")
        if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
            print(f"error: not a digest: {spec}", file=sys.stderr)
            return 2
        if image not in by_image:
            print(
                f"error: {image} is not pinned in {dockerfile}; "
                "use the exact image name from --list",
                file=sys.stderr,
            )
            return 2
        old = by_image[image]
        if old == digest:
            print(f"ok:   {image} is already pinned to {digest[:19]}...")
            continue
        stale.append((image, old, digest))

    if not stale:
        print("nothing to move")
        return 0
    changed = rewrite_pins(dockerfile, stale)
    for image, old, new in stale:
        print(f"moved {image}: {old[:19]}... -> {new[:19]}...")
    print(f"updated {changed} pin(s) in {dockerfile}")
    return 0


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
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the pinned images as 'image@digest', one per line, and exit",
    )
    ap.add_argument(
        "--resolve",
        action="store_true",
        help=(
            "print what each pinned tag resolves to right now as "
            "'image@digest', one per line, and exit. This is the candidate "
            "an upgrade would move to -- whether it is worth moving is a "
            "question about CVEs, not about the tag having moved."
        ),
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="IMAGE@DIGEST",
        help=(
            "pin IMAGE to DIGEST in the Dockerfile (repeatable). This is "
            "the deliberate move: it is what an upgrade does once a CVE "
            "says the pin has to move."
        ),
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

    if args.set:
        return apply_sets(args.dockerfile, pins, args.set)

    if args.list:
        # The Dockerfile is the single source of truth for which images are
        # pinned; a workflow that needs those references reads them from
        # here instead of repeating the digests.
        for image, digest in pins:
            print(f"{image}@{digest}")
        return 0

    if args.resolve:
        failed = False
        for image, _digest in pins:
            _reg, repo_and_tag = split_image(image)
            repo, tag = split_repo_tag(repo_and_tag)
            try:
                current = manifest_digest(args.registry, repo, tag)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                print(f"warn: cannot reach {args.registry} for {repo}:{tag}: {exc}", file=sys.stderr)
                failed = True
                continue
            print(f"{image}@{current}")
        if failed:
            print("error: could not resolve every pinned tag", file=sys.stderr)
            return 2
        return 0

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
        # A moved tag is a fact, not a defect. The pin does not follow
        # the tag; it follows a CVE. Say so and let the CVE scan decide.
        print(
            "note: the tag moved. The pin stays until a CVE says it has "
            f"to move -- see the base-image-cve job in refresh-pins.yml.",
            file=sys.stderr,
        )
        return 1
    if unreachable and len(unreachable) == len(pins):
        print(
            f"error: could not reach any registry ({args.registry}); the pins were NOT verified",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- the exit code is the message
        # Anything unexpected is "could not check", not "nothing to do".
        # The caller distinguishes the two by the exit code.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
