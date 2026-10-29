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
    tools/check-base-digests.py --selftest           # check the exit codes offline

Exit status
-----------
0   done
1   a digest has drifted (the tag moved) AND every pin was verified
    against the registry. On its own this is not an error: a moved tag
    is only worth acting on when it carries a fix.
2   the pins could not be verified (network problem, not a drift), or
    a pin is malformed. Partial verification counts as no verification:
    a pin that was not checked is a pin that is not verified.

The two failure modes are deliberately different exit codes: "your pin
is stale" and "we could not check" must not look the same to a human
reading a red CI run.

When both happen at once -- one pin drifted and another could not be
checked -- the answer is 2, not 1. The drift is real, but a pin that was
never verified is a pin that is not verified, and exit 1 is treated by
the caller as "the check succeeded, the tag merely moved". Reporting 1
there would let a registry outage hide behind a moved tag and leave an
unchecked pin green. So the "could not check" case is decided first and
wins.

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
    r"(?:@(?P<ref>\S+))?",
    re.IGNORECASE | re.MULTILINE,
)

# The only shape of digest this repository accepts: `sha256:` followed by
# 64 lowercase hex characters. The check is a *full match*, so a digest
# with trailing junk ("sha256:<64 hex>#comment") or uppercase hex is
# rejected rather than half-parsed.
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")

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
        # "I cannot check this registry" is not "the pin drifted":
        # it is the exit code 2 case.
        print(f"error: unsupported registry: {registry}", file=sys.stderr)
        sys.exit(2)
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = json.load(resp)
    token = body.get("token") or body.get("access_token")
    if not token:
        print(f"error: no token in the response from {url}", file=sys.stderr)
        sys.exit(2)
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


def split_digest(image: str) -> tuple[str, str | None]:
    """Split "name@sha256:..." into (name, digest).

    A reference that carries *something* after the `@` but is not a
    valid sha256 digest is an error, not a reference without a pin: the
    difference is the difference between "this image is pinned" and
    "this image is not pinned at all", and only the second one may be
    handled by saying "nothing to do".
    """
    if "@" not in image:
        return image, None
    name, _, digest = image.partition("@")
    if not DIGEST_RE.fullmatch(digest):
        raise ValueError(
            f"{image!r} carries a suffix that is not a sha256 digest "
            "(expected 'sha256:' followed by 64 lowercase hex characters)"
        )
    return name, digest


def pinned_images(dockerfile: str) -> list[tuple[str, str]]:
    """Return [(image, digest)] for every FROM line that pins a digest.

    The Dockerfile names its base images through build ARGs
    (FROM ${RUST_IMAGE}), so the ARG defaults are substituted first --
    otherwise the pins would be invisible to this check.

    Raises ValueError when a FROM line carries a digest-looking suffix
    that is not a valid digest: such a line is a broken pin, and it must
    not be silently skipped.
    """
    args: dict[str, str] = {}
    for m in ARG_RE.finditer(dockerfile):
        if m.group("value"):
            args[m.group("name")] = m.group("value")

    out = []
    for m in FROM_RE.finditer(dockerfile):
        image = m.group("image")
        ref = m.group("ref")
        if "${" in image:
            for var in VAR_RE.finditer(image):
                image = image.replace(var.group(0), args.get(var.group("name"), ""))
        # A digest inside the ARG value: FROM ${RUST_IMAGE} has no
        # "@sha256:" of its own, the pin lives in the ARG default.
        image, arg_digest = split_digest(image)
        if ref is not None:
            if not DIGEST_RE.fullmatch(ref):
                raise ValueError(
                    f"{image}@{ref} is not a valid digest "
                    "(expected 'sha256:' followed by 64 lowercase hex characters)"
                )
            digest = ref
        else:
            digest = arg_digest
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
        if not DIGEST_RE.fullmatch(digest):
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


def check_pins(
    pins: list[tuple[str, str]], registry: str, fetcher, dockerfile: str = "the Dockerfile"
) -> tuple[list, list]:
    """Compare every pinned digest against what the registry serves now.

    `fetcher(registry, repo, tag)` returns the digest the tag currently
    points at, or raises. The two outcomes are kept apart:

      stale      -- the registry answered and the digest moved
      unreachable -- the registry did not answer, so the pin was NOT
                     checked at all

    The second list is the fail-closed case: a pin that could not be
    checked is a pin that is not verified.
    """
    stale = []
    unreachable = []
    for image, digest in pins:
        _img_registry, repo_and_tag = split_image(image)
        repo, tag = split_repo_tag(repo_and_tag)
        try:
            current = fetcher(registry, repo, tag)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"warn: cannot reach {registry} for {repo}:{tag}: {exc}", file=sys.stderr)
            unreachable.append((image, digest, str(exc)))
            continue

        if current == digest:
            print(f"ok:   {image} -> {digest[:19]}... still matches {registry}")
        else:
            print(
                f"DRIFT: {image}\n"
                f"       pinned: {digest}\n"
                f"       current: {current}\n"
                f"       the tag moved. Update the pin in {dockerfile}."
            )
            stale.append((image, digest, current))
    return stale, unreachable


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
        "--selftest",
        action="store_true",
        help=(
            "check the exit-code logic offline (no registry access) and "
            "exit: 0 = every case behaved as expected"
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

    if args.selftest:
        return selftest()

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

    stale, unreachable = check_pins(pins, args.registry, manifest_digest, args.dockerfile)
    return report(pins, stale, unreachable, args.registry)


def report(pins: list[tuple[str, str]], stale: list, unreachable: list, registry: str) -> int:
    """Turn the outcome of the comparison into an exit code.

    The "could not check" case is decided FIRST, on purpose.
    """
    # A pin that could not be verified is a pin that is not verified. If
    # one pin drifted and another one could not be reached at all, the
    # answer is 2, not 1: exit 1 means "every pin was checked and one
    # merely moved", and the caller treats it as a non-failure. Reporting
    # 1 there would let a registry outage hide behind a moved tag and
    # leave the unchecked pin green.
    if unreachable:
        # Fail closed: a pin that could not be checked is a pin that is
        # not verified. Reporting "ok" for the subset that answered
        # would let a registry outage look like a clean audit.
        print(
            f"error: could not verify {len(unreachable)} of {len(pins)} pinned "
            f"image(s) against {registry}; the pins were NOT verified",
            file=sys.stderr,
        )
        return 2
    if stale:
        # A moved tag is a fact, not a defect. The pin does not follow
        # the tag; it follows a CVE. Say so and let the CVE scan decide.
        print(
            "note: the tag moved. The pin stays until a CVE says it has "
            f"to move -- see the base-image-cve job in refresh-pins.yml.",
            file=sys.stderr,
        )
        return 1
    return 0


def selftest() -> int:
    """Check the exit-code logic without touching a registry.

        tools/check-base-digests.py --selftest

    The cases that matter here are the combinations of "the tag moved"
    and "the registry could not be reached", because the two have to
    stay distinguishable: exit 1 is a non-failure for the caller, exit 2
    is a failure, and a pin that was never checked must never end up
    under exit 1.
    """
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    digest_c = "sha256:" + "c" * 64
    pins = [("docker.io/library/alpha:1", digest_a), ("docker.io/library/beta:1", digest_b)]

    def unreachable(*_args, **_kwargs):
        raise urllib.error.URLError("mock: registry is down")

    def drifted(_registry, repo, _tag):
        # `alpha` moved, `beta` still matches.
        return digest_c if repo == "library/alpha" else digest_b

    def fine(_registry, repo, _tag):
        return digest_a if repo == "library/alpha" else digest_b

    cases = [
        ("every pin verified, nothing moved", pins, fine, 0),
        ("one pin moved, all verified", pins, drifted, 1),
        ("one pin could not be reached", pins, unreachable, 2),
        (
            "one pin moved AND one pin could not be reached -> 2, not 1",
            pins,
            lambda registry, repo, tag: drifted(registry, repo, tag)
            if repo == "library/alpha"
            else unreachable(registry, repo, tag),
            2,
        ),
    ]

    failures = 0
    for name, case_pins, fetcher, expected in cases:
        stale, unreachable_pins = check_pins(case_pins, DOCKER_HUB, fetcher)
        rc = report(case_pins, stale, unreachable_pins, DOCKER_HUB)
        if rc == expected:
            print(f"ok:   {name} (exit {rc})")
        else:
            print(f"FAIL: {name}: exit {rc}, expected {expected}")
            failures += 1

    if failures:
        print(f"{failures} case(s) failed")
        return 1
    print("all cases passed")
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
