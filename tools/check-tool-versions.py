#!/usr/bin/env python3
"""Check the tool versions this repository pins, and update them.

What is pinned
--------------
The audit and coverage tooling is installed in CI with an exact version:

    cargo install --locked --version 0.20.2 cargo-deny

The exact version is deliberate. These tools run with full permissions,
exactly like the build scripts they audit, so bumping one is a decision
that somebody makes on purpose -- not a floating update that happens
underneath. But a version that never moves is a version that gets
stale, and a stale `cargo-deny` may not know about a new advisory class.

So this script does the half of the job that is mechanical -- "there is a
newer version" -- and leaves the other half, "is the new version
acceptable", to the pull request that a human merges.

Usage
-----
    tools/check-tool-versions.py            # report; exit 1 if outdated
    tools/check-tool-versions.py --update   # rewrite the pins in place

The pins live in two kinds of file:

  .github/workflows/ci.yml   cargo install --locked --version X cargo-...
  Dockerfile                 cargo install --locked --version X cargo-...

Both are rewritten in place by --update. The version stays exact
(no caret, no range) -- that is the whole point of pinning it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

# `cargo install --locked --version <VER> <crate>`
INSTALL_RE = re.compile(
    r"cargo\s+install\s+--locked\s+--version\s+(?P<ver>\S+)\s+(?P<crate>cargo-[\w-]+)"
)

CRATES_IO_API = "https://crates.io/api/v1/crates/{crate}"

# crates.io asks every client to identify itself.
USER_AGENT = "llmproxy-supply-chain-check (version pin refresher)"


def pinned_versions(paths: list[str]) -> list[tuple[str, str, str, int]]:
    """Return [(path, crate, version, line_no)] for every pinned install."""
    found = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                m = INSTALL_RE.search(line)
                if m:
                    found.append((path, m.group("crate"), m.group("ver"), lineno))
    return found


def latest_version(crate: str) -> str | None:
    """Return the newest stable version of `crate` on crates.io."""
    url = CRATES_IO_API.format(crate=crate)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data.get("crate", {}).get("max_stable_version")


def update_pins(paths: list[str], wanted: dict[tuple[str, str], str]) -> int:
    """Rewrite the pinned versions. Returns the number of lines changed."""
    changed = 0
    # path -> {crate: new version}
    by_path: dict[str, dict[str, str]] = {}
    for (path, crate), ver in wanted.items():
        by_path.setdefault(path, {})[crate] = ver

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        dirty = False
        for i, line in enumerate(lines):
            m = INSTALL_RE.search(line)
            if not m:
                continue
            crate = m.group("crate")
            new = by_path.get(path, {}).get(crate)
            if new is None or new == m.group("ver"):
                continue
            lines[i] = INSTALL_RE.sub(
                lambda _m, v=new, c=crate: f"cargo install --locked --version {v} {c}", line
            )
            changed += 1
            dirty = True
        if dirty:
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the pins in place instead of only reporting",
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=[".github/workflows/ci.yml", "Dockerfile"],
        help="files to scan for pinned installs",
    )
    args = ap.parse_args()

    pins = pinned_versions(args.files)
    if not pins:
        print("error: no pinned cargo installs found", file=sys.stderr)
        return 2

    # One lookup per crate, even if it is pinned in several files.
    crates = sorted({crate for _, crate, _, _ in pins})
    latest: dict[str, str | None] = {}
    unreachable = False
    for crate in crates:
        try:
            latest[crate] = latest_version(crate)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"warn: cannot reach crates.io for {crate}: {exc}", file=sys.stderr)
            latest[crate] = None
            unreachable = True

    outdated = []
    wanted: dict[tuple[str, str], str] = {}
    for path, crate, ver, lineno in pins:
        new = latest.get(crate)
        if new is None:
            continue
        if new != ver:
            outdated.append((path, lineno, crate, ver, new))
            wanted[(path, crate)] = new
        else:
            print(f"ok:   {crate} {ver} (pinned at {path}:{lineno})")

    if not outdated:
        if unreachable:
            print(
                "error: could not reach crates.io; the pins were NOT verified",
                file=sys.stderr,
            )
            return 2
        return 0

    for path, lineno, crate, ver, new in outdated:
        print(f"OUTDATED: {crate} {ver} -> {new}   ({path}:{lineno})")

    if args.update:
        changed = update_pins(args.files, wanted)
        print(f"updated {changed} pin(s) in place")
        # The pins are now current; the caller can treat this as success
        # and look at the diff to see what moved.
        return 0
    print("re-run with --update to rewrite the pins")
    return 1


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
