#!/usr/bin/env python3
"""Detect dependency *downgrades* in Cargo.lock.

Why this exists
---------------
`--locked` asserts that Cargo.lock agrees with Cargo.toml. It does **not**
stop a dependency from being pinned to an *older* version -- a downgrade is
still a perfectly consistent lock file. Downgrades are the interesting case
for supply chain: an attacker who can get a lock-file edit merged (or who
compromises an upstream release) can move a dependency back to a version
with a known CVE, and every consistency check stays green.

Dependabot only ever *raises* versions, so a decrease is either a mistake
or an attack. The check is a gate, so it needs an escape hatch: label the
PR `supply-chain-ok` when a downgrade is genuinely intended.

Usage:
    tools/lock-drift.py <base-ref> [Cargo.lock]

`<base-ref>` is any git ref whose Cargo.lock is the "before" state
(a base branch, a commit SHA). The lock file to compare defaults to
Cargo.lock in the current directory.

Exit codes: 0 = no downgrade, 1 = downgrade found, 2 = usage/parse error.

Two signals, because one is not enough
---------------------------------------
A lock file may hold several versions of the same crate (Cargo only
allows that when the semver requirements are incompatible -- tower-http
0.6 and 0.7 coexist happily). Comparing "the version of crate X" by
name alone therefore loses information, and gets it wrong in both
directions:

  - it reports a downgrade when a *newer* duplicate was simply removed
    (dropping the direct `tower-http 0.7.1` dependency while the
    transitive `tower-http 0.6.11` stays is a removal, not a downgrade);
  - it misses a downgrade when the older version was already present
    (the direct dependency moves 0.7.1 -> 0.6.11 while a transitive
    0.6.11 is already in the lock).

So:

  1. **Direct dependencies.** The workspace's own entry in the lock file
     lists its dependencies, and Cargo writes the version there when it
     is ambiguous. Each direct dependency is resolved against its own
     lock file and the two versions compared. This catches the second
     case above, where the name alone is not enough to tell which
     version the workspace actually got.

  2. **Package sets.** For every (crate, source): a version that
     appeared while a *higher* version disappeared is a replacement --
     a dependency was moved back. This catches the transitive case.

A removal is not a downgrade, and a version that appears alongside the
existing ones is not either: only a replacement counts.

What is still not covered: a transitive downgrade that happens to be
masked by an identical older version already in the lock file, in a
branch that does not touch the workspace's own dependency list. The
lock file does not record *which* dependency requires *which* version,
so that case is indistinguishable from a removal. It is called out
here rather than papered over.
"""

import subprocess
import sys
import tomllib

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"


class Lock:
    """The packages in one Cargo.lock, and the workspace's own entry."""

    def __init__(self, text: str):
        data = tomllib.loads(text)
        # (name, version, source) -> the package's dependency list
        self.packages: dict[tuple[str, str, str], list[str]] = {}
        # name -> the versions present
        self.versions: dict[str, set[str]] = {}
        # name -> source, for the (rare) case of one name from two sources
        self.sources: dict[str, set[str]] = {}
        # the workspace member: the package without a source
        self.root: list[str] = []

        for pkg in data.get("package", []):
            source = pkg.get("source")
            deps = pkg.get("dependencies", [])
            if not source:
                # A path/workspace package is built from this repository
                # and cannot be a supply-chain vector. Its dependency
                # list, though, is exactly what signal 1 needs.
                if not self.root:
                    self.root = deps
                continue
            version = pkg["version"]
            self.packages[(pkg["name"], version, source)] = deps
            self.versions.setdefault(pkg["name"], set()).add(version)
            self.sources.setdefault(pkg["name"], set()).add(source)

    def dep_name(self, dep: str) -> str:
        """The crate name of a dependency entry ("foo" or "foo 1.2.3")."""
        return dep.split()[0]

    def resolve(self, dep: str) -> str | None:
        """The version a dependency entry refers to in this lock file.

        Cargo writes "foo" when only one version of foo is in the lock,
        and "foo 1.2.3" when several are. Returns None when the entry
        is ambiguous -- the caller cannot compare what it cannot pin
        down.
        """
        parts = dep.split()
        name = parts[0]
        if len(parts) > 1 and parts[1][0].isdigit():
            return parts[1]
        candidates = self.versions.get(name, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        return None


def prerelease_key(pre: str) -> tuple:
    """Order a prerelease suffix the way semver does.

    A prerelease is a dot-separated list of identifiers. Numeric
    identifiers compare numerically and always have *lower* precedence
    than alphanumeric ones; alphanumeric identifiers compare
    lexically (ASCII order); a larger set of identifiers wins when all
    the preceding ones are equal.

    So "rc.10" > "rc.9" -- comparing the suffix as a plain string would
    say the opposite, and a downgrade from rc.10 to rc.9 would go
    unnoticed.
    """
    parts = []
    for ident in pre.split("."):
        if ident.isdigit():
            # (0, value, "") -- numeric identifiers rank below alphanumerics.
            parts.append((0, int(ident), ""))
        else:
            parts.append((1, 0, ident))
    return tuple(parts)


def version_key(v: str):
    """Order versions the way semver does, approximately.

    Numeric components compare numerically (1.10 > 1.9); a prerelease
    sorts before its release (1.0.0-rc.1 < 1.0.0). Build metadata
    ("+sha") is ignored, as semver requires.
    """
    core, _, rest = v.partition("+")
    core, _, pre = core.partition("-")
    nums = [int(p) if p.isdigit() else 0 for p in core.split(".")]
    nums = (nums + [0, 0, 0])[:3]
    # (0, pre) sorts before (1, "") -- i.e. prerelease before release.
    return (tuple(nums), 0 if pre else 1, prerelease_key(pre))


def direct_downgrades(old: Lock, new: Lock) -> list[tuple[str, str, str]]:
    """Direct dependencies whose version went backwards."""
    new_deps = {new.dep_name(d): d for d in new.root}
    out = []
    for dep in old.root:
        name = old.dep_name(dep)
        if name not in new_deps:
            continue  # the dependency was removed -- not a downgrade
        old_v = old.resolve(dep)
        new_v = new.resolve(new_deps[name])
        if old_v is None or new_v is None:
            continue  # ambiguous in one of the two locks
        if version_key(new_v) < version_key(old_v):
            out.append((name, old_v, new_v))
    return out


def transitive_downgrades(old: Lock, new: Lock) -> list[tuple[str, str, str]]:
    """Packages where a lower version replaced a higher one.

    Grouped by (name, source): two versions of a crate from different
    registries are unrelated packages, and comparing them would be
    meaningless.
    """
    def by_name_source(lock: Lock) -> dict[tuple[str, str], set[str]]:
        out: dict[tuple[str, str], set[str]] = {}
        for name, version, source in lock.packages:
            out.setdefault((name, source), set()).add(version)
        return out

    old_sets = by_name_source(old)
    new_sets = by_name_source(new)
    out = []
    for key, new_versions in sorted(new_sets.items()):
        old_versions = old_sets.get(key, set())
        appeared = new_versions - old_versions
        disappeared = old_versions - new_versions
        for new_v in sorted(appeared):
            for old_v in sorted(disappeared):
                if version_key(new_v) < version_key(old_v):
                    out.append((key[0], old_v, new_v))
    return out


def selftest() -> int:
    """Check the semantics on hand-written lock files.

    This check's first version reported a downgrade whenever a crate's
    version went down *by name*, and that is wrong in both directions.
    A false positive here blocks legitimate pull requests -- it blocked
    one of ours, over a dependency that was being removed -- and a false
    negative is the whole thing this check exists to prevent. So the
    cases are written down, and can be run without a git repository:

        tools/lock-drift.py --selftest
    """

    def lock(pkgs, root_deps=()):
        text = '[[package]]\nname = "app"\nversion = "0.1.0"\ndependencies = [\n'
        for dep in root_deps:
            text += f' "{dep}",\n'
        text += "]\n"
        for name, version, deps in pkgs:
            text += (
                f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
                f'source = "{REGISTRY}"\ndependencies = [\n'
            )
            for dep in deps:
                text += f' "{dep}",\n'
            text += "]\n"
        return text

    def downgrades(old_text, new_text):
        old, new = Lock(old_text), Lock(new_text)
        found = direct_downgrades(old, new)
        for item in transitive_downgrades(old, new):
            if item not in found:
                found.append(item)
        return sorted(found)

    cases = [
        (
            "a transitive dependency moved to an older version",
            lock([("foo", "1.0.0", [])]),
            lock([("foo", "0.9.0", [])]),
            [("foo", "1.0.0", "0.9.0")],
        ),
        (
            "a direct dependency moved to an older version",
            lock([("foo", "1.0.0", [])], ["foo 1.0.0"]),
            lock([("foo", "0.9.0", [])], ["foo 0.9.0"]),
            [("foo", "1.0.0", "0.9.0")],
        ),
        (
            "a direct dependency moved to an older version that was "
            "already in the lock file",
            lock([("tw", "0.6.11", []), ("tw", "0.7.1", [])], ["tw 0.7.1"]),
            lock([("tw", "0.6.11", [])], ["tw"]),
            [("tw", "0.7.1", "0.6.11")],
        ),
        (
            "a dependency was removed, and the older duplicate stayed",
            lock([("tw", "0.6.11", []), ("tw", "0.7.1", [])], ["tw 0.7.1"]),
            lock([("tw", "0.6.11", [])]),
            [],
        ),
        (
            "an ordinary version bump",
            lock([("foo", "1.0.0", [])]),
            lock([("foo", "1.1.0", [])]),
            [],
        ),
        (
            "a dependency was removed entirely",
            lock([("foo", "1.0.0", [])], ["foo 1.0.0"]),
            lock([]),
            [],
        ),
        (
            "one of two versions was replaced by an older one",
            lock([("foo", "0.9.0", []), ("foo", "1.0.0", [])]),
            lock([("foo", "0.8.0", []), ("foo", "1.0.0", [])]),
            [("foo", "0.9.0", "0.8.0")],
        ),
        (
            "an older version appeared alongside the existing one",
            lock([("foo", "1.0.0", [])]),
            lock([("foo", "0.8.0", []), ("foo", "1.0.0", [])]),
            [],
        ),
        (
            "a prerelease moved to an older prerelease (rc.10 -> rc.9)",
            lock([("foo", "1.0.0-rc.10", [])]),
            lock([("foo", "1.0.0-rc.9", [])]),
            [("foo", "1.0.0-rc.10", "1.0.0-rc.9")],
        ),
        (
            "a prerelease moved to a newer prerelease (rc.9 -> rc.10)",
            lock([("foo", "1.0.0-rc.9", [])]),
            lock([("foo", "1.0.0-rc.10", [])]),
            [],
        ),
        (
            "a prerelease moved to its release (1.0.0-rc.1 -> 1.0.0)",
            lock([("foo", "1.0.0-rc.1", [])]),
            lock([("foo", "1.0.0", [])]),
            [],
        ),
        (
            "a release moved back to a prerelease (1.0.0 -> 1.0.0-rc.1)",
            lock([("foo", "1.0.0", [])]),
            lock([("foo", "1.0.0-rc.1", [])]),
            [("foo", "1.0.0", "1.0.0-rc.1")],
        ),
    ]

    failures = 0
    for name, old_text, new_text, expected in cases:
        found = downgrades(old_text, new_text)
        if found == expected:
            print(f"ok:   {name}")
        else:
            failures += 1
            print(f"FAIL: {name}\n      expected {expected}\n      got      {found}")
    if failures:
        print(f"\n{failures} case(s) failed")
        return 1
    print(f"\nall {len(cases)} cases passed")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--"]
    if args and args[0] == "--selftest":
        return selftest()
    if len(args) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2

    base_ref = args[0]
    lock_path = args[1] if len(args) == 2 else "Cargo.lock"
    try:
        with open(lock_path, "rb") as fh:
            new_text = fh.read().decode()
    except OSError as exc:
        print(f"lock-drift: cannot read {lock_path}: {exc}", file=sys.stderr)
        return 2

    # Resolve the base ref to a full object name first. A ref that
    # starts with "-" would otherwise be parsed by git as an option;
    # `--end-of-options` tells git that everything after it is a
    # revision, and the resolved SHA never starts with "-".
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0:
        print(
            f"lock-drift: cannot resolve base ref {base_ref!r}: "
            f"{resolved.stderr.strip()}",
            file=sys.stderr,
        )
        return 2
    base = resolved.stdout.strip()

    proc = subprocess.run(
        ["git", "show", f"{base}:Cargo.lock"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(
            f"lock-drift: cannot read Cargo.lock at {base_ref!r}: "
            f"{proc.stderr.strip()}",
            file=sys.stderr,
        )
        return 2

    try:
        old = Lock(proc.stdout)
        new = Lock(new_text)
    except tomllib.TOMLDecodeError as exc:
        print(f"lock-drift: cannot parse Cargo.lock: {exc}", file=sys.stderr)
        return 2

    downgrades = direct_downgrades(old, new)
    for name, old_v, new_v in transitive_downgrades(old, new):
        if (name, old_v, new_v) not in downgrades:
            downgrades.append((name, old_v, new_v))

    if not downgrades:
        print(
            f"lock-drift: no downgrades against {base_ref} "
            f"({len(new.packages)} packages)"
        )
        return 0

    print(f"lock-drift: DEPENDENCY DOWNGRADE detected against {base_ref}:")
    for name, old_v, new_v in sorted(downgrades):
        print(f"  {name}: {old_v} -> {new_v}  (DOWNGRADE)")
    print("")
    print("A downgrade keeps Cargo.lock self-consistent, so `--locked` and")
    print("`cargo deny` will not catch it. If this is intended, label the PR")
    print("`supply-chain-ok` and add a note explaining why.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
