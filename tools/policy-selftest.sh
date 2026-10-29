#!/bin/sh
# Prove that the supply-chain checks actually go RED when they should.
#
# Why this exists
# ---------------
# Every control in this repository is a gate that is supposed to fail on a
# violation. A gate that has never been seen failing is indistinguishable
# from a gate that is silently broken: `cargo deny check` exits 0 on a
# policy with no teeth, `cargo vet` exits 0 on a ledger that matches
# nothing, `lock-drift.py` exits 0 on a lock file it never understood.
#
# So the checks are tested the way the sandbox canary is tested: in BOTH
# directions. The green path is the control (the unmodified repository
# passes), and every red case injects exactly one violation and asserts
# that the tool refuses it.
#
# Each case runs in its own scratch copy of the repository, so a
# deliberately broken Cargo.lock or deny.toml never touches the real tree.
#
# Usage:
#     tools/policy-selftest.sh
#
# Requires cargo-deny and cargo-vet on PATH (the versions CI pins:
# cargo-deny 0.20.2, cargo-vet 0.10.3). The lock-drift case additionally
# needs python3; without it that case is reported as SKIP -- CI treats a
# SKIP as a failure, because there the interpreter is always present.
#
# POSIX sh on purpose: the build host has no bash and no python, and a
# check that only runs on one machine is a check that quietly stops
# running.
#
# Exit codes: 0 = every case that ran behaved as expected, 1 = a case did
# not. (A SKIP does not fail the script; the CI step greps for it.)

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0
SKIP=0

ok()  { printf 'ok:   %s\n' "$1"; PASS=$((PASS + 1)); }
bad() { printf 'FAIL: %s\n' "$1"; FAIL=$((FAIL + 1)); }
skip(){ printf 'SKIP: %s\n' "$1"; SKIP=$((SKIP + 1)); }

# Copy the repository into a scratch directory. The build directory is
# never copied: it is gigabytes and none of the tools need it.
scratch_copy() {
    dest=$1
    mkdir -p "$dest"
    for item in Cargo.toml Cargo.lock deny.toml Dockerfile src tests supply-chain tools; do
        if [ -e "$REPO_ROOT/$item" ]; then
            cp -R "$REPO_ROOT/$item" "$dest/"
        fi
    done
    # .cargo/config.toml pins CARGO_TARGET_DIR to a shared path; inside a
    # scratch copy the tools must not share a target directory with the
    # real tree.
    rm -rf "$dest/.cargo"
    return 0
}

# Run a command inside a scratch copy, keep its output and exit code.
run_in() {
    dir=$1
    shift
    ( cd "$dir" && "$@" ) >"$SCRATCH/out.txt" 2>&1
    printf '%s' "$?" > "$SCRATCH/rc.txt"
    return 0
}

last_rc()  { cat "$SCRATCH/rc.txt"; }
last_out() { cat "$SCRATCH/out.txt"; }

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'error: %s is not on PATH; the self-test cannot run\n' "$1" >&2
        exit 2
    fi
}

need cargo-deny
need cargo-vet
need cargo
need awk
need sed

# ---------------------------------------------------------------------------
# Control: the green path. If the unmodified repository does NOT pass, the
# red cases below would prove nothing -- the tool might simply be broken.
# ---------------------------------------------------------------------------
green_dir=$SCRATCH/green
scratch_copy "$green_dir"
run_in "$green_dir" cargo deny check
if [ "$(last_rc)" -eq 0 ]; then
    ok "control: cargo deny check passes on the unmodified tree"
else
    bad "control: cargo deny check FAILS on the unmodified tree (the red cases below would prove nothing)"
    printf -- '---- cargo deny output ----\n%s\n---------------------------\n' "$(last_out)"
fi

run_in "$green_dir" cargo vet
if [ "$(last_rc)" -eq 0 ]; then
    ok "control: cargo vet passes on the unmodified tree"
else
    bad "control: cargo vet FAILS on the unmodified tree (the red cases below would prove nothing)"
    printf -- '---- cargo vet output ----\n%s\n---------------------------\n' "$(last_out)"
fi

# ---------------------------------------------------------------------------
# Red case 1: cargo-deny, licences.
#
# The allow-list is narrowed to a single licence. The check must refuse the
# graph. (An allow-list that is too wide passes; one that is too narrow
# must fail -- that is the property "the list = the licences actually in
# use" rests on.)
# ---------------------------------------------------------------------------
case_dir=$SCRATCH/deny-licenses
scratch_copy "$case_dir"
awk '
    /^allow = \[/ { print "allow = [\"MIT\"]"; skip = 1; next }
    skip && /^\]/ { skip = 0; next }
    !skip { print }
' "$case_dir/deny.toml" > "$case_dir/deny.toml.new" && mv "$case_dir/deny.toml.new" "$case_dir/deny.toml"
run_in "$case_dir" cargo deny check licenses
rc=$(last_rc)
if [ "$rc" -ne 0 ] && last_out | grep -qiE 'not allowed|license'; then
    ok "cargo-deny rejects a dependency graph whose licence is not allowed"
else
    bad "cargo-deny did NOT reject the narrowed licence allow-list (exit $rc)"
    printf -- '---- cargo deny output ----\n%s\n---------------------------\n' "$(last_out)"
fi

# ---------------------------------------------------------------------------
# Red case 2: cargo-deny, sources.
#
# The crates.io registry is removed from the allowed sources, so every
# dependency comes from an unknown source and must be denied. This is the
# control that matters for git/path dependencies, which the lock file's
# sha256 does NOT cover.
# ---------------------------------------------------------------------------
case_dir=$SCRATCH/deny-sources
scratch_copy "$case_dir"
sed 's|^allow-registry = .*|allow-registry = []|' "$case_dir/deny.toml" > "$case_dir/deny.toml.new" \
    && mv "$case_dir/deny.toml.new" "$case_dir/deny.toml"
run_in "$case_dir" cargo deny check sources
rc=$(last_rc)
if [ "$rc" -ne 0 ] && last_out | grep -qiE 'source|registry'; then
    ok "cargo-deny rejects a dependency graph from an unallowed source"
else
    bad "cargo-deny did NOT reject the unallowed registry (exit $rc)"
    printf -- '---- cargo deny output ----\n%s\n---------------------------\n' "$(last_out)"
fi

# ---------------------------------------------------------------------------
# Red case 3: cargo-vet.
#
# One dependency is moved to a version the ledger has no record of. The
# ledger is per (crate, version, source), so any version change must make
# the check fail until somebody records an audit for the new version.
# ---------------------------------------------------------------------------
case_dir=$SCRATCH/vet
scratch_copy "$case_dir"
run_in "$case_dir" cargo update -p anyhow --precise 1.0.86
if [ "$(last_rc)" -ne 0 ]; then
    bad "could not move anyhow to an unvetted version; the red case did not run"
    printf -- '---- cargo update output ----\n%s\n---------------------------\n' "$(last_out)"
else
    run_in "$case_dir" cargo vet
    rc=$(last_rc)
    if [ "$rc" -ne 0 ] && last_out | grep -qiE 'unvetted|not audited|no audit|audit record|not vetted'; then
        ok "cargo-vet rejects a dependency version the ledger has no record of"
    else
        bad "cargo-vet did NOT reject an unvetted version (exit $rc)"
        printf -- '---- cargo vet output ----\n%s\n---------------------------\n' "$(last_out)"
    fi
fi

# ---------------------------------------------------------------------------
# Red case 4: the downgrade gate, end to end.
#
# lock-drift.py has its own unit-level self-test (--selftest); this case
# exercises the whole path -- a real lock file, a real base ref -- on a
# copy of the repository, so the CI step that calls it is covered too.
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    skip "lock-drift end-to-end case (python3 is not available on this host)"
else
    case_dir=$SCRATCH/drift
    scratch_copy "$case_dir"
    (
        cd "$case_dir" &&
        git init -q . &&
        git add -A &&
        git -c user.email=selftest@example.invalid -c user.name=selftest commit -qm base
    ) >/dev/null 2>&1
    # Move one crate to an older version. The checksum is left alone on
    # purpose: this case tests the *version* comparison, not cargo's
    # checksum enforcement (which has its own experiment).
    sed '0,/^version = "1\.[0-9]*\.[0-9]*"$/!b' "$case_dir/Cargo.lock" >/dev/null 2>&1
    awk '
        /^name = "anyhow"$/ { inpkg = 1 }
        inpkg && /^version = "/ && !done {
            print "version = \"1.0.80\""; done = 1; next
        }
        { print }
    ' "$case_dir/Cargo.lock" > "$case_dir/Cargo.lock.new" && mv "$case_dir/Cargo.lock.new" "$case_dir/Cargo.lock"
    run_in "$case_dir" python3 "$REPO_ROOT/tools/lock-drift.py" HEAD Cargo.lock
    rc=$(last_rc)
    if [ "$rc" -eq 1 ] && last_out | grep -qi 'DOWNGRADE'; then
        ok "lock-drift rejects a dependency moved to an older version"
    else
        bad "lock-drift did NOT reject the downgraded lock file (exit $rc)"
        printf -- '---- lock-drift output ----\n%s\n---------------------------\n' "$(last_out)"
    fi
fi

# ---------------------------------------------------------------------------
# Red case 5: the base-image pin check.
#
# A pin that is not a valid sha256 digest is not a pin, and a FROM line
# without any digest is not pinned at all. In both cases the check must
# refuse the Dockerfile (exit 2) instead of skipping the line and
# reporting "nothing to do" -- that would turn a broken pin into a
# green check.
#
# Both cases are offline: the check fails before it reaches a registry.
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    skip "base image pin cases (python3 is not available on this host)"
else
    case_dir=$SCRATCH/pin-malformed
    scratch_copy "$case_dir"
    sed 's|@sha256:[0-9a-f]\{64\}|@sha256:NOT-A-DIGEST|g' "$case_dir/Dockerfile" > "$case_dir/Dockerfile.new" \
        && mv "$case_dir/Dockerfile.new" "$case_dir/Dockerfile"
    run_in "$case_dir" python3 "$REPO_ROOT/tools/check-base-digests.py" --dockerfile Dockerfile
    rc=$(last_rc)
    if [ "$rc" -eq 2 ]; then
        ok "check-base-digests refuses a malformed base image digest"
    else
        bad "check-base-digests did NOT refuse the malformed digest (exit $rc)"
        printf -- '---- check-base-digests output ----\n%s\n-------------------------------\n' "$(last_out)"
    fi

    case_dir=$SCRATCH/pin-missing
    scratch_copy "$case_dir"
    sed 's|@sha256:[0-9a-f]\{64\}||g' "$case_dir/Dockerfile" > "$case_dir/Dockerfile.new" \
        && mv "$case_dir/Dockerfile.new" "$case_dir/Dockerfile"
    run_in "$case_dir" python3 "$REPO_ROOT/tools/check-base-digests.py" --dockerfile Dockerfile
    rc=$(last_rc)
    if [ "$rc" -eq 2 ]; then
        ok "check-base-digests refuses a Dockerfile whose base image is not pinned"
    else
        bad "check-base-digests did NOT refuse the unpinned base image (exit $rc)"
        printf -- '---- check-base-digests output ----\n%s\n-------------------------------\n' "$(last_out)"
    fi
fi

printf '\n%d case(s) passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
