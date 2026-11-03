#!/bin/sh
# Container entrypoint: get the proxy's secrets into the container
# without ever putting them in the container's environment.
#
# Two modes, decided by the fallback file:
#
#   TPM mode (normal)   /secrets/llmproxy.env is empty or absent ->
#                       unseal the chunks in /secrets/seal into tmpfs.
#   fallback mode       /secrets/llmproxy.env is non-empty -> copy it
#                       into tmpfs and skip the TPM entirely.
#
# Plaintext only ever lands in /run/llmproxy/, which the compose file
# mounts as a tmpfs. Without a tmpfs there the unsealed key would be
# written into the container's overlay *diff* layer -- i.e. onto the
# host's disk -- which is the one thing this whole mechanism exists to
# prevent. Both modes therefore refuse to run when the target directory
# is not on a tmpfs.
# The fallback mode additionally honours ALLOW_PLAINTEXT_FALLBACK=1 as
# an explicit "yes, I accept the plaintext on disk" opt-out.
#
# /run/llmproxy/ and not /run/secrets/: podman mounts its own
# per-container directories at /run/secrets and /run/.containerenv, and
# when the rootfs is read-only podman forces those mounts read-only
# (libpod/container_internal_common.go: `if c.IsReadOnly() { ...
# "ro" ... }`). Writing the unsealed payload to /run/secrets then fails
# with EROFS even though /run itself is a writable tmpfs. Nothing under
# /run/llmproxy/ is reserved by podman, so it stays writable.
#
# In TPM mode the unsealed payload is compared against the digest
# recorded by deploy/seal.sh (llmproxy.env.sha256). A seal directory
# without that file keeps the old behaviour: a warning, not a refusal.
#
# The proxy reads the file itself (`secrets.provider: file` in
# config.yaml); nothing is exported into the environment, because an
# exported key is readable from /proc/<pid>/environ and shows up in
# `podman inspect`.
#
# The paths below are overridable ONLY so the unseal logic can be
# exercised on the host. In particular `LLMPROXY_MOUNTS` (the mount
# table the tmpfs check reads) must NOT be set in production: it is a
# test knob, and it is also a switch for turning the tmpfs check off.
# See docs/tpm-sealed-secrets-deploy.md.
#
# See plans/feat-vault-secret-backend.md (A-Phase 3) for the design.

set -eu

# The paths are overridable so the unseal logic can be exercised on the
# host (where /secrets and /run/llmproxy do not exist) without rebuilding
# the image. The defaults are the production layout.
#
# /run/llmproxy/ is this project's own directory inside the container's
# tmpfs: the unsealed payload and the TPM policy session both live
# there. It is deliberately NOT /run/secrets -- see the header.
KEYS=${LLMPROXY_KEYS:-/run/llmproxy/llmproxy.env}
FALLBACK=${LLMPROXY_FALLBACK:-/secrets/llmproxy.env}
SEAL=${LLMPROXY_SEAL:-/secrets/seal}

# The loaded TPM policy session. It sits next to the payload so that
# everything this script writes lands on the same tmpfs: with a
# read-only rootfs there is nowhere else to put it.
SCRATCH=${LLMPROXY_SCRATCH:-$(dirname "$KEYS")/seal.ctx}

# The mount table the tmpfs checks read. Overridable only so the
# behaviour can be exercised on the host (where the target directory
# may or may not be on a tmpfs); production always reads the real
# /proc/mounts.
#
# WARNING: production must **not** set this variable. It exists only so the
# tmpfs check can be exercised on a host (see
# docs/tpm-sealed-secrets-deploy.md). It is also a switch that turns the T12
# check off: point it at a file with no tmpfs for the target directory and
# the entrypoint treats the target as a non-tmpfs (more conservative on the
# plaintext path); point it at a forged mount table instead and plaintext can
# land on the host disk undetected. So it is a test knob, not a
# configuration option.
MOUNTS=${LLMPROXY_MOUNTS:-/proc/mounts}

# The PCRs the sealed chunks are bound to: 0 = firmware code,
# 7 = Secure Boot policy. Must match deploy/seal.sh.
PCR_LIST="sha256:0,7"

# Glob expansion must be byte-ordered, not locale-ordered: the chunk
# index is a zero-padded number, and "llmproxy.0010" has to sort after
# "llmproxy.0009".
LC_ALL=C
export LC_ALL

# `exec "$@"` with no arguments is a silent no-op that exits 0, which
# turns a misconfigured compose file into an invisible restart loop.
if [ "$#" -eq 0 ]; then
    echo "ERROR: entrypoint was given no command to exec" >&2
    echo "       set \`command:\` on the service (or rely on the image CMD)" >&2
    exit 1
fi

# umask before the mkdir, so the directory holding the plaintext is
# created 0700 and not 0755.
umask 077

KEYS_DIR=$(dirname "$KEYS")

# The mount that contains a given path: the deepest mount whose path is
# a prefix of it. A path can sit under several mounts (a tmpfs at /run
# with a read-only one at /run/secrets underneath it), and the one that
# decides whether a write succeeds is the deepest.
target_mount() {
    awk -v p="$1" '
        {
            mp = $2
            if (mp == "/" || p == mp || index(p, mp "/") == 1) {
                if (length(mp) > len) {
                    len = length(mp); fstype = $3; opts = $4
                }
            }
        }
        END { print fstype "|" opts }
    ' "$MOUNTS"
}

# Is the path on a tmpfs? The plaintext payload must land on a tmpfs:
# a writable overlay (the container's own layer) or a disk filesystem
# would put the key material on the host disk.
target_is_tmpfs() {
    case "$(target_mount "$1")" in
        tmpfs\|*) return 0 ;;
        *) return 1 ;;
    esac
}

# Is the path on a writable mount? "Is /run a tmpfs" is not enough: a
# *deeper* mount can shadow it. podman mounts its own per-container
# directories at /run/secrets and /run/.containerenv, and when the
# rootfs is read-only podman forces those mounts read-only. What you
# get is a writable tmpfs at /run with a read-only /run/secrets
# underneath it -- the write fails with EROFS while a tmpfs check on
# /run alone happily passes.
target_is_writable() {
    info=$(target_mount "$1")
    opts=${info#*|}
    case ",$opts," in
        *,ro,*) return 1 ;;
        *) return 0 ;;
    esac
}

require_writable_target() {
    if target_is_writable "$1"; then
        return 0
    fi
    echo "ERROR: $1 is on a read-only mount; the unsealed payload" >&2
    echo "       cannot be written there (EROFS)." >&2
    echo "       /run being a tmpfs is not enough -- a deeper mount" >&2
    echo "       can shadow it. podman reserves /run/secrets and" >&2
    echo "       /run/.containerenv and mounts them read-only when the" >&2
    echo "       rootfs is read-only." >&2
    echo "       Use a directory of your own (e.g. /run/llmproxy) and" >&2
    echo "       point config.yaml's secrets.file.path at it." >&2
    exit 1
}

# The payload must land on a tmpfs. The check is about the TARGET
# directory, not about /run: the path is configurable, and a tmpfs
# somewhere other than /run is just as good.
require_tmpfs_target() {
    if target_is_tmpfs "$1"; then
        return 0
    fi
    echo "ERROR: $1 is not on a tmpfs; refusing to write the plaintext" >&2
    echo "       there -- it would land on the host disk, in the" >&2
    echo "       container's writable layer. Add to docker-compose.yaml:" >&2
    echo "         tmpfs:" >&2
    echo "           - /run:rw,noexec,nosuid,nodev,size=4m" >&2
    exit 1
}

# Check the mount BEFORE creating the directory: on a read-only mount
# the mkdir itself would fail with EROFS and hide the real cause.
require_writable_target "$KEYS_DIR"

mkdir -p "$KEYS_DIR"

if [ -f "$FALLBACK" ] && [ -s "$FALLBACK" ]; then
    # Fallback: an operator hand-placed a plaintext .env. Say so loudly
    # on every start -- in this mode the host disk does hold plaintext,
    # and that must never look like the normal state.
    #
    # `-f` matters: podman creates an empty *directory* for a bind mount
    # whose host path does not exist, and `-s` is true for a non-empty
    # directory too. Without `-f` that empty mountpoint would be taken
    # for a fallback file and the cp below would fail.
    echo "WARN: using fallback plaintext env file; TPM unseal skipped" >&2
    if ! target_is_tmpfs "$KEYS_DIR"; then
        # Without a tmpfs the copy lands in the container's writable
        # layer, i.e. on the host disk -- the exact thing this
        # mechanism exists to prevent. The TPM branch below refuses to
        # start in that situation; the fallback branch used to only
        # warn, which made "plaintext on the host disk" a quiet
        # default. It now takes the same opt-out: set
        # ALLOW_PLAINTEXT_FALLBACK=1 to accept the plaintext anyway.
        if [ "${ALLOW_PLAINTEXT_FALLBACK:-0}" != "1" ]; then
            echo "ERROR: $KEYS_DIR is not on a tmpfs and the fallback path" >&2
            echo "       would write the plaintext env into the container's" >&2
            echo "       writable layer (and thus onto the host disk). Add to" >&2
            echo "       docker-compose.yaml:" >&2
            echo "         tmpfs:" >&2
            echo "           - /run:rw,noexec,nosuid,nodev,size=4m" >&2
            echo "       or set ALLOW_PLAINTEXT_FALLBACK=1 to accept the" >&2
            echo "       plaintext on disk anyway." >&2
            exit 1
        fi
        echo "WARN: $KEYS_DIR is not on a tmpfs; the copy lands in the" >&2
        echo "      container's writable layer as well" >&2
        echo "      (ALLOW_PLAINTEXT_FALLBACK=1)" >&2
    fi
    rm -f "$KEYS"
    cp "$FALLBACK" "$KEYS"
elif ls "$SEAL"/llmproxy.*.pub > /dev/null 2>&1; then
    # The unsealed payload must land on a tmpfs -- that is the whole
    # point of the mechanism. There is no opt-out here: the fallback
    # file exists for exactly this case (an operator accepting the
    # plaintext on disk, with the knowledge that they do).
    require_tmpfs_target "$KEYS_DIR"

    # The parent must be the PERSISTENT primary at 0x81000001, not a
    # freshly created one: tpm2_createprimary derives a different
    # sensitive seed every time, so a new primary cannot unseal a blob
    # sealed under an older one.
    #
    # The trap flushes the TPM's transient-object slots on every exit
    # path -- success, a failed load, or a failed unseal. swtpm has only
    # three slots, so a leak makes the *third* start fail with 0x902
    # while the first two look fine.
    trap 'tpm2_flushcontext -t 2>/dev/null || true; rm -f "$SCRATCH"' EXIT

    # The .env does not fit in one sealed object: a TPM caps a sealed
    # payload at TPM2B_SENSITIVE_DATA, which is 128 bytes on Infineon
    # parts. The file is therefore sealed as a series of 128-byte
    # chunks, each with its own public/private pair, and the plaintext
    # is the concatenation of the unsealed chunks in index order. See
    # deploy/seal.sh.
    #
    # LC_ALL=C above makes the glob byte-ordered, and the index is
    # zero-padded, so the chunks come back in the order they were
    # sealed.
    # The previous run left the file mode 0400; a plain truncate would
    # fail for a non-root owner, so it is removed first. Inside the
    # container the entrypoint runs as root and could overwrite it
    # anyway -- but the same script is used on the host for checks, and
    # "start twice in a row must work" has to hold in both places.
    rm -f "$KEYS"
    # The scratch file defaults to the payload's own directory, but
    # LLMPROXY_SCRATCH may point anywhere -- make sure its directory
    # exists, so a bad override fails with a clear message instead of
    # a bare "No such file or directory" from tpm2_load.
    mkdir -p "$(dirname "$SCRATCH")"
    for pub in "$SEAL"/llmproxy.*.pub; do
        priv="${pub%.pub}.priv"
        if [ ! -f "$priv" ]; then
            echo "ERROR: $priv is missing (private half of $pub)" >&2
            exit 1
        fi
        if ! tpm2_load -C 0x81000001 \
            -u "$pub" \
            -r "$priv" \
            -c "$SCRATCH" > /dev/null; then
            echo "ERROR: tpm2_load failed for $pub" >&2
            echo "       (the private half must be readable, and the" >&2
            echo "        scratch file's directory -- $SCRATCH -- must be" >&2
            echo "        on a writable mount)" >&2
            exit 1
        fi
        tpm2_unseal -c "$SCRATCH" -p "pcr:$PCR_LIST" >> "$KEYS"
        tpm2_flushcontext -t
        rm -f "$SCRATCH"
    done

    trap - EXIT

    # The seal step records the sha256 of the plaintext it sealed
    # (deploy/seal.sh -> llmproxy.env.sha256). Comparing it with the
    # digest of what we just unsealed means a truncated, reordered or
    # tampered seal directory cannot start the proxy with half a key
    # set -- the unseal itself is not an integrity check.
    #
    # A seal directory written before this check existed has no digest
    # file: warn and continue (the old behaviour) rather than refusing
    # to start an otherwise perfectly good deployment.
    SUM_FILE="$SEAL/llmproxy.env.sha256"
    if [ -f "$SUM_FILE" ]; then
        if ! command -v sha256sum > /dev/null 2>&1; then
            echo "ERROR: sha256sum is not available; cannot verify the unsealed" >&2
            echo "       payload against $SUM_FILE" >&2
            rm -f "$KEYS"
            exit 1
        fi
        EXPECTED=$(cut -d' ' -f1 < "$SUM_FILE" | tr -d ' \t\r\n')
        ACTUAL=$(sha256sum < "$KEYS" | cut -d' ' -f1 | tr -d ' \t\r\n')
        if [ -z "$EXPECTED" ] || [ "$EXPECTED" != "$ACTUAL" ]; then
            echo "ERROR: the unsealed payload does not match the sealed digest" >&2
            echo "       expected sha256 $EXPECTED" >&2
            echo "       actual   sha256 $ACTUAL" >&2
            echo "       (the seal directory is incomplete or has been modified)" >&2
            rm -f "$KEYS"
            exit 1
        fi
        echo "OK: the unsealed payload matches the sealed digest"
    else
        echo "WARN: $SUM_FILE is missing; the unsealed payload was not" >&2
        echo "      verified against a digest (re-run deploy/seal.sh to add one)" >&2
    fi
else
    echo "ERROR: no sealed chunks in $SEAL and no fallback at $FALLBACK" >&2
    echo "       either run the seal step (deploy/seal.sh) or put a" >&2
    echo "       plaintext .env at deploy/secrets/llmproxy.env" >&2
    exit 1
fi

# The file is the contract between this script and the proxy, so check
# it rather than trusting the exit status alone: an empty or malformed
# file would start the proxy with no keys (and an empty server.api_key
# is now itself rejected, but failing here names the right cause).
if [ ! -s "$KEYS" ] || ! grep -q '=' "$KEYS"; then
    echo "ERROR: $KEYS is empty or contains no NAME=VALUE lines" >&2
    echo "       (did the unseal or the fallback copy produce anything?" >&2
    echo "       run the seal/unseal check in docs/tpm-sealed-secrets-deploy.md)" >&2
    rm -f "$KEYS"
    exit 1
fi

chmod 0400 "$KEYS"

exec "$@"