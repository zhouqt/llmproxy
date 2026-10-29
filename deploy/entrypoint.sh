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
# Plaintext only ever lands in /run/secrets/, which the compose file
# mounts as a tmpfs. Without that mount the unsealed key would be
# written into the container's overlay *diff* layer -- i.e. onto the
# host's disk -- which is the one thing this whole mechanism exists to
# prevent. Both modes therefore refuse to run when /run is not tmpfs.
# The fallback mode additionally honours ALLOW_PLAINTEXT_FALLBACK=1 as
# an explicit "yes, I accept the plaintext on disk" opt-out.
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
# host (where /secrets and /run/secrets do not exist) without rebuilding
# the image. The defaults are the production layout.
KEYS=${LLMPROXY_KEYS:-/run/secrets/llmproxy.env}
FALLBACK=${LLMPROXY_FALLBACK:-/secrets/llmproxy.env}
SEAL=${LLMPROXY_SEAL:-/secrets/seal}

# The mount table the tmpfs check reads. Overridable only so the
# behaviour can be exercised on the host (where /run may or may not be
# a tmpfs); production always reads the real /proc/mounts.
#
# WARNING: production must **not** set this variable. It exists only so the
# tmpfs check can be exercised on a host (see
# docs/tpm-sealed-secrets-deploy.md). It is also a switch that turns the T12
# check off: point it at a file with no /run tmpfs and the entrypoint treats
# /run as a non-tmpfs (more conservative on the plaintext path); point it at
# a forged mount table instead and plaintext can land on the host disk
# undetected. So it is a test knob, not a configuration option.
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

mkdir -p "$(dirname "$KEYS")"
umask 077

run_is_tmpfs() {
    awk '$2 == "/run" && $3 == "tmpfs" { found = 1 } END { exit !found }' \
        "$MOUNTS"
}

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
    if ! run_is_tmpfs; then
        # Without a tmpfs the copy lands in the container's writable
        # layer, i.e. on the host disk -- the exact thing this
        # mechanism exists to prevent. The TPM branch below refuses to
        # start in that situation; the fallback branch used to only
        # warn, which made "plaintext on the host disk" a quiet
        # default. It now takes the same opt-out: set
        # ALLOW_PLAINTEXT_FALLBACK=1 to accept the plaintext anyway.
        if [ "${ALLOW_PLAINTEXT_FALLBACK:-0}" != "1" ]; then
            echo "ERROR: /run is not a tmpfs and the fallback path would write" >&2
            echo "       the plaintext env into the container's writable layer" >&2
            echo "       (and thus onto the host disk). Add to docker-compose.yaml:" >&2
            echo "         tmpfs:" >&2
            echo "           - /run:rw,noexec,nosuid,nodev,size=4m" >&2
            echo "       or set ALLOW_PLAINTEXT_FALLBACK=1 to accept the" >&2
            echo "       plaintext on disk anyway." >&2
            exit 1
        fi
        echo "WARN: /run is not a tmpfs; the copy lands in the container's" >&2
        echo "      writable layer as well (ALLOW_PLAINTEXT_FALLBACK=1)" >&2
    fi
    rm -f "$KEYS"
    cp "$FALLBACK" "$KEYS"
elif ls "$SEAL"/llmproxy.*.pub > /dev/null 2>&1; then
    if ! run_is_tmpfs; then
        echo "ERROR: /run is not a tmpfs; refusing to unseal onto the" >&2
        echo "       container's writable layer (the plaintext would end" >&2
        echo "       up on the host disk). Add to docker-compose.yaml:" >&2
        echo "         tmpfs:" >&2
        echo "           - /run:rw,noexec,nosuid,nodev,size=4m" >&2
        exit 1
    fi

    # The parent must be the PERSISTENT primary at 0x81000001, not a
    # freshly created one: tpm2_createprimary derives a different
    # sensitive seed every time, so a new primary cannot unseal a blob
    # sealed under an older one.
    #
    # The trap flushes the TPM's transient-object slots on every exit
    # path -- success, a failed load, or a failed unseal. swtpm has only
    # three slots, so a leak makes the *third* start fail with 0x902
    # while the first two look fine.
    trap 'tpm2_flushcontext -t 2>/dev/null || true; rm -f /tmp/seal.ctx' EXIT

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
    for pub in "$SEAL"/llmproxy.*.pub; do
        priv="${pub%.pub}.priv"
        if [ ! -f "$priv" ]; then
            echo "ERROR: $priv is missing (private half of $pub)" >&2
            exit 1
        fi
        tpm2_load -C 0x81000001 \
            -u "$pub" \
            -r "$priv" \
            -c /tmp/seal.ctx > /dev/null
        tpm2_unseal -c /tmp/seal.ctx -p "pcr:$PCR_LIST" >> "$KEYS"
        tpm2_flushcontext -t
        rm -f /tmp/seal.ctx
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