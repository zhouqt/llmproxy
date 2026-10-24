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
# prevent. TPM mode therefore refuses to run when /run is not tmpfs.
#
# The proxy reads the file itself (`secrets.provider: file` in
# config.yaml); nothing is exported into the environment, because an
# exported key is readable from /proc/<pid>/environ and shows up in
# `podman inspect`.
#
# See plans/feat-vault-secret-backend.md (A-Phase 3) for the design.

set -eu

# The paths are overridable so the unseal logic can be exercised on the
# host (where /secrets and /run/secrets do not exist) without rebuilding
# the image. The defaults are the production layout.
KEYS=${LLMPROXY_KEYS:-/run/secrets/llmproxy.env}
FALLBACK=${LLMPROXY_FALLBACK:-/secrets/llmproxy.env}
SEAL=${LLMPROXY_SEAL:-/secrets/seal}

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
        /proc/mounts
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
        echo "WARN: /run is not a tmpfs; the copy lands in the container's" >&2
        echo "      writable layer as well" >&2
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