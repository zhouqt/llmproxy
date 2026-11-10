#!/bin/sh
# Host-side preflight for a sealed secrets directory.
#
# Why this exists
# ---------------
# A seal is bound to two things at once: the TPM that holds it (the
# persistent primary 0x81000001) and the exact PCR 0/7 values that
# were current when it was made. When either no longer matches, the
# unseal fails -- and the TPM's own error for that case,
# TPM_RC_POLICY_FAIL (0x99D), says nothing about the cause. This
# script turns it into an answer before the container is even
# started: which binding broke, and what to do about it.
#
# It is read-only with respect to the secrets: the unsealed bytes go
# to /dev/null and never reach a file, a pipe or the terminal.
#
# Usage:
#     deploy/seal-check.sh [seal-dir]
#
# seal-dir defaults to <directory of this script>/secrets/seal
#
# Exit codes:
#     0  the seal is intact and unseals on this machine
#     1  the seal cannot be used here (see the report)
#     2  the seal directory or the tooling is missing
set -eu

PARENT=0x81000001
PCR_LIST="sha256:0,7"

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
SEAL=${1:-$SELF_DIR/secrets/seal}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# The current values of the PCRs the policy is bound to, normalized to
# one "sha256:<index>=<hex>" line per PCR. tpm2_pcrread prints a
# different layout in different tpm2-tools releases, so the values are
# normalized before they are compared with the recorded ones.
current_pcrs() {
    command -v tpm2_pcrread > /dev/null 2>&1 || return 0
    tpm2_pcrread "$PCR_LIST" 2>/dev/null | awk '
        {
            if (match($0, /0x[0-9A-Fa-f]+/)) {
                val = substr($0, RSTART, RLENGTH)
                pre = substr($0, 1, RSTART - 1)
                if (match(pre, /[0-9]+[ \t]*[:=]?[ \t]*$/)) {
                    idx = substr(pre, RSTART, RLENGTH)
                    gsub(/[^0-9]/, "", idx)
                    printf "sha256:%s=%s\n", idx, tolower(val)
                }
            }
        }
    ' | sort
}

fail() {
    echo "FAIL: $1" >&2
    exit "${2:-1}"
}

echo "==> seal directory: $SEAL"

[ -d "$SEAL" ] || fail "$SEAL does not exist" 2
CHUNKS=$(ls "$SEAL"/llmproxy.*.pub 2>/dev/null | wc -l)
[ "$CHUNKS" -gt 0 ] || fail "no sealed chunks in $SEAL" 2
echo "    chunks: $CHUNKS"

for tool in tpm2_load tpm2_unseal; do
    command -v "$tool" > /dev/null 2>&1 \
        || fail "$tool is not installed (tpm2-tools)" 2
done

# --- the TPM side ------------------------------------------------------
echo
echo "==> TPM"
if command -v tpm2_getcap > /dev/null 2>&1; then
    if tpm2_getcap handles-persistent 2>/dev/null | grep -q "$PARENT"; then
        echo "    persistent parent $PARENT: present"
        PARENT_OK=1
    else
        echo "    persistent parent $PARENT: MISSING"
        PARENT_OK=0
    fi
else
    echo "    (tpm2_getcap is not installed; skipping the parent check)"
    PARENT_OK=0
fi

# --- the boot state ----------------------------------------------------
echo
echo "==> PCR binding"
CURRENT=$(current_pcrs || true)
if [ -f "$SEAL/llmproxy.pcrs" ]; then
    SEALED=$(sort "$SEAL/llmproxy.pcrs")
    echo "    sealed:  $(echo "$SEALED" | tr '\n' ' ')"
    echo "    current: $(echo "$CURRENT" | tr '\n' ' ')"
    if [ -z "$CURRENT" ]; then
        echo "    UNKNOWN -- the current PCR values could not be read"
        echo "    (no tpm2_pcrread, or the TPM refused the read); the boot"
        echo "    state could not be compared"
        PCR_OK=1
    elif [ "$SEALED" != "$CURRENT" ]; then
        echo "    MISMATCH -- the boot state changed since the seal"
        PCR_OK=0
    else
        echo "    match"
        PCR_OK=1
    fi
else
    echo "    (this seal predates the PCR record; nothing to compare)"
    PCR_OK=1
fi

# --- the machine -------------------------------------------------------
echo
echo "==> host"
if [ -f "$SEAL/llmproxy.host" ] && [ -r /etc/machine-id ] \
    && command -v sha256sum > /dev/null 2>&1; then
    SEALED_HOST=$(tr -d ' \t\r\n' < "$SEAL/llmproxy.host")
    CURRENT_HOST=$(sha256sum /etc/machine-id | cut -d' ' -f1)
    if [ "$SEALED_HOST" = "$CURRENT_HOST" ]; then
        echo "    sealed on this machine"
        HOST_OK=1
    else
        echo "    sealed on a DIFFERENT machine -- its chunks live under"
        echo "    that machine's TPM and cannot be loaded here"
        HOST_OK=0
    fi
else
    echo "    (no host fingerprint to compare)"
    HOST_OK=1
fi

# --- the actual unseal -------------------------------------------------
echo
echo "==> unseal test"
if [ "$PARENT_OK" -ne 1 ]; then
    echo "    skipped (no persistent parent $PARENT on this TPM)"
    fail "the TPM does not hold the parent this seal was made under" 1
fi

PUB=$(ls "$SEAL"/llmproxy.*.pub | head -1)
PRIV="${PUB%.pub}.priv"
[ -f "$PRIV" ] || fail "the private half of $PUB is missing" 2

UNSEAL_OK=0
if tpm2_load -C "$PARENT" -u "$PUB" -r "$PRIV" -c "$TMP/seal.ctx" > /dev/null 2>&1; then
    if tpm2_unseal -c "$TMP/seal.ctx" -p "pcr:$PCR_LIST" > /dev/null 2>&1; then
        UNSEAL_OK=1
    fi
fi
tpm2_flushcontext -t 2>/dev/null || true

if [ "$UNSEAL_OK" -eq 1 ]; then
    echo "    OK -- the chunk unseals on this machine"
    # The unseal is the authoritative signal: the TPM either released
    # the blob or it did not. A mismatch in the recorded metadata while
    # the unseal works means the *record* is stale (it was written by
    # an older seal.sh, or the host fingerprint was taken on a
    # different machine-id), not that the seal is broken -- say so
    # instead of failing, and offer the refresh.
    if [ "$PCR_OK" -ne 1 ]; then
        echo "    note: the recorded PCR values differ from the current"
        echo "          ones, but the unseal worked -- the record is"
        echo "          stale. Re-sealing refreshes it."
    fi
    if [ "$HOST_OK" -ne 1 ]; then
        echo "    note: the recorded host fingerprint differs, but the"
        echo "          unseal worked -- the seal was made under this"
        echo "          TPM. Re-sealing refreshes the fingerprint."
    fi
else
    echo "    FAILED -- the TPM refused the chunk"
    if [ "$PCR_OK" -ne 1 ]; then
        echo "    cause: the PCR values the seal was made under no longer"
        echo "           match this machine's (firmware update, Secure Boot"
        echo "           policy or dbx change, or a seal from another machine)"
    elif [ "$HOST_OK" -ne 1 ]; then
        echo "    cause: the seal was made on another machine"
    else
        echo "    cause: unknown -- the chunk does not match this TPM's"
        echo "           persistent parent, or the TPM state was cleared"
    fi
    echo "    remedy: re-seal on this host and recreate the container:"
    echo "      deploy/seal.sh /path/to/plaintext.env deploy/secrets/seal"
    fail "the seal cannot be used on this machine" 1
fi

echo
echo "OK: the seal is intact and unseals here."
echo "    (the container will do the same at start-up)"
