#!/bin/sh
# Seal a plaintext .env into chunked TPM-sealed blobs.
#
# Why chunks: a sealed object stores its payload in TPM2B_SENSITIVE_DATA,
# which Infineon parts cap at 128 bytes (measured on this host: 128 bytes
# seal fine, 129 fail with a TPM error). A real .env is ~1.7 kB, so it
# does not fit in one object. The file is therefore split into 128-byte
# pieces, each sealed as its own object under the same persistent parent.
# The entrypoint unseals the pieces in index order and concatenates them;
# the resulting byte stream is identical to the input file.
#
# The host keeps ciphertext only: llmproxy.NNNN.pub + llmproxy.NNNN.priv
# per chunk, plus the sha256 of the sealed plaintext
# (llmproxy.env.sha256) that the entrypoint verifies the unsealed
# payload against. The plaintext is piped, never copied -- it is never
# written to a temporary file, never put on a command line, and never
# enters the shell history.
#
# Usage:
#   deploy/seal.sh <plaintext.env> [seal-dir]
#
# Defaults: seal-dir = <directory of this script>/secrets/seal
#
# See plans/feat-vault-secret-backend.md (A-Phase 1) for the design.
set -eu

# The persistent parent. Every chunk is sealed under it, and the
# container unseals under it too -- tpm2_createprimary derives a
# different sensitive seed on every run, so a primary created at
# container start cannot unseal a blob sealed under an earlier one.
PARENT=0x81000001

# Payload size per sealed object. 128 is the maximum this TPM accepts;
# a smaller value would work as well but wastes objects.
CHUNK=128

# The PCRs the sealed objects are bound to: 0 = firmware code,
# 7 = Secure Boot policy. See docs/tpm-sealed-secrets-deploy.md.
PCR_LIST="sha256:0,7"

usage() {
    echo "usage: $0 <plaintext.env> [seal-dir]" >&2
    exit 2
}

[ $# -ge 1 ] && [ $# -le 2 ] || usage

SRC=$1
# The default seal directory is next to this script, not next to the
# caller: "deploy/secrets/seal" is correct only when the script is run
# from the repository root. Run from anywhere else (a cron job, a
# recovery shell) the relative default would write the ciphertext into
# an unrelated directory -- and the entrypoint would then find no seal
# where it looks for one.
SELF_DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${2:-$SELF_DIR/secrets/seal}

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC is not a file" >&2
    exit 1
fi
if [ ! -s "$SRC" ]; then
    echo "ERROR: $SRC is empty" >&2
    exit 1
fi
if ! grep -q '=' "$SRC"; then
    echo "ERROR: $SRC contains no NAME=VALUE lines" >&2
    exit 1
fi

# The scratch directory holds the transient context files and the PCR
# policy. mktemp -d gives it mode 0700.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT"
chmod 700 "$OUT"

# Stale chunks from an earlier seal would be concatenated with the new
# ones and corrupt the plaintext, so the target is emptied first. Only
# the llmproxy.* blobs are removed -- anything else in the directory is
# left alone. The bookkeeping files of an earlier seal go with them: a
# PCR record or a digest that outlives its chunks would describe a
# payload that no longer exists.
rm -f "$OUT"/llmproxy.*.pub "$OUT"/llmproxy.*.priv \
      "$OUT"/llmproxy.pcrs "$OUT"/llmproxy.host "$OUT"/llmproxy.env.sha256

if tpm2_getcap handles-persistent | grep -q "$PARENT"; then
    echo "persistent parent $PARENT already present; reusing"
else
    echo "creating persistent parent $PARENT"
    tpm2_createprimary -c "$TMP/primary.ctx"
    tpm2_evictcontrol -C o -c "$TMP/primary.ctx" "$PARENT"
    tpm2_flushcontext -t
    rm -f "$TMP/primary.ctx"
fi

# The policy binds each chunk to the PCRs. On later runs this same
# policy digest is reused, which is what lets an old blob be unsealed
# after a reboot of the host.
tpm2_createpolicy --policy-pcr -l "$PCR_LIST" -L "$TMP/pcr.policy"

# Record the PCR values the policy is bound to, next to the chunks.
#
# A PolicyPCR assertion names the values that were current when the
# policy was written, so the object's policy is the digest of exactly
# those values. They stay the same across reboots of the same
# machine, but a firmware update, a Secure Boot policy or dbx change,
# or a different machine all move them -- and then the TPM refuses
# the policy session with TPM_RC_POLICY_FAIL (0x99D), which says
# nothing about the cause. With the values recorded, the unseal side
# can compare them and report what actually happened.
#
# The output format of tpm2_pcrread differs between tpm2-tools
# releases, and the unseal runs with the image's tools while the seal
# runs with the host's, so the values are normalized into one
# "sha256:<index>=<hex>" line per PCR.
if command -v tpm2_pcrread > /dev/null 2>&1; then
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
    ' | sort > "$OUT/llmproxy.pcrs"
    chmod 600 "$OUT/llmproxy.pcrs"
fi

# A fingerprint of the machine that made the seal, so a seal directory
# that travels to another machine is recognisable as foreign: its
# chunks live under this machine's TPM and cannot be loaded there.
# The machine-id is hashed, not stored -- the file has to be
# comparable across hosts, but the host's identity is nobody else's
# business.
if [ -r /etc/machine-id ]; then
    sha256sum /etc/machine-id | cut -d' ' -f1 > "$OUT/llmproxy.host"
    chmod 600 "$OUT/llmproxy.host"
fi

SIZE=$(wc -c < "$SRC")
NCHUNKS=$(( (SIZE + CHUNK - 1) / CHUNK ))
echo "sealing $SIZE bytes in $NCHUNKS chunk(s) of up to $CHUNK bytes"

i=0
while [ "$i" -lt "$NCHUNKS" ]; do
    idx=$(printf '%04d' "$i")
    # dd reads one chunk straight into the pipe; the plaintext never
    # reaches the filesystem. tpm2_create's exit status is the pipeline's
    # status (last command), so a TPM failure aborts the script.
    dd if="$SRC" bs="$CHUNK" skip="$i" count=1 2>/dev/null \
        | tpm2_create -C "$PARENT" -g sha256 \
                -u "$OUT/llmproxy.$idx.pub" \
                -r "$OUT/llmproxy.$idx.priv" \
                -L "$TMP/pcr.policy" -i- > /dev/null
    # Transient slots are scarce (swtpm has three); a leak makes the
    # next seal or the next container start fail with 0x902.
    tpm2_flushcontext -t
    i=$((i + 1))
done

chmod 600 "$OUT"/llmproxy.*.pub "$OUT"/llmproxy.*.priv

# --- round-trip check -------------------------------------------------
# Unseal the chunks again and compare digests. The unsealed bytes go
# into a pipe, never into a file, so the check does not put the
# plaintext back on the disk.
unseal_stream() {
    for pub in "$OUT"/llmproxy.*.pub; do
        priv="${pub%.pub}.priv"
        if [ ! -f "$priv" ]; then
            echo "ERROR: missing private half $priv" >&2
            exit 1
        fi
        tpm2_load -C "$PARENT" -u "$pub" -r "$priv" -c "$TMP/seal.ctx" > /dev/null
        tpm2_unseal -c "$TMP/seal.ctx" -p "pcr:$PCR_LIST"
        tpm2_flushcontext -t
        rm -f "$TMP/seal.ctx"
    done
}

SRC_SUM=$(sha256sum < "$SRC" | cut -d' ' -f1)
OUT_SUM=$(unseal_stream | sha256sum | cut -d' ' -f1)

if [ "$SRC_SUM" != "$OUT_SUM" ]; then
    echo "ERROR: seal/unseal round trip does not match" >&2
    echo "       source sha256 $SRC_SUM" >&2
    echo "       unsealed sha256 $OUT_SUM" >&2
    exit 1
fi

# Record the digest of the sealed plaintext next to the chunks, so the
# entrypoint can verify what it unseals (deploy/entrypoint.sh). The
# unseal itself proves only that the TPM released the blob, not that
# the blob set is complete and unmodified.
#
# The file is written with the plain "<hash>  <name>" layout that
# sha256sum -c understands, but the entrypoint only reads the hash
# field, so the name is a label rather than a path.
printf '%s  llmproxy.env\n' "$SRC_SUM" > "$OUT/llmproxy.env.sha256"
chmod 600 "$OUT/llmproxy.env.sha256"

# The variable count is printed, the values are not: enough to tell
# "the seal has all ten keys" from "the seal has half of them".
VARS=$(unseal_stream | grep -c '^[A-Za-z_][A-Za-z0-9_]*=' || true)

echo "OK: $NCHUNKS chunk(s) sealed and verified (sha256 $SRC_SUM)"
echo "    $VARS NAME=VALUE line(s) in the sealed payload"
echo "    ciphertext in $OUT/llmproxy.*.{pub,priv}"
