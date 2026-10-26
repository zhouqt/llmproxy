# Deployment secrets

The container ships with an entrypoint that materialises the proxy's
secrets **inside the container only**. The host disk holds a TPM-sealed
blob (ciphertext); the entrypoint unseals it into a tmpfs at
`/run/secrets/llmproxy.env`; llmproxy reads that file through its
`secrets:` config block. No key ever reaches the container environment,
`podman inspect`, or the host filesystem.

The cost of that property is a one-time seal step, and a re-seal
whenever the firmware or Secure Boot policy changes. If you would
rather not pay it, see [Fallback mode](#fallback-mode).

```
host                                    container (tmpfs)
deploy/secrets/seal/llmproxy.NNNN.* ┐
  (one .pub/.priv pair per chunk)   ┴─► tpm2_load + tpm2_unseal ─► /run/secrets/llmproxy.env
                                        (parent 0x81000001)              │
                                                                         ▼
                                                    llmproxy --config /etc/llmproxy/config.yaml
```

## Layout

| Path | Role |
|---|---|
| `deploy/entrypoint.sh` | Unseals (or copies) the secrets, then `exec`s the proxy |
| `deploy/seal.sh` | Host-side: seals a plaintext `.env` into the chunked blob, then verifies the round trip |
| `deploy/secrets/seal/llmproxy.NNNN.{pub,priv}` | The sealed chunks — **ciphertext only**, gitignored |
| `deploy/secrets/llmproxy.env` | Fallback plaintext — **empty in normal operation**, gitignored |
| `deploy/secrets/llmproxy.env.example` | Template for the fallback file (tracked) |

### Why the blob is chunked

A sealed object stores its payload in `TPM2B_SENSITIVE_DATA`, which
Infineon parts cap at **128 bytes** (measured on this host: a 128-byte
payload seals, 129 bytes fails with a TPM error). A real `.env` is
~1.7 kB, so it does not fit in one object.

`deploy/seal.sh` therefore splits the file into 128-byte pieces and
seals each one as its own object under the same persistent parent. The
entrypoint unseals the pieces in index order and concatenates them; the
resulting byte stream is identical to the input file. The host still
holds ciphertext only — chunking changes the file count, not the
security property.

## Host prerequisites (once)

Needs root for package installation and group membership.

```bash
# Debian / Ubuntu. Note: there is no `tpm2-tss` metapackage on trixie;
# the udev rule comes from `tpm-udev`.
apt-get install -y tpm2-tools libtss2-tcti-device0 tpm-udev
# Alpine host: apk add tpm2-tools tpm2-tss-tcti-device

ls -l /dev/tpmrm0                  # crw-rw---- tss tss
getent group tss
usermod -aG tss "$USER"            # log out and back in afterwards
tpm2_getrandom 16 --hex            # as the normal user, not with sudo
podman info --format '{{.Host.OCIRuntime.Name}}'   # keep-groups needs crun
```

The container reaches the TPM through the compose `devices:` entry for
`/dev/tpmrm0` plus `group_add: keep-groups` (crun carries the host's
supplementary groups — including `tss` — into the container; a numeric
gid would be re-mapped to a subgid instead).

## One-time: seal

Keep a copy of the plaintext `.env` in your password manager first.
**That copy is the root backup** — if the vTPM state is ever lost, it is
the only way back.

```bash
cd deploy
./seal.sh /path/to/plaintext.env deploy/secrets/seal
```

The script creates the persistent parent if it is missing, seals the
file chunk by chunk, and then unseals it again to compare digests — it
fails loudly rather than leaving a half-sealed directory behind. The
plaintext is piped, never copied: it is never written to a temporary
file, never put on a command line, and never enters the shell history.

<details>
<summary>The manual equivalent (what the script runs)</summary>

```bash
mkdir -p deploy/secrets/seal && chmod 700 deploy/secrets deploy/secrets/seal
cd deploy/secrets/seal

# Create the persistent parent once. On later runs the handle already
# exists and `tpm2_evictcontrol` fails with 0x14C ("persistent object
# already defined"), so check first instead of re-creating it.
if tpm2_getcap handles-persistent | grep -q 0x81000001; then
    echo "persistent parent 0x81000001 already present; reusing"
else
    tpm2_createprimary -c primary.ctx
    tpm2_evictcontrol -C o -c primary.ctx 0x81000001
    tpm2_flushcontext -t
    rm -f primary.ctx
fi

tpm2_createpolicy --policy-pcr -l sha256:0,7 -L pcr.policy

# Seal the file in 128-byte chunks (the TPM's TPM2B_SENSITIVE_DATA cap).
# `dd` feeds each chunk straight into `tpm2_create`, so the plaintext
# never reaches the filesystem. The index is zero-padded because the
# entrypoint concatenates the chunks in byte order.
size=$(wc -c < "$PLAINTEXT_ENV")
nchunks=$(( (size + 127) / 128 ))
i=0
while [ "$i" -lt "$nchunks" ]; do
    idx=$(printf '%04d' "$i")
    dd if="$PLAINTEXT_ENV" bs=128 skip="$i" count=1 2>/dev/null \
      | tpm2_create -C 0x81000001 -g sha256 \
                    -u "llmproxy.$idx.pub" -r "llmproxy.$idx.priv" \
                    -L pcr.policy -i- > /dev/null
    tpm2_flushcontext -t
    i=$((i + 1))
done

tpm2_flushcontext -t
rm -f pcr.policy
chmod 600 llmproxy.*.pub llmproxy.*.priv
```

</details>

Two details that are easy to get wrong:

- **The parent must be the persistent handle.** `tpm2_createprimary`
  derives a different sensitive seed on every run: its public key is
  stable but the object is not, so a primary created at container start
  cannot unseal a blob sealed under an earlier one.
- **Flush transient slots.** `tpm2_flushcontext -t` after every command.
  The TPM has very few transient slots (swtpm has three); a leak makes
  the third container start fail with `0x902` while the first two look
  fine. The entrypoint flushes on every exit path for the same reason.

Verify the round trip before starting the container:

```bash
# Unseal the chunks again and compare digests. The unsealed bytes go
# into a pipe, never into a file.
unseal_stream() {
    for pub in llmproxy.*.pub; do
        tpm2_load -C 0x81000001 -u "$pub" -r "${pub%.pub}.priv" -c /tmp/seal.ctx > /dev/null
        tpm2_unseal -c /tmp/seal.ctx -p pcr:sha256:0,7
        tpm2_flushcontext -t
        rm -f /tmp/seal.ctx
    done
}
[ "$(unseal_stream | sha256sum)" = "$(sha256sum < "$PLAINTEXT_ENV")" ] \
    && echo "OK: seal/unseal round trip matches" || echo "MISMATCH"
```

The same check is what `deploy/seal.sh` runs automatically after sealing.

### Which PCRs

The policy is bound to PCR 0 and 7:

| PCR | Meaning | Changes on |
|---|---|---|
| 0 | Firmware code | BIOS/UEFI update |
| 7 | Secure Boot policy | Secure Boot mode / key change |

Deliberately **not** bound to PCR 2/4 (kernel, initramfs), so ordinary
kernel upgrades keep working. The trade-off: an unseal fails after a
firmware upgrade or a Secure Boot policy change, and those are exactly
the events a re-seal should follow. On a machine with Secure Boot off,
PCR 7 is constant zero and the policy reduces to "PCR 0 unchanged".

## Configure llmproxy

`config.yaml` uses `${VAR}` references as before; point the `secrets:`
block at the tmpfs file:

```yaml
secrets:
  provider: file
  file:
    path: /run/secrets/llmproxy.env
    required: true      # default: refuse to start without it
```

`required: true` is what makes a broken unseal a startup failure rather
than a proxy that starts with empty keys. See `config.example.yaml` for
the parsing rules (same `.env` dialect as the fallback file).

`server.api_key: "${LLMPROXY_API_KEY}"` must resolve to a non-empty
value: the proxy rejects an empty key at startup rather than treating it
as "auth disabled" (a misspelled variable must not silently open the
endpoint).

## Container hardening

The compose service runs with a read-only root filesystem, every Linux
capability dropped, and `no-new-privileges` set (security audit L5):

```yaml
read_only: true
cap_drop: [ALL]
security_opt: [no-new-privileges]
tmpfs:
  - /run:rw,noexec,nosuid,nodev,size=4m
  - /tmp:rw,noexec,nosuid,nodev,size=4m
```

The writable surfaces are exactly the ones the design already needs:
`/run` (the unsealed plaintext, tmpfs — never on disk), `/tmp` (the
entrypoint's scratch file `/tmp/seal.ctx`, also tmpfs — with a read-only
rootfs the unseal would have nowhere to load the policy session
otherwise), and the `llmproxy-data` volume (Copilot token store). An
exploited proxy process therefore cannot modify the image, persist a
payload, or escalate; TPM access comes from `keep-groups` (the `tss`
group on `/dev/tpmrm0`), not from any capability, so `cap_drop ALL`
does not affect the unseal path.

## Normal operation

```bash
podman-compose up -d
podman-compose logs -f          # expect no key material, ever
```

The unsealed file is `0400`, owned by root in the container's user
namespace.

### Rotating a key

```bash
# 1. Issue the new key and update the master copy in your password manager.
# 2. Re-seal. The persistent parent is reused as-is; only the policy and
#    the chunks are written again. seal.sh clears the old chunks first,
#    so a shorter file cannot leave stale ones behind.
./deploy/seal.sh /path/to/plaintext.env deploy/secrets/seal
# 3. Recreate the container so the entrypoint unseals again.
podman-compose up -d --force-recreate llmproxy
```

If you changed the source rather than the secrets, add `--build` (with a
local `build:` in the compose file), otherwise the existing image is
reused.

Rotating a leaked LLM key takes about a minute and is almost always a
better first response than investigating how it leaked.

## Encrypting the token store

The sealed `.env` protects the *provider* keys, but the Copilot OAuth
token does not live there: it is written to
`github_token.json` inside the `llmproxy-data` named volume, which sits
on the **unencrypted** host disk. So the volume still holds a live
GitHub access token and a live Copilot API token in the clear.

Close that half of the story with `token_store_key`:

```bash
# 1. Generate a key and put it in the plaintext .env next to the others.
openssl rand -hex 32
# -> 64 hex characters, e.g. 9f2c...  (32 bytes)

# 2. Add it to the source .env, then re-seal:
#      LLMPROXY_TOKEN_STORE_KEY=<64 hex chars>
./deploy/seal.sh /path/to/plaintext.env deploy/secrets/seal

# 3. Point the provider at it in config.yaml:
#      providers:
#        - name: copilot
#          type: github_copilot
#          token_store_key: "${LLMPROXY_TOKEN_STORE_KEY}"

# 4. Recreate the container so it re-unseals and re-encrypts.
podman-compose up -d --force-recreate llmproxy
```

From then on the volume holds only ciphertext: the store is sealed with
AES-256-GCM under that key, with a fresh nonce per write. A store
written *before* the key was configured is still read as-is (plaintext
is accepted on load) and is rewritten encrypted on the next successful
authentication.

**Back up the key in the same password manager as the `.env` backup.**
Losing it means losing the stored tokens — which costs one device-flow
re-authentication (`POST /admin/copilot/auth`), nothing more. The
proxy never fails to start over a missing key: an encrypted store with
no configured key is treated as empty, with a warning in the log.

## Fallback mode

Fill `deploy/secrets/llmproxy.env` and recreate the container; the
entrypoint detects a non-empty file, logs a WARN on every start, and
skips the TPM entirely. Empty the file again to return to TPM mode.

```bash
cp deploy/secrets/llmproxy.env.example deploy/secrets/llmproxy.env
chmod 600 deploy/secrets/llmproxy.env
$EDITOR deploy/secrets/llmproxy.env
podman-compose up -d --force-recreate llmproxy
```

**In this mode the plaintext is on the host disk**, so the property this
whole mechanism exists to provide does not hold. Use it to keep the
service up while you diagnose a TPM problem, not as a permanent
configuration.

## Verification

Run through this after the first deploy and after any seal change.

| Check | Expected |
|---|---|
| `podman inspect <c> \| grep -i 'sk-'` | empty |
| `podman inspect <c>` → `Config.Env` | `RUST_LOG=...` only |
| `podman inspect <c>` → `Mounts` | seal dir + empty placeholder, no `.env` |
| `podman exec <c> ls -l /run/secrets/` | `llmproxy.env`, `0400` |
| `podman exec <c> env \| grep -i key` | empty (nothing came via the environment) |
| `podman logs <c>` | no key material; at most `from=file` source markers |
| `podman exec <c> mount \| grep ' /run '` | **`tmpfs on /run`** — see below |
| `podman diff <c>` | no `/run/secrets/llmproxy.env` |
| `grep -rl 'sk-' ~/.local/share/containers/storage/overlay/` | empty |
| `podman-compose restart llmproxy` (twice in a row) | both starts succeed |
| `tpm2_getcap handles-persistent` (host) | `0x81000001` present |
| `ls deploy/secrets/seal/llmproxy.*.pub \| wc -l` (host) | the chunk count (a 1.7 kB `.env` → 14) |

The `tmpfs` row is not optional. **rootless podman does not give `/run` a
tmpfs of its own**; without the compose `tmpfs:` entry the unsealed file
is a regular file in the container's overlay *diff* layer, i.e. on the
host disk at `~/.local/share/containers/storage/overlay/<id>/diff/run/`.
The entrypoint refuses to unseal when `/run` is not a tmpfs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not load tcti, got: "(null)"` | Device TCTI package missing | Install the TCTI package (`tpm2-tss-tcti-device` on Alpine, `libtss2-tcti-device0` on Debian) |
| `tpm2_load` fails, handle invalid | Persistent `0x81000001` gone (vTPM state lost) | Re-seal, then recreate the container |
| `tpm2_unseal` fails, policy check | PCR 0/7 changed (firmware or Secure Boot update) | Confirm with `tpm2_pcrread sha256:0,7`, re-seal |
| Third consecutive start fails, `0x902` | Transient slots leaked | The entrypoint flushes on exit; if it recurs, check for other `tpm2_*` users on the host |
| `ERROR: /run is not a tmpfs` | `tmpfs:` missing from the compose service | Restore it; do not "fix" this by disabling the check |
| Container restart-loops every second, log shows only `name: ...` | The compose file sets `entrypoint:` (podman-compose 1.3 then drops the image CMD, leaving the entrypoint nothing to exec) | Remove the `entrypoint:` override; keep `command:` |
| `ERROR: ... is empty or contains no NAME=VALUE lines` | Unseal or copy produced nothing | Check the seal with the round-trip command above, then re-seal |
| Container restarts with empty API keys | `secrets.file.required: false` | Set it back to `true` |
| `Permission denied` on `/dev/tpmrm0` | `keep-groups` unsupported (runc) or user not in `tss` | Use crun; re-login after `usermod -aG tss` |
| `ERROR: no sealed blob ... and no fallback` | Neither mode is provisioned | Run the seal step, or fill the fallback file |
| `File "..." size is larger than buffer, got 1704 expected less than or equal to 256` | The whole `.env` was handed to one `tpm2_create` | Seal it in chunks — `deploy/seal.sh` does that; a single object holds at most 128 bytes |
| The TPM rejects a payload of 129 bytes | `TPM2B_SENSITIVE_DATA` is 128 bytes on Infineon parts | Same as above: chunk the file |
| `ERROR: <file>.priv is missing (private half of <file>.pub)` | A chunk pair was deleted | Re-seal; the pairs are useless on their own |
| The unsealed file is one key too long / truncated | A chunk is missing or the index order is broken | Re-seal; `seal.sh` verifies the round trip by digest |

A plain restart does **not** lose the seal: the persistent handle lives
in the vTPM's NV state, which survives VM reboots. Rebuilding the vTPM
(or its swtpm state directory on the PVE host) does lose it — that is
the vTPM trust boundary, not a bug. Restore from the password-manager
copy and re-seal.