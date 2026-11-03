# deploy/

Everything that belongs to a **running deployment**: the scripts that go
into the image, and the ones an operator runs against a deployed system.
Build- and check-time helpers live in [`../tools/`](../tools/).

| Path | What it does |
|---|---|
| `entrypoint.sh` | The container entrypoint: unseals (or copies) the secrets, then `exec`s the proxy. Copied into the image by the `Dockerfile`. |
| `seal.sh` | Host-side: seals a plaintext `.env` into the chunked TPM blob under `secrets/seal/`, then verifies the round trip. |
| `verify-image.sh` | Verifies a published image with `cosign` against its signature. |
| `secrets/llmproxy.env.example` | Template for the fallback plaintext secrets file. The real files under `secrets/` are gitignored. |

`docker-compose.yaml` bind-mounts `./deploy/secrets/seal` into the
container, so this directory's layout is part of the deployment's
contract: moving `secrets/` breaks an existing deployment's volumes.
