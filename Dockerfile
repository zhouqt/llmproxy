# syntax=docker/dockerfile:1
FROM docker.io/library/rust:1-alpine AS builder

RUN apk update && apk add pkgconf openssl-dev \
    && rm -rf /var/cache/apk/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

COPY src/ src/
RUN cargo build --release && \
    cp target/release/llmproxy /llmproxy

FROM docker.io/library/alpine:3
# tpm2-tools provides tpm2_load/tpm2_unseal; tpm2-tss-tcti-device is the
# device TCTI the tools need to talk to /dev/tpmrm0. The two are
# separate packages on Alpine -- installing only the first fails at
# runtime with `Could not load tcti, got: "(null)"`.
RUN apk update && apk add --no-cache ca-certificates \
        tpm2-tools tpm2-tss-tcti-device \
    && rm -rf /var/cache/apk/*

COPY --from=builder /llmproxy /usr/local/bin/llmproxy
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

EXPOSE 8080
# The entrypoint unseals the TPM-sealed secrets into /run/secrets and
# then execs the command below -- see deploy/entrypoint.sh.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["llmproxy", "--config", "/etc/llmproxy/config.yaml"]
