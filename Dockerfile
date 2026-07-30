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
RUN apk update && apk add ca-certificates \
    && rm -rf /var/cache/apk/*

COPY --from=builder /llmproxy /usr/local/bin/llmproxy

EXPOSE 8080
ENTRYPOINT ["llmproxy"]
CMD ["--config", "/etc/llmproxy/config.yaml"]
