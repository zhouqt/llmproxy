# syntax=docker/dockerfile:1
FROM docker.io/library/rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

COPY src/ src/
RUN cargo build --release && \
    cp target/release/llmproxy /llmproxy

FROM docker.io/library/debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /llmproxy /usr/local/bin/llmproxy

EXPOSE 8080
ENTRYPOINT ["llmproxy"]
CMD ["--config", "/etc/llmproxy/config.yaml"]
