FROM python:3-alpine
WORKDIR /app
RUN apk add --no-cache ca-certificates && \
    python3 -m pip install --no-cache-dir aiohttp==3.14.3
COPY mock_server.py .
EXPOSE 9000
# Default env matches debug-env/config.dev.yaml's MOCK_BASE fallback
# (compose sets MOCK_BASE=http://mock:9000, so 0.0.0.0:9000 inside container is reachable).
ENV MOCK_PORT=9000 MOCK_LATENCY_MS=80 MOCK_TTFT_MS=80 MOCK_INTER_TOKEN_MS=5 MOCK_N_TOKENS=64
HEALTHCHECK --interval=5s --timeout=2s --start-period=2s --retries=3 \
    CMD wget -qO- http://127.0.0.1:9000/health || exit 1
CMD ["python3", "-u", "mock_server.py"]