# syntax=docker/dockerfile:1
#
# 镜像构建分四段,核心目的是**让编译期执行的第三方代码没有网络、没有凭据**,
# 并且让镜像里的二进制**可以被外部扫描器读懂**。
#
# 为什么分段(实测结论见 plans/harden-cargo-deps.md §2):
#   - `cargo build --offline` **不是沙箱** —— build.rs 照样出网、照样读到全部环境变量。
#   - 容器 `--network=none` 是真的内核级隔离(TCP/DNS 全断),但**挡不住环境变量**。
#   → 所以:有网只允许"下载依赖 / 装工具"这一步,编译那一步必须 `RUN --network=none`,
#     并且编译环境里不能有任何凭据。
#
# 基础镜像说明:直接用 Docker Hub(docker.io)。GitHub Actions 能直连 docker.io,
# 生产的镜像也应当来自同一个源 —— 中间再套一层 ECR 镜像多一次转换、多一个信任点。
# 按 digest 钉死(可变 tag 是没被钉住的输入)。
#
# 构建机(rustbuilder)到 docker.io 的 TLS 被 reset,拉不动。它走**代理**:
#   HTTPS_PROXY=http://192.168.11.1:12345 podman build …
# 或者用 tools/build-image.sh(里面带了同样的设置)。走代理拉的是同一个 docker.io,
# 不是镜像站,所以构建输入与 CI 完全一致(实测:两个 registry 上同一 tag 的
# manifest 字节相同,digest 相同)。
#
# Dockerfile 里必须写**字面量**的 FROM:Dependabot 靠解析 FROM 行来更新 digest,
# 写成 ARG(${VAR})它认不出来,tag 漂移就没人管了。
#
#   rust:1-alpine  → sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87
#   alpine:3       → sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a

# ---- 阶段 0:工具(有网)----------------------------------------------------
# cargo-auditable / cargo-cyclonedx 本身也是 crates.io 产物,以全权限运行 ——
# 与被审计的依赖是同一类风险。缓解手段只有两条:版本钉死 + `--locked` 安装
# (用 crate 自带的 Cargo.lock 校验字节)。升级这两个版本是一次**显式的决定**。
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS tools
RUN cargo install --locked --version 0.7.7 cargo-auditable \
 && cargo install --locked --version 0.5.9 cargo-cyclonedx

# ---- 阶段 1:取依赖(有网)----------------------------------------------------
# `cargo fetch` 只下载并按 Cargo.lock 的 sha256 校验,**不编译** → 这一段里
# 一个第三方的 build.rs 都不会被执行。
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS fetch
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
# cargo refuses to parse a manifest with no targets, so the fetch stage
# needs a placeholder source file. It is thrown away with the stage.
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs
RUN cargo fetch --locked

# ---- 阶段 2:编译(编译期无网)----------------------------------------------
FROM docker.io/library/rust:1-alpine@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS builder
RUN apk add --no-cache pkgconf openssl-dev

WORKDIR /app
# 从 fetch 阶段搬已下载的 registry(含 index 缓存 —— cargo 校验 checksum 时要用)。
COPY --from=fetch /usr/local/cargo /usr/local/cargo
# 只搬两个工具的二进制,不搬它们的依赖 —— 工具本身不参与编译产物。
COPY --from=tools /usr/local/cargo/bin/cargo-auditable /usr/local/cargo/bin/
COPY --from=tools /usr/local/cargo/bin/cargo-cyclonedx /usr/local/cargo/bin/
COPY Cargo.toml Cargo.lock ./

# 可复现性:把编译期写进二进制的路径去掉。/app 是本 Dockerfile 固定的 WORKDIR,
# /usr/local/cargo 是 registry 源码所在目录 —— 两个都是"在哪台机器上编译"的信息,
# 去掉之后同一个 lock 在任何路径下编出来的二进制哈希一致。
ENV RUSTFLAGS="--remap-path-prefix=/app=. --remap-path-prefix=/usr/local/cargo=deps"

# 依赖编译层:先用一个空的 main 把依赖全部编译出来,buildkit 会缓存这一层。
# `--offline` 保证不会再从网络取任何东西;`--network=none` 保证这一步没有出网能力。
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs
RUN --network=none cargo build --release --offline --locked

# 沙箱自检:确认这一层确实是隔离的。破口 → 构建立刻失败。
COPY tools/sandbox-canary/ tools/sandbox-canary/
RUN --network=none cargo build --release --offline --locked --manifest-path tools/sandbox-canary/Cargo.toml

# 真实源码:只有 llmproxy 自身需要重编,依赖层直接复用上面的缓存。
COPY src/ src/
# 用 cargo-auditable 而不是 cargo build:它把依赖清单嵌进二进制,
# 镜像扫描器(Trivy/Grype)才能报出 ring / rustls 这些 crate 的版本与 CVE ——
# 否则一个 strip 过的二进制对扫描器是全黑的。
RUN --network=none cargo auditable build --release --offline --locked \
    && cp target/release/llmproxy /llmproxy

# SBOM:CycloneDX,随镜像一起分发(不进仓库 —— 进仓必然随升级变旧,变成噪声源)。
#
# cargo-cyclonedx 没有 --output-file;`--override-filename` 不允许带路径分隔符,
# 文件写在当前目录(/app),格式扩展名由工具自己补上 → 产物是 sbom.json,
# 下面改名成 sbom.cdx.json,和运行时镜像里的路径保持一致。
RUN --network=none cargo cyclonedx --format json --override-filename sbom \
    && mv sbom.json /app/sbom.cdx.json

# ---- 阶段 3:运行时 ---------------------------------------------------------
FROM docker.io/library/alpine:3@sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a
# tpm2-tools provides tpm2_load/tpm2_unseal; tpm2-tss-tcti-device is the
# device TCTI the tools need to talk to /dev/tpmrm0. The two are
# separate packages on Alpine -- installing only the first fails at
# runtime with `Could not load tcti, got: "(null)"`.
RUN apk add --no-cache ca-certificates \
        tpm2-tools tpm2-tss-tcti-device

COPY --from=builder /llmproxy /usr/local/bin/llmproxy
COPY --from=builder /app/sbom.cdx.json /usr/local/share/llmproxy/sbom.cdx.json
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

EXPOSE 8080
# The entrypoint unseals the TPM-sealed secrets into /run/secrets and
# then execs the command below -- see deploy/entrypoint.sh.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["llmproxy", "--config", "/etc/llmproxy/config.yaml"]

# ---------------------------------------------------------------------------
# 刷新基础镜像 digest 的方法:
#   tools/check-base-digests.py            # 核对当前 digest 是否仍与 tag 一致
#   tools/check-base-digests.py --registry public.ecr.aws   # 构建机上用(ECR 可达)
# digest 漂移时,Dependabot(docker ecosystem)会开 PR 改这一行。
# ---------------------------------------------------------------------------
