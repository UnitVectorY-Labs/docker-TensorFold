# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.5.0-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG TENSORFOLD_GPU=spark
ARG TENSORFOLD_REF=

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    curl \
    jq \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN set -eux; \
    if [ -n "${TENSORFOLD_REF}" ]; then \
        echo "Building TensorFold ref: ${TENSORFOLD_REF}"; \
        git clone --depth 1 --branch "${TENSORFOLD_REF}" https://github.com/ashhart/TensorFold.git TensorFold; \
    else \
        TENSORFOLD_TAG="$(curl -fsSL https://api.github.com/repos/ashhart/TensorFold/releases/latest | jq -er '.tag_name')"; \
        test -n "${TENSORFOLD_TAG}"; \
        echo "Building TensorFold release tag: ${TENSORFOLD_TAG}"; \
        git clone --depth 1 --branch "${TENSORFOLD_TAG}" https://github.com/ashhart/TensorFold.git TensorFold; \
    fi

RUN set -eux; \
    ZIG_VERSION="$(tr -d '\r\n' < /src/TensorFold/.zig-version)"; \
    ARCH="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${ARCH}" in \
        amd64) ZIG_ARCH="x86_64" ;; \
        arm64) ZIG_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;; \
    esac; \
    INDEX_JSON="$(curl -fsSL https://ziglang.org/download/index.json)"; \
    ZIG_URL="$(printf '%s' "${INDEX_JSON}" | jq -er --arg v "${ZIG_VERSION}" --arg k "${ZIG_ARCH}-linux" '.[$v][$k].tarball')"; \
    ZIG_SHA="$(printf '%s' "${INDEX_JSON}" | jq -er --arg v "${ZIG_VERSION}" --arg k "${ZIG_ARCH}-linux" '.[$v][$k].shasum')"; \
    curl -fsSL "${ZIG_URL}" -o /tmp/zig.tar.xz; \
    echo "${ZIG_SHA}  /tmp/zig.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/zig; \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1; \
    rm -f /tmp/zig.tar.xz; \
    /opt/zig/zig version

WORKDIR /src/TensorFold

RUN set -eux; \
    ARCH="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${ARCH}" in \
        amd64) ZIG_ARCH="x86_64" ;; \
        arm64) ZIG_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;; \
    esac; \
    VERSION_ARG=""; \
    if [ -z "${TENSORFOLD_REF}" ]; then \
        TAG="$(git describe --tags --exact-match 2>/dev/null || true)"; \
        VER="${TAG#v}"; \
        if echo "${VER}" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then \
            VERSION_ARG="-Dversion=${VER}"; \
        fi; \
    fi; \
    /opt/zig/zig build \
        -Dgpu="${TENSORFOLD_GPU}" \
        -Dtarget="${ZIG_ARCH}-linux-gnu.2.28" \
        -Dnvcc=/usr/local/cuda/bin/nvcc \
        ${VERSION_ARG}

RUN set -eux; \
    ARCH="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${ARCH}" in \
        amd64) ZIG_ARCH="x86_64" ;; \
        arm64) ZIG_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;; \
    esac; \
    XGRAMMAR_TAG="$(sed -n 's/.*pinned = "\([^"]*\)".*/\1/p' zig/src/platform/grammar_library.zig | head -n 1)"; \
    test -n "${XGRAMMAR_TAG}"; \
    git clone --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules --shallow-submodules https://github.com/mlc-ai/xgrammar.git /src/xgrammar; \
    /opt/zig/zig build grammar \
        -Dtarget="${ZIG_ARCH}-linux-gnu.2.28" \
        -Dxgrammar=/src/xgrammar

FROM nvidia/cuda:13.5.0-runtime-ubuntu24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=tensorfold
ARG APP_HOME=/home/tensorfold
ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if ! getent group "${APP_GID}" >/dev/null; then \
        groupadd --gid "${APP_GID}" "${APP_USER}"; \
    fi; \
    if ! getent passwd "${APP_UID}" >/dev/null; then \
        useradd \
            --uid "${APP_UID}" \
            --gid "${APP_GID}" \
            --create-home \
            --home-dir "${APP_HOME}" \
            --shell /usr/sbin/nologin \
            "${APP_USER}"; \
    fi; \
    install -d -o "${APP_UID}" -g "${APP_GID}" \
        "${APP_HOME}" \
        "${APP_HOME}/.cache/tensorfold" \
        "${APP_HOME}/.cache/huggingface"

WORKDIR ${APP_HOME}

COPY --from=builder /src/TensorFold/zig-out/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
COPY --from=builder /src/TensorFold/zig-out/native/lib/libtfgrammar.so /usr/local/lib/libtfgrammar.so
COPY --from=builder /src/TensorFold/LICENSE /usr/local/share/licenses/tensorfold/LICENSE
COPY --from=builder /src/TensorFold/NOTICE /usr/local/share/licenses/tensorfold/NOTICE
COPY --from=builder /src/TensorFold/THIRD_PARTY_NOTICES.md /usr/local/share/licenses/tensorfold/THIRD_PARTY_NOTICES.md
COPY --from=builder /src/xgrammar/LICENSE /usr/local/share/licenses/tensorfold/xgrammar-LICENSE

RUN ldconfig && /usr/local/bin/tensorfold-native --version

ENV HOME=${APP_HOME}
ENV XDG_CACHE_HOME=${APP_HOME}/.cache
ENV HF_HOME=${APP_HOME}/.cache/huggingface
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER ${APP_UID}:${APP_GID}

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/tensorfold-native"]
CMD ["--version"]
