FROM debian:bookworm-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build --target neuriplo-kserve-runtime

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /home/runtime --shell /usr/sbin/nologin runtime

COPY --from=build /build/neuriplo-kserve-runtime /usr/local/bin/neuriplo-kserve-runtime

USER runtime
EXPOSE 8080

ENTRYPOINT ["neuriplo-kserve-runtime"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
