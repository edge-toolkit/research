# Build only the evaluation driver; reuse the installed Core hub unchanged.
ARG CORE_IMAGE=et-ws-server-core:local
FROM rust:1-bookworm AS build
ENV CARGO_BUILD_JOBS=2
WORKDIR /driver
COPY driver/Cargo.toml driver/Cargo.lock ./
COPY driver/src ./src
RUN cargo test --locked && cargo build --release --locked && rustc --version > /build-compiler.txt

FROM ${CORE_IMAGE}
USER 0
RUN apt-get update && apt-get install -y --no-install-recommends python3 iproute2 ethtool ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /driver/target/release/hub-network-agents /usr/local/bin/hub-network-agents
COPY --from=build /build-compiler.txt /build-compiler.txt
COPY tc_setup.py /usr/local/lib/hub-network/tc_setup.py
COPY monitor.py /usr/local/lib/hub-network/monitor.py
COPY diagnostics.py /usr/local/lib/hub-network/diagnostics.py
WORKDIR /app
CMD ["sleep", "infinity"]
