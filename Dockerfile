FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/minuspod-jev-proxy /minuspod-jev-proxy
USER nobody
EXPOSE 8787
ENV PORT=8787
ENTRYPOINT ["/minuspod-jev-proxy"]
