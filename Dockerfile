# NexaCore — Multi-stage Docker build
# Produces a minimal image with the nexa CLI binary and MCP server.
#
# Build:  docker build -t nexa-core .
# Run:    docker run -i nexa-core            # MCP server over STDIO
# CLI:    docker run --entrypoint nexa nexa-core encode input.txt
#
# Environment variables:
#   NEXA_BIN — path to nexa binary (default: /usr/local/bin/nexa)

# --- Build stage: compile Rust binary ---
FROM rust:1.82-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release -p nexa-cli && \
    strip /app/target/release/nexa

# --- Runtime stage: Python + compiled binary ---
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /app/target/release/nexa /usr/local/bin/nexa

COPY mcp/ mcp/
RUN pip install --no-cache-dir -r mcp/requirements.txt

ENV NEXA_BIN=/usr/local/bin/nexa

# Default: run MCP server over STDIO
CMD ["python", "mcp/server.py"]
