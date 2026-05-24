# --- Build stage: compile Rust binary ---
FROM rust:1.94-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release -p nexa-cli

# --- Runtime stage: Python + compiled binary ---
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /app/target/release/nexa /usr/local/bin/nexa

COPY mcp/ mcp/
RUN pip install --no-cache-dir -r mcp/requirements.txt

ENV NEXA_BIN=/usr/local/bin/nexa

CMD ["python", "mcp/server.py"]
