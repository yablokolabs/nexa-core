# Contributing to NexaCore

Thank you for your interest in contributing to NexaCore.

## Getting Started

```bash
git clone https://github.com/yablokolabs/nexa-core.git
cd nexa-core
cargo build --workspace
cargo test --workspace
```

## Development Workflow

1. Fork the repository and create a feature branch from `main`
2. Write tests for your changes before implementing
3. Ensure all tests pass: `cargo test --workspace`
4. Ensure no warnings: `cargo build --workspace 2>&1 | grep warning`
5. Submit a pull request with a clear description of the change

## Code Standards

- Follow existing code style and module organization
- All public APIs must have doc comments
- New features require tests that verify behavior, not implementation details
- Keep dependencies minimal — justify any new dependency in the PR description

## Running Benchmarks

```bash
cargo bench --package nexa-bench
```

## Project Structure

Each crate in `crates/` has a focused responsibility. See the [README](README.md) for the full breakdown. When adding functionality, place it in the appropriate crate rather than expanding `nexa-core`.

## Reporting Issues

Open an issue with:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Rust version (`rustc --version`)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
