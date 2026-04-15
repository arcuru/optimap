# OptiMap development recipes

# Default: run tests
default: test

# --- Development ---

# Quick feedback loop: check + test + clippy
dev: check test lint

# Auto-fix what can be fixed
fix:
    cargo clippy --fix --allow-dirty
    cargo fmt

# --- Building ---

# Type-check without codegen
check:
    cargo check --all-targets

# Build in release mode
build:
    cargo build --release

# --- Testing ---

# Run tests (accepts filter, e.g. `just test splitsies`)
test *args:
    cargo nextest run {{ args }}

# Run doc tests only
test-doc:
    cargo test --doc

# Run all tests (unit + integration + doc)
test-all: test test-doc

# --- Linting ---

# Run clippy (allows unsafe-op-in-unsafe-fn until edition 2024 migration is done)
lint:
    cargo clippy --all-targets -- -D warnings -A unsafe-op-in-unsafe-fn -A clippy::too_many_arguments

# Check formatting
fmt-check:
    cargo fmt -- --check

# Format code
fmt:
    cargo fmt

# --- Benchmarks ---

# Run all benchmarks
bench:
    cargo bench

# Run a specific benchmark file (e.g. `just bench-file throughput`)
bench-file name:
    cargo bench --bench {{ name }}

# --- Documentation ---

# Build API docs
doc:
    cargo doc --no-deps

# Build API docs and open in browser
doc-open:
    cargo doc --no-deps --open

# Build the mdbook
book:
    mdbook build docs

# Serve the mdbook with live reload
book-serve:
    mdbook serve docs

# Test code examples in the mdbook
book-test:
    cargo build
    mdbook test docs -L target/debug/deps

# Test all documentation (rustdoc + mdbook)
doc-test: test-doc book-test

# --- CI-like checks ---

# Run everything CI would run
ci: fmt-check lint test-all doc book
