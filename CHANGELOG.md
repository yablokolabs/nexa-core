# Changelog

All notable changes to NexaCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-24

### Added

- **nexa-core**: Hypervector engine with BinaryHV, BipolarHV, RealHV, and SparseHV types
- **nexa-core**: SIMD-optimized XOR binding, Hamming distance, cosine similarity
- **nexa-core**: `.nexa` binary file format with CRC32 checksums and memory-mappable layout
- **nexa-hdc**: Codebook-based symbol encoding, sequence/set/n-gram encoders, role-filler bindings
- **nexa-holography**: FFT-based circular convolution and correlation for holographic binding
- **nexa-memory**: Cleanup memory, sparse distributed memory (SDM), and associative memory
- **nexa-encoder**: Universal encoder for text, JSON, CSV, and binary data
- **nexa-decoder**: Four decoder modes — exact, approximate, cleanup, and symbolic
- **nexa-decoder**: Corruption engine for resilience testing (bit-flip, truncation, block zeroing)
- **nexa-runtime**: HDC classifier, vector search, anomaly detection, clustering, homomorphic ops
- **nexa-topology**: Model graph DAG, graph encoder, topology analyzer with MLP/CNN builders
- **nexa-cli**: Seven commands — encode, decode, inspect, similarity, benchmark, recover, topology
- **nexa-bench**: Criterion benchmark suite for core operations
- **MCP server**: Python-based MCP server wrapping the CLI for tool integration
- **MCPize support**: Dockerfile and `mcpize.yaml` for containerized deployment
