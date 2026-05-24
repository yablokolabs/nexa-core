"""
NexaCore MCP Server — exposes the NexaCore representation runtime as MCP tools.

Wraps the compiled `nexa` CLI binary via subprocess and exposes each command
as a callable MCP tool over STDIO transport.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "NexaCore",
    instructions="Universal representation runtime — hyperdimensional computing, holographic memory, and encoded-space compute",
)

NEXA_BIN = os.environ.get("NEXA_BIN", shutil.which("nexa") or "nexa")


def _run_nexa(*args: str, stdin_data: bytes | None = None) -> str:
    """Run the nexa CLI binary and return combined stdout+stderr."""
    try:
        result = subprocess.run(
            [NEXA_BIN, *args],
            capture_output=True,
            text=True,
            timeout=120,
            input=stdin_data.decode() if stdin_data else None,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output.strip()
    except FileNotFoundError:
        return f"Error: nexa binary not found at '{NEXA_BIN}'. Set NEXA_BIN env var to the path of the compiled nexa binary."
    except subprocess.TimeoutExpired:
        return "Error: nexa command timed out after 120 seconds."


@server.tool()
def encode(content: str, content_type: str = "text", dim: int = 10000) -> str:
    """Encode data into NexaCore hypervector space.

    Transforms input data into a high-dimensional holographic hypervector
    and saves it as a .nexa file. Supports text, JSON, and CSV input.

    Args:
        content: The data to encode (text string, JSON string, or CSV string).
        content_type: One of "text", "json", "csv". Determines encoding strategy.
        dim: Hypervector dimensionality (default 10000). Higher = more capacity.
    """
    ext_map = {"text": ".txt", "json": ".json", "csv": ".csv"}
    ext = ext_map.get(content_type, ".txt")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{ext}"
        output_path = Path(tmpdir) / "encoded.nexa"
        input_path.write_text(content)

        result = _run_nexa("encode", str(input_path), "-o", str(output_path), "-d", str(dim))

        if output_path.exists():
            size = output_path.stat().st_size
            result += f"\n\nEncoded file size: {size} bytes"
            result += f"\nOutput: {output_path.name}"

        return result


@server.tool()
def inspect(file_path: str) -> str:
    """Inspect a .nexa file and display its metadata.

    Shows magic bytes verification, version, dimension count, vector count,
    encoding type, metadata, and checksum validity.

    Args:
        file_path: Path to the .nexa file to inspect.
    """
    return _run_nexa("inspect", file_path)


@server.tool()
def similarity(content_a: str, content_b: str, dim: int = 10000) -> str:
    """Compute similarity between two pieces of content in hypervector space.

    Encodes both inputs as hypervectors and computes their Hamming similarity.
    Returns a score between 0.0 (completely dissimilar) and 1.0 (identical).

    Args:
        content_a: First text/data to compare.
        content_b: Second text/data to compare.
        dim: Hypervector dimensionality (default 10000).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_a = Path(tmpdir) / "a.txt"
        input_b = Path(tmpdir) / "b.txt"
        nexa_a = Path(tmpdir) / "a.nexa"
        nexa_b = Path(tmpdir) / "b.nexa"

        input_a.write_text(content_a)
        input_b.write_text(content_b)

        _run_nexa("encode", str(input_a), "-o", str(nexa_a), "-d", str(dim))
        _run_nexa("encode", str(input_b), "-o", str(nexa_b), "-d", str(dim))

        if not nexa_a.exists() or not nexa_b.exists():
            return "Error: failed to encode one or both inputs."

        return _run_nexa("similarity", str(nexa_a), str(nexa_b))


@server.tool()
def benchmark(dim: int = 10000) -> str:
    """Run NexaCore performance benchmarks.

    Measures throughput of core hypervector operations: XOR binding,
    Hamming distance computation, and vector bundling at the given dimension.

    Args:
        dim: Hypervector dimensionality for benchmarks (default 10000).
    """
    return _run_nexa("benchmark", "-d", str(dim))


@server.tool()
def topology(model_json: str) -> str:
    """Encode a neural network architecture topology into hypervector space.

    Takes a JSON model graph definition and encodes its structure
    (layer types, connections, parameters) as a hyperdimensional representation.

    Args:
        model_json: JSON string defining the model graph (ModelGraph format).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "model.json"
        input_path.write_text(model_json)
        return _run_nexa("topology", str(input_path))


@server.tool()
def encode_and_inspect(content: str, content_type: str = "text", dim: int = 10000) -> str:
    """Encode data and immediately inspect the resulting .nexa file.

    Combines encode + inspect into a single operation for convenience.

    Args:
        content: The data to encode.
        content_type: One of "text", "json", "csv".
        dim: Hypervector dimensionality (default 10000).
    """
    ext_map = {"text": ".txt", "json": ".json", "csv": ".csv"}
    ext = ext_map.get(content_type, ".txt")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{ext}"
        output_path = Path(tmpdir) / "encoded.nexa"
        input_path.write_text(content)

        encode_result = _run_nexa("encode", str(input_path), "-o", str(output_path), "-d", str(dim))

        if output_path.exists():
            inspect_result = _run_nexa("inspect", str(output_path))
            return f"{encode_result}\n\n--- Inspection ---\n\n{inspect_result}"

        return encode_result


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
