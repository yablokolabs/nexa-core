//! NexaCompress — lossless compression for hypervector data and `.nexa` files.
//!
//! Provides multiple compression strategies optimised for binary hypervectors:
//!
//! * **Deflate** — general-purpose compression via zlib/deflate
//! * **Delta** — XOR successive vectors → compress low-entropy deltas
//! * **MetadataStrip** — remove redundant `original_data` from encoding records
//! * **Auto** — try all strategies and pick the smallest result

use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use nexa_core::{NexaError, NexaFormat, NexaHeader};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

pub type Result<T> = std::result::Result<T, NexaError>;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Compression strategy for `.nexa` data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strategy {
    /// No compression — passthrough.
    None,
    /// General-purpose deflate compression.
    Deflate,
    /// Delta encoding: XOR successive vectors, then deflate the deltas.
    Delta,
    /// Strip `original_data` from metadata records before deflate.
    MetadataStrip,
    /// Try all strategies and keep the smallest result.
    Auto,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::None => write!(f, "none"),
            Strategy::Deflate => write!(f, "deflate"),
            Strategy::Delta => write!(f, "delta"),
            Strategy::MetadataStrip => write!(f, "metadata-strip"),
            Strategy::Auto => write!(f, "auto"),
        }
    }
}

impl Strategy {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "none" => Strategy::None,
            "deflate" => Strategy::Deflate,
            "delta" => Strategy::Delta,
            "metadata-strip" | "metadatastrip" | "strip" => Strategy::MetadataStrip,
            _ => Strategy::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// CompressedData
// ---------------------------------------------------------------------------

/// A compressed payload with metadata about the compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    pub strategy: Strategy,
    pub original_size: usize,
    pub compressed_size: usize,
    pub data: Vec<u8>,
    /// Number of vectors (for delta decoding).
    pub vector_count: u32,
    /// Bytes per vector (for delta decoding).
    pub vector_stride: u32,
}

// ---------------------------------------------------------------------------
// CompressionStats
// ---------------------------------------------------------------------------

/// Human-readable compression statistics.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub ratio: f64,
    pub bits_per_byte: f64,
    pub original_size: usize,
    pub compressed_size: usize,
    pub strategy: String,
    pub duration_ms: f64,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Strategy: {}\n  Original:   {} bytes\n  Compressed: {} bytes\n  Ratio:      {:.2}x\n  Bits/byte:  {:.3}\n  Time:       {:.1}ms",
            self.strategy, self.original_size, self.compressed_size,
            self.ratio, self.bits_per_byte, self.duration_ms
        )
    }
}

// ---------------------------------------------------------------------------
// Core compression / decompression
// ---------------------------------------------------------------------------

/// Compress raw bytes using deflate.
pub fn deflate_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(data)
        .map_err(|e| NexaError::EncodingError(format!("deflate write: {e}")))?;
    encoder
        .finish()
        .map_err(|e| NexaError::EncodingError(format!("deflate finish: {e}")))
}

/// Decompress deflated bytes.
pub fn deflate_decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = DeflateDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| NexaError::EncodingError(format!("inflate: {e}")))?;
    Ok(output)
}

/// Delta-encode a sequence of equal-length vectors: each vector is XORed with
/// the previous one (the first is kept as-is), then the deltas are deflated.
pub fn delta_encode(vectors: &[Vec<u8>]) -> Result<Vec<u8>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }
    let stride = vectors[0].len();
    let mut deltas: Vec<u8> = Vec::with_capacity(stride * vectors.len());

    // First vector verbatim.
    deltas.extend_from_slice(&vectors[0]);

    // Subsequent vectors as XOR delta against predecessor.
    for i in 1..vectors.len() {
        if vectors[i].len() != stride {
            return Err(NexaError::EncodingError(
                "delta encoding requires equal-length vectors".into(),
            ));
        }
        for j in 0..stride {
            deltas.push(vectors[i][j] ^ vectors[i - 1][j]);
        }
    }

    deflate_compress(&deltas)
}

/// Reverse of `delta_encode`: inflate then reconstruct vectors from deltas.
pub fn delta_decode(compressed: &[u8], vector_count: usize, stride: usize) -> Result<Vec<Vec<u8>>> {
    let deltas = deflate_decompress(compressed)?;
    let expected = vector_count * stride;
    if deltas.len() < expected {
        return Err(NexaError::EncodingError(format!(
            "delta decode: expected {expected} bytes, got {}",
            deltas.len()
        )));
    }

    let mut vectors: Vec<Vec<u8>> = Vec::with_capacity(vector_count);
    vectors.push(deltas[..stride].to_vec());

    for i in 1..vector_count {
        let offset = i * stride;
        let prev = &vectors[i - 1];
        let current: Vec<u8> = (0..stride)
            .map(|j| deltas[offset + j] ^ prev[j])
            .collect();
        vectors.push(current);
    }

    Ok(vectors)
}

/// Strip `original_data` from JSON metadata records (if present), keeping
/// only the structural fields. Returns the cleaned metadata JSON.
pub fn strip_metadata(metadata: &str) -> String {
    if metadata.is_empty() {
        return String::new();
    }

    // Parse as array of objects and remove `original_data` field.
    if let Ok(mut arr) = serde_json::from_str::<Vec<serde_json::Value>>(metadata) {
        for item in &mut arr {
            if let Some(obj) = item.as_object_mut() {
                obj.remove("original_data");
            }
        }
        serde_json::to_string(&arr).unwrap_or_else(|_| metadata.to_string())
    } else {
        metadata.to_string()
    }
}

// ---------------------------------------------------------------------------
// Compress / decompress a full `.nexa` payload
// ---------------------------------------------------------------------------

/// Compress raw byte data with the given strategy.
pub fn compress(data: &[u8], strategy: Strategy) -> Result<CompressedData> {
    match strategy {
        Strategy::None => Ok(CompressedData {
            strategy,
            original_size: data.len(),
            compressed_size: data.len(),
            data: data.to_vec(),
            vector_count: 0,
            vector_stride: 0,
        }),
        Strategy::Deflate | Strategy::MetadataStrip => {
            let compressed = deflate_compress(data)?;
            Ok(CompressedData {
                strategy,
                original_size: data.len(),
                compressed_size: compressed.len(),
                data: compressed,
                vector_count: 0,
                vector_stride: 0,
            })
        }
        Strategy::Delta => {
            // Delta without vector structure falls back to plain deflate.
            let compressed = deflate_compress(data)?;
            Ok(CompressedData {
                strategy: Strategy::Deflate,
                original_size: data.len(),
                compressed_size: compressed.len(),
                data: compressed,
                vector_count: 0,
                vector_stride: 0,
            })
        }
        Strategy::Auto => {
            let deflated = compress(data, Strategy::Deflate)?;
            Ok(deflated) // without vector info, deflate is the best we can do
        }
    }
}

/// Decompress data produced by `compress`.
pub fn decompress(cd: &CompressedData) -> Result<Vec<u8>> {
    match cd.strategy {
        Strategy::None => Ok(cd.data.clone()),
        Strategy::Deflate | Strategy::MetadataStrip | Strategy::Auto => {
            deflate_decompress(&cd.data)
        }
        Strategy::Delta => {
            // Should not happen in raw compress; handled in compress_vectors.
            deflate_decompress(&cd.data)
        }
    }
}

/// Compress a sequence of vectors using the chosen strategy.
pub fn compress_vectors(vectors: &[Vec<u8>], strategy: Strategy) -> Result<CompressedData> {
    if vectors.is_empty() {
        return Ok(CompressedData {
            strategy: Strategy::None,
            original_size: 0,
            compressed_size: 0,
            data: Vec::new(),
            vector_count: 0,
            vector_stride: 0,
        });
    }

    let stride = vectors[0].len();
    let total: usize = vectors.iter().map(|v| v.len()).sum();

    match strategy {
        Strategy::Delta => {
            let compressed = delta_encode(vectors)?;
            Ok(CompressedData {
                strategy: Strategy::Delta,
                original_size: total,
                compressed_size: compressed.len(),
                data: compressed,
                vector_count: vectors.len() as u32,
                vector_stride: stride as u32,
            })
        }
        Strategy::Auto => {
            // Try all strategies, pick smallest.
            let flat: Vec<u8> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
            let deflated = deflate_compress(&flat)?;
            let delta = delta_encode(vectors)?;

            if delta.len() < deflated.len() {
                Ok(CompressedData {
                    strategy: Strategy::Delta,
                    original_size: total,
                    compressed_size: delta.len(),
                    data: delta,
                    vector_count: vectors.len() as u32,
                    vector_stride: stride as u32,
                })
            } else {
                Ok(CompressedData {
                    strategy: Strategy::Deflate,
                    original_size: total,
                    compressed_size: deflated.len(),
                    data: deflated,
                    vector_count: vectors.len() as u32,
                    vector_stride: stride as u32,
                })
            }
        }
        other => {
            let flat: Vec<u8> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
            let mut cd = compress(&flat, other)?;
            cd.vector_count = vectors.len() as u32;
            cd.vector_stride = stride as u32;
            Ok(cd)
        }
    }
}

/// Decompress vectors produced by `compress_vectors`.
pub fn decompress_vectors(cd: &CompressedData) -> Result<Vec<Vec<u8>>> {
    if cd.vector_count == 0 {
        return Ok(Vec::new());
    }
    let count = cd.vector_count as usize;
    let stride = cd.vector_stride as usize;

    match cd.strategy {
        Strategy::Delta => delta_decode(&cd.data, count, stride),
        _ => {
            let flat = decompress(cd)?;
            let mut vectors = Vec::with_capacity(count);
            for i in 0..count {
                let start = i * stride;
                let end = start + stride;
                if end > flat.len() {
                    return Err(NexaError::EncodingError(
                        "decompressed data shorter than expected".into(),
                    ));
                }
                vectors.push(flat[start..end].to_vec());
            }
            Ok(vectors)
        }
    }
}

// ---------------------------------------------------------------------------
// File-level compress / decompress
// ---------------------------------------------------------------------------

/// Magic bytes for compressed `.nexa` files.
pub const NEXC_MAGIC: [u8; 4] = *b"NEXC";
pub const NEXC_MAGIC_END: [u8; 4] = *b"CXEN";

/// Compress a `.nexa` file into a compressed `.nexa` file.
///
/// The compressed file uses a simple envelope:
///   NEXC magic (4) | strategy (1) | vector_count (4) | vector_stride (4) |
///   metadata_len (4) | metadata (var) |
///   compressed_data_len (4) | compressed_data (var) |
///   CXEN magic (4)
pub fn compress_nexa_file(
    input: &Path,
    output: &Path,
    strategy: Strategy,
) -> Result<CompressionStats> {
    let start = Instant::now();

    // Read original .nexa file
    let mut file = std::fs::File::open(input)?;
    let (header, vectors) = NexaFormat::read(&mut file)?;
    let original_file_size = std::fs::metadata(input)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    // Determine effective strategy and metadata
    let effective_strategy = if strategy == Strategy::Auto || strategy == Strategy::MetadataStrip {
        strategy
    } else {
        strategy
    };

    let metadata = if effective_strategy == Strategy::MetadataStrip
        || effective_strategy == Strategy::Auto
    {
        strip_metadata(&header.metadata)
    } else {
        header.metadata.clone()
    };

    // Compress vectors
    let cd = compress_vectors(&vectors, effective_strategy)?;

    // Compress metadata
    let meta_bytes = metadata.as_bytes();
    let compressed_meta = deflate_compress(meta_bytes)?;

    // Write compressed file
    let mut out = std::fs::File::create(output)?;
    out.write_all(&NEXC_MAGIC)?;

    let strat_byte: u8 = match cd.strategy {
        Strategy::None => 0,
        Strategy::Deflate => 1,
        Strategy::Delta => 2,
        Strategy::MetadataStrip => 3,
        Strategy::Auto => 4,
    };
    out.write_all(&[strat_byte])?;
    out.write_all(&header.dimension.to_le_bytes())?;
    out.write_all(&cd.vector_count.to_le_bytes())?;
    out.write_all(&cd.vector_stride.to_le_bytes())?;

    // Compressed metadata
    out.write_all(&(compressed_meta.len() as u32).to_le_bytes())?;
    out.write_all(&compressed_meta)?;

    // Compressed vector data
    out.write_all(&(cd.data.len() as u32).to_le_bytes())?;
    out.write_all(&cd.data)?;

    out.write_all(&NEXC_MAGIC_END)?;

    let compressed_file_size = std::fs::metadata(output)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    let elapsed = start.elapsed();
    let ratio = if compressed_file_size > 0 {
        original_file_size as f64 / compressed_file_size as f64
    } else {
        1.0
    };
    let bpb = if original_file_size > 0 {
        (compressed_file_size as f64 * 8.0) / original_file_size as f64
    } else {
        8.0
    };

    Ok(CompressionStats {
        ratio,
        bits_per_byte: bpb,
        original_size: original_file_size,
        compressed_size: compressed_file_size,
        strategy: cd.strategy.to_string(),
        duration_ms: elapsed.as_secs_f64() * 1000.0,
    })
}

/// Decompress a compressed `.nexa` file back to a standard `.nexa` file.
pub fn decompress_nexa_file(input: &Path, output: &Path) -> Result<()> {
    let data = std::fs::read(input)?;

    if data.len() < 8 || &data[0..4] != NEXC_MAGIC {
        return Err(NexaError::InvalidMagic);
    }

    let mut pos = 4;

    let strat_byte = data[pos];
    pos += 1;

    let dimension = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let vector_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let vector_stride = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // Compressed metadata
    let meta_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let compressed_meta = &data[pos..pos + meta_len];
    pos += meta_len;
    let meta_bytes = deflate_decompress(compressed_meta)?;
    let metadata = String::from_utf8(meta_bytes)
        .map_err(|e| NexaError::EncodingError(format!("metadata utf8: {e}")))?;

    // Compressed vector data
    let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let compressed_data = &data[pos..pos + data_len];
    pos += data_len;

    // Verify end magic
    if pos + 4 > data.len() || &data[pos..pos + 4] != NEXC_MAGIC_END {
        return Err(NexaError::InvalidMagic);
    }

    let strategy = match strat_byte {
        0 => Strategy::None,
        1 => Strategy::Deflate,
        2 => Strategy::Delta,
        3 => Strategy::MetadataStrip,
        _ => Strategy::Deflate,
    };

    let cd = CompressedData {
        strategy,
        original_size: (vector_count as usize) * (vector_stride as usize),
        compressed_size: compressed_data.len(),
        data: compressed_data.to_vec(),
        vector_count,
        vector_stride,
    };

    let vectors = decompress_vectors(&cd)?;

    // Write standard .nexa file
    let header =
        NexaHeader::new(dimension, vectors.len() as u32).with_metadata(metadata);
    let mut out = std::fs::File::create(output)?;
    NexaFormat::write(&mut out, &header, &vectors)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflate_roundtrip() {
        let original = b"hello nexacore compression test! repeated repeated repeated";
        let compressed = deflate_compress(original).unwrap();
        assert!(compressed.len() < original.len());
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn delta_encode_decode_roundtrip() {
        let vectors = vec![
            vec![0u8, 1, 2, 3, 4, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 6, 8], // differs by 1 byte
            vec![0, 1, 2, 3, 4, 5, 6, 9],
            vec![0, 1, 2, 3, 4, 5, 7, 0],
        ];
        let compressed = delta_encode(&vectors).unwrap();
        let decompressed = delta_decode(&compressed, 4, 8).unwrap();
        assert_eq!(decompressed, vectors);
    }

    #[test]
    fn delta_better_than_flat_for_similar_vectors() {
        // Vectors that differ by only a few bytes should delta-compress better.
        let base: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut vectors = vec![base.clone()];
        for i in 1..10 {
            let mut v = base.clone();
            v[i * 100] ^= 0xFF;
            vectors.push(v);
        }

        let flat: Vec<u8> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_compressed = deflate_compress(&flat).unwrap();
        let delta_compressed = delta_encode(&vectors).unwrap();

        assert!(
            delta_compressed.len() <= flat_compressed.len(),
            "delta ({}) should be <= flat ({}) for similar vectors",
            delta_compressed.len(),
            flat_compressed.len()
        );
    }

    #[test]
    fn compress_vectors_auto_picks_best() {
        let base: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let mut vectors = vec![base.clone()];
        for i in 1..5 {
            let mut v = base.clone();
            v[i * 50] ^= 0xFF;
            vectors.push(v);
        }

        let cd = compress_vectors(&vectors, Strategy::Auto).unwrap();
        assert!(cd.compressed_size < cd.original_size);
        let decompressed = decompress_vectors(&cd).unwrap();
        assert_eq!(decompressed, vectors);
    }

    #[test]
    fn strip_metadata_removes_original_data() {
        let meta = r#"[{"id":"text_0","original_data":[104,101,108,108,111],"data_type":"Text"}]"#;
        let stripped = strip_metadata(meta);
        assert!(!stripped.contains("original_data"));
        assert!(stripped.contains("text_0"));
        assert!(stripped.contains("Text"));
    }

    #[test]
    fn compress_decompress_raw_roundtrip() {
        let data = b"The quick brown fox jumps over the lazy dog. Repeated data is key to compression. Repeated data is key to compression.";
        let cd = compress(data, Strategy::Deflate).unwrap();
        assert!(cd.compressed_size < data.len());
        let recovered = decompress(&cd).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn nexa_file_compress_decompress_roundtrip() {
        let header = NexaHeader::new(10000, 2)
            .with_metadata(r#"[{"id":"t0","original_data":[72,73],"data_type":"Text"}]"#.to_string());
        let vectors = vec![vec![1u8; 1250], vec![2u8; 1250]];

        let dir = std::env::temp_dir();
        let orig = dir.join("compress_test_orig.nexa");
        let comp = dir.join("compress_test_comp.nexc");
        let restored = dir.join("compress_test_restored.nexa");

        // Write original .nexa
        let mut f = std::fs::File::create(&orig).unwrap();
        NexaFormat::write(&mut f, &header, &vectors).unwrap();

        // Compress
        let stats = compress_nexa_file(&orig, &comp, Strategy::Auto).unwrap();
        assert!(stats.ratio > 1.0, "should achieve some compression, ratio={}", stats.ratio);

        // Decompress
        decompress_nexa_file(&comp, &restored).unwrap();

        // Verify roundtrip
        let mut f2 = std::fs::File::open(&restored).unwrap();
        let (h2, v2) = NexaFormat::read(&mut f2).unwrap();
        assert_eq!(h2.dimension, 10000);
        assert_eq!(v2.len(), 2);
        assert_eq!(v2, vectors);

        // Cleanup
        let _ = std::fs::remove_file(&orig);
        let _ = std::fs::remove_file(&comp);
        let _ = std::fs::remove_file(&restored);
    }
}
