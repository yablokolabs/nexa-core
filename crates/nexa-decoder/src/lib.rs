use std::collections::HashMap;

use nexa_core::{BinaryHV, NexaError};
use nexa_hdc::Codebook;
use nexa_memory::{CleanupMemory, CleanupResult};
use nexa_encoder::{DataType, NexaEncoder};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub type Result<T> = std::result::Result<T, NexaError>;

fn data_type_label(dt: &DataType) -> &'static str {
    match dt {
        DataType::Text => "text",
        DataType::Json => "json",
        DataType::Csv => "csv",
        DataType::Binary => "binary",
    }
}

// ---------------------------------------------------------------------------
// DecodedData
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DecodedData {
    pub data: Vec<u8>,
    pub data_type: String,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// ExactDecoder
// ---------------------------------------------------------------------------

pub struct ExactDecoder {
    registry: Vec<(BinaryHV, Vec<u8>, String)>,
}

impl ExactDecoder {
    pub fn new() -> Self {
        Self { registry: Vec::new() }
    }

    pub fn register(&mut self, hv: BinaryHV, original: Vec<u8>, data_type: &str) {
        self.registry.push((hv, original, data_type.to_string()));
    }

    pub fn from_encoder(encoder: &NexaEncoder) -> Self {
        let mut dec = Self::new();
        for (hv, rec) in encoder.records() {
            dec.register(
                hv.clone(),
                rec.original_data.clone(),
                data_type_label(&rec.data_type),
            );
        }
        dec
    }

    pub fn decode(&self, query: &BinaryHV) -> Result<DecodedData> {
        for (hv, data, dt) in &self.registry {
            if hv == query {
                return Ok(DecodedData {
                    data: data.clone(),
                    data_type: dt.clone(),
                    confidence: 1.0,
                });
            }
        }
        Err(NexaError::NotFound("No exact match found".into()))
    }

    pub fn decode_fuzzy(&self, query: &BinaryHV, threshold: f64) -> Result<DecodedData> {
        let mut best: Option<(f64, usize)> = None;
        for (i, (hv, _, _)) in self.registry.iter().enumerate() {
            let sim = query.hamming_similarity(hv)?;
            if sim >= threshold && best.as_ref().map_or(true, |(s, _)| sim > *s) {
                best = Some((sim, i));
            }
        }
        match best {
            Some((sim, idx)) => {
                let (_, data, dt) = &self.registry[idx];
                Ok(DecodedData {
                    data: data.clone(),
                    data_type: dt.clone(),
                    confidence: sim,
                })
            }
            None => Err(NexaError::NotFound("No match above threshold".into())),
        }
    }
}

// ---------------------------------------------------------------------------
// ApproxResult / ApproxDecoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ApproxResult {
    pub data: Vec<u8>,
    pub label: String,
    pub similarity: f64,
}

pub struct ApproxDecoder {
    cleanup: CleanupMemory,
    data_store: HashMap<String, Vec<u8>>,
}

impl ApproxDecoder {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            cleanup: CleanupMemory::new(dim)?,
            data_store: HashMap::new(),
        })
    }

    pub fn register(&mut self, label: &str, hv: BinaryHV, original: Vec<u8>) -> Result<()> {
        self.cleanup.store(label, hv)?;
        self.data_store.insert(label.to_string(), original);
        Ok(())
    }

    pub fn from_encoder(encoder: &NexaEncoder) -> Result<Self> {
        let mut dec = Self::new(encoder.dim())?;
        for (hv, rec) in encoder.records() {
            dec.register(&rec.id, hv.clone(), rec.original_data.clone())?;
        }
        Ok(dec)
    }

    pub fn decode(&self, query: &BinaryHV) -> Result<Option<ApproxResult>> {
        match self.cleanup.query(query)? {
            Some(cr) => {
                let data = self.data_store.get(&cr.label).cloned().unwrap_or_default();
                Ok(Some(ApproxResult {
                    data,
                    label: cr.label,
                    similarity: cr.similarity,
                }))
            }
            None => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// CleanupDecoder
// ---------------------------------------------------------------------------

pub struct CleanupDecoder {
    cleanup: CleanupMemory,
    prototypes: HashMap<String, BinaryHV>,
}

impl CleanupDecoder {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            cleanup: CleanupMemory::new(dim)?,
            prototypes: HashMap::new(),
        })
    }

    pub fn register(&mut self, label: &str, hv: BinaryHV) -> Result<()> {
        self.cleanup.store(label, hv.clone())?;
        self.prototypes.insert(label.to_string(), hv);
        Ok(())
    }

    pub fn restore(&self, noisy: &BinaryHV) -> Result<Option<CleanupResult>> {
        Ok(self.cleanup.query(noisy)?)
    }

    /// Repeatedly query cleanup memory, replacing the query with the matched
    /// prototype, until the returned label stabilises or `max_iters` is reached.
    pub fn restore_iterative(
        &self,
        noisy: &BinaryHV,
        max_iters: usize,
    ) -> Result<Option<CleanupResult>> {
        let mut current = noisy.clone();
        let mut last_label: Option<String> = None;

        for _ in 0..max_iters {
            match self.cleanup.query(&current)? {
                Some(result) => {
                    if let Some(ref prev) = last_label {
                        if prev == &result.label {
                            return Ok(Some(result));
                        }
                    }
                    last_label = Some(result.label.clone());
                    match self.prototypes.get(&result.label) {
                        Some(proto) => current = proto.clone(),
                        None => return Ok(Some(result)),
                    }
                }
                None => return Ok(None),
            }
        }

        Ok(self.cleanup.query(&current)?)
    }
}

// ---------------------------------------------------------------------------
// SymbolicResult / SymbolicDecoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SymbolicResult {
    pub symbol: String,
    pub similarity: f64,
}

pub struct SymbolicDecoder {
    codebook: Codebook,
    known_symbols: Vec<String>,
}

impl SymbolicDecoder {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            codebook: Codebook::new(dim, seed),
            known_symbols: Vec::new(),
        }
    }

    pub fn with_codebook(codebook: Codebook) -> Self {
        Self {
            codebook,
            known_symbols: Vec::new(),
        }
    }

    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    pub fn codebook_mut(&mut self) -> &mut Codebook {
        &mut self.codebook
    }

    pub fn register_symbol(&mut self, symbol: &str) {
        self.codebook.get_or_insert(symbol);
        if !self.known_symbols.iter().any(|s| s == symbol) {
            self.known_symbols.push(symbol.to_string());
        }
    }

    /// Unbind `known` from `bound` and look up the nearest symbol.
    pub fn decode_binding(
        &self,
        bound: &BinaryHV,
        known: &BinaryHV,
    ) -> Option<SymbolicResult> {
        let unbound = bound.unbind(known).ok()?;
        let (sym, sim) = self.codebook.nearest(&unbound)?;
        Some(SymbolicResult {
            symbol: sym.to_string(),
            similarity: sim,
        })
    }

    /// Try unbinding every known symbol from `bound` and return matches whose
    /// nearest neighbour is a *different* symbol with similarity > 0.55.
    pub fn decode_relation(&self, bound: &BinaryHV) -> Vec<SymbolicResult> {
        let mut results = Vec::new();
        for sym_name in &self.known_symbols {
            if let Some(hv) = self.codebook.get(sym_name) {
                if let Ok(unbound) = bound.unbind(hv) {
                    if let Some((nearest, sim)) = self.codebook.nearest(&unbound) {
                        if sim > 0.55 && nearest != sym_name {
                            results.push(SymbolicResult {
                                symbol: nearest.to_string(),
                                similarity: sim,
                            });
                        }
                    }
                }
            }
        }
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ---------------------------------------------------------------------------
// CorruptionEngine
// ---------------------------------------------------------------------------

pub struct CorruptionEngine;

impl CorruptionEngine {
    pub fn corrupt(hv: &BinaryHV, rate: f64, seed: u64) -> BinaryHV {
        hv.corrupt(rate, seed)
    }

    pub fn truncate(hv: &BinaryHV, start_frac: f64, end_frac: f64) -> BinaryHV {
        hv.truncate_block(start_frac, end_frac)
    }

    pub fn permute_scramble(hv: &BinaryHV, seed: u64) -> BinaryHV {
        let dim = hv.dim();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..dim).collect();
        indices.shuffle(&mut rng);

        let mut result = BinaryHV::zeros(dim).expect("valid dim");
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            result.set_bit(new_idx, hv.get_bit(old_idx));
        }
        result
    }

    pub fn block_drop(hv: &BinaryHV, block_count: usize, seed: u64) -> BinaryHV {
        let dim = hv.dim();
        if block_count == 0 {
            return hv.clone();
        }
        let total_blocks = (block_count * 3).max(8);
        let block_size = dim / total_blocks;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut block_indices: Vec<usize> = (0..total_blocks).collect();
        block_indices.shuffle(&mut rng);

        let to_drop: std::collections::HashSet<usize> = block_indices
            [..block_count.min(total_blocks)]
            .iter()
            .copied()
            .collect();

        let mut result = hv.clone();
        for &bi in &to_drop {
            let start = bi * block_size;
            let end = ((bi + 1) * block_size).min(dim);
            for i in start..end {
                result.set_bit(i, false);
            }
        }
        result
    }

    pub fn measure_fidelity(original: &BinaryHV, corrupted: &BinaryHV) -> Result<f64> {
        original.hamming_similarity(corrupted)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 10_000;

    #[test]
    fn exact_decoder_roundtrip_text() {
        let mut encoder = NexaEncoder::new(DIM, 42);
        let hv = encoder.encode_text("apple").unwrap();
        let decoder = ExactDecoder::from_encoder(&encoder);
        let result = decoder.decode(&hv).unwrap();
        assert_eq!(String::from_utf8_lossy(&result.data), "apple");
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.data_type, "text");
    }

    #[test]
    fn exact_decoder_fuzzy_with_noise() {
        let mut encoder = NexaEncoder::new(DIM, 42);
        let hv = encoder.encode_text("apple").unwrap();
        let noisy = CorruptionEngine::corrupt(&hv, 0.05, 99);
        let decoder = ExactDecoder::from_encoder(&encoder);
        let result = decoder.decode_fuzzy(&noisy, 0.9).unwrap();
        assert_eq!(String::from_utf8_lossy(&result.data), "apple");
        assert!(result.confidence >= 0.9);
    }

    #[test]
    fn approx_decoder_recovers_from_corruption() {
        let mut encoder = NexaEncoder::new(DIM, 42);
        let hv_apple = encoder.encode_text("apple").unwrap();
        let _hv_banana = encoder.encode_text("banana").unwrap();
        let _hv_cherry = encoder.encode_text("cherry").unwrap();

        let decoder = ApproxDecoder::from_encoder(&encoder).unwrap();
        let noisy = CorruptionEngine::corrupt(&hv_apple, 0.15, 77);
        let result = decoder.decode(&noisy).unwrap().unwrap();
        assert_eq!(String::from_utf8_lossy(&result.data), "apple");
    }

    #[test]
    fn cleanup_decoder_converges() {
        let mut decoder = CleanupDecoder::new(DIM).unwrap();
        for i in 0u64..50 {
            let hv = BinaryHV::random(DIM, i).unwrap();
            decoder.register(&format!("proto_{}", i), hv).unwrap();
        }
        // Re-generate the same deterministic vector for proto_7
        let target = BinaryHV::random(DIM, 7).unwrap();
        let noisy = CorruptionEngine::corrupt(&target, 0.20, 99);
        let result = decoder.restore_iterative(&noisy, 10).unwrap().unwrap();
        assert_eq!(result.label, "proto_7");
    }

    #[test]
    fn symbolic_decoder_unbinds_relation() {
        let mut decoder = SymbolicDecoder::new(DIM, 42);
        decoder.register_symbol("DOG");
        decoder.register_symbol("BARK");

        let dog_hv = decoder.codebook().get("DOG").unwrap().clone();
        let bark_hv = decoder.codebook().get("BARK").unwrap().clone();
        let bound = dog_hv.bind(&bark_hv).unwrap();

        let result = decoder.decode_binding(&bound, &dog_hv).unwrap();
        assert_eq!(result.symbol, "BARK");
        assert!(result.similarity > 0.9);
    }

    #[test]
    fn symbolic_decoder_decode_relation() {
        let mut decoder = SymbolicDecoder::new(DIM, 42);
        decoder.register_symbol("DOG");
        decoder.register_symbol("BARK");

        let dog_hv = decoder.codebook().get("DOG").unwrap().clone();
        let bark_hv = decoder.codebook().get("BARK").unwrap().clone();
        let bound = dog_hv.bind(&bark_hv).unwrap();

        let results = decoder.decode_relation(&bound);
        let symbols: Vec<&str> = results.iter().map(|r| r.symbol.as_str()).collect();
        assert!(symbols.contains(&"DOG"), "DOG should appear in results");
        assert!(symbols.contains(&"BARK"), "BARK should appear in results");
    }

    #[test]
    fn corruption_engine_fidelity_decreases() {
        let hv = BinaryHV::random(DIM, 42).unwrap();
        let rates = [0.05, 0.10, 0.20, 0.30];
        let fidelities: Vec<f64> = rates
            .iter()
            .map(|&rate| {
                let corrupted = CorruptionEngine::corrupt(&hv, rate, 99);
                CorruptionEngine::measure_fidelity(&hv, &corrupted).unwrap()
            })
            .collect();

        for i in 1..fidelities.len() {
            assert!(
                fidelities[i] < fidelities[i - 1],
                "Fidelity at {:.0}% ({:.4}) should be less than at {:.0}% ({:.4})",
                rates[i] * 100.0,
                fidelities[i],
                rates[i - 1] * 100.0,
                fidelities[i - 1],
            );
        }
    }

    #[test]
    fn holographic_partial_recovery() {
        let mut encoder = NexaEncoder::new(DIM, 42);
        let hv = encoder.encode_text("quantum").unwrap();

        // Zero out the first 30% of the vector
        let truncated = CorruptionEngine::truncate(&hv, 0.0, 0.3);

        let decoder = ExactDecoder::from_encoder(&encoder);
        let result = decoder.decode_fuzzy(&truncated, 0.5).unwrap();
        assert_eq!(String::from_utf8_lossy(&result.data), "quantum");
    }
}
