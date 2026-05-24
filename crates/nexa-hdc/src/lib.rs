//! Hyperdimensional computing library — codebooks, sequence/set/n-gram encoding, role-filler bindings.

use std::collections::HashMap;
use nexa_core::{BinaryHV, NexaError};

pub type Result<T> = std::result::Result<T, NexaError>;

pub const DEFAULT_DIMENSION: usize = 10_000;

/// Derive a deterministic per-symbol seed from base_seed and the symbol string.
fn symbol_seed(base_seed: u64, symbol: &str) -> u64 {
    let mut hash: u64 = base_seed;
    for byte in symbol.bytes() {
        hash = hash.wrapping_mul(6364136223846793005).wrapping_add(byte as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// Maps string symbols to deterministic random `BinaryHV`s.
pub struct Codebook {
    dim: usize,
    base_seed: u64,
    map: HashMap<String, BinaryHV>,
}

impl Codebook {
    pub fn new(dim: usize, base_seed: u64) -> Self {
        Self {
            dim,
            base_seed,
            map: HashMap::new(),
        }
    }

    /// Return existing HV or generate a deterministic one from the symbol hash.
    pub fn get_or_insert(&mut self, symbol: &str) -> &BinaryHV {
        let dim = self.dim;
        let seed = symbol_seed(self.base_seed, symbol);
        self.map
            .entry(symbol.to_string())
            .or_insert_with(|| BinaryHV::random(dim, seed).expect("valid dimension"))
    }

    pub fn get(&self, symbol: &str) -> Option<&BinaryHV> {
        self.map.get(symbol)
    }

    /// Find the nearest symbol by Hamming similarity.
    pub fn nearest(&self, query: &BinaryHV) -> Option<(&str, f64)> {
        self.map
            .iter()
            .filter_map(|(sym, hv)| {
                hv.hamming_similarity(query).ok().map(|sim| (sym.as_str(), sim))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ---------------------------------------------------------------------------
// SequenceEncoder
// ---------------------------------------------------------------------------

/// Encode ordered sequences using permutation + binding, preserving token order.
pub struct SequenceEncoder;

impl SequenceEncoder {
    /// Encode an ordered sequence of tokens.
    ///
    /// Algorithm: permute each token HV by its position index, then bundle all.
    pub fn encode(codebook: &mut Codebook, tokens: &[&str]) -> Result<BinaryHV> {
        if tokens.is_empty() {
            return Err(NexaError::EmptyInput);
        }
        let permuted: Vec<BinaryHV> = tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| {
                let hv = codebook.get_or_insert(tok);
                hv.permute(i as isize)
            })
            .collect();
        let refs: Vec<&BinaryHV> = permuted.iter().collect();
        BinaryHV::bundle(&refs)
    }
}

// ---------------------------------------------------------------------------
// SetEncoder
// ---------------------------------------------------------------------------

/// Encode unordered sets via bundling (order-invariant).
pub struct SetEncoder;

impl SetEncoder {
    pub fn encode(codebook: &mut Codebook, tokens: &[&str]) -> Result<BinaryHV> {
        if tokens.is_empty() {
            return Err(NexaError::EmptyInput);
        }
        // Collect clones so we can form a slice of references without borrow issues.
        let hvs: Vec<BinaryHV> = tokens
            .iter()
            .map(|tok| codebook.get_or_insert(tok).clone())
            .collect();
        let refs: Vec<&BinaryHV> = hvs.iter().collect();
        BinaryHV::bundle(&refs)
    }
}

// ---------------------------------------------------------------------------
// RoleFiller
// ---------------------------------------------------------------------------

/// Role-filler binding for structured key-value data.
pub struct RoleFiller;

impl RoleFiller {
    /// Bind a role HV with a filler HV.
    pub fn bind_role(role: &BinaryHV, filler: &BinaryHV) -> Result<BinaryHV> {
        role.bind(filler)
    }

    /// Extract the filler by unbinding the role.
    pub fn unbind_role(bound: &BinaryHV, role: &BinaryHV) -> Result<BinaryHV> {
        bound.unbind(role)
    }

    /// Encode a set of (role, filler) string pairs: bind each pair then bundle.
    pub fn encode_structure(
        codebook: &mut Codebook,
        pairs: &[(&str, &str)],
    ) -> Result<BinaryHV> {
        if pairs.is_empty() {
            return Err(NexaError::EmptyInput);
        }
        let bound: Vec<BinaryHV> = pairs
            .iter()
            .map(|(role, filler)| {
                let r = codebook.get_or_insert(role).clone();
                let f = codebook.get_or_insert(filler).clone();
                r.bind(&f).expect("same dimension")
            })
            .collect();
        let refs: Vec<&BinaryHV> = bound.iter().collect();
        BinaryHV::bundle(&refs)
    }
}

// ---------------------------------------------------------------------------
// NGramEncoder
// ---------------------------------------------------------------------------

/// Encode n-grams via shifted binding.
pub struct NGramEncoder;

impl NGramEncoder {
    /// Encode n-grams from a token sequence.
    ///
    /// For each window of size `n`, bind tokens shifted by their position
    /// within the window, then bundle all resulting n-gram HVs.
    pub fn encode(codebook: &mut Codebook, tokens: &[&str], n: usize) -> Result<BinaryHV> {
        if tokens.is_empty() || n == 0 {
            return Err(NexaError::EmptyInput);
        }
        if tokens.len() < n {
            return Err(NexaError::EncodingError(
                "token sequence shorter than n-gram size".into(),
            ));
        }

        // Pre-fetch all token HVs.
        let hvs: Vec<BinaryHV> = tokens
            .iter()
            .map(|tok| codebook.get_or_insert(tok).clone())
            .collect();

        let mut ngram_hvs: Vec<BinaryHV> = Vec::with_capacity(tokens.len() - n + 1);
        for window in hvs.windows(n) {
            // Within each window: permute position 0 by (n-1), position 1 by (n-2), …, last by 0
            // then bind them all together.
            let mut acc = window[0].permute((n - 1) as isize);
            for (j, hv) in window.iter().enumerate().skip(1) {
                let shifted = hv.permute((n - 1 - j) as isize);
                acc = acc.bind(&shifted)?;
            }
            ngram_hvs.push(acc);
        }

        let refs: Vec<&BinaryHV> = ngram_hvs.iter().collect();
        BinaryHV::bundle(&refs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = DEFAULT_DIMENSION;
    const SEED: u64 = 42;

    #[test]
    fn codebook_deterministic() {
        let mut cb1 = Codebook::new(DIM, SEED);
        let mut cb2 = Codebook::new(DIM, SEED);
        let hv1 = cb1.get_or_insert("alpha").clone();
        let hv2 = cb2.get_or_insert("alpha").clone();
        assert_eq!(hv1.words(), hv2.words());
    }

    #[test]
    fn codebook_quasi_orthogonal() {
        let mut cb = Codebook::new(DIM, SEED);
        let a = cb.get_or_insert("alpha").clone();
        let b = cb.get_or_insert("beta").clone();
        let sim = a.hamming_similarity(&b).unwrap();
        // Random binary vectors should have ~0.5 similarity.
        assert!((sim - 0.5).abs() < 0.05, "similarity {sim} not near 0.5");
    }

    #[test]
    fn sequence_encodes_order() {
        let mut cb = Codebook::new(DIM, SEED);
        let abc = SequenceEncoder::encode(&mut cb, &["A", "B", "C"]).unwrap();
        let cba = SequenceEncoder::encode(&mut cb, &["C", "B", "A"]).unwrap();
        let sim = abc.hamming_similarity(&cba).unwrap();
        // Different orderings should yield low similarity (near 0.5).
        assert!(sim < 0.7, "sequence similarity {sim} too high — order not captured");
    }

    #[test]
    fn set_order_invariant() {
        let mut cb = Codebook::new(DIM, SEED);
        let abc = SetEncoder::encode(&mut cb, &["A", "B", "C"]).unwrap();
        let cba = SetEncoder::encode(&mut cb, &["C", "B", "A"]).unwrap();
        let sim = abc.hamming_similarity(&cba).unwrap();
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "set encoding not order-invariant: similarity {sim}"
        );
    }

    #[test]
    fn role_filler_roundtrip() {
        let mut cb = Codebook::new(DIM, SEED);
        let role = cb.get_or_insert("color").clone();
        let filler = cb.get_or_insert("red").clone();
        let bound = RoleFiller::bind_role(&role, &filler).unwrap();
        let recovered = RoleFiller::unbind_role(&bound, &role).unwrap();
        let sim = recovered.hamming_similarity(&filler).unwrap();
        // XOR bind/unbind is exact for binary HVs.
        assert!(
            sim > 0.95,
            "role-filler roundtrip similarity {sim} too low"
        );
    }

    #[test]
    fn ngram_captures_structure() {
        let mut cb = Codebook::new(DIM, SEED);
        let abc = NGramEncoder::encode(&mut cb, &["A", "B", "C"], 2).unwrap();
        let abd = NGramEncoder::encode(&mut cb, &["A", "B", "D"], 2).unwrap();
        let xyz = NGramEncoder::encode(&mut cb, &["X", "Y", "Z"], 2).unwrap();

        let sim_close = abc.hamming_similarity(&abd).unwrap();
        let sim_far = abc.hamming_similarity(&xyz).unwrap();
        // abc and abd share the bigram (A,B), so should be more similar than abc vs xyz.
        assert!(
            sim_close > sim_far,
            "n-gram similarity: close={sim_close} should exceed far={sim_far}"
        );
    }

    #[test]
    fn codebook_nearest_finds_symbol() {
        let mut cb = Codebook::new(DIM, SEED);
        cb.get_or_insert("cat");
        cb.get_or_insert("dog");
        cb.get_or_insert("fish");
        let query = cb.get("cat").unwrap().clone();
        let (sym, sim) = cb.nearest(&query).unwrap();
        assert_eq!(sym, "cat");
        assert!((sim - 1.0).abs() < 1e-9);
    }
}
