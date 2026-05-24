use crate::simd;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension(pub usize);

impl Dimension {
    pub fn new(d: usize) -> crate::error::Result<Self> {
        if d == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        Ok(Dimension(d))
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

// ──────────────────────────────────────────────
// Binary Hypervector (bit-packed in u64 words)
// ──────────────────────────────────────────────

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryHV {
    words: Vec<u64>,
    dim: usize,
}

impl fmt::Debug for BinaryHV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryHV(dim={})", self.dim)
    }
}

impl BinaryHV {
    pub fn zeros(dim: usize) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let n_words = (dim + 63) / 64;
        Ok(BinaryHV {
            words: vec![0u64; n_words],
            dim,
        })
    }

    pub fn random(dim: usize, seed: u64) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let n_words = (dim + 63) / 64;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut words: Vec<u64> = (0..n_words).map(|_| rng.gen()).collect();
        // Mask trailing bits in the last word
        let trailing = dim % 64;
        if trailing != 0 {
            words[n_words - 1] &= (1u64 << trailing) - 1;
        }
        Ok(BinaryHV { words, dim })
    }

    pub fn from_words(words: Vec<u64>, dim: usize) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let needed = (dim + 63) / 64;
        if words.len() != needed {
            return Err(crate::NexaError::DimensionMismatch {
                expected: needed,
                got: words.len(),
            });
        }
        Ok(BinaryHV { words, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    fn check_dim(&self, other: &BinaryHV) -> crate::error::Result<()> {
        if self.dim != other.dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        Ok(())
    }

    /// XOR binding — self-inverse
    pub fn bind(&self, other: &BinaryHV) -> crate::error::Result<BinaryHV> {
        self.check_dim(other)?;
        let words = simd::xor_words(&self.words, &other.words);
        Ok(BinaryHV { words, dim: self.dim })
    }

    /// XOR unbinding (same as bind for binary)
    pub fn unbind(&self, other: &BinaryHV) -> crate::error::Result<BinaryHV> {
        self.bind(other)
    }

    /// Majority-rule bundling
    pub fn bundle(vectors: &[&BinaryHV]) -> crate::error::Result<BinaryHV> {
        if vectors.is_empty() {
            return Err(crate::NexaError::EmptyInput);
        }
        let dim = vectors[0].dim;
        let n_words = vectors[0].n_words();
        for v in vectors.iter().skip(1) {
            if v.dim != dim {
                return Err(crate::NexaError::DimensionMismatch {
                    expected: dim,
                    got: v.dim,
                });
            }
        }

        let mut result_words = vec![0u64; n_words];
        let threshold = vectors.len() / 2;

        for bit_idx in 0..dim {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            let mut count = 0usize;
            for v in vectors {
                if (v.words[word_idx] >> bit_pos) & 1 == 1 {
                    count += 1;
                }
            }
            if count > threshold {
                result_words[word_idx] |= 1u64 << bit_pos;
            }
        }

        Ok(BinaryHV { words: result_words, dim })
    }

    /// Circular permutation by `shifts` positions
    pub fn permute(&self, shifts: isize) -> BinaryHV {
        let dim = self.dim;
        let shifts = ((shifts % dim as isize) + dim as isize) as usize % dim;
        if shifts == 0 {
            return self.clone();
        }

        let mut result = vec![0u64; self.n_words()];
        for i in 0..dim {
            let src_word = i / 64;
            let src_bit = i % 64;
            let dst = (i + shifts) % dim;
            let dst_word = dst / 64;
            let dst_bit = dst % 64;
            if (self.words[src_word] >> src_bit) & 1 == 1 {
                result[dst_word] |= 1u64 << dst_bit;
            }
        }

        BinaryHV { words: result, dim }
    }

    /// Hamming distance (number of differing bits)
    pub fn hamming_distance(&self, other: &BinaryHV) -> crate::error::Result<u32> {
        self.check_dim(other)?;
        Ok(simd::hamming_distance_words(&self.words, &other.words))
    }

    /// Normalized Hamming distance in [0, 1]
    pub fn hamming_similarity(&self, other: &BinaryHV) -> crate::error::Result<f64> {
        let dist = self.hamming_distance(other)?;
        Ok(1.0 - (dist as f64 / self.dim as f64))
    }

    /// Flip random bits with given corruption rate [0.0, 1.0]
    pub fn corrupt(&self, rate: f64, seed: u64) -> BinaryHV {
        let mut result = self.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for i in 0..self.dim {
            if rng.gen::<f64>() < rate {
                let word_idx = i / 64;
                let bit_pos = i % 64;
                result.words[word_idx] ^= 1u64 << bit_pos;
            }
        }
        result
    }

    /// Zero out a contiguous block
    pub fn truncate_block(&self, start_frac: f64, end_frac: f64) -> BinaryHV {
        let mut result = self.clone();
        let start = (self.dim as f64 * start_frac) as usize;
        let end = (self.dim as f64 * end_frac).min(self.dim as f64) as usize;
        for i in start..end {
            let word_idx = i / 64;
            let bit_pos = i % 64;
            result.words[word_idx] &= !(1u64 << bit_pos);
        }
        result
    }

    /// Population count (number of 1-bits)
    pub fn popcount(&self) -> u32 {
        simd::popcount_words(&self.words)
    }

    /// Get bit at index
    pub fn get_bit(&self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_pos = idx % 64;
        (self.words[word_idx] >> bit_pos) & 1 == 1
    }

    /// Set bit at index
    pub fn set_bit(&mut self, idx: usize, val: bool) {
        let word_idx = idx / 64;
        let bit_pos = idx % 64;
        if val {
            self.words[word_idx] |= 1u64 << bit_pos;
        } else {
            self.words[word_idx] &= !(1u64 << bit_pos);
        }
    }

    /// Convert to bipolar representation
    pub fn to_bipolar(&self) -> BipolarHV {
        let data: Vec<i8> = (0..self.dim)
            .map(|i| if self.get_bit(i) { 1 } else { -1 })
            .collect();
        BipolarHV { data, dim: self.dim }
    }

    /// Convert to real-valued representation
    pub fn to_real(&self) -> RealHV {
        let data: Vec<f32> = (0..self.dim)
            .map(|i| if self.get_bit(i) { 1.0 } else { -1.0 })
            .collect();
        RealHV { data, dim: self.dim }
    }
}

// ──────────────────────────────────────────────
// Bipolar Hypervector {-1, +1}
// ──────────────────────────────────────────────

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BipolarHV {
    data: Vec<i8>,
    dim: usize,
}

impl fmt::Debug for BipolarHV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BipolarHV(dim={})", self.dim)
    }
}

impl BipolarHV {
    pub fn random(dim: usize, seed: u64) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<i8> = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        Ok(BipolarHV { data, dim })
    }

    pub fn from_data(data: Vec<i8>, dim: usize) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        if data.len() != dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: dim,
                got: data.len(),
            });
        }
        Ok(BipolarHV { data, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    fn check_dim(&self, other: &BipolarHV) -> crate::error::Result<()> {
        if self.dim != other.dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        Ok(())
    }

    /// Element-wise multiply binding
    pub fn bind(&self, other: &BipolarHV) -> crate::error::Result<BipolarHV> {
        self.check_dim(other)?;
        let data: Vec<i8> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Ok(BipolarHV { data, dim: self.dim })
    }

    /// Unbind (same as bind for bipolar — multiply is self-inverse)
    pub fn unbind(&self, other: &BipolarHV) -> crate::error::Result<BipolarHV> {
        self.bind(other)
    }

    /// Element-wise sum then threshold to {-1, +1}
    pub fn bundle(vectors: &[&BipolarHV]) -> crate::error::Result<BipolarHV> {
        if vectors.is_empty() {
            return Err(crate::NexaError::EmptyInput);
        }
        let dim = vectors[0].dim;
        for v in vectors.iter().skip(1) {
            if v.dim != dim {
                return Err(crate::NexaError::DimensionMismatch {
                    expected: dim,
                    got: v.dim,
                });
            }
        }

        let mut sums = vec![0i32; dim];
        for v in vectors {
            for (i, &val) in v.data.iter().enumerate() {
                sums[i] += val as i32;
            }
        }

        let data: Vec<i8> = sums.iter()
            .map(|&s| if s >= 0 { 1 } else { -1 })
            .collect();
        Ok(BipolarHV { data, dim })
    }

    /// Circular permutation
    pub fn permute(&self, shifts: isize) -> BipolarHV {
        let dim = self.dim;
        let shifts = ((shifts % dim as isize) + dim as isize) as usize % dim;
        if shifts == 0 {
            return self.clone();
        }
        let mut result = vec![0i8; dim];
        for i in 0..dim {
            result[(i + shifts) % dim] = self.data[i];
        }
        BipolarHV { data: result, dim }
    }

    /// Cosine similarity
    pub fn cosine_similarity(&self, other: &BipolarHV) -> crate::error::Result<f64> {
        self.check_dim(other)?;
        let dot: i64 = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a as i64 * b as i64)
            .sum();
        Ok(dot as f64 / self.dim as f64)
    }

    pub fn to_binary(&self) -> BinaryHV {
        let n_words = (self.dim + 63) / 64;
        let mut words = vec![0u64; n_words];
        for (i, &val) in self.data.iter().enumerate() {
            if val > 0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BinaryHV { words, dim: self.dim }
    }

    pub fn corrupt(&self, rate: f64, seed: u64) -> BipolarHV {
        let mut result = self.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for i in 0..self.dim {
            if rng.gen::<f64>() < rate {
                result.data[i] *= -1;
            }
        }
        result
    }
}

// ──────────────────────────────────────────────
// Real-valued Hypervector (f32)
// ──────────────────────────────────────────────

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RealHV {
    data: Vec<f32>,
    dim: usize,
}

impl fmt::Debug for RealHV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RealHV(dim={})", self.dim)
    }
}

impl RealHV {
    pub fn random_normal(dim: usize, seed: u64) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Box-Muller transform for normal distribution
        let data: Vec<f32> = (0..dim)
            .map(|_| {
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (z / (dim as f64).sqrt()) as f32
            })
            .collect();
        Ok(RealHV { data, dim })
    }

    pub fn zeros(dim: usize) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        Ok(RealHV {
            data: vec![0.0; dim],
            dim,
        })
    }

    pub fn from_data(data: Vec<f32>, dim: usize) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        if data.len() != dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: dim,
                got: data.len(),
            });
        }
        Ok(RealHV { data, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn check_dim(&self, other: &RealHV) -> crate::error::Result<()> {
        if self.dim != other.dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        Ok(())
    }

    /// Element-wise multiplication binding
    pub fn bind(&self, other: &RealHV) -> crate::error::Result<RealHV> {
        self.check_dim(other)?;
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Ok(RealHV { data, dim: self.dim })
    }

    /// Element-wise addition
    pub fn add(&self, other: &RealHV) -> crate::error::Result<RealHV> {
        self.check_dim(other)?;
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(RealHV { data, dim: self.dim })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &RealHV) -> crate::error::Result<RealHV> {
        self.check_dim(other)?;
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(RealHV { data, dim: self.dim })
    }

    /// Scale by scalar
    pub fn scale(&self, s: f32) -> RealHV {
        let data: Vec<f32> = self.data.iter().map(|&x| x * s).collect();
        RealHV { data, dim: self.dim }
    }

    /// L2 normalization
    pub fn normalize(&self) -> RealHV {
        let norm = self.l2_norm();
        if norm < 1e-10 {
            return self.clone();
        }
        self.scale(1.0 / norm as f32)
    }

    pub fn l2_norm(&self) -> f64 {
        let sum_sq: f64 = self.data.iter().map(|&x| (x as f64) * (x as f64)).sum();
        sum_sq.sqrt()
    }

    /// Cosine similarity
    pub fn cosine_similarity(&self, other: &RealHV) -> crate::error::Result<f64> {
        self.check_dim(other)?;
        let dot: f64 = simd::dot_product_f32(&self.data, &other.data);
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        let denom = norm_a * norm_b;
        if denom < 1e-10 {
            return Ok(0.0);
        }
        Ok(dot / denom)
    }

    /// Dot product
    pub fn dot(&self, other: &RealHV) -> crate::error::Result<f64> {
        self.check_dim(other)?;
        Ok(simd::dot_product_f32(&self.data, &other.data))
    }

    /// Circular permutation
    pub fn permute(&self, shifts: isize) -> RealHV {
        let dim = self.dim;
        let shifts = ((shifts % dim as isize) + dim as isize) as usize % dim;
        if shifts == 0 {
            return self.clone();
        }
        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            result[(i + shifts) % dim] = self.data[i];
        }
        RealHV { data: result, dim }
    }

    /// Add Gaussian noise
    pub fn add_noise(&self, std_dev: f64, seed: u64) -> RealHV {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = self.data.iter()
            .map(|&x| {
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * std_dev;
                x + noise as f32
            })
            .collect();
        RealHV { data, dim: self.dim }
    }
}

// ──────────────────────────────────────────────
// Sparse Hypervector
// ──────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseHV {
    indices: Vec<usize>,
    values: Vec<f32>,
    dim: usize,
}

impl SparseHV {
    pub fn new(dim: usize, indices: Vec<usize>, values: Vec<f32>) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        if indices.len() != values.len() {
            return Err(crate::NexaError::DimensionMismatch {
                expected: indices.len(),
                got: values.len(),
            });
        }
        for &idx in &indices {
            if idx >= dim {
                return Err(crate::NexaError::DimensionMismatch {
                    expected: dim,
                    got: idx + 1,
                });
            }
        }
        Ok(SparseHV { indices, values, dim })
    }

    pub fn random(dim: usize, sparsity: f64, seed: u64) -> crate::error::Result<Self> {
        if dim == 0 {
            return Err(crate::NexaError::ZeroDimension);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_nonzero = ((dim as f64) * sparsity).round() as usize;
        let mut indices: Vec<usize> = (0..dim).collect();
        // Fisher-Yates partial shuffle
        for i in 0..n_nonzero.min(dim) {
            let j = i + (rng.gen::<usize>() % (dim - i));
            indices.swap(i, j);
        }
        indices.truncate(n_nonzero);
        indices.sort();
        let values: Vec<f32> = (0..indices.len())
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        Ok(SparseHV { indices, values, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Convert to dense RealHV
    pub fn to_real(&self) -> RealHV {
        let mut data = vec![0.0f32; self.dim];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            data[idx] = val;
        }
        RealHV { data, dim: self.dim }
    }

    /// Dot product with another sparse vector
    pub fn dot(&self, other: &SparseHV) -> crate::error::Result<f64> {
        if self.dim != other.dim {
            return Err(crate::NexaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        let mut result = 0.0f64;
        let (mut i, mut j) = (0, 0);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] as f64 * other.values[j] as f64;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Binary HV tests ──

    #[test]
    fn binary_bind_then_unbind_recovers_original() {
        let a = BinaryHV::random(10_000, 42).unwrap();
        let b = BinaryHV::random(10_000, 43).unwrap();
        let bound = a.bind(&b).unwrap();
        let recovered = bound.unbind(&a).unwrap();
        assert_eq!(recovered, b);
    }

    #[test]
    fn binary_bundle_of_same_vectors_equals_original() {
        let v = BinaryHV::random(10_000, 42).unwrap();
        let bundled = BinaryHV::bundle(&[&v, &v, &v]).unwrap();
        assert_eq!(bundled, v);
    }

    #[test]
    fn binary_permute_then_inverse_recovers_original() {
        let v = BinaryHV::random(10_000, 42).unwrap();
        let shifted = v.permute(37);
        let recovered = shifted.permute(-37);
        assert_eq!(recovered, v);
    }

    #[test]
    fn binary_random_vectors_are_quasi_orthogonal() {
        let a = BinaryHV::random(10_000, 100).unwrap();
        let b = BinaryHV::random(10_000, 200).unwrap();
        let sim = a.hamming_similarity(&b).unwrap();
        assert!((sim - 0.5).abs() < 0.05,
            "Expected ~0.5 similarity, got {}", sim);
    }

    #[test]
    fn binary_similarity_is_symmetric() {
        let a = BinaryHV::random(10_000, 42).unwrap();
        let b = BinaryHV::random(10_000, 43).unwrap();
        let sim_ab = a.hamming_similarity(&b).unwrap();
        let sim_ba = b.hamming_similarity(&a).unwrap();
        assert!((sim_ab - sim_ba).abs() < 1e-10);
    }

    #[test]
    fn binary_deterministic_seeded_generation() {
        let a = BinaryHV::random(10_000, 42).unwrap();
        let b = BinaryHV::random(10_000, 42).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn binary_corruption_reduces_similarity() {
        let original = BinaryHV::random(10_000, 42).unwrap();
        let corrupted = original.corrupt(0.1, 99);
        let sim = original.hamming_similarity(&corrupted).unwrap();
        assert!(sim > 0.85 && sim < 0.95,
            "Expected ~0.9 similarity at 10% corruption, got {}", sim);
    }

    #[test]
    fn binary_zero_dim_rejected() {
        assert!(BinaryHV::random(0, 42).is_err());
    }

    #[test]
    fn binary_dimension_mismatch_rejected() {
        let a = BinaryHV::random(100, 42).unwrap();
        let b = BinaryHV::random(200, 42).unwrap();
        assert!(a.bind(&b).is_err());
    }

    // ── Bipolar HV tests ──

    #[test]
    fn bipolar_bind_then_unbind_recovers_original() {
        let a = BipolarHV::random(10_000, 42).unwrap();
        let b = BipolarHV::random(10_000, 43).unwrap();
        let bound = a.bind(&b).unwrap();
        let recovered = bound.unbind(&a).unwrap();
        assert_eq!(recovered, b);
    }

    #[test]
    fn bipolar_random_vectors_quasi_orthogonal() {
        let a = BipolarHV::random(10_000, 100).unwrap();
        let b = BipolarHV::random(10_000, 200).unwrap();
        let sim = a.cosine_similarity(&b).unwrap();
        assert!(sim.abs() < 0.05,
            "Expected ~0 cosine similarity, got {}", sim);
    }

    #[test]
    fn bipolar_bundle_preserves_majority() {
        let a = BipolarHV::random(10_000, 1).unwrap();
        let b = BipolarHV::random(10_000, 2).unwrap();
        let c = BipolarHV::random(10_000, 3).unwrap();
        let bundled = BipolarHV::bundle(&[&a, &a, &a, &b, &c]).unwrap();
        let sim_a = a.cosine_similarity(&bundled).unwrap();
        let sim_b = b.cosine_similarity(&bundled).unwrap();
        assert!(sim_a > sim_b,
            "Majority vector should be most similar: sim_a={}, sim_b={}", sim_a, sim_b);
    }

    // ── Real HV tests ──

    #[test]
    fn real_cosine_similarity_of_identical_is_one() {
        let v = RealHV::random_normal(10_000, 42).unwrap();
        let sim = v.cosine_similarity(&v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6, "Self-similarity should be 1.0, got {}", sim);
    }

    #[test]
    fn real_random_vectors_quasi_orthogonal() {
        let a = RealHV::random_normal(10_000, 100).unwrap();
        let b = RealHV::random_normal(10_000, 200).unwrap();
        let sim = a.cosine_similarity(&b).unwrap();
        assert!(sim.abs() < 0.05,
            "Expected ~0 cosine similarity, got {}", sim);
    }

    #[test]
    fn real_normalize_produces_unit_vector() {
        let v = RealHV::random_normal(10_000, 42).unwrap();
        let n = v.normalize();
        assert!((n.l2_norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn real_add_noise_reduces_similarity() {
        let original = RealHV::random_normal(10_000, 42).unwrap();
        let noisy = original.add_noise(0.5, 99);
        let sim = original.cosine_similarity(&noisy).unwrap();
        assert!(sim > 0.0 && sim < 1.0,
            "Noisy vector should have reduced similarity: {}", sim);
    }

    // ── Sparse HV tests ──

    #[test]
    fn sparse_dot_product_correct() {
        let a = SparseHV::new(100, vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseHV::new(100, vec![5, 10, 20], vec![4.0, 5.0, 6.0]).unwrap();
        let dot = a.dot(&b).unwrap();
        assert!((dot - (2.0 * 4.0 + 3.0 * 5.0)).abs() < 1e-6);
    }

    #[test]
    fn sparse_to_real_roundtrip() {
        let s = SparseHV::random(1000, 0.1, 42).unwrap();
        let r = s.to_real();
        assert_eq!(r.dim(), 1000);
        let nnz_count = r.data().iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nnz_count, s.nnz());
    }
}
