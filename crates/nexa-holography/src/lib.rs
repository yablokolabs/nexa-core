use nexa_core::{NexaError, RealHV};
use std::f64::consts::PI;
use std::ops;

// ── Complex number for FFT ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }
}

impl ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

// ── FFT (radix-2 Cooley-Tukey) ─────────────────────────────────────────────

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }

    let even: Vec<Complex> = input.iter().step_by(2).copied().collect();
    let odd: Vec<Complex> = input.iter().skip(1).step_by(2).copied().collect();

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let mut result = vec![Complex::new(0.0, 0.0); n];
    for k in 0..n / 2 {
        let angle = -2.0 * PI * (k as f64) / (n as f64);
        let twiddle = Complex::new(angle.cos(), angle.sin()) * odd_fft[k];
        result[k] = even_fft[k] + twiddle;
        result[k + n / 2] = even_fft[k] - twiddle;
    }
    result
}

fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let conjugated: Vec<Complex> = input.iter().map(|c| c.conj()).collect();
    let transformed = fft(&conjugated);
    let scale = 1.0 / n as f64;
    transformed.iter().map(|c| Complex::new(c.re * scale, c.im * scale)).collect()
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Circular convolution — the binding operation for holographic reduced
/// representations.  Uses FFT for O(n log n) computation.
pub fn circular_convolution(a: &RealHV, b: &RealHV) -> Result<RealHV, NexaError> {
    let n = a.dim();
    if n != b.dim() {
        return Err(NexaError::DimensionMismatch { expected: n, got: b.dim() });
    }

    let padded = next_power_of_two(n);

    let mut ca: Vec<Complex> = a.data().iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    let mut cb: Vec<Complex> = b.data().iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    ca.resize(padded, Complex::new(0.0, 0.0));
    cb.resize(padded, Complex::new(0.0, 0.0));

    let fa = fft(&ca);
    let fb = fft(&cb);

    let fc: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).collect();
    let result = ifft(&fc);

    let data: Vec<f32> = result[..n].iter().map(|c| c.re as f32).collect();
    RealHV::from_data(data, n)
}

/// Circular correlation — the approximate inverse (unbinding) operation for
/// HRR.  correlation(a, b) = convolution(a_inverted, b).
pub fn circular_correlation(a: &RealHV, b: &RealHV) -> Result<RealHV, NexaError> {
    let n = a.dim();
    if n != b.dim() {
        return Err(NexaError::DimensionMismatch { expected: n, got: b.dim() });
    }

    let padded = next_power_of_two(n);

    let mut ca: Vec<Complex> = a.data().iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    let mut cb: Vec<Complex> = b.data().iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    ca.resize(padded, Complex::new(0.0, 0.0));
    cb.resize(padded, Complex::new(0.0, 0.0));

    let fa = fft(&ca);
    let fb = fft(&cb);

    // Correlation = conj(FFT(a)) * FFT(b), then IFFT
    let fc: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x.conj() * y).collect();
    let result = ifft(&fc);

    let data: Vec<f32> = result[..n].iter().map(|c| c.re as f32).collect();
    RealHV::from_data(data, n)
}

// ── HolographicStore ────────────────────────────────────────────────────────

/// Superposition-based holographic memory.
///
/// Stores key-value pairs by accumulating `convolution(key, value)` into a
/// single composite vector and retrieves via `correlation(key, composite)`.
pub struct HolographicStore {
    composite: RealHV,
    count: usize,
}

impl HolographicStore {
    /// Create a new empty store of the given dimensionality.
    pub fn new(dim: usize) -> Result<Self, NexaError> {
        let composite = RealHV::zeros(dim)?;
        Ok(Self { composite, count: 0 })
    }

    /// Store a key-value association by adding `convolution(key, value)` to
    /// the composite trace.
    pub fn store(&mut self, key: &RealHV, value: &RealHV) -> Result<(), NexaError> {
        let bound = circular_convolution(key, value)?;
        self.composite = self.composite.add(&bound)?;
        self.count += 1;
        Ok(())
    }

    /// Retrieve the value associated with `key` by correlating against the
    /// composite trace.
    pub fn retrieve(&self, key: &RealHV) -> Result<RealHV, NexaError> {
        circular_correlation(key, &self.composite)
    }

    /// Number of key-value pairs that have been stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_then_correlation_recovers_value() {
        let dim = 256;
        let key = RealHV::random_normal(dim, 1).unwrap();
        let value = RealHV::random_normal(dim, 2).unwrap();

        let conv = circular_convolution(&key, &value).unwrap();
        let recovered = circular_correlation(&key, &conv).unwrap();

        let sim = recovered.cosine_similarity(&value).unwrap();
        assert!(sim > 0.5, "expected similarity > 0.5, got {sim}");
    }

    #[test]
    fn holographic_store_multiple_pairs() {
        let dim = 256;
        let a = RealHV::random_normal(dim, 10).unwrap();
        let b = RealHV::random_normal(dim, 11).unwrap();
        let c = RealHV::random_normal(dim, 12).unwrap();
        let d = RealHV::random_normal(dim, 13).unwrap();

        let mut store = HolographicStore::new(dim).unwrap();
        store.store(&a, &b).unwrap();
        store.store(&c, &d).unwrap();
        assert_eq!(store.len(), 2);

        let retrieved = store.retrieve(&a).unwrap();
        let sim_b = retrieved.cosine_similarity(&b).unwrap();
        let sim_d = retrieved.cosine_similarity(&d).unwrap();

        assert!(
            sim_b > sim_d,
            "retrieved with A should be more similar to B ({sim_b}) than D ({sim_d})"
        );
        assert!(sim_b > 0.3, "expected similarity with B > 0.3, got {sim_b}");
    }

    #[test]
    fn holographic_retrieval_degrades_gracefully() {
        let dim = 512;
        let key = RealHV::random_normal(dim, 100).unwrap();
        let value = RealHV::random_normal(dim, 101).unwrap();

        let mut prev_sim = f64::MAX;
        for num_pairs in [1, 5, 20] {
            let mut store = HolographicStore::new(dim).unwrap();
            store.store(&key, &value).unwrap();
            // Add noise pairs
            for s in 1..num_pairs {
                let k = RealHV::random_normal(dim, 200 + s).unwrap();
                let v = RealHV::random_normal(dim, 300 + s).unwrap();
                store.store(&k, &v).unwrap();
            }

            let retrieved = store.retrieve(&key).unwrap();
            let sim = retrieved.cosine_similarity(&value).unwrap();

            assert!(
                sim <= prev_sim + 0.05,
                "similarity should degrade: {num_pairs} pairs gave {sim}, previous {prev_sim}"
            );
            // Even with 20 pairs the signal should be above random chance
            if num_pairs <= 20 {
                assert!(sim > 0.05, "with {num_pairs} pairs, sim {sim} should be above chance");
            }
            prev_sim = sim;
        }
    }

    #[test]
    fn correlation_is_approximate_inverse() {
        let dim = 256;
        let a = RealHV::random_normal(dim, 42).unwrap();
        let b = RealHV::random_normal(dim, 43).unwrap();

        let conv = circular_convolution(&a, &b).unwrap();
        let recovered = circular_correlation(&a, &conv).unwrap();

        let sim = recovered.cosine_similarity(&b).unwrap();
        assert!(sim > 0.5, "correlation should approximately invert convolution, got sim={sim}");
    }

    #[test]
    fn empty_store_returns_noise() {
        let dim = 256;
        let store = HolographicStore::new(dim).unwrap();
        let key = RealHV::random_normal(dim, 77).unwrap();
        let value = RealHV::random_normal(dim, 78).unwrap();

        let retrieved = store.retrieve(&key).unwrap();
        let sim = retrieved.cosine_similarity(&value).unwrap();

        assert!(
            sim.abs() < 0.3,
            "empty store retrieval should have low similarity, got {sim}"
        );
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }
}
