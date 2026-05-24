/// SIMD-optimized and scalar fallback operations for hypervector computation.

/// XOR two word arrays
#[inline]
pub fn xor_words(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

/// Hamming distance via popcount of XOR
#[inline]
pub fn hamming_distance_words(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Total popcount
#[inline]
pub fn popcount_words(words: &[u64]) -> u32 {
    words.iter().map(|w| w.count_ones()).sum()
}

/// Dot product for f32 slices
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f64 {
    // Process in chunks of 4 for better pipelining
    let chunks = a.len() / 4;
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] as f64 * b[base] as f64;
        sum1 += a[base + 1] as f64 * b[base + 1] as f64;
        sum2 += a[base + 2] as f64 * b[base + 2] as f64;
        sum3 += a[base + 3] as f64 * b[base + 3] as f64;
    }

    let mut total = sum0 + sum1 + sum2 + sum3;

    for i in (chunks * 4)..a.len() {
        total += a[i] as f64 * b[i] as f64;
    }

    total
}
