/// SIMD-optimized and scalar fallback operations for hypervector computation.
///
/// On x86_64 with AVX2 support, uses 256-bit SIMD intrinsics for XOR, popcount,
/// and dot product operations. Falls back to scalar implementations on other
/// architectures or when AVX2 is unavailable.

// ---------------------------------------------------------------------------
// AVX2 implementations (x86_64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(a: &[u64], b: &[u64]) -> Vec<u64> {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut result = vec![0u64; len];
    let chunks = len / 4; // 4 u64 = 256 bits

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm256_loadu_si256(a.as_ptr().add(base) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(base) as *const __m256i);
        let vr = _mm256_xor_si256(va, vb);
        _mm256_storeu_si256(result.as_mut_ptr().add(base) as *mut __m256i, vr);
    }

    for i in (chunks * 4)..len {
        result[i] = a[i] ^ b[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_words_avx2(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 4;
    let mut total = 0u32;

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm256_loadu_si256(a.as_ptr().add(base) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(base) as *const __m256i);
        let xored = _mm256_xor_si256(va, vb);

        // Byte-level popcount via lookup table
        let lo = _mm256_and_si256(xored, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
        let popcnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));

        // Horizontal sum: sad against zero accumulates bytes into u64 lanes
        let sad = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
        // Extract 4 u64 lanes and sum
        let lo128 = _mm256_castsi256_si128(sad);
        let hi128 = _mm256_extracti128_si256(sad, 1);
        total += (_mm_extract_epi64(lo128, 0) + _mm_extract_epi64(lo128, 1)
            + _mm_extract_epi64(hi128, 0) + _mm_extract_epi64(hi128, 1)) as u32;
    }

    for i in (chunks * 4)..len {
        total += (a[i] ^ b[i]).count_ones();
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_words_avx2(words: &[u64]) -> u32 {
    use std::arch::x86_64::*;
    let len = words.len();
    let chunks = len / 4;
    let mut total = 0u32;

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);

    for i in 0..chunks {
        let base = i * 4;
        let v = _mm256_loadu_si256(words.as_ptr().add(base) as *const __m256i);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let popcnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
        let sad = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
        let lo128 = _mm256_castsi256_si128(sad);
        let hi128 = _mm256_extracti128_si256(sad, 1);
        total += (_mm_extract_epi64(lo128, 0) + _mm_extract_epi64(lo128, 1)
            + _mm_extract_epi64(hi128, 0) + _mm_extract_epi64(hi128, 1)) as u32;
    }

    for i in (chunks * 4)..len {
        total += words[i].count_ones();
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_f32_avx(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 8; // 8 f32 = 256 bits
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    // Horizontal sum of 8 f32 lanes
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut total = _mm_cvtss_f32(sum32) as f64;

    for i in (chunks * 8)..len {
        total += a[i] as f64 * b[i] as f64;
    }
    total
}

// ---------------------------------------------------------------------------
// Scalar fallback implementations
// ---------------------------------------------------------------------------

fn xor_words_scalar(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

fn hamming_distance_words_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

fn popcount_words_scalar(words: &[u64]) -> u32 {
    words.iter().map(|w| w.count_ones()).sum()
}

fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f64 {
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

// ---------------------------------------------------------------------------
// Public API — dispatches to AVX2/AVX or scalar at runtime
// ---------------------------------------------------------------------------

/// XOR two word arrays. Uses AVX2 on supported x86_64 CPUs.
#[inline]
pub fn xor_words(a: &[u64], b: &[u64]) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { xor_words_avx2(a, b) };
        }
    }
    xor_words_scalar(a, b)
}

/// Hamming distance via popcount of XOR. Uses AVX2 on supported x86_64 CPUs.
#[inline]
pub fn hamming_distance_words(a: &[u64], b: &[u64]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_distance_words_avx2(a, b) };
        }
    }
    hamming_distance_words_scalar(a, b)
}

/// Total popcount. Uses AVX2 on supported x86_64 CPUs.
#[inline]
pub fn popcount_words(words: &[u64]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { popcount_words_avx2(words) };
        }
    }
    popcount_words_scalar(words)
}

/// Dot product for f32 slices. Uses AVX on supported x86_64 CPUs.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { dot_product_f32_avx(a, b) };
        }
    }
    dot_product_f32_scalar(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_words_correctness() {
        let a = vec![0xFF00FF00u64, 0x12345678, 0, u64::MAX];
        let b = vec![0x00FF00FFu64, 0x87654321, u64::MAX, u64::MAX];
        let result = xor_words(&a, &b);
        assert_eq!(result, vec![0xFFFFFFFF, 0x95511559, u64::MAX, 0]);
    }

    #[test]
    fn hamming_distance_correctness() {
        let a = vec![0u64; 4];
        let b = vec![u64::MAX; 4];
        assert_eq!(hamming_distance_words(&a, &b), 256);

        let c = vec![0u64; 4];
        assert_eq!(hamming_distance_words(&a, &c), 0);
    }

    #[test]
    fn popcount_correctness() {
        let words = vec![1u64, 3, 7, 15]; // 1 + 2 + 3 + 4 = 10
        assert_eq!(popcount_words(&words), 10);
    }

    #[test]
    fn dot_product_correctness() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let result = dot_product_f32(&a, &b);
        assert!((result - 35.0).abs() < 1e-6);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a: Vec<u64> = (0..157).map(|_| rng.gen()).collect();
        let b: Vec<u64> = (0..157).map(|_| rng.gen()).collect();

        let xor_s = xor_words_scalar(&a, &b);
        let ham_s = hamming_distance_words_scalar(&a, &b);
        let pop_s = popcount_words_scalar(&a);

        assert_eq!(xor_words(&a, &b), xor_s);
        assert_eq!(hamming_distance_words(&a, &b), ham_s);
        assert_eq!(popcount_words(&a), pop_s);

        let fa: Vec<f32> = (0..157).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let fb: Vec<f32> = (0..157).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let dot_s = dot_product_f32_scalar(&fa, &fb);
        let dot_v = dot_product_f32(&fa, &fb);
        assert!((dot_s - dot_v).abs() < 1e-2, "Dot mismatch: {} vs {}", dot_s, dot_v);
    }
}
