use nexa_core::{BinaryHV, NexaError};

pub type Result<T> = std::result::Result<T, NexaError>;

// ──────────────────────────────────────────────
// CleanupResult
// ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CleanupResult {
    pub label: String,
    pub similarity: f64,
}

// ──────────────────────────────────────────────
// CleanupMemory
// ──────────────────────────────────────────────

pub struct CleanupMemory {
    prototypes: Vec<(String, BinaryHV)>,
    dim: usize,
}

impl CleanupMemory {
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(NexaError::ZeroDimension);
        }
        Ok(Self {
            prototypes: Vec::new(),
            dim,
        })
    }

    pub fn store(&mut self, label: &str, vector: BinaryHV) -> Result<()> {
        if vector.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: vector.dim(),
            });
        }
        self.prototypes.push((label.to_string(), vector));
        Ok(())
    }

    pub fn query(&self, noisy: &BinaryHV) -> Result<Option<CleanupResult>> {
        if noisy.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: noisy.dim(),
            });
        }
        if self.prototypes.is_empty() {
            return Ok(None);
        }

        let mut best: Option<CleanupResult> = None;
        for (label, proto) in &self.prototypes {
            let sim = proto.hamming_similarity(noisy)?;
            if best.as_ref().map_or(true, |b| sim > b.similarity) {
                best = Some(CleanupResult {
                    label: label.clone(),
                    similarity: sim,
                });
            }
        }
        Ok(best)
    }

    pub fn query_topk(&self, noisy: &BinaryHV, k: usize) -> Result<Vec<CleanupResult>> {
        if noisy.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: noisy.dim(),
            });
        }
        let mut results: Vec<CleanupResult> = Vec::with_capacity(self.prototypes.len());
        for (label, proto) in &self.prototypes {
            let sim = proto.hamming_similarity(noisy)?;
            results.push(CleanupResult {
                label: label.clone(),
                similarity: sim,
            });
        }
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        Ok(results)
    }

    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }
}

// ──────────────────────────────────────────────
// SparseDistributedMemory (Kanerva SDM)
// ──────────────────────────────────────────────

pub struct SparseDistributedMemory {
    addresses: Vec<BinaryHV>,
    counters: Vec<Vec<i32>>,
    dim: usize,
    access_radius: u32,
}

impl SparseDistributedMemory {
    pub fn new(dim: usize, num_addresses: usize, access_radius: u32, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(NexaError::ZeroDimension);
        }
        let mut addresses = Vec::with_capacity(num_addresses);
        for i in 0..num_addresses {
            addresses.push(BinaryHV::random(dim, seed.wrapping_add(i as u64))?);
        }
        let counters = vec![vec![0i32; dim]; num_addresses];
        Ok(Self {
            addresses,
            counters,
            dim,
            access_radius,
        })
    }

    pub fn write(&mut self, address: &BinaryHV, data: &BinaryHV) -> Result<usize> {
        if address.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: address.dim(),
            });
        }
        if data.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: data.dim(),
            });
        }

        let mut activated = 0usize;
        for (i, hard_addr) in self.addresses.iter().enumerate() {
            let dist = hard_addr.hamming_distance(address)?;
            if dist <= self.access_radius {
                activated += 1;
                for bit in 0..self.dim {
                    if data.get_bit(bit) {
                        self.counters[i][bit] += 1;
                    } else {
                        self.counters[i][bit] -= 1;
                    }
                }
            }
        }
        Ok(activated)
    }

    pub fn read(&self, address: &BinaryHV) -> Result<BinaryHV> {
        if address.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: address.dim(),
            });
        }

        let mut sums = vec![0i64; self.dim];
        for (i, hard_addr) in self.addresses.iter().enumerate() {
            let dist = hard_addr.hamming_distance(address)?;
            if dist <= self.access_radius {
                for bit in 0..self.dim {
                    sums[bit] += self.counters[i][bit] as i64;
                }
            }
        }

        // Threshold: positive → 1, else → 0
        let mut result = BinaryHV::zeros(self.dim)?;
        for bit in 0..self.dim {
            if sums[bit] > 0 {
                result.set_bit(bit, true);
            }
        }
        Ok(result)
    }

    pub fn activated_count(&self, address: &BinaryHV) -> Result<usize> {
        if address.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: address.dim(),
            });
        }
        let mut count = 0usize;
        for hard_addr in &self.addresses {
            let dist = hard_addr.hamming_distance(address)?;
            if dist <= self.access_radius {
                count += 1;
            }
        }
        Ok(count)
    }
}

// ──────────────────────────────────────────────
// AssociativeMemory
// ──────────────────────────────────────────────

pub struct AssociativeMemory {
    pairs: Vec<(BinaryHV, BinaryHV)>,
    dim: usize,
}

impl AssociativeMemory {
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(NexaError::ZeroDimension);
        }
        Ok(Self {
            pairs: Vec::new(),
            dim,
        })
    }

    pub fn store(&mut self, key: BinaryHV, value: BinaryHV) -> Result<()> {
        if key.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: key.dim(),
            });
        }
        if value.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: value.dim(),
            });
        }
        self.pairs.push((key, value));
        Ok(())
    }

    pub fn recall(&self, cue: &BinaryHV, cleanup: &CleanupMemory) -> Result<Option<CleanupResult>> {
        if cue.dim() != self.dim {
            return Err(NexaError::DimensionMismatch {
                expected: self.dim,
                got: cue.dim(),
            });
        }
        if self.pairs.is_empty() {
            return Ok(None);
        }

        // Unbind cue from each stored pair to recover candidate values,
        // then bundle all candidates via majority rule.
        let candidates: Vec<BinaryHV> = self
            .pairs
            .iter()
            .map(|(k, v)| {
                let bound = k.bind(v).expect("dim checked at store");
                cue.unbind(&bound).expect("dim checked at store")
            })
            .collect();

        let refs: Vec<&BinaryHV> = candidates.iter().collect();
        let bundled = BinaryHV::bundle(&refs)?;

        cleanup.query(&bundled)
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_memory_recovers_from_noise() {
        let dim = 1000;
        let mut mem = CleanupMemory::new(dim).unwrap();

        for i in 0..100 {
            let v = BinaryHV::random(dim, i).unwrap();
            mem.store(&format!("vec_{}", i), v).unwrap();
        }

        // Pick a prototype, corrupt it 10%, query
        let target_seed = 42u64;
        let original = BinaryHV::random(dim, target_seed).unwrap();
        let noisy = original.corrupt(0.10, 9999);

        let result = mem.query(&noisy).unwrap().expect("should find a match");
        assert_eq!(result.label, format!("vec_{}", target_seed));
        assert!(result.similarity > 0.8, "similarity {} should be > 0.8", result.similarity);
    }

    #[test]
    fn cleanup_memory_returns_none_for_unrelated() {
        let dim = 1000;
        let mut mem = CleanupMemory::new(dim).unwrap();

        for i in 0..10 {
            let v = BinaryHV::random(dim, i).unwrap();
            mem.store(&format!("vec_{}", i), v).unwrap();
        }

        // Query with a completely unrelated random vector
        let unrelated = BinaryHV::random(dim, 99999).unwrap();
        let result = mem.query(&unrelated).unwrap().expect("still returns a result");
        // Similarity should be around 0.5 (random chance)
        assert!(
            result.similarity > 0.4 && result.similarity < 0.6,
            "similarity {} should be near 0.5",
            result.similarity
        );
    }

    #[test]
    fn cleanup_memory_topk_returns_ordered() {
        let dim = 1000;
        let mut mem = CleanupMemory::new(dim).unwrap();

        for i in 0..50 {
            let v = BinaryHV::random(dim, i).unwrap();
            mem.store(&format!("vec_{}", i), v).unwrap();
        }

        let query_vec = BinaryHV::random(dim, 10).unwrap().corrupt(0.15, 777);
        let topk = mem.query_topk(&query_vec, 5).unwrap();

        assert_eq!(topk.len(), 5);
        for w in topk.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "results not sorted: {} < {}",
                w[0].similarity,
                w[1].similarity
            );
        }
        // The top result should be the correct one
        assert_eq!(topk[0].label, "vec_10");
    }

    #[test]
    fn sdm_write_then_read_recovers() {
        let dim = 1000;
        let num_addresses = 1000;
        let access_radius = (dim as f64 * 0.45) as u32;

        let mut sdm =
            SparseDistributedMemory::new(dim, num_addresses, access_radius, 42).unwrap();

        let address = BinaryHV::random(dim, 100).unwrap();
        let data = BinaryHV::random(dim, 200).unwrap();

        let activated = sdm.write(&address, &data).unwrap();
        assert!(activated > 0, "should activate at least one address");

        let read_back = sdm.read(&address).unwrap();
        let sim = data.hamming_similarity(&read_back).unwrap();
        assert!(
            sim > 0.7,
            "read-back similarity {} should be > 0.7",
            sim
        );
    }

    #[test]
    fn sdm_capacity_degrades_gracefully() {
        let dim = 1000;
        let num_addresses = 1000;
        let access_radius = (dim as f64 * 0.45) as u32;

        let mut sdm =
            SparseDistributedMemory::new(dim, num_addresses, access_radius, 0).unwrap();

        let n_writes = 50;
        let mut addresses = Vec::new();
        let mut data_vecs = Vec::new();

        for i in 0..n_writes {
            let addr = BinaryHV::random(dim, 1000 + i).unwrap();
            let data = BinaryHV::random(dim, 2000 + i).unwrap();
            sdm.write(&addr, &data).unwrap();
            addresses.push(addr);
            data_vecs.push(data);
        }

        // Check first few entries are still recoverable above random chance
        let mut above_chance = 0;
        for i in 0..5 {
            let read_back = sdm.read(&addresses[i]).unwrap();
            let sim = data_vecs[i].hamming_similarity(&read_back).unwrap();
            if sim > 0.5 {
                above_chance += 1;
            }
        }
        assert!(
            above_chance >= 3,
            "at least 3 of 5 early entries should have similarity > 0.5"
        );
    }

    #[test]
    fn associative_memory_recall() {
        let dim = 1000;
        let mut assoc = AssociativeMemory::new(dim).unwrap();
        let mut cleanup = CleanupMemory::new(dim).unwrap();

        // Create key-value pairs
        let key = BinaryHV::random(dim, 10).unwrap();
        let value = BinaryHV::random(dim, 20).unwrap();

        cleanup.store("value_0", value.clone()).unwrap();
        assoc.store(key.clone(), value.clone()).unwrap();

        // Also store some distractors in cleanup memory
        for i in 1..10 {
            let distractor = BinaryHV::random(dim, 100 + i).unwrap();
            cleanup.store(&format!("value_{}", i), distractor).unwrap();
        }

        let result = assoc.recall(&key, &cleanup).unwrap().expect("should recall");
        assert_eq!(result.label, "value_0");
        assert!(
            result.similarity > 0.7,
            "recall similarity {} should be > 0.7",
            result.similarity
        );
    }
}
