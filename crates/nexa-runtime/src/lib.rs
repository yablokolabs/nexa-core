use nexa_core::BinaryHV;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub label: String,
    pub similarity: f64,
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// VectorSearch — similarity search over a corpus of labeled BinaryHVs
// ---------------------------------------------------------------------------
pub struct VectorSearch {
    corpus: Vec<(String, BinaryHV)>,
    _dim: usize,
}

impl VectorSearch {
    pub fn new(dim: usize) -> Self {
        Self {
            corpus: Vec::new(),
            _dim: dim,
        }
    }

    pub fn insert(&mut self, label: String, vector: BinaryHV) {
        self.corpus.push((label, vector));
    }

    pub fn search(&self, query: &BinaryHV, top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(String, f64)> = self
            .corpus
            .iter()
            .map(|(label, v)| {
                let sim = v.hamming_similarity(query).unwrap_or(0.0);
                (label.clone(), sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(i, (label, similarity))| SearchResult {
                label,
                similarity,
                rank: i + 1,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.corpus.len()
    }
}

// ---------------------------------------------------------------------------
// PredictionResult
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub class: String,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// HdcClassifier — HDC classifier using class prototypes
// ---------------------------------------------------------------------------
pub struct HdcClassifier {
    prototypes: Vec<(String, BinaryHV)>,
    dim: usize,
}

impl HdcClassifier {
    pub fn new(dim: usize) -> Self {
        Self {
            prototypes: Vec::new(),
            dim,
        }
    }

    pub fn train(&mut self, class: &str, examples: &[&BinaryHV]) {
        let bundled = BinaryHV::bundle(examples).expect("bundle failed");

        if let Some((_, proto)) = self.prototypes.iter_mut().find(|(c, _)| c == class) {
            // Merge: odd-count bundle (3) for real majority vote
            let all: Vec<&BinaryHV> = vec![&*proto, &bundled, &bundled];
            *proto = BinaryHV::bundle(&all).expect("merge bundle failed");
        } else {
            self.prototypes.push((class.to_string(), bundled));
        }
    }

    pub fn predict(&self, query: &BinaryHV) -> Option<PredictionResult> {
        self.prototypes
            .iter()
            .map(|(class, proto)| {
                let sim = proto.hamming_similarity(query).unwrap_or(0.0);
                PredictionResult {
                    class: class.clone(),
                    confidence: sim,
                }
            })
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Subtract query from wrong class prototype, add to correct class prototype.
    /// Uses individual bit voting: flip each bit toward/away from query.
    pub fn retrain(&mut self, query: &BinaryHV, wrong_class: &str, correct_class: &str) {
        let dim = self.dim;

        // Add query to correct class — move prototype toward query
        if let Some((_, proto)) = self.prototypes.iter_mut().find(|(c, _)| c == correct_class) {
            for i in 0..dim {
                if proto.get_bit(i) != query.get_bit(i) {
                    proto.set_bit(i, query.get_bit(i));
                }
            }
        }

        // Subtract query from wrong class — move prototype away from query
        if let Some((_, proto)) = self.prototypes.iter_mut().find(|(c, _)| c == wrong_class) {
            for i in 0..dim {
                if proto.get_bit(i) == query.get_bit(i) {
                    proto.set_bit(i, !query.get_bit(i));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AnomalyDetector — detect outlier HVs
// ---------------------------------------------------------------------------
pub struct AnomalyDetector {
    reference: Vec<BinaryHV>,
    threshold: f64,
    _dim: usize,
}

impl AnomalyDetector {
    pub fn new(dim: usize, threshold: f64) -> Self {
        Self {
            reference: Vec::new(),
            threshold,
            _dim: dim,
        }
    }

    pub fn add_reference(&mut self, vector: BinaryHV) {
        self.reference.push(vector);
    }

    /// Returns true if max similarity to all reference vectors is below threshold.
    pub fn is_anomaly(&self, query: &BinaryHV) -> bool {
        self.max_similarity(query) < self.threshold
    }

    /// Returns 1.0 - max_similarity (higher = more anomalous).
    pub fn anomaly_score(&self, query: &BinaryHV) -> f64 {
        1.0 - self.max_similarity(query)
    }

    fn max_similarity(&self, query: &BinaryHV) -> f64 {
        self.reference
            .iter()
            .map(|r| r.hamming_similarity(query).unwrap_or(0.0))
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// ClusterAssignment
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    pub cluster_id: usize,
    pub similarity: f64,
}

// ---------------------------------------------------------------------------
// Clusterer — K-means-style clustering in HV space
// ---------------------------------------------------------------------------
pub struct Clusterer {
    _dim: usize,
}

impl Clusterer {
    pub fn new(dim: usize) -> Self {
        Self { _dim: dim }
    }

    pub fn cluster(
        &self,
        vectors: &[&BinaryHV],
        k: usize,
        max_iters: usize,
        seed: u64,
    ) -> Vec<ClusterAssignment> {
        assert!(k > 0 && k <= vectors.len());

        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        let mut centroids: Vec<BinaryHV> =
            indices[..k].iter().map(|&i| vectors[i].clone()).collect();

        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..max_iters {
            let prev = assignments.clone();

            // Assign each vector to nearest centroid
            for (i, v) in vectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_sim = f64::NEG_INFINITY;
                for (j, c) in centroids.iter().enumerate() {
                    let sim = c.hamming_similarity(v).unwrap_or(0.0);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cluster = j;
                    }
                }
                assignments[i] = best_cluster;
            }

            if assignments == prev {
                break;
            }

            // Update centroids by bundling members
            for j in 0..k {
                let members: Vec<&BinaryHV> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, v)| *v)
                    .collect();

                if !members.is_empty() {
                    centroids[j] = BinaryHV::bundle(&members).expect("bundle failed");
                }
            }
        }

        // Final assignments with similarities
        vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let cid = assignments[i];
                let similarity = centroids[cid].hamming_similarity(v).unwrap_or(0.0);
                ClusterAssignment {
                    cluster_id: cid,
                    similarity,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// HomomorphicOps — structure-preserving operations demo
// ---------------------------------------------------------------------------
pub struct HomomorphicOps;

impl HomomorphicOps {
    /// Compute similarity between f(bind(a,b)) and bind(f(a), f(b)).
    /// Returns 1.0 for perfect homomorphism.
    pub fn verify_binding_homomorphism(
        a: &BinaryHV,
        b: &BinaryHV,
        f: impl Fn(&BinaryHV) -> BinaryHV,
    ) -> f64 {
        let ab = a.bind(b).expect("bind failed");
        let f_ab = f(&ab);

        let fa = f(a);
        let fb = f(b);
        let fa_fb = fa.bind(&fb).expect("bind failed");

        f_ab.hamming_similarity(&fa_fb).unwrap_or(0.0)
    }

    /// Search directly in encoded space, return sorted (index, similarity) pairs.
    pub fn encoded_similarity_search(
        query: &BinaryHV,
        corpus: &[&BinaryHV],
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let sim = query.hamming_similarity(v).unwrap_or(0.0);
                (i, sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ---------------------------------------------------------------------------
// LshIndex — Locality-Sensitive Hashing for approximate nearest neighbor
// ---------------------------------------------------------------------------

/// LSH index using random bit sampling for Hamming-space ANN search.
/// Each hash function selects `bits_per_hash` random bit positions from the HV
/// and uses those bits as a hash key. Multiple hash tables increase recall.
pub struct LshIndex {
    tables: Vec<LshTable>,
    corpus: Vec<(String, BinaryHV)>,
    dim: usize,
}

struct LshTable {
    bit_positions: Vec<usize>,
    buckets: std::collections::HashMap<u64, Vec<usize>>,
}

impl LshTable {
    fn hash(&self, hv: &BinaryHV) -> u64 {
        let mut h = 0u64;
        for (i, &pos) in self.bit_positions.iter().enumerate() {
            if hv.get_bit(pos) {
                h |= 1 << (i % 64);
            }
        }
        h
    }
}

impl LshIndex {
    /// Create an LSH index with `num_tables` hash tables, each sampling
    /// `bits_per_hash` random bit positions.
    pub fn new(dim: usize, num_tables: usize, bits_per_hash: usize, seed: u64) -> Self {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tables = (0..num_tables)
            .map(|_| {
                let bit_positions: Vec<usize> =
                    (0..bits_per_hash).map(|_| rng.gen_range(0..dim)).collect();
                LshTable {
                    bit_positions,
                    buckets: std::collections::HashMap::new(),
                }
            })
            .collect();

        Self {
            tables,
            corpus: Vec::new(),
            dim,
        }
    }

    /// Insert a labeled vector into the index.
    pub fn insert(&mut self, label: String, hv: BinaryHV) {
        let idx = self.corpus.len();
        self.corpus.push((label, hv.clone()));
        for table in &mut self.tables {
            let hash = table.hash(&hv);
            table.buckets.entry(hash).or_default().push(idx);
        }
    }

    /// Approximate nearest neighbor search. Returns candidates from matching
    /// LSH buckets, re-ranked by exact Hamming similarity.
    pub fn search(&self, query: &BinaryHV, top_k: usize) -> Vec<SearchResult> {
        let mut candidate_indices = std::collections::HashSet::new();
        for table in &self.tables {
            let hash = table.hash(query);
            if let Some(bucket) = table.buckets.get(&hash) {
                for &idx in bucket {
                    candidate_indices.insert(idx);
                }
            }
        }

        let mut scored: Vec<(String, f64)> = candidate_indices
            .into_iter()
            .map(|idx| {
                let (label, hv) = &self.corpus[idx];
                let sim = hv.hamming_similarity(query).unwrap_or(0.0);
                (label.clone(), sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(i, (label, similarity))| SearchResult {
                label,
                similarity,
                rank: i + 1,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// KnnClassifier — k-Nearest Neighbor classifier in HV space
// ---------------------------------------------------------------------------

/// k-NN classifier using Hamming similarity on BinaryHV space.
pub struct KnnClassifier {
    corpus: Vec<(String, BinaryHV)>,
    k: usize,
    _dim: usize,
}

impl KnnClassifier {
    pub fn new(dim: usize, k: usize) -> Self {
        Self {
            corpus: Vec::new(),
            k,
            _dim: dim,
        }
    }

    /// Add a labeled training example.
    pub fn insert(&mut self, label: String, hv: BinaryHV) {
        self.corpus.push((label, hv));
    }

    /// Predict class label by majority vote of k nearest neighbors.
    pub fn predict(&self, query: &BinaryHV) -> Option<PredictionResult> {
        if self.corpus.is_empty() {
            return None;
        }

        let mut scored: Vec<(&str, f64)> = self
            .corpus
            .iter()
            .map(|(label, hv)| {
                let sim = hv.hamming_similarity(query).unwrap_or(0.0);
                (label.as_str(), sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.k);

        // Majority vote with confidence = weighted similarity
        let mut votes: std::collections::HashMap<&str, (usize, f64)> =
            std::collections::HashMap::new();
        for (label, sim) in &scored {
            let entry = votes.entry(label).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += sim;
        }

        votes
            .into_iter()
            .max_by(|a, b| {
                a.1 .0
                    .cmp(&b.1 .0)
                    .then(a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(class, (count, total_sim))| PredictionResult {
                class: class.to_string(),
                confidence: total_sim / count as f64,
            })
    }

    pub fn len(&self) -> usize {
        self.corpus.len()
    }
}

// ---------------------------------------------------------------------------
// NexaCrypto — seed-based encryption for encoded hypervectors
// ---------------------------------------------------------------------------

/// Seed-based encryption for hypervectors. XORs the HV with a deterministic
/// PRNG stream generated from the seed, providing symmetric encryption.
/// The same seed encrypts and decrypts (XOR is its own inverse).
pub struct NexaCrypto;

impl NexaCrypto {
    /// Encrypt a BinaryHV using a seed-derived PRNG stream.
    pub fn encrypt(hv: &BinaryHV, seed: u64) -> BinaryHV {
        Self::xor_with_stream(hv, seed)
    }

    /// Decrypt a BinaryHV using the same seed (XOR is self-inverse).
    pub fn decrypt(hv: &BinaryHV, seed: u64) -> BinaryHV {
        Self::xor_with_stream(hv, seed)
    }

    /// Verify that a decrypted HV matches the original.
    pub fn verify(original: &BinaryHV, encrypted: &BinaryHV, seed: u64) -> bool {
        let decrypted = Self::decrypt(encrypted, seed);
        original.hamming_similarity(&decrypted).unwrap_or(0.0) > 0.999
    }

    fn xor_with_stream(hv: &BinaryHV, seed: u64) -> BinaryHV {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let words = hv.words();
        let encrypted_words: Vec<u64> = words.iter().map(|&w| w ^ rng.gen::<u64>()).collect();
        BinaryHV::from_words(encrypted_words, hv.dim()).expect("encryption failed")
    }
}

// ---------------------------------------------------------------------------
// TrainingPipeline — end-to-end encode → train → evaluate
// ---------------------------------------------------------------------------

/// Result from a training pipeline evaluation.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub accuracy: f64,
    pub total: usize,
    pub correct: usize,
    pub per_class: Vec<(String, f64)>,
}

/// End-to-end training pipeline using HdcClassifier.
pub struct TrainingPipeline {
    dim: usize,
}

impl TrainingPipeline {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Train an HDC classifier from labeled examples and evaluate on test set.
    /// Returns (trained classifier, evaluation results).
    pub fn train_and_evaluate(
        &self,
        train: &[(String, BinaryHV)],
        test: &[(String, BinaryHV)],
    ) -> (HdcClassifier, EvalResult) {
        let mut classifier = HdcClassifier::new(self.dim);

        // Group training examples by class
        let mut by_class: std::collections::HashMap<String, Vec<&BinaryHV>> =
            std::collections::HashMap::new();
        for (label, hv) in train {
            by_class.entry(label.clone()).or_default().push(hv);
        }

        // Train each class
        for (class, examples) in &by_class {
            classifier.train(class, examples);
        }

        // Evaluate
        let eval = self.evaluate(&classifier, test);
        (classifier, eval)
    }

    /// Evaluate classifier accuracy on a test set.
    pub fn evaluate(&self, classifier: &HdcClassifier, test: &[(String, BinaryHV)]) -> EvalResult {
        let mut correct = 0;
        let mut per_class_correct: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();

        for (true_label, hv) in test {
            let entry = per_class_correct
                .entry(true_label.clone())
                .or_insert((0, 0));
            entry.1 += 1;

            if let Some(pred) = classifier.predict(hv) {
                if pred.class == *true_label {
                    correct += 1;
                    entry.0 += 1;
                }
            }
        }

        let per_class: Vec<(String, f64)> = per_class_correct
            .into_iter()
            .map(|(class, (c, t))| (class, if t > 0 { c as f64 / t as f64 } else { 0.0 }))
            .collect();

        EvalResult {
            accuracy: if test.is_empty() {
                0.0
            } else {
                correct as f64 / test.len() as f64
            },
            total: test.len(),
            correct,
            per_class,
        }
    }

    /// Train with iterative retraining for misclassified examples.
    pub fn train_with_retraining(
        &self,
        train: &[(String, BinaryHV)],
        test: &[(String, BinaryHV)],
        retrain_epochs: usize,
    ) -> (HdcClassifier, EvalResult) {
        let (mut classifier, _) = self.train_and_evaluate(train, test);

        for _ in 0..retrain_epochs {
            for (true_label, hv) in train {
                if let Some(pred) = classifier.predict(hv) {
                    if pred.class != *true_label {
                        classifier.retrain(hv, &pred.class, true_label);
                    }
                }
            }
        }

        let eval = self.evaluate(&classifier, test);
        (classifier, eval)
    }
}

// ---------------------------------------------------------------------------
// EnsembleEvaluator — evaluate combinations of classifiers
// ---------------------------------------------------------------------------

/// Evaluate ensemble combinations of multiple classifiers via majority voting.
pub struct EnsembleEvaluator;

impl EnsembleEvaluator {
    /// Evaluate all non-empty subsets of classifiers (up to 2^n - 1 combinations).
    /// Each combination uses majority vote across its member classifiers.
    /// Returns results sorted by accuracy descending.
    pub fn evaluate_combinations(
        classifiers: &[&HdcClassifier],
        test: &[(String, BinaryHV)],
    ) -> Vec<(Vec<usize>, f64)> {
        let n = classifiers.len();
        if n > 16 {
            // Cap at 16 to avoid 2^n explosion
            return Vec::new();
        }

        let total_combos = (1u32 << n) - 1;
        let mut results: Vec<(Vec<usize>, f64)> = Vec::new();

        for mask in 1..=total_combos {
            let indices: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let mut correct = 0;

            for (true_label, hv) in test {
                // Collect predictions from each classifier in the combo
                let mut vote_counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for &idx in &indices {
                    if let Some(pred) = classifiers[idx].predict(hv) {
                        *vote_counts.entry(pred.class).or_insert(0) += 1;
                    }
                }

                // Majority vote
                if let Some((winner, _)) = vote_counts.into_iter().max_by_key(|(_, c)| *c) {
                    if winner == *true_label {
                        correct += 1;
                    }
                }
            }

            let accuracy = if test.is_empty() {
                0.0
            } else {
                correct as f64 / test.len() as f64
            };
            results.push((indices, accuracy));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use nexa_core::BinaryHV;

    const DIM: usize = 1000;

    #[test]
    fn vector_search_finds_exact_match() {
        let mut search = VectorSearch::new(DIM);

        for i in 0..100u64 {
            let v = BinaryHV::random(DIM, i).unwrap();
            search.insert(format!("vec_{}", i), v);
        }

        // Search for vec_42
        let query = BinaryHV::random(DIM, 42).unwrap();
        let results = search.search(&query, 5);

        assert!(!results.is_empty());
        assert_eq!(results[0].label, "vec_42");
        assert!((results[0].similarity - 1.0).abs() < 1e-10);
        assert_eq!(results[0].rank, 1);
    }

    #[test]
    fn classifier_trains_and_predicts() {
        let mut classifier = HdcClassifier::new(DIM);

        // Create 3 classes with correlated vectors (corrupted from a base)
        let base_a = BinaryHV::random(DIM, 1000).unwrap();
        let base_b = BinaryHV::random(DIM, 2000).unwrap();
        let base_c = BinaryHV::random(DIM, 3000).unwrap();

        let class_a: Vec<BinaryHV> = (0..10).map(|i| base_a.corrupt(0.1, i)).collect();
        let class_b: Vec<BinaryHV> = (0..10).map(|i| base_b.corrupt(0.1, 100 + i)).collect();
        let class_c: Vec<BinaryHV> = (0..10).map(|i| base_c.corrupt(0.1, 200 + i)).collect();

        let refs_a: Vec<&BinaryHV> = class_a.iter().collect();
        let refs_b: Vec<&BinaryHV> = class_b.iter().collect();
        let refs_c: Vec<&BinaryHV> = class_c.iter().collect();

        classifier.train("A", &refs_a);
        classifier.train("B", &refs_b);
        classifier.train("C", &refs_c);

        // Test with slightly corrupted versions of the base vectors
        let test_a = base_a.corrupt(0.15, 999);
        let test_b = base_b.corrupt(0.15, 998);
        let test_c = base_c.corrupt(0.15, 997);

        assert_eq!(classifier.predict(&test_a).unwrap().class, "A");
        assert_eq!(classifier.predict(&test_b).unwrap().class, "B");
        assert_eq!(classifier.predict(&test_c).unwrap().class, "C");
    }

    #[test]
    fn classifier_retrain_improves() {
        let mut classifier = HdcClassifier::new(DIM);

        let base_a = BinaryHV::random(DIM, 5000).unwrap();
        let base_b = BinaryHV::random(DIM, 6000).unwrap();

        let class_a: Vec<BinaryHV> = (0..10).map(|i| base_a.corrupt(0.1, i)).collect();
        let class_b: Vec<BinaryHV> = (0..10).map(|i| base_b.corrupt(0.1, 100 + i)).collect();

        let refs_a: Vec<&BinaryHV> = class_a.iter().collect();
        let refs_b: Vec<&BinaryHV> = class_b.iter().collect();

        classifier.train("A", &refs_a);
        classifier.train("B", &refs_b);

        // Query close to B — will be classified as B
        let query = base_b.corrupt(0.2, 777);
        let pred = classifier.predict(&query).unwrap();
        assert_eq!(pred.class, "B", "Expected initial classification as B");

        // Retrain: tell the classifier this was actually A
        classifier.retrain(&query, "B", "A");

        let pred_after = classifier.predict(&query).unwrap();
        assert_eq!(
            pred_after.class, "A",
            "Expected correct classification after retrain"
        );
    }

    #[test]
    fn anomaly_detector_catches_outliers() {
        let mut detector = AnomalyDetector::new(DIM, 0.7);

        // Add similar reference vectors (corruptions of the same base)
        let base = BinaryHV::random(DIM, 42).unwrap();
        for i in 0..10 {
            detector.add_reference(base.corrupt(0.05, i));
        }

        // A close vector should NOT be anomalous
        let normal = base.corrupt(0.1, 999);
        assert!(!detector.is_anomaly(&normal));
        assert!(detector.anomaly_score(&normal) < 0.3);

        // A completely random vector should be anomalous (~0.5 similarity)
        let outlier = BinaryHV::random(DIM, 9999).unwrap();
        assert!(detector.is_anomaly(&outlier));
        assert!(detector.anomaly_score(&outlier) > 0.3);
    }

    #[test]
    fn clusterer_separates_distinct_groups() {
        let clusterer = Clusterer::new(DIM);

        // Create 3 well-separated groups of correlated vectors
        let base_0 = BinaryHV::random(DIM, 100).unwrap();
        let base_1 = BinaryHV::random(DIM, 200).unwrap();
        let base_2 = BinaryHV::random(DIM, 300).unwrap();

        let mut all_vecs: Vec<BinaryHV> = Vec::new();
        let mut true_labels: Vec<usize> = Vec::new();

        for i in 0..10u64 {
            all_vecs.push(base_0.corrupt(0.05, 1000 + i));
            true_labels.push(0);
        }
        for i in 0..10u64 {
            all_vecs.push(base_1.corrupt(0.05, 2000 + i));
            true_labels.push(1);
        }
        for i in 0..10u64 {
            all_vecs.push(base_2.corrupt(0.05, 3000 + i));
            true_labels.push(2);
        }

        let refs: Vec<&BinaryHV> = all_vecs.iter().collect();
        let assignments = clusterer.cluster(&refs, 3, 20, 42);

        // Verify: within each true group, most vectors should share a cluster id
        for group in 0..3 {
            let group_clusters: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(i, _)| true_labels[*i] == group)
                .map(|(_, a)| a.cluster_id)
                .collect();

            // Find the most common cluster in this group
            let mut counts = [0usize; 3];
            for &c in &group_clusters {
                counts[c] += 1;
            }
            let majority = *counts.iter().max().unwrap();
            assert!(
                majority >= 7,
                "Group {} should have at least 7/10 in same cluster, got {}",
                group,
                majority
            );
        }
    }

    #[test]
    fn homomorphic_binding_preserves_structure() {
        let a = BinaryHV::random(DIM, 10).unwrap();
        let b = BinaryHV::random(DIM, 20).unwrap();

        // Permutation is a homomorphic transform for XOR binding:
        // permute(a XOR b) == permute(a) XOR permute(b)
        let sim = HomomorphicOps::verify_binding_homomorphism(&a, &b, |v| v.permute(3));

        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Permutation should be perfectly homomorphic for XOR binding, got {}",
            sim
        );
    }

    #[test]
    fn lsh_index_finds_exact_match() {
        let mut index = LshIndex::new(DIM, 10, 12, 42);

        for i in 0..100u64 {
            let v = BinaryHV::random(DIM, i).unwrap();
            index.insert(format!("vec_{}", i), v);
        }

        let query = BinaryHV::random(DIM, 42).unwrap();
        let results = index.search(&query, 5);

        // LSH should find the exact match (with high probability)
        assert!(!results.is_empty());
        assert_eq!(results[0].label, "vec_42");
        assert!((results[0].similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn knn_classifier_predicts_correctly() {
        let mut knn = KnnClassifier::new(DIM, 3);

        let base_a = BinaryHV::random(DIM, 1000).unwrap();
        let base_b = BinaryHV::random(DIM, 2000).unwrap();

        for i in 0..20u64 {
            knn.insert("A".to_string(), base_a.corrupt(0.1, i));
            knn.insert("B".to_string(), base_b.corrupt(0.1, 100 + i));
        }

        let test_a = base_a.corrupt(0.15, 999);
        let test_b = base_b.corrupt(0.15, 998);

        assert_eq!(knn.predict(&test_a).unwrap().class, "A");
        assert_eq!(knn.predict(&test_b).unwrap().class, "B");
    }

    #[test]
    fn crypto_encrypt_decrypt_roundtrip() {
        let original = BinaryHV::random(DIM, 42).unwrap();
        let seed = 123456789u64;

        let encrypted = NexaCrypto::encrypt(&original, seed);
        // Encrypted should be different from original
        let sim = original.hamming_similarity(&encrypted).unwrap();
        assert!((sim - 0.5).abs() < 0.1, "Encrypted should look random, got sim={sim}");

        let decrypted = NexaCrypto::decrypt(&encrypted, seed);
        assert!(NexaCrypto::verify(&original, &encrypted, seed));
        assert_eq!(original.hamming_similarity(&decrypted).unwrap(), 1.0);
    }

    #[test]
    fn crypto_wrong_seed_fails() {
        let original = BinaryHV::random(DIM, 42).unwrap();
        let encrypted = NexaCrypto::encrypt(&original, 111);
        assert!(!NexaCrypto::verify(&original, &encrypted, 222));
    }

    #[test]
    fn training_pipeline_basic() {
        let pipeline = TrainingPipeline::new(DIM);
        let base_a = BinaryHV::random(DIM, 100).unwrap();
        let base_b = BinaryHV::random(DIM, 200).unwrap();

        let train: Vec<(String, BinaryHV)> = (0..20u64)
            .map(|i| ("A".to_string(), base_a.corrupt(0.1, i)))
            .chain((0..20u64).map(|i| ("B".to_string(), base_b.corrupt(0.1, 100 + i))))
            .collect();

        let test: Vec<(String, BinaryHV)> = (0..5u64)
            .map(|i| ("A".to_string(), base_a.corrupt(0.15, 500 + i)))
            .chain((0..5u64).map(|i| ("B".to_string(), base_b.corrupt(0.15, 600 + i))))
            .collect();

        let (_, eval) = pipeline.train_and_evaluate(&train, &test);
        assert!(eval.accuracy >= 0.8, "Expected good accuracy, got {}", eval.accuracy);
        assert_eq!(eval.total, 10);
    }

    #[test]
    fn ensemble_evaluator_basic() {
        let base_a = BinaryHV::random(DIM, 100).unwrap();
        let base_b = BinaryHV::random(DIM, 200).unwrap();

        let mut c1 = HdcClassifier::new(DIM);
        let mut c2 = HdcClassifier::new(DIM);

        let a_examples: Vec<BinaryHV> = (0..10u64).map(|i| base_a.corrupt(0.1, i)).collect();
        let b_examples: Vec<BinaryHV> = (0..10u64).map(|i| base_b.corrupt(0.1, 50 + i)).collect();

        let a_refs: Vec<&BinaryHV> = a_examples.iter().collect();
        let b_refs: Vec<&BinaryHV> = b_examples.iter().collect();

        c1.train("A", &a_refs);
        c1.train("B", &b_refs);
        c2.train("A", &a_refs);
        c2.train("B", &b_refs);

        let test: Vec<(String, BinaryHV)> = vec![
            ("A".to_string(), base_a.corrupt(0.15, 999)),
            ("B".to_string(), base_b.corrupt(0.15, 998)),
        ];

        let results = EnsembleEvaluator::evaluate_combinations(&[&c1, &c2], &test);
        assert!(!results.is_empty());
        // Best combo should have high accuracy
        assert!(results[0].1 >= 0.5);
    }
}
