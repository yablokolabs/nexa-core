use nexa_core::BinaryHV;

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
    dim: usize,
}

impl VectorSearch {
    pub fn new(dim: usize) -> Self {
        Self {
            corpus: Vec::new(),
            dim,
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
    dim: usize,
}

impl AnomalyDetector {
    pub fn new(dim: usize, threshold: f64) -> Self {
        Self {
            reference: Vec::new(),
            threshold,
            dim,
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
    dim: usize,
}

impl Clusterer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
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
}
