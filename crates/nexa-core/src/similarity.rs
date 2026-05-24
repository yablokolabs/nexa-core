#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    Hamming,
    Cosine,
    DotProduct,
}
