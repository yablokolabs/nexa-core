mod error;
mod hypervector;
mod simd;
mod format;
mod similarity;

pub use error::NexaError;
pub use hypervector::{BinaryHV, BipolarHV, RealHV, SparseHV, Dimension};
pub use similarity::SimilarityMetric;
pub use format::{NexaHeader, NexaFormat, NEXA_MAGIC, NEXA_MAGIC_END};

pub const DEFAULT_DIMENSION: usize = 10_000;
