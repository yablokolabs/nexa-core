use thiserror::Error;

#[derive(Error, Debug)]
pub enum NexaError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("zero dimension vectors are not allowed")]
    ZeroDimension,

    #[error("empty input")]
    EmptyInput,

    #[error("encoding error: {0}")]
    EncodingError(String),

    #[error("decoding error: {0}")]
    DecodingError(String),

    #[error("format error: {0}")]
    FormatError(String),

    #[error("checksum mismatch: expected {expected:#010x}, got {got:#010x}")]
    ChecksumMismatch { expected: u32, got: u32 },

    #[error("invalid magic bytes")]
    InvalidMagic,

    #[error("unsupported version: {0}")]
    UnsupportedVersion(u16),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),
}

pub type Result<T> = std::result::Result<T, NexaError>;
