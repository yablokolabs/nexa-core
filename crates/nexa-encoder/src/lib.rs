use nexa_core::{BinaryHV, NexaError, NexaFormat, NexaHeader};
use nexa_hdc::{Codebook, SequenceEncoder};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub type Result<T> = std::result::Result<T, NexaError>;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DataType {
    Text,
    Json,
    Csv,
    Binary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingRecord {
    pub id: String,
    pub original_data: Vec<u8>,
    pub data_type: DataType,
}

// ---------------------------------------------------------------------------
// NexaEncoder
// ---------------------------------------------------------------------------

pub struct NexaEncoder {
    codebook: Codebook,
    dim: usize,
    records: Vec<(BinaryHV, EncodingRecord)>,
}

impl NexaEncoder {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            codebook: Codebook::new(dim, seed),
            dim,
            records: Vec::new(),
        }
    }

    // -- Text ---------------------------------------------------------------

    /// Character-level sequence encoding (no record stored).
    fn encode_text_inner(&mut self, text: &str) -> Result<BinaryHV> {
        if text.is_empty() {
            return Ok(self.codebook.get_or_insert("__empty__").clone());
        }
        let mut positioned = Vec::new();
        for (i, c) in text.chars().enumerate() {
            let char_hv = self.codebook.get_or_insert(&c.to_string()).clone();
            positioned.push(char_hv.permute(i as isize));
        }
        let refs: Vec<&BinaryHV> = positioned.iter().collect();
        BinaryHV::bundle(&refs)
    }

    pub fn encode_text(&mut self, text: &str) -> Result<BinaryHV> {
        let hv = self.encode_text_inner(text)?;
        let record = EncodingRecord {
            id: format!("text_{}", self.records.len()),
            original_data: text.as_bytes().to_vec(),
            data_type: DataType::Text,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- JSON ---------------------------------------------------------------

    fn encode_json_value(&mut self, value: &serde_json::Value) -> Result<BinaryHV> {
        use serde_json::Value;
        match value {
            Value::Object(map) => {
                if map.is_empty() {
                    return Ok(self.codebook.get_or_insert("__empty_object__").clone());
                }
                let mut pairs = Vec::new();
                for (key, val) in map {
                    let key_hv = self.codebook.get_or_insert(key).clone();
                    let val_hv = self.encode_json_value(val)?;
                    pairs.push(key_hv.bind(&val_hv)?);
                }
                let refs: Vec<&BinaryHV> = pairs.iter().collect();
                BinaryHV::bundle(&refs)
            }
            Value::Array(arr) => {
                if arr.is_empty() {
                    return Ok(self.codebook.get_or_insert("__empty_array__").clone());
                }
                let mut positioned = Vec::new();
                for (i, elem) in arr.iter().enumerate() {
                    let elem_hv = self.encode_json_value(elem)?;
                    positioned.push(elem_hv.permute(i as isize));
                }
                let refs: Vec<&BinaryHV> = positioned.iter().collect();
                BinaryHV::bundle(&refs)
            }
            Value::String(s) => self.encode_text_inner(s),
            Value::Number(n) => self.encode_text_inner(&n.to_string()),
            Value::Bool(b) => self.encode_text_inner(if *b { "true" } else { "false" }),
            Value::Null => Ok(self.codebook.get_or_insert("__null__").clone()),
        }
    }

    pub fn encode_json(&mut self, json_str: &str) -> Result<BinaryHV> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| NexaError::EncodingError(e.to_string()))?;
        let hv = self.encode_json_value(&value)?;
        let record = EncodingRecord {
            id: format!("json_{}", self.records.len()),
            original_data: json_str.as_bytes().to_vec(),
            data_type: DataType::Json,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- CSV ----------------------------------------------------------------

    pub fn encode_csv_row(&mut self, fields: &[&str]) -> Result<BinaryHV> {
        if fields.is_empty() {
            return Err(NexaError::EmptyInput);
        }
        let mut pairs = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            let col_hv = self
                .codebook
                .get_or_insert(&format!("__col_{}__", i))
                .clone();
            let val_hv = self.codebook.get_or_insert(field).clone();
            pairs.push(col_hv.bind(&val_hv)?);
        }
        let refs: Vec<&BinaryHV> = pairs.iter().collect();
        let hv = BinaryHV::bundle(&refs)?;
        let record = EncodingRecord {
            id: format!("csv_{}", self.records.len()),
            original_data: fields.join(",").into_bytes(),
            data_type: DataType::Csv,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- Binary -------------------------------------------------------------

    pub fn encode_bytes(&mut self, data: &[u8]) -> Result<BinaryHV> {
        if data.is_empty() {
            return Err(NexaError::EmptyInput);
        }
        let tokens: Vec<String> = data.iter().map(|b| format!("byte_{}", b)).collect();
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let hv = SequenceEncoder::encode(&mut self.codebook, &token_refs)?;
        let record = EncodingRecord {
            id: format!("bytes_{}", self.records.len()),
            original_data: data.to_vec(),
            data_type: DataType::Binary,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- Accessors ----------------------------------------------------------

    pub fn records(&self) -> &[(BinaryHV, EncodingRecord)] {
        &self.records
    }

    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    pub fn codebook_mut(&mut self) -> &mut Codebook {
        &mut self.codebook
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// .nexa file I/O
// ---------------------------------------------------------------------------

pub fn write_nexa_file(path: &Path, encoder: &NexaEncoder, vectors: &[&BinaryHV]) -> Result<()> {
    let records: Vec<&EncodingRecord> = encoder.records().iter().map(|(_, r)| r).collect();
    let metadata =
        serde_json::to_string(&records).map_err(|e| NexaError::Serialization(e.to_string()))?;
    let header = NexaHeader::new(encoder.dim() as u32, vectors.len() as u32)
        .with_metadata(metadata);
    let raw_vectors: Vec<Vec<u8>> = vectors
        .iter()
        .map(|v| v.words().iter().flat_map(|w| w.to_le_bytes()).collect())
        .collect();
    let mut file = std::fs::File::create(path)?;
    NexaFormat::write(&mut file, &header, &raw_vectors)?;
    Ok(())
}

pub fn read_nexa_file(path: &Path) -> Result<(NexaHeader, Vec<Vec<u8>>)> {
    let mut file = std::fs::File::open(path)?;
    NexaFormat::read(&mut file)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 10_000;
    const SEED: u64 = 42;

    #[test]
    fn text_encode_deterministic() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        let hv1 = enc1.encode_text("hello").unwrap();
        let hv2 = enc2.encode_text("hello").unwrap();
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn different_texts_produce_dissimilar_encodings() {
        let mut enc = NexaEncoder::new(DIM, SEED);
        let hv1 = enc.encode_text("hello").unwrap();
        let hv2 = enc.encode_text("world").unwrap();
        let sim = hv1.hamming_similarity(&hv2).unwrap();
        assert!(
            (sim - 0.5).abs() < 0.1,
            "Similarity {sim} should be near 0.5 for unrelated texts"
        );
    }

    #[test]
    fn json_encode_preserves_structure() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        let hv_a1 = enc1.encode_json(r#"{"a":1}"#).unwrap();
        let hv_a2 = enc2.encode_json(r#"{"a":1}"#).unwrap();
        assert_eq!(hv_a1, hv_a2, "Same JSON must produce identical encoding");

        let mut enc3 = NexaEncoder::new(DIM, SEED);
        let hv_b = enc3.encode_json(r#"{"b":2}"#).unwrap();
        let sim = hv_a1.hamming_similarity(&hv_b).unwrap();
        assert!(
            (sim - 0.5).abs() < 0.15,
            "Different JSON objects should be dissimilar, got {sim}"
        );
    }

    #[test]
    fn csv_row_encode_deterministic() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        let hv1 = enc1.encode_csv_row(&["a", "b", "c"]).unwrap();
        let hv2 = enc2.encode_csv_row(&["a", "b", "c"]).unwrap();
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn bytes_encode_roundtrip_recorded() {
        let mut enc = NexaEncoder::new(DIM, SEED);
        let data = b"hello bytes";
        let _hv = enc.encode_bytes(data).unwrap();
        let records = enc.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1.original_data, data.to_vec());
        assert!(matches!(records[0].1.data_type, DataType::Binary));
    }

    #[test]
    fn nexa_file_write_read_roundtrip() {
        let mut enc = NexaEncoder::new(DIM, SEED);
        let hv1 = enc.encode_text("hello").unwrap();
        let hv2 = enc.encode_text("world").unwrap();

        let path = std::env::temp_dir().join("nexa_encoder_roundtrip_test.nexa");
        write_nexa_file(&path, &enc, &[&hv1, &hv2]).unwrap();

        let (header, raw_vectors) = read_nexa_file(&path).unwrap();
        assert_eq!(header.dimension, DIM as u32);
        assert_eq!(header.vector_count, 2);
        assert_eq!(raw_vectors.len(), 2);

        let expected1: Vec<u8> = hv1.words().iter().flat_map(|w| w.to_le_bytes()).collect();
        let expected2: Vec<u8> = hv2.words().iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(raw_vectors[0], expected1);
        assert_eq!(raw_vectors[1], expected2);

        let _ = std::fs::remove_file(&path);
    }
}
