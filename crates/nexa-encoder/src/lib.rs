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
    Image,
    Audio,
    Sensor,
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

    // -- Image --------------------------------------------------------------

    /// Encode a grayscale image as a hypervector.
    ///
    /// `pixels` is a flat row-major array of pixel values (0-255).
    /// `width` and `height` define the image dimensions.
    /// Divides the image into non-overlapping patches and encodes each patch
    /// with positional binding, then bundles all patches.
    pub fn encode_image(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<BinaryHV> {
        if pixels.is_empty() || width == 0 || height == 0 {
            return Err(NexaError::EmptyInput);
        }
        if pixels.len() != width * height {
            return Err(NexaError::EncodingError(format!(
                "pixel count {} != {}x{}", pixels.len(), width, height
            )));
        }

        let patch_size = 4.min(width).min(height).max(1);
        let patches_x = (width + patch_size - 1) / patch_size;
        let patches_y = (height + patch_size - 1) / patch_size;
        let mut patch_hvs = Vec::new();

        for py in 0..patches_y {
            for px in 0..patches_x {
                let patch_idx = py * patches_x + px;
                let mut patch_tokens = Vec::new();
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let y = py * patch_size + dy;
                        let x = px * patch_size + dx;
                        if y < height && x < width {
                            let pixel = pixels[y * width + x];
                            // Quantize to 16 levels for manageable codebook size
                            let level = pixel / 16;
                            patch_tokens.push(format!("px_{}", level));
                        }
                    }
                }
                let token_refs: Vec<&str> = patch_tokens.iter().map(|s| s.as_str()).collect();
                let patch_hv = SequenceEncoder::encode(&mut self.codebook, &token_refs)?;
                patch_hvs.push(patch_hv.permute(patch_idx as isize));
            }
        }

        let refs: Vec<&BinaryHV> = patch_hvs.iter().collect();
        let hv = BinaryHV::bundle(&refs)?;
        let record = EncodingRecord {
            id: format!("image_{}", self.records.len()),
            original_data: pixels.to_vec(),
            data_type: DataType::Image,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- Audio --------------------------------------------------------------

    /// Encode audio samples as a hypervector.
    ///
    /// `samples` is a flat array of audio sample values (e.g., i16 range as bytes).
    /// `frame_size` is the number of samples per frame (e.g., 256).
    /// Each frame is encoded as a byte sequence, bound with temporal position, then bundled.
    pub fn encode_audio(&mut self, samples: &[u8], frame_size: usize) -> Result<BinaryHV> {
        if samples.is_empty() || frame_size == 0 {
            return Err(NexaError::EmptyInput);
        }

        let num_frames = (samples.len() + frame_size - 1) / frame_size;
        let mut frame_hvs = Vec::new();

        for f in 0..num_frames {
            let start = f * frame_size;
            let end = (start + frame_size).min(samples.len());
            let frame = &samples[start..end];

            // Encode frame: quantize bytes into spectral-band tokens
            // Group every 4 bytes and compute average for compression
            let chunk_size = 4.min(frame.len()).max(1);
            let mut tokens = Vec::new();
            for chunk in frame.chunks(chunk_size) {
                let avg: u16 = chunk.iter().map(|&b| b as u16).sum::<u16>() / chunk.len() as u16;
                tokens.push(format!("af_{}", avg / 8)); // 32 quantization levels
            }
            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            let frame_hv = SequenceEncoder::encode(&mut self.codebook, &token_refs)?;
            frame_hvs.push(frame_hv.permute(f as isize));
        }

        let refs: Vec<&BinaryHV> = frame_hvs.iter().collect();
        let hv = BinaryHV::bundle(&refs)?;
        let record = EncodingRecord {
            id: format!("audio_{}", self.records.len()),
            original_data: samples.to_vec(),
            data_type: DataType::Audio,
        };
        self.records.push((hv.clone(), record));
        Ok(hv)
    }

    // -- Sensor -------------------------------------------------------------

    /// Encode multi-channel sensor data as a hypervector.
    ///
    /// `channels` is a slice of named channels, each containing a time series of f32 values.
    /// Each channel is encoded by quantizing values, binding with channel identity,
    /// then bundling across channels and timesteps.
    pub fn encode_sensor(
        &mut self,
        channels: &[(&str, &[f32])],
    ) -> Result<BinaryHV> {
        if channels.is_empty() {
            return Err(NexaError::EmptyInput);
        }

        let max_len = channels.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Err(NexaError::EmptyInput);
        }

        let mut timestep_hvs = Vec::new();

        for t in 0..max_len {
            let mut channel_hvs = Vec::new();
            for (name, values) in channels {
                if t < values.len() {
                    let channel_hv = self.codebook.get_or_insert(&format!("__ch_{}__", name)).clone();
                    // Quantize f32 to 64 levels in [-1, 1] range
                    let clamped = values[t].clamp(-1.0, 1.0);
                    let level = ((clamped + 1.0) * 31.5) as u8;
                    let val_hv = self.codebook.get_or_insert(&format!("sv_{}", level)).clone();
                    channel_hvs.push(channel_hv.bind(&val_hv)?);
                }
            }
            if !channel_hvs.is_empty() {
                let refs: Vec<&BinaryHV> = channel_hvs.iter().collect();
                let bundled = BinaryHV::bundle(&refs)?;
                timestep_hvs.push(bundled.permute(t as isize));
            }
        }

        let refs: Vec<&BinaryHV> = timestep_hvs.iter().collect();
        let hv = BinaryHV::bundle(&refs)?;

        // Serialize channel info for the record
        let sensor_data: Vec<u8> = channels
            .iter()
            .flat_map(|(name, vals)| {
                let mut bytes = name.as_bytes().to_vec();
                bytes.push(b':');
                bytes.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
                bytes.push(b';');
                bytes
            })
            .collect();

        let record = EncodingRecord {
            id: format!("sensor_{}", self.records.len()),
            original_data: sensor_data,
            data_type: DataType::Sensor,
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

    #[test]
    fn image_encode_deterministic() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        // 8x8 grayscale image
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let hv1 = enc1.encode_image(&pixels, 8, 8).unwrap();
        let hv2 = enc2.encode_image(&pixels, 8, 8).unwrap();
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn image_different_images_dissimilar() {
        let mut enc = NexaEncoder::new(DIM, SEED);
        let bright: Vec<u8> = vec![200u8; 64];
        let dark: Vec<u8> = vec![10u8; 64];
        let hv1 = enc.encode_image(&bright, 8, 8).unwrap();
        let hv2 = enc.encode_image(&dark, 8, 8).unwrap();
        let sim = hv1.hamming_similarity(&hv2).unwrap();
        // Different images should not be identical
        assert!(sim < 0.95, "Different images should be dissimilar, got {sim}");
    }

    #[test]
    fn audio_encode_deterministic() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        let samples: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let hv1 = enc1.encode_audio(&samples, 64).unwrap();
        let hv2 = enc2.encode_audio(&samples, 64).unwrap();
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn sensor_encode_deterministic() {
        let mut enc1 = NexaEncoder::new(DIM, SEED);
        let mut enc2 = NexaEncoder::new(DIM, SEED);
        let accel = vec![0.1f32, 0.5, -0.3, 0.8];
        let gyro = vec![0.0f32, -0.2, 0.7, -0.5];
        let channels = vec![("accel_x", accel.as_slice()), ("gyro_z", gyro.as_slice())];
        let hv1 = enc1.encode_sensor(&channels).unwrap();
        let hv2 = enc2.encode_sensor(&channels).unwrap();
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn sensor_different_data_dissimilar() {
        let mut enc = NexaEncoder::new(DIM, SEED);
        let ch1 = vec![0.9f32; 10];
        let ch2 = vec![-0.9f32; 10];
        let hv1 = enc.encode_sensor(&[("x", ch1.as_slice())]).unwrap();
        let hv2 = enc.encode_sensor(&[("x", ch2.as_slice())]).unwrap();
        let sim = hv1.hamming_similarity(&hv2).unwrap();
        assert!(sim < 0.95, "Different sensor data should be dissimilar, got {sim}");
    }
}
