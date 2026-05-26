//! NexaCore — Python bindings via PyO3.
//!
//! Provides Pythonic access to hypervector operations, encoding, compression,
//! search, and classification.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nexa_core::BinaryHV;
use nexa_encoder::NexaEncoder;
use nexa_runtime::{VectorSearch, HdcClassifier, KnnClassifier, NexaCrypto, LshIndex};

// ---------------------------------------------------------------------------
// PyBinaryHV — wraps BinaryHV for Python
// ---------------------------------------------------------------------------

#[pyclass]
struct PyBinaryHV {
    inner: BinaryHV,
}

#[pymethods]
impl PyBinaryHV {
    #[staticmethod]
    fn random(dim: usize, seed: u64) -> PyResult<Self> {
        BinaryHV::random(dim, seed)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn hamming_distance(&self, other: &PyBinaryHV) -> PyResult<u32> {
        self.inner
            .hamming_distance(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn hamming_similarity(&self, other: &PyBinaryHV) -> PyResult<f64> {
        self.inner
            .hamming_similarity(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn bind(&self, other: &PyBinaryHV) -> PyResult<Self> {
        self.inner
            .bind(&other.inner)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn permute(&self, amount: isize) -> Self {
        PyBinaryHV {
            inner: self.inner.permute(amount),
        }
    }

    fn popcount(&self) -> u32 {
        self.inner.popcount()
    }

    fn corrupt(&self, noise: f64, seed: u64) -> Self {
        PyBinaryHV {
            inner: self.inner.corrupt(noise, seed),
        }
    }

    fn encrypt(&self, seed: u64) -> Self {
        PyBinaryHV {
            inner: NexaCrypto::encrypt(&self.inner, seed),
        }
    }

    fn decrypt(&self, seed: u64) -> Self {
        PyBinaryHV {
            inner: NexaCrypto::decrypt(&self.inner, seed),
        }
    }
}

// ---------------------------------------------------------------------------
// PyEncoder — wraps NexaEncoder
// ---------------------------------------------------------------------------

#[pyclass]
struct PyEncoder {
    inner: NexaEncoder,
}

#[pymethods]
impl PyEncoder {
    #[new]
    fn new(dim: usize, seed: u64) -> Self {
        PyEncoder {
            inner: NexaEncoder::new(dim, seed),
        }
    }

    fn encode_text(&mut self, text: &str) -> PyResult<PyBinaryHV> {
        self.inner
            .encode_text(text)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn encode_json(&mut self, json_str: &str) -> PyResult<PyBinaryHV> {
        self.inner
            .encode_json(json_str)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn encode_csv_row(&mut self, fields: Vec<String>) -> PyResult<PyBinaryHV> {
        let refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
        self.inner
            .encode_csv_row(&refs)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn encode_bytes(&mut self, data: Vec<u8>) -> PyResult<PyBinaryHV> {
        self.inner
            .encode_bytes(&data)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn encode_image(&mut self, pixels: Vec<u8>, width: usize, height: usize) -> PyResult<PyBinaryHV> {
        self.inner
            .encode_image(&pixels, width, height)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn encode_audio(&mut self, samples: Vec<u8>, frame_size: usize) -> PyResult<PyBinaryHV> {
        self.inner
            .encode_audio(&samples, frame_size)
            .map(|hv| PyBinaryHV { inner: hv })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

// ---------------------------------------------------------------------------
// PyVectorSearch — wraps VectorSearch
// ---------------------------------------------------------------------------

#[pyclass]
struct PyVectorSearch {
    inner: VectorSearch,
}

#[pymethods]
impl PyVectorSearch {
    #[new]
    fn new(dim: usize) -> Self {
        PyVectorSearch {
            inner: VectorSearch::new(dim),
        }
    }

    fn insert(&mut self, label: String, vector: &PyBinaryHV) {
        self.inner.insert(label, vector.inner.clone());
    }

    fn search(&self, query: &PyBinaryHV, top_k: usize) -> Vec<(String, f64)> {
        self.inner
            .search(&query.inner, top_k)
            .into_iter()
            .map(|r| (r.label, r.similarity))
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ---------------------------------------------------------------------------
// PyLshIndex — wraps LshIndex
// ---------------------------------------------------------------------------

#[pyclass]
struct PyLshIndex {
    inner: LshIndex,
}

#[pymethods]
impl PyLshIndex {
    #[new]
    fn new(dim: usize, num_tables: usize, bits_per_hash: usize, seed: u64) -> Self {
        PyLshIndex {
            inner: LshIndex::new(dim, num_tables, bits_per_hash, seed),
        }
    }

    fn insert(&mut self, label: String, vector: &PyBinaryHV) {
        self.inner.insert(label, vector.inner.clone());
    }

    fn search(&self, query: &PyBinaryHV, top_k: usize) -> Vec<(String, f64)> {
        self.inner
            .search(&query.inner, top_k)
            .into_iter()
            .map(|r| (r.label, r.similarity))
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ---------------------------------------------------------------------------
// PyHdcClassifier — wraps HdcClassifier
// ---------------------------------------------------------------------------

#[pyclass]
struct PyHdcClassifier {
    inner: HdcClassifier,
}

#[pymethods]
impl PyHdcClassifier {
    #[new]
    fn new(dim: usize) -> Self {
        PyHdcClassifier {
            inner: HdcClassifier::new(dim),
        }
    }

    fn train(&mut self, class_name: &str, examples: Vec<PyRef<PyBinaryHV>>) {
        let hvs: Vec<&BinaryHV> = examples.iter().map(|e| &e.inner).collect();
        self.inner.train(class_name, &hvs);
    }

    fn predict(&self, query: &PyBinaryHV) -> Option<(String, f64)> {
        self.inner
            .predict(&query.inner)
            .map(|r| (r.class, r.confidence))
    }
}

// ---------------------------------------------------------------------------
// PyKnnClassifier — wraps KnnClassifier
// ---------------------------------------------------------------------------

#[pyclass]
struct PyKnnClassifier {
    inner: KnnClassifier,
}

#[pymethods]
impl PyKnnClassifier {
    #[new]
    fn new(dim: usize, k: usize) -> Self {
        PyKnnClassifier {
            inner: KnnClassifier::new(dim, k),
        }
    }

    fn insert(&mut self, label: String, vector: &PyBinaryHV) {
        self.inner.insert(label, vector.inner.clone());
    }

    fn predict(&self, query: &PyBinaryHV) -> Option<(String, f64)> {
        self.inner
            .predict(&query.inner)
            .map(|r| (r.class, r.confidence))
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn nexa_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBinaryHV>()?;
    m.add_class::<PyEncoder>()?;
    m.add_class::<PyVectorSearch>()?;
    m.add_class::<PyLshIndex>()?;
    m.add_class::<PyHdcClassifier>()?;
    m.add_class::<PyKnnClassifier>()?;
    Ok(())
}
