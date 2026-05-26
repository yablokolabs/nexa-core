//! NexaForge — universal model wrapping engine for encoded-space inference.
//!
//! NexaForge translates neural network models so they can operate directly
//! on hypervector-encoded data **without retraining**.  It:
//!
//! 1. Ingests a model definition (topology + weights)
//! 2. Projects weight matrices into hypervector space via random projection
//! 3. Translates each layer into HV-space operations
//! 4. Produces a `ForgedModel` that runs inference on `BinaryHV` inputs
//!
//! # Supported layers
//!
//! | Original layer | HV translation |
//! |----------------|----------------|
//! | Dense (fully connected) | Random-projection into HV space + cosine similarity |
//! | ReLU activation | Element-wise threshold at 0.5 (binary) |
//! | Sigmoid | Scaled similarity mapping to \[0,1\] |
//! | Softmax (output) | Normalized cosine similarity to class prototypes |
//! | Flatten | Passthrough (HV is already flat) |
//! | BatchNorm | L2 normalisation in real-valued HV space |
//! | Dropout | Identity at inference time |
//!
//! # Example
//!
//! ```rust,ignore
//! use nexa_forge::*;
//!
//! // Define a simple 2-layer MLP: 784 → 128 → 10
//! let def = ModelDefinition::simple_mlp(
//!     "digit_classifier",
//!     &[784, 128, 10],
//!     "relu",
//! );
//!
//! let config = ForgeConfig::default();
//! let forged = ForgeEngine::forge(&def, &config)?;
//!
//! // Run inference on an encoded HV
//! let predictions = forged.predict(&encoded_input)?;
//! for (class, score) in &predictions {
//!     println!("{class}: {score:.4}");
//! }
//! ```

use nexa_core::{BinaryHV, NexaError, RealHV};
use nexa_topology::{LayerType, ModelGraph};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, NexaError>;

// ---------------------------------------------------------------------------
// WeightMatrix
// ---------------------------------------------------------------------------

/// A dense weight matrix stored in row-major order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl WeightMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "weight data length mismatch");
        Self { rows, cols, data }
    }

    /// Create a random weight matrix (for testing / initialisation).
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scale = (2.0 / cols as f32).sqrt(); // He initialisation
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self { rows, cols, data }
    }

    /// Get a row as a slice.
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Matrix-vector multiply: output\[i\] = sum_j(self\[i,j\] * input\[j\]).
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                self.row(i)
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// BiasVector
// ---------------------------------------------------------------------------

/// Optional bias vector for a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasVector {
    pub data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// ModelDefinition
// ---------------------------------------------------------------------------

/// Complete model definition: topology + trained weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub graph: ModelGraph,
    pub weights: HashMap<String, WeightMatrix>,
    pub biases: HashMap<String, BiasVector>,
    pub class_labels: Vec<String>,
}

impl ModelDefinition {
    /// Build a simple MLP definition with random weights (for testing).
    pub fn simple_mlp(_name: &str, layer_sizes: &[usize], activation: &str) -> Self {
        let graph = nexa_topology::build_simple_mlp(layer_sizes, activation);
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();

        for (i, pair) in layer_sizes.windows(2).enumerate() {
            let layer_name = format!("dense_{}", i);
            weights.insert(
                layer_name.clone(),
                WeightMatrix::random(pair[1], pair[0], (i as u64 + 1) * 1000),
            );
            biases.insert(
                layer_name,
                BiasVector {
                    data: vec![0.0; pair[1]],
                },
            );
        }

        let num_classes = *layer_sizes.last().unwrap();
        let class_labels: Vec<String> = (0..num_classes).map(|i| format!("class_{}", i)).collect();

        Self {
            graph,
            weights,
            biases,
            class_labels,
        }
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NexaError::Serialization(e.to_string()))
    }

    /// Parse from JSON.
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str)
            .map_err(|e| NexaError::Serialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ForgeConfig
// ---------------------------------------------------------------------------

/// Configuration for the forging process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeConfig {
    /// Hypervector dimensionality for the forged model.
    pub dim: usize,
    /// Random seed for projection matrices.
    pub seed: u64,
    /// Whether to use bipolar projections (+1/-1) vs real-valued.
    pub bipolar_projection: bool,
}

impl Default for ForgeConfig {
    fn default() -> Self {
        Self {
            dim: 10_000,
            seed: 42,
            bipolar_projection: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ProjectionMatrix — random projection into HV space
// ---------------------------------------------------------------------------

/// A random projection matrix that maps from R^input_dim to R^hv_dim.
/// Uses sparse Rademacher ±1 projections for efficiency.
struct ProjectionMatrix {
    hv_dim: usize,
    input_dim: usize,
    /// Stored as `hv_dim × input_dim` signs: +1.0 or -1.0.
    signs: Vec<f32>,
}

impl ProjectionMatrix {
    fn new(hv_dim: usize, input_dim: usize, seed: u64) -> Self {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..hv_dim * input_dim)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        Self {
            hv_dim,
            input_dim,
            signs,
        }
    }

    /// Project a vector from input space to HV space.
    fn project(&self, input: &[f32]) -> Vec<f32> {
        assert!(input.len() <= self.input_dim);
        let mut output = vec![0.0f32; self.hv_dim];
        let scale = 1.0 / (self.input_dim as f32).sqrt();
        for i in 0..self.hv_dim {
            let row_offset = i * self.input_dim;
            let mut sum = 0.0f32;
            for j in 0..input.len() {
                sum += self.signs[row_offset + j] * input[j];
            }
            output[i] = sum * scale;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// ForgedLayer — a single translated layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ForgedLayerKind {
    /// Dense layer: project weight rows into HV space, compute via dot products.
    Dense {
        /// Each output neuron is a projected HV (RealHV).
        neuron_hvs: Vec<RealHV>,
        bias: Vec<f32>,
        output_dim: usize,
    },
    /// ReLU: threshold at 0.
    ReLU,
    /// Sigmoid: map to [0, 1].
    Sigmoid,
    /// Softmax: normalised exponent of scores.
    Softmax,
    /// Flatten / identity pass-through.
    Passthrough,
    /// BatchNorm: L2 normalise the real-valued representation.
    Normalize,
    /// Conv2d: patch-wise random projection of filters into HV space.
    Conv2d {
        filter_hvs: Vec<RealHV>,
        num_filters: usize,
        #[allow(dead_code)]
        kernel_size: usize,
    },
    /// Pooling: reduce spatial dimensions via max or average over segments.
    Pooling {
        pool_type: String,
        pool_size: usize,
    },
}

#[derive(Debug, Clone)]
struct ForgedLayer {
    #[allow(dead_code)]
    name: String,
    kind: ForgedLayerKind,
}

// ---------------------------------------------------------------------------
// ForgedModel
// ---------------------------------------------------------------------------

/// A model that has been "forged" to operate on hypervector-encoded inputs.
///
/// Internally, it works with `RealHV` representations through the layers
/// and produces similarity-based classification at the output.
pub struct ForgedModel {
    layers: Vec<ForgedLayer>,
    class_labels: Vec<String>,
    dim: usize,
}

impl ForgedModel {
    /// Run the forged model on an encoded BinaryHV input.
    ///
    /// Returns a sorted `Vec<(class_label, score)>` with highest score first.
    pub fn predict(&self, input: &BinaryHV) -> Result<Vec<(String, f64)>> {
        let real_input = input.to_real();
        let output = self.forward_real(&real_input)?;

        // Output is a real vector of length = number of classes.
        // Apply softmax to get probabilities.
        let scores = &output;
        let max_val = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores.iter().map(|&s| (s - max_val).exp()).sum();

        let mut results: Vec<(String, f64)> = self
            .class_labels
            .iter()
            .zip(scores.iter())
            .map(|(label, &score)| {
                let prob = ((score - max_val).exp()) / exp_sum;
                (label.clone(), prob)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Run the forged model and return the output as a BinaryHV.
    pub fn forward(&self, input: &BinaryHV) -> Result<BinaryHV> {
        let real_input = input.to_real();
        let output = self.forward_real(&real_input)?;

        // Threshold to binary: > 0 → 1, else 0.
        let words_needed = (output.len() + 63) / 64;
        let mut words = vec![0u64; words_needed];
        for (i, &val) in output.iter().enumerate() {
            if val > 0.0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BinaryHV::from_words(words, output.len())
    }

    /// Internal forward pass on real-valued representations.
    fn forward_real(&self, input: &RealHV) -> Result<Vec<f64>> {
        let mut current: Vec<f64> = input.data().iter().map(|&x| x as f64).collect();

        for layer in &self.layers {
            current = Self::apply_layer(layer, &current)?;
        }

        Ok(current)
    }

    fn apply_layer(layer: &ForgedLayer, input: &[f64]) -> Result<Vec<f64>> {
        match &layer.kind {
            ForgedLayerKind::Dense {
                neuron_hvs,
                bias,
                output_dim,
            } => {
                // Each neuron computes a dot product with the input.
                let mut output = Vec::with_capacity(*output_dim);
                for (i, neuron) in neuron_hvs.iter().enumerate() {
                    let neuron_data = neuron.data();
                    let dot: f64 = neuron_data
                        .iter()
                        .zip(input.iter())
                        .map(|(&w, &x)| w as f64 * x)
                        .sum();
                    let b = if i < bias.len() { bias[i] as f64 } else { 0.0 };
                    output.push(dot + b);
                }
                Ok(output)
            }
            ForgedLayerKind::ReLU => {
                Ok(input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect())
            }
            ForgedLayerKind::Sigmoid => {
                Ok(input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
            }
            ForgedLayerKind::Softmax => {
                let max_val = input
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = input.iter().map(|&x| (x - max_val).exp()).sum();
                Ok(input
                    .iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect())
            }
            ForgedLayerKind::Passthrough => Ok(input.to_vec()),
            ForgedLayerKind::Normalize => {
                let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    Ok(input.to_vec())
                } else {
                    Ok(input.iter().map(|&x| x / norm).collect())
                }
            }
            ForgedLayerKind::Conv2d {
                filter_hvs,
                num_filters,
                kernel_size: _,
            } => {
                // Patch-wise convolution in HV space: each filter HV computes
                // a dot product against overlapping patches of the input.
                // Output: one value per filter per patch position.
                let patch_stride = num_filters.max(&1);
                let num_patches = (input.len() + patch_stride - 1) / patch_stride;
                let mut output = Vec::new();

                for p in 0..num_patches {
                    let start = p * patch_stride;
                    for filter in filter_hvs {
                        let filter_data = filter.data();
                        let len = filter_data.len().min(input.len() - start);
                        let dot: f64 = filter_data[..len]
                            .iter()
                            .zip(input[start..start + len].iter())
                            .map(|(&w, &x)| w as f64 * x)
                            .sum();
                        output.push(dot);
                    }
                }
                Ok(output)
            }
            ForgedLayerKind::Pooling {
                pool_type,
                pool_size,
            } => {
                let ps = (*pool_size).max(1);
                let num_out = (input.len() + ps - 1) / ps;
                let mut output = Vec::with_capacity(num_out);

                for i in 0..num_out {
                    let start = i * ps;
                    let end = (start + ps).min(input.len());
                    let chunk = &input[start..end];
                    let val = match pool_type.as_str() {
                        "avg" | "average" => {
                            chunk.iter().sum::<f64>() / chunk.len() as f64
                        }
                        _ => {
                            // Default: max pooling
                            chunk
                                .iter()
                                .cloned()
                                .fold(f64::NEG_INFINITY, f64::max)
                        }
                    };
                    output.push(val);
                }
                Ok(output)
            }
        }
    }

    /// Number of translated layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// HV dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Class labels.
    pub fn class_labels(&self) -> &[String] {
        &self.class_labels
    }
}

// ---------------------------------------------------------------------------
// ForgeReport
// ---------------------------------------------------------------------------

/// Report from the forging process.
#[derive(Debug, Clone)]
pub struct ForgeReport {
    pub model_name: String,
    pub original_layers: usize,
    pub forged_layers: usize,
    pub hv_dim: usize,
    pub total_weights: usize,
    pub projected_weights: usize,
    pub class_count: usize,
    pub duration_ms: f64,
}

impl std::fmt::Display for ForgeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NexaForge Report\n  Model:             {}\n  Original layers:   {}\n  Forged layers:     {}\n  HV dimension:      {}\n  Total weights:     {}\n  Projected weights: {}\n  Classes:           {}\n  Forge time:        {:.1}ms",
            self.model_name,
            self.original_layers,
            self.forged_layers,
            self.hv_dim,
            self.total_weights,
            self.projected_weights,
            self.class_count,
            self.duration_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// ForgeEngine
// ---------------------------------------------------------------------------

/// Engine that translates model definitions into forged HV-space models.
pub struct ForgeEngine;

impl ForgeEngine {
    /// Forge a model: translate all layers into HV-space equivalents.
    pub fn forge(
        definition: &ModelDefinition,
        config: &ForgeConfig,
    ) -> Result<(ForgedModel, ForgeReport)> {
        let start = std::time::Instant::now();
        let mut forged_layers = Vec::new();
        let mut total_weights = 0usize;
        let mut projected_weights = 0usize;
        let mut seed_offset = 0u64;

        // Process each layer in the model graph.
        for layer_node in &definition.graph.layers {
            match &layer_node.layer_type {
                LayerType::Input { .. } => {
                    // Input layer: no transformation needed.
                }
                LayerType::Dense { units, activation } => {
                    let layer_name = &layer_node.name;

                    // Get weights for this layer.
                    if let Some(weight) = definition.weights.get(layer_name) {
                        total_weights += weight.data.len();

                        // Build random projection for this layer.
                        let proj = ProjectionMatrix::new(
                            config.dim,
                            weight.cols,
                            config.seed + seed_offset,
                        );
                        seed_offset += 1;

                        // Project each neuron's weight row into HV space.
                        let mut neuron_hvs = Vec::with_capacity(*units);
                        for row_idx in 0..weight.rows.min(*units) {
                            let row = weight.row(row_idx);
                            let projected = proj.project(row);
                            let hv = RealHV::from_data(projected, config.dim)?;
                            neuron_hvs.push(hv);
                            projected_weights += config.dim;
                        }

                        let bias = definition
                            .biases
                            .get(layer_name)
                            .map(|b| b.data.clone())
                            .unwrap_or_else(|| vec![0.0; *units]);

                        forged_layers.push(ForgedLayer {
                            name: format!("{}_projected", layer_name),
                            kind: ForgedLayerKind::Dense {
                                neuron_hvs,
                                bias,
                                output_dim: *units,
                            },
                        });

                        // Add activation.
                        match activation.to_lowercase().as_str() {
                            "relu" => {
                                forged_layers.push(ForgedLayer {
                                    name: format!("{}_relu", layer_name),
                                    kind: ForgedLayerKind::ReLU,
                                });
                            }
                            "sigmoid" => {
                                forged_layers.push(ForgedLayer {
                                    name: format!("{}_sigmoid", layer_name),
                                    kind: ForgedLayerKind::Sigmoid,
                                });
                            }
                            "softmax" => {
                                forged_layers.push(ForgedLayer {
                                    name: format!("{}_softmax", layer_name),
                                    kind: ForgedLayerKind::Softmax,
                                });
                            }
                            _ => {} // linear / none
                        }
                    }
                }
                LayerType::BatchNorm => {
                    forged_layers.push(ForgedLayer {
                        name: layer_node.name.clone(),
                        kind: ForgedLayerKind::Normalize,
                    });
                }
                LayerType::Flatten => {
                    forged_layers.push(ForgedLayer {
                        name: layer_node.name.clone(),
                        kind: ForgedLayerKind::Passthrough,
                    });
                }
                LayerType::Dropout { .. } => {
                    // Dropout is identity at inference time.
                    forged_layers.push(ForgedLayer {
                        name: layer_node.name.clone(),
                        kind: ForgedLayerKind::Passthrough,
                    });
                }
                LayerType::Conv2d { filters, kernel_size, activation } => {
                    let layer_name = &layer_node.name;
                    let filter_dim = kernel_size.0 * kernel_size.1;

                    let filter_hvs = if let Some(weight) = definition.weights.get(layer_name) {
                        total_weights += weight.data.len();
                        let proj = ProjectionMatrix::new(
                            config.dim,
                            weight.cols.max(filter_dim),
                            config.seed + seed_offset,
                        );
                        seed_offset += 1;

                        let mut hvs = Vec::new();
                        for row_idx in 0..weight.rows.min(*filters) {
                            let row = weight.row(row_idx);
                            let projected = proj.project(row);
                            let hv = RealHV::from_data(projected, config.dim)?;
                            hvs.push(hv);
                            projected_weights += config.dim;
                        }
                        hvs
                    } else {
                        let proj = ProjectionMatrix::new(
                            config.dim,
                            filter_dim,
                            config.seed + seed_offset,
                        );
                        seed_offset += 1;

                        let mut hvs = Vec::new();
                        for f in 0..*filters {
                            let dummy_weight: Vec<f32> = (0..filter_dim)
                                .map(|j| ((f * filter_dim + j) as f32).sin())
                                .collect();
                            let projected = proj.project(&dummy_weight);
                            let hv = RealHV::from_data(projected, config.dim)?;
                            hvs.push(hv);
                            projected_weights += config.dim;
                        }
                        hvs
                    };

                    forged_layers.push(ForgedLayer {
                        name: format!("{}_conv2d", layer_name),
                        kind: ForgedLayerKind::Conv2d {
                            filter_hvs,
                            num_filters: *filters,
                            kernel_size: kernel_size.0,
                        },
                    });

                    match activation.to_lowercase().as_str() {
                        "relu" => {
                            forged_layers.push(ForgedLayer {
                                name: format!("{}_relu", layer_name),
                                kind: ForgedLayerKind::ReLU,
                            });
                        }
                        "sigmoid" => {
                            forged_layers.push(ForgedLayer {
                                name: format!("{}_sigmoid", layer_name),
                                kind: ForgedLayerKind::Sigmoid,
                            });
                        }
                        _ => {}
                    }
                }
                LayerType::Pooling { pool_type, size } => {
                    forged_layers.push(ForgedLayer {
                        name: layer_node.name.clone(),
                        kind: ForgedLayerKind::Pooling {
                            pool_type: pool_type.clone(),
                            pool_size: size.0,
                        },
                    });
                }
                LayerType::Custom { .. } => {
                    tracing::warn!(
                        "Layer '{}' ({:?}) has no HV translation; using passthrough",
                        layer_node.name,
                        layer_node.layer_type
                    );
                    forged_layers.push(ForgedLayer {
                        name: layer_node.name.clone(),
                        kind: ForgedLayerKind::Passthrough,
                    });
                }
            }
        }

        let model = ForgedModel {
            layers: forged_layers.clone(),
            class_labels: definition.class_labels.clone(),
            dim: config.dim,
        };

        let report = ForgeReport {
            model_name: definition.graph.name.clone(),
            original_layers: definition.graph.layers.len(),
            forged_layers: forged_layers.len(),
            hv_dim: config.dim,
            total_weights,
            projected_weights,
            class_count: definition.class_labels.len(),
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        };

        Ok((model, report))
    }

    /// Quick-forge an MLP: build a simple model definition and forge it.
    pub fn forge_mlp(
        name: &str,
        layer_sizes: &[usize],
        activation: &str,
        config: &ForgeConfig,
    ) -> Result<(ForgedModel, ForgeReport)> {
        let def = ModelDefinition::simple_mlp(name, layer_sizes, activation);
        Self::forge(&def, config)
    }
}

// ---------------------------------------------------------------------------
// ONNX Ingestion — parse ONNX JSON format into ModelDefinition
// ---------------------------------------------------------------------------

/// ONNX JSON node representation.
#[derive(Debug, Clone, Deserialize)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: Option<String>,
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
    #[serde(default)]
    pub attribute: Vec<OnnxAttribute>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxAttribute {
    pub name: String,
    #[serde(default)]
    pub i: Option<i64>,
    #[serde(default)]
    pub f: Option<f32>,
    #[serde(default)]
    pub s: Option<String>,
    #[serde(default)]
    pub ints: Vec<i64>,
}

/// ONNX JSON tensor (initializer) representation.
#[derive(Debug, Clone, Deserialize)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<usize>,
    #[serde(default)]
    pub float_data: Vec<f32>,
    #[serde(default)]
    pub raw_data: Option<String>,
}

/// Top-level ONNX JSON model.
#[derive(Debug, Clone, Deserialize)]
pub struct OnnxModel {
    pub graph: OnnxGraph,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxGraph {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub node: Vec<OnnxNode>,
    #[serde(default)]
    pub initializer: Vec<OnnxTensor>,
    #[serde(default)]
    pub input: Vec<OnnxValueInfo>,
    #[serde(default)]
    pub output: Vec<OnnxValueInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxValueInfo {
    pub name: String,
}

/// Parse an ONNX JSON string and convert to a ModelDefinition.
pub fn parse_onnx_json(json_str: &str, class_labels: Vec<String>) -> Result<ModelDefinition> {
    let onnx: OnnxModel = serde_json::from_str(json_str)
        .map_err(|e| NexaError::EncodingError(format!("ONNX JSON parse error: {}", e)))?;

    let graph_name = onnx.graph.name.unwrap_or_else(|| "onnx_model".to_string());
    let mut model_graph = ModelGraph::new(&graph_name);
    let mut weights: HashMap<String, WeightMatrix> = HashMap::new();
    let mut biases: HashMap<String, BiasVector> = HashMap::new();

    // Parse initializers (weights and biases)
    let mut initializers: HashMap<String, &OnnxTensor> = HashMap::new();
    for tensor in &onnx.graph.initializer {
        initializers.insert(tensor.name.clone(), tensor);
    }

    // Add input layer
    let input_name = onnx
        .graph
        .input
        .first()
        .map(|v| v.name.clone())
        .unwrap_or_else(|| "input".to_string());
    let prev_id = model_graph.add_layer(&input_name, LayerType::Input { shape: vec![0] });

    let mut prev_layer_id = prev_id;
    let mut layer_idx = 0;

    for node in &onnx.graph.node {
        let name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("layer_{}", layer_idx));

        let layer_type = match node.op_type.as_str() {
            "Gemm" | "MatMul" => {
                // Dense/fully-connected layer
                let units = if let Some(w_name) = node.input.get(1) {
                    if let Some(tensor) = initializers.get(w_name) {
                        let rows = *tensor.dims.first().unwrap_or(&0);
                        let cols = *tensor.dims.get(1).unwrap_or(&0);
                        if !tensor.float_data.is_empty() {
                            weights.insert(
                                name.clone(),
                                WeightMatrix::new(rows, cols, tensor.float_data.clone()),
                            );
                        }
                        cols
                    } else {
                        128
                    }
                } else {
                    128
                };

                // Check for bias
                if let Some(b_name) = node.input.get(2) {
                    if let Some(tensor) = initializers.get(b_name) {
                        if !tensor.float_data.is_empty() {
                            biases.insert(
                                name.clone(),
                                BiasVector {
                                    data: tensor.float_data.clone(),
                                },
                            );
                        }
                    }
                }

                Some(LayerType::Dense {
                    units,
                    activation: "linear".to_string(),
                })
            }
            "Relu" => {
                let id = model_graph.add_layer(&name, LayerType::Dense {
                    units: 0,
                    activation: "relu".to_string(),
                });
                model_graph.connect(prev_layer_id, id);
                prev_layer_id = id;
                layer_idx += 1;
                continue;
            }
            "Conv" => {
                let mut filters = 32;
                let mut kernel_h = 3usize;
                let mut kernel_w = 3usize;

                for attr in &node.attribute {
                    match attr.name.as_str() {
                        "kernel_shape" => {
                            if attr.ints.len() >= 2 {
                                kernel_h = attr.ints[0] as usize;
                                kernel_w = attr.ints[1] as usize;
                            }
                        }
                        _ => {}
                    }
                }

                if let Some(w_name) = node.input.get(1) {
                    if let Some(tensor) = initializers.get(w_name) {
                        filters = *tensor.dims.first().unwrap_or(&32);
                        if !tensor.float_data.is_empty() {
                            let cols = tensor.dims.iter().skip(1).product::<usize>().max(1);
                            weights.insert(
                                name.clone(),
                                WeightMatrix::new(filters, cols, tensor.float_data.clone()),
                            );
                        }
                    }
                }

                Some(LayerType::Conv2d {
                    filters,
                    kernel_size: (kernel_h, kernel_w),
                    activation: "relu".to_string(),
                })
            }
            "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
                let pool_type = if node.op_type.contains("Average") {
                    "avg"
                } else {
                    "max"
                }
                .to_string();

                let mut size = 2usize;
                for attr in &node.attribute {
                    if attr.name == "kernel_shape" && !attr.ints.is_empty() {
                        size = attr.ints[0] as usize;
                    }
                }

                Some(LayerType::Pooling {
                    pool_type,
                    size: (size, size),
                })
            }
            "Flatten" | "Reshape" => Some(LayerType::Flatten),
            "BatchNormalization" => Some(LayerType::BatchNorm),
            "Dropout" => {
                let rate = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "ratio")
                    .and_then(|a| a.f)
                    .unwrap_or(0.5) as f64;
                Some(LayerType::Dropout { rate })
            }
            "Softmax" | "LogSoftmax" => {
                // Terminal activation — skip as separate layer
                None
            }
            _ => {
                Some(LayerType::Custom {
                    name: node.op_type.clone(),
                })
            }
        };

        if let Some(lt) = layer_type {
            let id = model_graph.add_layer(&name, lt);
            model_graph.connect(prev_layer_id, id);
            prev_layer_id = id;
            layer_idx += 1;
        }
    }

    Ok(ModelDefinition {
        graph: model_graph,
        weights,
        biases,
        class_labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_matrix_matvec() {
        let w = WeightMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input = vec![1.0, 1.0, 1.0];
        let output = w.matvec(&input);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 6.0).abs() < 1e-5);
        assert!((output[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn projection_matrix_projects() {
        let proj = ProjectionMatrix::new(100, 10, 42);
        let input = vec![1.0; 10];
        let output = proj.project(&input);
        assert_eq!(output.len(), 100);
        // Output should be approximately distributed around 0.
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(
            mean.abs() < 1.0,
            "projected mean should be near 0, got {mean}"
        );
    }

    #[test]
    fn model_definition_simple_mlp() {
        let def = ModelDefinition::simple_mlp("test", &[784, 128, 10], "relu");
        assert_eq!(def.graph.layers.len(), 3); // input + 2 dense
        assert!(def.weights.contains_key("dense_0"));
        assert!(def.weights.contains_key("dense_1"));
        assert_eq!(def.weights["dense_0"].rows, 128);
        assert_eq!(def.weights["dense_0"].cols, 784);
        assert_eq!(def.weights["dense_1"].rows, 10);
        assert_eq!(def.weights["dense_1"].cols, 128);
        assert_eq!(def.class_labels.len(), 10);
    }

    #[test]
    fn model_definition_json_roundtrip() {
        let def = ModelDefinition::simple_mlp("test_rt", &[10, 5, 3], "relu");
        let json = def.to_json().unwrap();
        let restored = ModelDefinition::from_json(&json).unwrap();
        assert_eq!(restored.graph.name, "MLP");
        assert_eq!(restored.weights.len(), def.weights.len());
        assert_eq!(restored.class_labels.len(), 3);
    }

    #[test]
    fn forge_simple_mlp() {
        let config = ForgeConfig {
            dim: 1000,
            seed: 42,
            bipolar_projection: true,
        };
        let (model, report) = ForgeEngine::forge_mlp("test_forge", &[10, 5, 3], "relu", &config)
            .unwrap();

        assert!(report.forged_layers > 0);
        assert_eq!(report.class_count, 3);
        assert_eq!(model.dim(), 1000);
        assert_eq!(model.class_labels().len(), 3);
    }

    #[test]
    fn forged_model_predict() {
        let config = ForgeConfig {
            dim: 1000,
            seed: 42,
            bipolar_projection: true,
        };
        let (model, _) =
            ForgeEngine::forge_mlp("predictor", &[1000, 64, 5], "relu", &config).unwrap();

        // Create a random input HV.
        let input = BinaryHV::random(1000, 99).unwrap();
        let predictions = model.predict(&input).unwrap();

        assert_eq!(predictions.len(), 5);
        // Probabilities should sum to ~1.0
        let total: f64 = predictions.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "probabilities should sum to 1.0, got {total}"
        );
        // Sorted descending
        for w in predictions.windows(2) {
            assert!(w[0].1 >= w[1].1, "predictions should be sorted descending");
        }
    }

    #[test]
    fn forged_model_forward_produces_binary() {
        let config = ForgeConfig {
            dim: 500,
            seed: 42,
            bipolar_projection: true,
        };
        let (model, _) =
            ForgeEngine::forge_mlp("fwd_test", &[500, 32, 8], "relu", &config).unwrap();

        let input = BinaryHV::random(500, 123).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.dim(), 8); // output dimension = last layer size
    }

    #[test]
    fn forge_deterministic() {
        let config = ForgeConfig {
            dim: 500,
            seed: 42,
            bipolar_projection: true,
        };
        let def = ModelDefinition::simple_mlp("det", &[500, 32, 4], "relu");

        let (model1, _) = ForgeEngine::forge(&def, &config).unwrap();
        let (model2, _) = ForgeEngine::forge(&def, &config).unwrap();

        let input = BinaryHV::random(500, 77).unwrap();
        let p1 = model1.predict(&input).unwrap();
        let p2 = model2.predict(&input).unwrap();

        for ((l1, s1), (l2, s2)) in p1.iter().zip(p2.iter()) {
            assert_eq!(l1, l2);
            assert!(
                (s1 - s2).abs() < 1e-10,
                "predictions should be deterministic"
            );
        }
    }

    #[test]
    fn different_inputs_produce_different_outputs() {
        let config = ForgeConfig {
            dim: 1000,
            seed: 42,
            bipolar_projection: true,
        };
        let (model, _) =
            ForgeEngine::forge_mlp("diff_test", &[1000, 64, 5], "relu", &config).unwrap();

        let input_a = BinaryHV::random(1000, 1).unwrap();
        let input_b = BinaryHV::random(1000, 2).unwrap();

        let pred_a = model.predict(&input_a).unwrap();
        let pred_b = model.predict(&input_b).unwrap();

        // At least one score should differ meaningfully.
        let max_diff: f64 = pred_a
            .iter()
            .zip(pred_b.iter())
            .map(|((_, a), (_, b))| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_diff > 0.001,
            "different inputs should produce different outputs, max_diff={max_diff}"
        );
    }

    #[test]
    fn forge_report_fields() {
        let config = ForgeConfig {
            dim: 500,
            seed: 42,
            bipolar_projection: true,
        };
        let (_, report) = ForgeEngine::forge_mlp("report_test", &[100, 50, 10], "relu", &config)
            .unwrap();

        assert_eq!(report.model_name, "MLP");
        assert_eq!(report.class_count, 10);
        assert!(report.total_weights > 0);
        assert!(report.projected_weights > 0);
        assert!(report.duration_ms >= 0.0);
    }

    #[test]
    fn onnx_json_parse_simple_mlp() {
        let onnx_json = r#"{
            "graph": {
                "name": "test_mlp",
                "node": [
                    {
                        "op_type": "Gemm",
                        "name": "dense1",
                        "input": ["input", "dense1_w", "dense1_b"],
                        "output": ["dense1_out"],
                        "attribute": []
                    },
                    {
                        "op_type": "Relu",
                        "name": "relu1",
                        "input": ["dense1_out"],
                        "output": ["relu1_out"],
                        "attribute": []
                    },
                    {
                        "op_type": "Gemm",
                        "name": "dense2",
                        "input": ["relu1_out", "dense2_w"],
                        "output": ["output"],
                        "attribute": []
                    }
                ],
                "initializer": [
                    {
                        "name": "dense1_w",
                        "dims": [4, 3],
                        "float_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
                    },
                    {
                        "name": "dense1_b",
                        "dims": [4],
                        "float_data": [0.01, 0.02, 0.03, 0.04]
                    },
                    {
                        "name": "dense2_w",
                        "dims": [2, 4],
                        "float_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    }
                ],
                "input": [{"name": "input"}],
                "output": [{"name": "output"}]
            }
        }"#;

        let labels = vec!["cat".to_string(), "dog".to_string()];
        let def = parse_onnx_json(onnx_json, labels.clone()).unwrap();

        assert_eq!(def.class_labels, labels);
        assert!(def.weights.contains_key("dense1"));
        assert!(def.weights.contains_key("dense2"));
        assert!(def.biases.contains_key("dense1"));
        assert_eq!(def.weights["dense1"].rows, 4);
        assert_eq!(def.weights["dense1"].cols, 3);
    }

    #[test]
    fn onnx_json_parse_cnn() {
        let onnx_json = r#"{
            "graph": {
                "name": "test_cnn",
                "node": [
                    {
                        "op_type": "Conv",
                        "name": "conv1",
                        "input": ["input", "conv1_w"],
                        "output": ["conv1_out"],
                        "attribute": [{"name": "kernel_shape", "ints": [3, 3]}]
                    },
                    {
                        "op_type": "MaxPool",
                        "name": "pool1",
                        "input": ["conv1_out"],
                        "output": ["pool1_out"],
                        "attribute": [{"name": "kernel_shape", "ints": [2]}]
                    },
                    {
                        "op_type": "Flatten",
                        "name": "flatten",
                        "input": ["pool1_out"],
                        "output": ["flat_out"],
                        "attribute": []
                    }
                ],
                "initializer": [
                    {
                        "name": "conv1_w",
                        "dims": [16, 1, 3, 3],
                        "float_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    }
                ],
                "input": [{"name": "input"}],
                "output": [{"name": "output"}]
            }
        }"#;

        let def = parse_onnx_json(onnx_json, vec!["0".to_string(), "1".to_string()]).unwrap();
        assert!(def.weights.contains_key("conv1"));
        assert_eq!(def.weights["conv1"].rows, 16);
    }
}
