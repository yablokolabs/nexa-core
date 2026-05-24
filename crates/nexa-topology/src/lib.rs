//! Model architecture analysis and topology encoding into hyperdimensional space.

use nexa_core::{BinaryHV, NexaError};
use nexa_hdc::Codebook;
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, NexaError>;

// ---------------------------------------------------------------------------
// LayerType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Dense { units: usize, activation: String },
    Conv2d { filters: usize, kernel_size: (usize, usize), activation: String },
    Pooling { pool_type: String, size: (usize, usize) },
    Dropout { rate: f64 },
    BatchNorm,
    Flatten,
    Input { shape: Vec<usize> },
    Custom { name: String },
}

impl LayerType {
    fn type_name(&self) -> &str {
        match self {
            LayerType::Dense { .. } => "Dense",
            LayerType::Conv2d { .. } => "Conv2d",
            LayerType::Pooling { .. } => "Pooling",
            LayerType::Dropout { .. } => "Dropout",
            LayerType::BatchNorm => "BatchNorm",
            LayerType::Flatten => "Flatten",
            LayerType::Input { .. } => "Input",
            LayerType::Custom { name } => name.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNode {
    pub id: usize,
    pub name: String,
    pub layer_type: LayerType,
    pub connections: Vec<usize>,
}

// ---------------------------------------------------------------------------
// ModelGraph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    pub name: String,
    pub layers: Vec<LayerNode>,
}

impl ModelGraph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, name: &str, layer_type: LayerType) -> usize {
        let id = self.layers.len();
        self.layers.push(LayerNode {
            id,
            name: name.to_string(),
            layer_type,
            connections: Vec::new(),
        });
        id
    }

    pub fn connect(&mut self, from: usize, to: usize) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == from) {
            if !layer.connections.contains(&to) {
                layer.connections.push(to);
            }
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str)
            .map_err(|e| NexaError::Serialization(e.to_string()))
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NexaError::Serialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// GraphEncoder
// ---------------------------------------------------------------------------

pub struct GraphEncoder {
    codebook: Codebook,
    _dim: usize,
}

impl GraphEncoder {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            codebook: Codebook::new(dim, seed),
            _dim: dim,
        }
    }

    pub fn encode_layer(&mut self, layer: &LayerNode) -> Result<BinaryHV> {
        let type_name = layer.layer_type.type_name();
        let type_hv = self.codebook.get_or_insert(type_name).clone();

        match &layer.layer_type {
            LayerType::Dense { units, .. } => {
                let units_sym = format!("units_{}", units);
                let units_hv = self.codebook.get_or_insert(&units_sym).clone();
                type_hv.bind(&units_hv)
            }
            LayerType::Conv2d { filters, kernel_size, .. } => {
                let filters_sym = format!("filters_{}", filters);
                let filters_hv = self.codebook.get_or_insert(&filters_sym).clone();
                let kernel_sym = format!("kernel_{}x{}", kernel_size.0, kernel_size.1);
                let kernel_hv = self.codebook.get_or_insert(&kernel_sym).clone();
                let bound = type_hv.bind(&filters_hv)?;
                bound.bind(&kernel_hv)
            }
            LayerType::Pooling { pool_type, size } => {
                let pool_sym = format!("pool_{}", pool_type);
                let pool_hv = self.codebook.get_or_insert(&pool_sym).clone();
                let size_sym = format!("poolsize_{}x{}", size.0, size.1);
                let size_hv = self.codebook.get_or_insert(&size_sym).clone();
                let bound = type_hv.bind(&pool_hv)?;
                bound.bind(&size_hv)
            }
            LayerType::Input { shape } => {
                let shape_sym = format!("shape_{:?}", shape);
                let shape_hv = self.codebook.get_or_insert(&shape_sym).clone();
                type_hv.bind(&shape_hv)
            }
            _ => Ok(type_hv),
        }
    }

    pub fn encode_graph(&mut self, graph: &ModelGraph) -> Result<BinaryHV> {
        if graph.layers.is_empty() {
            return Err(NexaError::EmptyInput);
        }

        let mut layer_hvs: Vec<BinaryHV> = Vec::with_capacity(graph.layers.len());
        for (i, layer) in graph.layers.iter().enumerate() {
            let hv = self.encode_layer(layer)?;
            layer_hvs.push(hv.permute(i as isize));
        }

        let refs: Vec<&BinaryHV> = layer_hvs.iter().collect();
        BinaryHV::bundle(&refs)
    }
}

// ---------------------------------------------------------------------------
// TopologyAnalyzer
// ---------------------------------------------------------------------------

pub struct TopologyAnalyzer {
    encoder: GraphEncoder,
}

impl TopologyAnalyzer {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            encoder: GraphEncoder::new(dim, seed),
        }
    }

    pub fn similarity(&mut self, a: &ModelGraph, b: &ModelGraph) -> Result<f64> {
        let hv_a = self.encoder.encode_graph(a)?;
        let hv_b = self.encoder.encode_graph(b)?;
        hv_a.hamming_similarity(&hv_b)
    }

    pub fn encode(&mut self, graph: &ModelGraph) -> Result<BinaryHV> {
        self.encoder.encode_graph(graph)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn build_simple_mlp(layer_sizes: &[usize], activation: &str) -> ModelGraph {
    let mut graph = ModelGraph::new("MLP");

    let input_id = graph.add_layer(
        "input",
        LayerType::Input { shape: vec![layer_sizes[0]] },
    );

    let mut prev_id = input_id;
    for (i, &units) in layer_sizes[1..].iter().enumerate() {
        let name = format!("dense_{}", i);
        let layer_id = graph.add_layer(
            &name,
            LayerType::Dense {
                units,
                activation: activation.to_string(),
            },
        );
        graph.connect(prev_id, layer_id);
        prev_id = layer_id;
    }

    graph
}

pub fn build_simple_cnn(num_conv_layers: usize, dense_units: usize) -> ModelGraph {
    let mut graph = ModelGraph::new("CNN");

    let input_id = graph.add_layer(
        "input",
        LayerType::Input { shape: vec![28, 28, 1] },
    );

    let mut prev_id = input_id;
    for i in 0..num_conv_layers {
        let filters = 32 * (i + 1);
        let conv_name = format!("conv_{}", i);
        let conv_id = graph.add_layer(
            &conv_name,
            LayerType::Conv2d {
                filters,
                kernel_size: (3, 3),
                activation: "relu".to_string(),
            },
        );
        graph.connect(prev_id, conv_id);

        let pool_name = format!("pool_{}", i);
        let pool_id = graph.add_layer(
            &pool_name,
            LayerType::Pooling {
                pool_type: "max".to_string(),
                size: (2, 2),
            },
        );
        graph.connect(conv_id, pool_id);
        prev_id = pool_id;
    }

    let flatten_id = graph.add_layer("flatten", LayerType::Flatten);
    graph.connect(prev_id, flatten_id);

    let dense_id = graph.add_layer(
        "dense_out",
        LayerType::Dense {
            units: dense_units,
            activation: "relu".to_string(),
        },
    );
    graph.connect(flatten_id, dense_id);

    graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 1000;
    const SEED: u64 = 42;

    #[test]
    fn same_architecture_produces_same_encoding() {
        let mlp1 = build_simple_mlp(&[784, 128, 10], "relu");
        let mlp2 = build_simple_mlp(&[784, 128, 10], "relu");

        let mut enc1 = GraphEncoder::new(DIM, SEED);
        let mut enc2 = GraphEncoder::new(DIM, SEED);
        let hv1 = enc1.encode_graph(&mlp1).unwrap();
        let hv2 = enc2.encode_graph(&mlp2).unwrap();

        assert_eq!(hv1.words(), hv2.words(), "identical architectures must produce identical encodings");
    }

    #[test]
    fn different_architectures_are_dissimilar() {
        let mlp_small = build_simple_mlp(&[784, 128, 10], "relu");
        let mlp_large = build_simple_mlp(&[784, 512, 256, 10], "relu");

        let mut analyzer = TopologyAnalyzer::new(DIM, SEED);
        let sim = analyzer.similarity(&mlp_small, &mlp_large).unwrap();

        assert!(sim < 0.7, "different architectures should be dissimilar, got {sim}");
    }

    #[test]
    fn mlp_more_layers_distinct() {
        let mlp2 = build_simple_mlp(&[784, 10], "relu");
        let mlp5 = build_simple_mlp(&[784, 512, 256, 128, 10], "relu");

        let mut analyzer = TopologyAnalyzer::new(DIM, SEED);
        let sim = analyzer.similarity(&mlp2, &mlp5).unwrap();

        assert!(sim < 0.7, "2-layer vs 5-layer MLP should be distinct, got {sim}");
    }

    #[test]
    fn topology_similarity_correlates_with_structure() {
        let mlp_a = build_simple_mlp(&[784, 128, 10], "relu");
        let mlp_b = build_simple_mlp(&[784, 128, 64, 10], "relu");
        let cnn = build_simple_cnn(2, 10);

        let mut analyzer = TopologyAnalyzer::new(DIM, SEED);
        let sim_ab = analyzer.similarity(&mlp_a, &mlp_b).unwrap();

        let mut analyzer2 = TopologyAnalyzer::new(DIM, SEED);
        let sim_ac = analyzer2.similarity(&mlp_a, &cnn).unwrap();

        assert!(
            sim_ab > sim_ac,
            "similar MLPs ({sim_ab}) should be more similar than MLP vs CNN ({sim_ac})"
        );
    }

    #[test]
    fn graph_json_roundtrip() {
        let original = build_simple_mlp(&[784, 128, 10], "relu");
        let json = original.to_json().unwrap();
        let restored = ModelGraph::from_json(&json).unwrap();

        assert_eq!(original.name, restored.name);
        assert_eq!(original.layers.len(), restored.layers.len());
        for (o, r) in original.layers.iter().zip(restored.layers.iter()) {
            assert_eq!(o.id, r.id);
            assert_eq!(o.name, r.name);
            assert_eq!(o.layer_type, r.layer_type);
            assert_eq!(o.connections, r.connections);
        }
    }

    #[test]
    fn cnn_vs_mlp_are_dissimilar() {
        let mlp = build_simple_mlp(&[784, 256, 128, 10], "relu");
        let cnn = build_simple_cnn(2, 10);

        let mut analyzer = TopologyAnalyzer::new(DIM, SEED);
        let sim = analyzer.similarity(&mlp, &cnn).unwrap();

        assert!(sim < 0.7, "CNN vs MLP should be dissimilar, got {sim}");
    }
}
