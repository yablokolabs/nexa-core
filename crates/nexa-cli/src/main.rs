use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nexa_core::BinaryHV;
use nexa_encoder::{read_nexa_file, write_nexa_file, NexaEncoder};
use nexa_topology::{ModelGraph, TopologyAnalyzer};
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nexa", about = "NexaCore — Universal Representation Runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a file into hypervector space
    Encode {
        input: String,
        #[arg(short, long, default_value = "output.nexa")]
        output: String,
        #[arg(short, long, default_value_t = 10000)]
        dim: usize,
    },
    /// Decode a .nexa file back to original data
    Decode {
        input: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Inspect a .nexa file metadata
    Inspect { input: String },
    /// Compute similarity between two .nexa files
    Similarity { file_a: String, file_b: String },
    /// Run built-in benchmarks
    Benchmark {
        #[arg(short, long, default_value_t = 10000)]
        dim: usize,
    },
    /// Recover a corrupted .nexa file
    Recover {
        input: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Encode model topology from JSON config
    Topology { input: String },
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Encode { input, output, dim } => cmd_encode(&input, &output, dim),
        Commands::Decode { input, output } => cmd_decode(&input, output.as_deref()),
        Commands::Inspect { input } => cmd_inspect(&input),
        Commands::Similarity { file_a, file_b } => cmd_similarity(&file_a, &file_b),
        Commands::Benchmark { dim } => cmd_benchmark(dim),
        Commands::Recover { input, output } => cmd_recover(&input, output.as_deref()),
        Commands::Topology { input } => cmd_topology(&input),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn detect_type(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("txt") => "Text",
        Some("json") => "Json",
        Some("csv") => "Csv",
        _ => "Binary",
    }
}

fn raw_to_hv(raw: &[u8], dim: usize) -> Result<BinaryHV> {
    let words: Vec<u64> = raw
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    BinaryHV::from_words(words, dim).context("Failed to reconstruct BinaryHV from raw data")
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_encode(input: &str, output: &str, dim: usize) -> Result<()> {
    let data = std::fs::read(input).with_context(|| format!("Cannot read {input}"))?;
    let data_type = detect_type(input);
    let mut encoder = NexaEncoder::new(dim, 42);

    let hv = match data_type {
        "Text" => {
            let text = String::from_utf8(data.clone())
                .context("Input file is not valid UTF-8 for text encoding")?;
            encoder
                .encode_text(&text)
                .context("Text encoding failed")?
        }
        "Json" => {
            let text = String::from_utf8(data.clone())
                .context("Input file is not valid UTF-8 for JSON encoding")?;
            encoder
                .encode_json(&text)
                .context("JSON encoding failed")?
        }
        "Csv" => {
            let text = String::from_utf8(data.clone())
                .context("Input file is not valid UTF-8 for CSV encoding")?;
            let mut all_hvs = Vec::new();
            for line in text.lines() {
                let fields: Vec<&str> = line.split(',').collect();
                if !fields.is_empty() {
                    all_hvs.push(
                        encoder
                            .encode_csv_row(&fields)
                            .context("CSV row encoding failed")?,
                    );
                }
            }
            if all_hvs.is_empty() {
                anyhow::bail!("CSV file is empty");
            }
            let refs: Vec<&BinaryHV> = all_hvs.iter().collect();
            BinaryHV::bundle(&refs).context("Bundle of CSV rows failed")?
        }
        _ => encoder
            .encode_bytes(&data)
            .context("Binary encoding failed")?,
    };

    write_nexa_file(Path::new(output), &encoder, &[&hv])
        .context("Failed to write .nexa file")?;

    println!("NexaCore Encoder");
    println!("  Input:     {input} ({} bytes)", data.len());
    println!("  Type:      {data_type}");
    println!("  Dimension: {dim}");
    println!("  Output:    {output}");
    println!("  Status:    ✓ Encoded successfully");
    Ok(())
}

fn cmd_decode(input: &str, output: Option<&str>) -> Result<()> {
    let (header, vectors) =
        read_nexa_file(Path::new(input)).with_context(|| format!("Cannot read {input}"))?;

    println!("NexaCore Decoder");
    println!("  File:       {input}");
    println!("  Version:    {}", header.version);
    println!("  Dimension:  {}", header.dimension);
    println!("  Vectors:    {}", vectors.len());
    println!("  Encoding:   {}", header.encoding_type);
    println!("  Metadata:   {}", truncate_str(&header.metadata, 200));

    if let Some(out_path) = output {
        // Extract original data from metadata records
        if !header.metadata.is_empty() {
            if let Ok(records) =
                serde_json::from_str::<Vec<nexa_encoder::EncodingRecord>>(&header.metadata)
            {
                let mut combined = Vec::new();
                for record in &records {
                    combined.extend_from_slice(&record.original_data);
                }
                std::fs::write(out_path, &combined)
                    .with_context(|| format!("Cannot write {out_path}"))?;
                println!("  Output:     {out_path} ({} bytes recovered from metadata)", combined.len());
                return Ok(());
            }
        }
        println!("  Note:       Full decode requires encoder registry; metadata-only recovery written");
    } else {
        println!("  Note:       Use -o <file> to write recovered data from metadata");
    }

    Ok(())
}

fn cmd_inspect(input: &str) -> Result<()> {
    let (header, vectors) =
        read_nexa_file(Path::new(input)).with_context(|| format!("Cannot read {input}"))?;

    println!("NexaCore Inspector");
    println!("  Magic:        NEXA ✓");
    println!("  Version:      {}", header.version);
    println!("  Dimension:    {}", header.dimension);
    println!("  Vectors:      {}", vectors.len());
    println!("  Encoding:     {} (default)", header.encoding_type);
    println!("  Metadata:     {}", truncate_str(&header.metadata, 200));
    println!("  Checksum:     ✓ Valid");
    Ok(())
}

fn cmd_similarity(file_a: &str, file_b: &str) -> Result<()> {
    let (header_a, vectors_a) =
        read_nexa_file(Path::new(file_a)).with_context(|| format!("Cannot read {file_a}"))?;
    let (header_b, vectors_b) =
        read_nexa_file(Path::new(file_b)).with_context(|| format!("Cannot read {file_b}"))?;

    anyhow::ensure!(!vectors_a.is_empty(), "{file_a} contains no vectors");
    anyhow::ensure!(!vectors_b.is_empty(), "{file_b} contains no vectors");

    let hv_a = raw_to_hv(&vectors_a[0], header_a.dimension as usize)?;
    let hv_b = raw_to_hv(&vectors_b[0], header_b.dimension as usize)?;

    let sim = hv_a
        .hamming_similarity(&hv_b)
        .context("Similarity computation failed")?;

    println!("NexaCore Similarity");
    println!("  File A:      {file_a} (dim={})", header_a.dimension);
    println!("  File B:      {file_b} (dim={})", header_b.dimension);
    println!("  Similarity:  {sim:.6}");
    Ok(())
}

fn cmd_benchmark(dim: usize) -> Result<()> {
    println!("NexaCore Benchmark (dim={dim})");

    let hv_a = BinaryHV::random(dim, 1).context("Failed to create random HV")?;
    let hv_b = BinaryHV::random(dim, 2).context("Failed to create random HV")?;

    // XOR binding
    let iters = 1000u64;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = hv_a.bind(&hv_b);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = iters as f64 / elapsed.as_secs_f64();
    println!(
        "  XOR Binding:      {iters} ops in {:.1}ms ({:.1}K ops/sec)",
        elapsed.as_secs_f64() * 1000.0,
        ops_per_sec / 1000.0
    );

    // Hamming distance
    let start = Instant::now();
    for _ in 0..iters {
        let _ = hv_a.hamming_distance(&hv_b);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = iters as f64 / elapsed.as_secs_f64();
    println!(
        "  Hamming Distance: {iters} ops in {:.1}ms ({:.1}K ops/sec)",
        elapsed.as_secs_f64() * 1000.0,
        ops_per_sec / 1000.0
    );

    // Bundle of 10 vectors
    let bundle_vecs: Vec<BinaryHV> = (0..10)
        .map(|i| BinaryHV::random(dim, 100 + i).unwrap())
        .collect();
    let bundle_refs: Vec<&BinaryHV> = bundle_vecs.iter().collect();
    let bundle_iters = 100u64;
    let start = Instant::now();
    for _ in 0..bundle_iters {
        let _ = BinaryHV::bundle(&bundle_refs);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = bundle_iters as f64 / elapsed.as_secs_f64();
    println!(
        "  Bundle (10 vecs): {bundle_iters} ops in {:.1}ms ({:.1}K ops/sec)",
        elapsed.as_secs_f64() * 1000.0,
        ops_per_sec / 1000.0
    );

    Ok(())
}

fn cmd_recover(input: &str, output: Option<&str>) -> Result<()> {
    println!("NexaCore Recovery");
    println!("  Input:      {input}");

    match read_nexa_file(Path::new(input)) {
        Ok((header, vectors)) => {
            println!("  Status:     File is valid (no corruption detected)");
            println!("  Version:    {}", header.version);
            println!("  Dimension:  {}", header.dimension);
            println!("  Vectors:    {}", vectors.len());
            println!("  Checksum:   ✓ Valid");

            if let Some(out_path) = output {
                std::fs::copy(input, out_path)
                    .with_context(|| format!("Cannot copy to {out_path}"))?;
                println!("  Output:     {out_path} (copied as-is)");
            }
        }
        Err(e) => {
            println!("  Status:     ✗ Corruption detected");
            println!("  Error:      {e}");

            // Perform partial recovery by reading raw file
            let raw = std::fs::read(input).with_context(|| format!("Cannot read {input}"))?;
            let magic_ok = raw.len() >= 4 && &raw[0..4] == b"NEXA";
            println!(
                "  Magic:      {}",
                if magic_ok { "NEXA ✓" } else { "✗ Invalid" }
            );
            println!("  File size:  {} bytes", raw.len());

            if magic_ok && raw.len() >= 18 {
                let version = u16::from_le_bytes([raw[4], raw[5]]);
                let dimension = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
                let vector_count = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
                println!("  Version:    {version} (from header)");
                println!("  Dimension:  {dimension} (from header)");
                println!("  Vectors:    {vector_count} (from header)");
            }

            if let Some(out_path) = output {
                std::fs::copy(input, out_path)
                    .with_context(|| format!("Cannot copy to {out_path}"))?;
                println!("  Output:     {out_path} (raw copy for manual recovery)");
            }
        }
    }

    Ok(())
}

fn cmd_topology(input: &str) -> Result<()> {
    let json_str =
        std::fs::read_to_string(input).with_context(|| format!("Cannot read {input}"))?;

    let graph =
        ModelGraph::from_json(&json_str).context("Failed to parse model graph from JSON")?;

    let mut analyzer = TopologyAnalyzer::new(10_000, 42);
    let hv = analyzer
        .encode(&graph)
        .context("Failed to encode topology")?;

    println!("NexaCore Topology");
    println!("  Input:      {input}");
    println!("  Model:      {}", graph.name);
    println!("  Layers:     {}", graph.layer_count());
    println!("  HV Dim:     {}", hv.dim());
    println!("  Popcount:   {} / {}", hv.popcount(), hv.dim());
    println!("  Status:     ✓ Topology encoded successfully");
    Ok(())
}
