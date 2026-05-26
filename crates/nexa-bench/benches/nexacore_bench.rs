use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nexa_core::{BinaryHV, RealHV};
use nexa_encoder::NexaEncoder;
use nexa_memory::CleanupMemory;
use nexa_runtime::VectorSearch;

fn bench_xor_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_binding");
    for &dim in &[1_000, 10_000, 100_000] {
        let a = BinaryHV::random(dim, 42).unwrap();
        let b = BinaryHV::random(dim, 43).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| a.bind(&b).unwrap());
        });
    }
    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_distance");
    for &dim in &[1_000, 10_000, 100_000] {
        let a = BinaryHV::random(dim, 42).unwrap();
        let b = BinaryHV::random(dim, 43).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| a.hamming_distance(&b).unwrap());
        });
    }
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    for &dim in &[1_000, 10_000] {
        let a = RealHV::random_normal(dim, 42).unwrap();
        let b = RealHV::random_normal(dim, 43).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| a.cosine_similarity(&b).unwrap());
        });
    }
    group.finish();
}

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle");
    let dim = 10_000;
    for &count in &[3, 7, 15] {
        let vecs: Vec<BinaryHV> = (0..count)
            .map(|i| BinaryHV::random(dim, i as u64).unwrap())
            .collect();
        let refs: Vec<&BinaryHV> = vecs.iter().collect();
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |bench, _| {
            bench.iter(|| BinaryHV::bundle(&refs).unwrap());
        });
    }
    group.finish();
}

fn bench_cleanup_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("cleanup_memory");
    let dim = 1_000;
    for &n in &[100, 500, 1_000] {
        let mut mem = CleanupMemory::new(dim).unwrap();
        for i in 0..n {
            mem.store(
                &format!("v{}", i),
                BinaryHV::random(dim, i as u64).unwrap(),
            )
            .unwrap();
        }
        let query = BinaryHV::random(dim, 9999).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| mem.query(&query).unwrap());
        });
    }
    group.finish();
}

fn bench_text_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_encoding");
    let dim = 10_000;
    for &len in &[10, 100, 1_000] {
        let text = "a".repeat(len);
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bench, _| {
            bench.iter(|| {
                let mut enc = NexaEncoder::new(dim, 42);
                enc.encode_text(&text).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_search");
    let dim = 1_000;
    for &n in &[100, 500, 1_000] {
        let mut vs = VectorSearch::new(dim);
        for i in 0..n {
            vs.insert(
                format!("v{}", i),
                BinaryHV::random(dim, i as u64).unwrap(),
            );
        }
        let query = BinaryHV::random(dim, 9999).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| vs.search(&query, 10));
        });
    }
    group.finish();
}

fn bench_lsh_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_search");
    let dim = 1_000;
    for &n in &[100, 500, 1_000] {
        let mut lsh = nexa_runtime::LshIndex::new(dim, 10, 12, 42);
        for i in 0..n {
            lsh.insert(
                format!("v{}", i),
                BinaryHV::random(dim, i as u64).unwrap(),
            );
        }
        let query = BinaryHV::random(dim, 9999).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| lsh.search(&query, 10));
        });
    }
    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    let dim = 10_000;
    for &count in &[10, 50, 100] {
        let vecs: Vec<BinaryHV> = (0..count)
            .map(|i| BinaryHV::random(dim, i as u64).unwrap())
            .collect();
        let raw: Vec<u8> = vecs
            .iter()
            .flat_map(|v| v.words().iter().flat_map(|w| w.to_le_bytes()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("deflate", count),
            &count,
            |bench, _| {
                bench.iter(|| nexa_compress::deflate_compress(&raw));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("delta", count),
            &count,
            |bench, _| {
                let stride = (dim + 63) / 64 * 8;
                bench.iter(|| {
                    let delta = nexa_compress::delta_encode(&raw, stride);
                    nexa_compress::deflate_compress(&delta)
                });
            },
        );
    }
    group.finish();
}

fn bench_knn_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_classify");
    let dim = 1_000;
    let base_a = BinaryHV::random(dim, 1000).unwrap();
    let base_b = BinaryHV::random(dim, 2000).unwrap();

    for &n in &[50, 200, 500] {
        let mut knn = nexa_runtime::KnnClassifier::new(dim, 3);
        for i in 0..(n / 2) as u64 {
            knn.insert("A".to_string(), base_a.corrupt(0.1, i));
            knn.insert("B".to_string(), base_b.corrupt(0.1, 1000 + i));
        }
        let query = base_a.corrupt(0.15, 9999);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| knn.predict(&query));
        });
    }
    group.finish();
}

fn bench_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("encryption");
    for &dim in &[1_000, 10_000] {
        let hv = BinaryHV::random(dim, 42).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| nexa_runtime::NexaCrypto::encrypt(&hv, 12345));
        });
    }
    group.finish();
}

fn bench_image_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_encoding");
    let dim = 10_000;
    // 28x28 like MNIST
    let pixels: Vec<u8> = (0..784).map(|i| (i % 256) as u8).collect();
    group.bench_function("28x28", |bench| {
        bench.iter(|| {
            let mut enc = NexaEncoder::new(dim, 42);
            enc.encode_image(&pixels, 28, 28).unwrap()
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_xor_binding,
    bench_hamming_distance,
    bench_cosine_similarity,
    bench_bundle,
    bench_cleanup_memory,
    bench_text_encoding,
    bench_similarity_search,
    bench_lsh_search,
    bench_compression,
    bench_knn_classify,
    bench_encryption,
    bench_image_encoding,
);
criterion_main!(benches);
