use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use nexa_core::BinaryHV;

fn bench_xor_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_binding");
    for dim in [1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let a = BinaryHV::random(dim, 42).unwrap();
            let v = BinaryHV::random(dim, 43).unwrap();
            b.iter(|| a.bind(&v).unwrap());
        });
    }
    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_distance");
    for dim in [1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let a = BinaryHV::random(dim, 42).unwrap();
            let v = BinaryHV::random(dim, 43).unwrap();
            b.iter(|| a.hamming_distance(&v).unwrap());
        });
    }
    group.finish();
}

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle");
    let dim = 10_000;
    for count in [3, 7, 15] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vecs: Vec<BinaryHV> = (0..count)
                .map(|i| BinaryHV::random(dim, i as u64).unwrap())
                .collect();
            let refs: Vec<&BinaryHV> = vecs.iter().collect();
            b.iter(|| BinaryHV::bundle(&refs).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_xor_binding, bench_hamming_distance, bench_bundle);
criterion_main!(benches);
