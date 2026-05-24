use criterion::{criterion_group, criterion_main, Criterion};
fn stub(c: &mut Criterion) { c.bench_function("stub", |b| b.iter(|| 1 + 1)); }
criterion_group!(benches, stub);
criterion_main!(benches);
