use blstrs::Scalar as Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::PrimeField;
use generic_array::typenum;
use neptune::poseidon::{HashMode, PoseidonConstants};
use neptune::*;
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256, Sha512};

fn bench_hash_bls<A>(c: &mut Criterion)
where
    A: Arity<Fr>,
{
    bench_hash::<Fr, A>(c, "bls");
}

fn bench_hash<F, A>(c: &mut Criterion, field_name: &str)
where
    F: PrimeField,
    A: Arity<F>,
{
    let scalars: Vec<F> = std::iter::repeat(())
        .take(1000)
        .enumerate()
        .map(|(i, _)| F::from(i as u64))
        .collect();

    let mut group = c.benchmark_group(format!("hash-{}-{}", field_name, A::to_usize() * 32));

    group.bench_with_input(
        BenchmarkId::new("Sha2 256", "Generated scalars"),
        &scalars,
        |b, s| {
            let mut h = Sha256::new();
            b.iter(|| {
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.update(scalar.to_repr().as_ref());
                    });

                h.finalize_reset()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Sha2 512", "Generated scalars"),
        &scalars,
        |b, s| {
            let mut h = Sha512::new();

            b.iter(|| {
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.update(scalar.to_repr().as_ref());
                    });

                h.finalize_reset()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Poseidon hash", "Generated scalars"),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Standard);
            let mut h = Poseidon::<F, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::Correct);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Poseidon hash optimized", "Generated scalars"),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Standard);
            let mut h = Poseidon::<F, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::OptimizedStatic);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "Poseidon hash optimized (strengthened)",
            "Generated scalars",
        ),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new_with_strength(Strength::Strengthened);
            let mut h = Poseidon::<F, A>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(A::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        h.input(*scalar).unwrap();
                    });

                h.hash_in_mode(HashMode::OptimizedStatic);
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = hash_bls;

    config = Criterion::default();

    targets = bench_hash_bls::<typenum::U2>, bench_hash_bls::<typenum::U4>,
    bench_hash_bls::<typenum::U8>, bench_hash_bls::<typenum::U11>
}

criterion_main!(hash_bls);
