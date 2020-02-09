use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::PrimeField;
use generic_array::{typenum, ArrayLength};
use neptune::poseidon::{HashMode, PoseidonConstants};
use neptune::*;
use paired::bls12_381::{Bls12, Fr};
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256, Sha512};

fn bench_hash<Arity>(c: &mut Criterion)
where
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<Fr>,
{
    let scalars: Vec<Scalar> = std::iter::repeat(())
        .take(1000)
        .enumerate()
        .map(|(i, _)| scalar_from_u64::<Bls12>(i as u64))
        .collect();

    let mut group = c.benchmark_group(format!("hash-{}", Arity::to_usize() * 32));

    group.bench_with_input(
        BenchmarkId::new("Sha2 256", "Generated scalars"),
        &scalars,
        |b, s| {
            b.iter(|| {
                let mut h = Sha256::new();

                std::iter::repeat(())
                    .take(Arity::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        for val in scalar.into_repr().as_ref() {
                            h.input(&val.to_le_bytes());
                        }
                    });

                h.result();
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Sha2 512", "Generated scalars"),
        &scalars,
        |b, s| {
            b.iter(|| {
                let mut h = Sha512::new();

                std::iter::repeat(())
                    .take(Arity::to_usize())
                    .map(|_| s.choose(&mut OsRng).unwrap())
                    .for_each(|scalar| {
                        for val in scalar.into_repr().as_ref() {
                            h.input(&val.to_le_bytes());
                        }
                    });

                h.result();
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Poseidon hash", "Generated scalars"),
        &scalars,
        |b, s| {
            let constants = PoseidonConstants::new();
            let mut h = Poseidon::<Bls12, Arity>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(Arity::to_usize())
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
            let constants = PoseidonConstants::new();
            let mut h = Poseidon::<Bls12, Arity>::new(&constants);
            b.iter(|| {
                h.reset();
                std::iter::repeat(())
                    .take(Arity::to_usize())
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
    name = hash;

    config = Criterion::default();

    targets = bench_hash::<typenum::U2>, bench_hash::<typenum::U4>, bench_hash::<typenum::U8>, bench_hash::<typenum::U11>
}
criterion_main!(hash);
