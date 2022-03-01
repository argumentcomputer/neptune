use blstrs::Scalar as Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use generic_array::typenum;
use neptune::poseidon::{HashMode, PoseidonConstants};
use neptune::*;
use pasta_curves::{Fp, Fq as Fv};
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256, Sha512};
use typenum::{U11, U2, U4, U8};

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

fn bench_bls_and_pasta_fields_for_arity<A>(c: &mut Criterion)
where
    A: Arity<Fr> + Arity<Fp> + Arity<Fv>,
{
    let arity = A::to_usize();

    let mut group = c.benchmark_group(format!("arity-{}", arity));

    let preimage = vec![Fr::one(); arity];
    let consts = PoseidonConstants::<Fr, A>::new();
    group.bench_function("bls", |b| {
        b.iter(|| Poseidon::new_with_preimage(&preimage, &consts).hash())
    });

    let preimage = vec![Fp::one(); arity];
    let consts = PoseidonConstants::<Fp, A>::new();
    group.bench_function("pallas", |b| {
        b.iter(|| Poseidon::new_with_preimage(&preimage, &consts).hash())
    });

    let preimage = vec![Fv::one(); arity];
    let consts = PoseidonConstants::<Fv, A>::new();
    group.bench_function("vesta", |b| {
        b.iter(|| Poseidon::new_with_preimage(&preimage, &consts).hash())
    });
}

criterion_group!(
    name = bench_all_fields_for_common_arities;

    config = Criterion::default();

    targets = bench_bls_and_pasta_fields_for_arity::<U2>,
    bench_bls_and_pasta_fields_for_arity::<U4>, bench_bls_and_pasta_fields_for_arity::<U8>,
    bench_bls_and_pasta_fields_for_arity::<U11>,
);

criterion_main!(hash_bls, bench_all_fields_for_common_arities);
