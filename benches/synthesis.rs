use crate::poseidon::{Arity, PoseidonConstants};

use bellpepper::util_cs::bench_cs::BenchCS;
use bellpepper::util_cs::witness_cs::WitnessCS;
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::ConstraintSystem;
use blstrs::Scalar as Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use generic_array::typenum;
use neptune::circuit::{poseidon_hash_circuit, CircuitType};
use neptune::circuit2::poseidon_hash_allocated;
use neptune::*;
use rand::thread_rng;
use std::marker::PhantomData;

struct BenchCircuit<'a, A: Arity<Fr>> {
    n: usize,
    circuit_type: Option<&'a CircuitType>,
    _a: PhantomData<A>,
}

impl<A: Arity<Fr>> BenchCircuit<'_, A> {
    fn synthesize<CS: ConstraintSystem<Fr>>(
        self,
        cs: &mut CS,
        data: &[AllocatedNum<Fr>],
        constants: &PoseidonConstants<Fr, A>,
    ) {
        for _ in 0..self.n {
            if self.circuit_type.is_some() {
                poseidon_hash_circuit(
                    &mut cs.namespace(|| "{i}"),
                    *self.circuit_type.unwrap(),
                    data.to_vec(),
                    constants,
                )
                .expect("poseidon hashing failed");
            } else {
                poseidon_hash_allocated(&mut cs.namespace(|| "{i}"), data.to_vec(), constants)
                    .expect("poseidon hashing failed");
            };
        }
    }

    fn data<CS: ConstraintSystem<Fr>>(cs: &mut CS) -> Vec<AllocatedNum<Fr>> {
        let mut rng = thread_rng();
        let arity = A::to_usize();

        let mut fr_data = vec![Fr::random(&mut rng); arity];
        let data: Vec<AllocatedNum<Fr>> = (0..arity)
            .map(|i| {
                let fr = Fr::random(&mut rng);
                fr_data[i] = fr;
                AllocatedNum::alloc_infallible(cs.namespace(|| format!("data {}", i)), || fr)
            })
            .collect::<Vec<_>>();
        data
    }
}

fn bench_synthesis<A>(c: &mut Criterion)
where
    A: Arity<Fr>,
{
    let mut group = c.benchmark_group(format!("synthesis-{}", A::to_usize()));
    let constants = PoseidonConstants::<Fr, A>::new();
    for i in 0..4 {
        let num_hashes = 10usize.pow(i);
        for circuit_type in &[CircuitType::Legacy, CircuitType::OptimalAllocated] {
            group.bench_with_input(
                BenchmarkId::new(
                    circuit_type.label(),
                    format!("arity: {}, count: {}", A::to_usize(), num_hashes),
                ),
                &num_hashes,
                |b, n| {
                    let mut cs = BenchCS::<Fr>::new();
                    let data = BenchCircuit::<A>::data(&mut cs);
                    b.iter(|| {
                        let circuit = BenchCircuit::<A> {
                            n: *n,
                            circuit_type: Some(circuit_type),
                            _a: PhantomData::<A>,
                        };
                        circuit.synthesize(&mut cs, &data, &constants)
                    })
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new(
                "hash_allocated_witness",
                format!("arity: {}, count: {}", A::to_usize(), num_hashes),
            ),
            &num_hashes,
            |b, n| {
                let mut cs = WitnessCS::<Fr>::new();
                let data = BenchCircuit::<A>::data(&mut cs);
                b.iter(|| {
                    let circuit = BenchCircuit::<A> {
                        n: *n,
                        circuit_type: None,
                        _a: PhantomData::<A>,
                    };
                    circuit.synthesize(&mut cs, &data, &constants)
                })
            },
        );
    }
}

criterion_group! {
    name = synthesis;

    config = Criterion::default().sample_size(10);

    targets = bench_synthesis::<typenum::U8>
}
criterion_main!(synthesis);
