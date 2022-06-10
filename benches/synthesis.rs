use crate::poseidon::{Arity, PoseidonConstants};
use bellperson::gadgets::num::AllocatedNum;
use bellperson::util_cs::bench_cs::BenchCS;
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use blstrs::Scalar as Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use generic_array::typenum;
use neptune::circuit::{poseidon_hash_circuit, CircuitType};
use neptune::*;
use rand::thread_rng;
use std::marker::PhantomData;

struct BenchCircuit<'a, A: Arity<Fr>> {
    n: usize,
    circuit_type: &'a CircuitType,
    _a: PhantomData<A>,
}

impl<A: Arity<Fr>> Circuit<Fr> for BenchCircuit<'_, A> {
    fn synthesize<CS: ConstraintSystem<Fr>>(self, mut cs: &mut CS) -> Result<(), SynthesisError> {
        let mut rng = thread_rng();
        let arity = A::to_usize();
        let constants = PoseidonConstants::<Fr, A>::new();

        for _ in 0..self.n {
            let mut i = 0;
            let mut fr_data = vec![Fr::random(&mut rng); arity];
            let data: Vec<AllocatedNum<Fr>> = (0..arity)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    fr_data[i] = fr;
                    i += 1;
                    AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
                })
                .collect::<Vec<_>>();
            let _ = poseidon_hash_circuit(&mut cs, *self.circuit_type, data, &constants)
                .expect("poseidon hashing failed");
        }
        Ok(())
    }
}

fn bench_synthesis<A>(c: &mut Criterion)
where
    A: Arity<Fr>,
{
    let mut group = c.benchmark_group(format!("synthesis-{}", A::to_usize()));
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
                    b.iter(|| {
                        let mut cs = BenchCS::<Fr>::new();
                        let circuit = BenchCircuit::<A> {
                            n: *n,
                            circuit_type,
                            _a: PhantomData::<A>,
                        };
                        circuit.synthesize(&mut cs)
                    })
                },
            );
        }
        // num_hashes *= 10;
    }
}

criterion_group! {
    name = synthesis;

    config = Criterion::default().sample_size(10);

    targets = bench_synthesis::<typenum::U8>
}
criterion_main!(synthesis);
