use crate::{FULL_ROUNDS, PARTIAL_ROUNDS, WIDTH};

use bellperson::gadgets::num;
use bellperson::gadgets::num::AllocatedNum;
use bellperson::{ConstraintSystem, SynthesisError};
use ff::Field;
use paired::bls12_381::Bls12;
use paired::Engine;

#[derive(Clone)]
pub struct PoseidonCircuit<E: Engine> {
    constants_offset: usize,
    round_constants: Vec<E::Fr>, // &'a [E::Fr],
    width: usize,
    leaves: Vec<AllocatedNum<E>>,
    pos: usize,
    full_rounds: usize,
    partial_rounds: usize,
    mds_matrix: Vec<Vec<E::Fr>>,
}

impl<E: Engine> PoseidonCircuit<E> {
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(preimage: Vec<AllocatedNum<E>>, matrix: Vec<Vec<E::Fr>>) -> Self {
        PoseidonCircuit {
            constants_offset: 0,
            round_constants: (0..1000).map(|_| E::Fr::zero()).collect::<Vec<_>>(), // FIXME
            width: WIDTH,
            leaves: preimage,
            pos: WIDTH,
            full_rounds: FULL_ROUNDS,
            partial_rounds: PARTIAL_ROUNDS,
            mds_matrix: matrix,
        }
    }

    fn hash<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<E>, SynthesisError> {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        for i in 0..self.full_rounds / 2 {
            self.full_round(cs.namespace(|| format!("initial full round {}", i)))?;
        }

        for i in 0..self.partial_rounds {
            self.partial_round(cs.namespace(|| format!("partial round {}", i)))?;
        }

        for i in 0..self.full_rounds / 2 {
            self.full_round(cs.namespace(|| format!("final full round {}", i)))?;
        }

        Ok(self.leaves[0].clone())
    }

    fn full_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants(cs.namespace(|| "add r"))?;

        let mut i = 0;
        // Apply the quintic S-Box to all elements
        self.leaves = self
            .leaves
            .iter()
            .map(|l| {
                let res =
                    quintic_s_box(cs.namespace(|| format!("quintic s-box {}", i)), &l).unwrap();
                i += 1;
                res
            }) // FIXME: Don't unwrap.
            .collect::<Vec<_>>();

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"))?;

        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants(cs.namespace(|| "add r"))?;

        // Apply the quintic S-Box to the first element.
        self.leaves[0] = quintic_s_box(cs.namespace(|| "quintic s-box"), &self.leaves[0])?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"))?;

        Ok(())
    }

    fn add_round_constants<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        let mut i = 0;
        self.leaves = self
            .leaves
            .iter()
            .map(|l| {
                // TODO: include indexes in namespace names.
                let num =
                    AllocatedNum::alloc(cs.namespace(|| format!("round constraint {}", i)), || {
                        Ok(self.round_constants[constants_offset])
                    })
                    .unwrap();
                constants_offset += 1;
                let res = add(cs.namespace(|| format!("add round key {}", i)), l, &num).unwrap();
                i += 1;

                res
            })
            .collect();

        self.constants_offset = constants_offset;

        Ok(())
    }

    fn product_mds<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let mut res: Vec<AllocatedNum<E>> = Vec::with_capacity(self.width);

        for j in 0..WIDTH {
            res.push(
                AllocatedNum::alloc(cs.namespace(|| format!("initial sum {}", j)), || {
                    Ok(E::Fr::zero())
                })
                .unwrap(),
            );
            for k in 0..WIDTH {
                // TODO: Allocate these once and reuse across rounds (and even between hashes).
                let v = AllocatedNum::alloc(
                    cs.namespace(|| format!("matrix tmp ({}, {})", j, k)),
                    || Ok(self.mds_matrix[j][k]),
                )?;

                v.mul(
                    cs.namespace(|| format!("multiply matrix element ({}, {})", j, k)),
                    &self.leaves[k],
                )?;

                add(
                    cs.namespace(|| format!("add to sum ({},{})", j, k)),
                    &res[j],
                    &v,
                )?;
            }
        }

        self.leaves = res;
        Ok(())
    }

    fn allocated_matrix<CS: ConstraintSystem<E>>(
        &mut self,
        mut _cs: CS,
    ) -> Vec<Vec<AllocatedNum<E>>> {
        let _mat: Vec<Vec<AllocatedNum<E>>>;
        unimplemented!()
    }
}

fn quintic_s_box<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    l: &AllocatedNum<E>,
) -> Result<AllocatedNum<E>, SynthesisError> {
    let l2 = l.square(cs.namespace(|| "l^2"))?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = l4.mul(cs.namespace(|| "l^5"), &l);

    l5
}

/// Adds a constraint to CS, enforcing a add relationship between the allocated numbers a, b, and sum.
///
/// a + b = sum
pub fn sum<E: Engine, A, AR, CS: ConstraintSystem<E>>(
    cs: &mut CS,
    annotation: A,
    a: &num::AllocatedNum<E>,
    b: &num::AllocatedNum<E>,
    sum: &num::AllocatedNum<E>,
) where
    A: FnOnce() -> AR,
    AR: Into<String>,
{
    // (a + b) * 1 = sum
    cs.enforce(
        annotation,
        |lc| lc + a.get_variable() + b.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
    );
}

pub fn add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    a: &num::AllocatedNum<E>,
    b: &num::AllocatedNum<E>,
) -> Result<num::AllocatedNum<E>, SynthesisError> {
    let res = num::AllocatedNum::alloc(cs.namespace(|| "add_num"), || {
        let mut tmp = a
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(
            &b.get_value()
                .ok_or_else(|| SynthesisError::AssignmentMissing)?,
        );

        Ok(tmp)
    })?;

    // a + b = res
    sum(&mut cs, || "sum constraint", &a, &b, &res);

    Ok(res)
}

fn poseidon_hash<CS: ConstraintSystem<Bls12>>(
    cs: CS,
    preimage: Vec<AllocatedNum<Bls12>>,
    matrix: Vec<Vec<<paired::bls12_381::Bls12 as ff::ScalarEngine>::Fr>>,
) -> Result<AllocatedNum<Bls12>, SynthesisError> {
    let mut p = PoseidonCircuit::new(preimage, matrix);
    p.hash(cs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_mds, WIDTH};

    use crate::test::TestConstraintSystem;
    use bellperson::ConstraintSystem;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let t = WIDTH;
        let cases = [(3, 1560)];

        let matrix = generate_mds(WIDTH);

        for (_, constraints) in &cases {
            let mut cs = TestConstraintSystem::<Bls12>::new();
            let mut i = 0;
            let data: Vec<AllocatedNum<Bls12>> = (0..t)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    i += 1;
                    AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
                })
                .collect::<Vec<_>>();

            let out =
                poseidon_hash(&mut cs, data, matrix.clone()).expect("poseidon hashing failed");

            assert!(cs.is_satisfied(), "constraints not satisfied");
            dbg!(cs.num_constraints());
            assert_eq!(
                cs.num_constraints(),
                *constraints,
                "constraint size changed",
            );

            // let expected: Fr;

            // assert_eq!(
            //     expected,
            //     out.get_value().unwrap(),
            //     "circuit and non circuit do not match"
            // );
        }
    }
}
