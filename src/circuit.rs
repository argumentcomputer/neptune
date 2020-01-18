use crate::{FULL_ROUNDS, PARTIAL_ROUNDS, ROUND_CONSTANTS, WIDTH};

use bellperson::gadgets::num;
use bellperson::{ConstraintSystem, SynthesisError};
use paired::Engine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseidonCircuit<E: Engine> {
    constants_offset: usize,
    leaves: [E::Fr; WIDTH],
    pos: usize,
}

impl<E: Engine> PoseidonCircuit<E> {
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(preimage: [E::Fr; WIDTH]) -> Self {
        PoseidonCircuit {
            constants_offset: 0,
            leaves: preimage,
            pos: WIDTH,
        }
    }

    fn hash<CS: ConstraintSystem<E>>(
        mut cs: CS,
        preimage: &num::AllocatedNum<E>,
    ) -> Result<num::AllocatedNum<E>, SynthesisError> {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WIDTH;

    use crate::circuit::test::TestConstraintSystem;
    use bellperson::ConstraintSystem;
    use paired::bls12_381::{Bls12, Fr};
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let t = WIDTH;

        let cases = [123];

        for (bytes, constraints) in &cases {
            let mut cs = TestConstraintSystem::<Bls12>::new();
            let data: Vec<Fr> = (0..t).map(|_| rng.gen()).collect();

            let out = poseidon_hash(&mut cs, &data).expect("poseidon hashing failed");

            assert!(cs.is_satisfied(), "constraints not satisfied");
            assert_eq!(
                cs.num_constraints(),
                *constraints,
                "constraint size changed",
            );

            let expected;

            unimplemented!();

            assert_eq!(
                expected,
                out.get_value().unwrap(),
                "circuit and non circuit do not match"
            );
        }
    }
}
