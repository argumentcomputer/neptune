use crate::poseidon::{arity_tag, PoseidonConstants};
use crate::{generate_mds, round_constants, ARITY, FULL_ROUNDS, PARTIAL_ROUNDS, WIDTH};

use bellperson::gadgets::num::AllocatedNum;
use bellperson::{ConstraintSystem, SynthesisError};
use ff::Field;
use paired::Engine;

#[derive(Clone)]
/// Circuit for Poseidon hash.
pub struct PoseidonCircuit<E: Engine> {
    constants_offset: usize,
    width: usize,
    elements: Vec<AllocatedNum<E>>,
    pos: usize,
    full_rounds: usize,
    partial_rounds: usize,
    constants: PoseidonConstants<E>,
}

/// PoseidonCircuit implementation.
impl<E: Engine> PoseidonCircuit<E> {
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(elements: Vec<AllocatedNum<E>>, constants: PoseidonConstants<E>) -> Self {
        let width = WIDTH;

        PoseidonCircuit {
            constants_offset: 0,
            width,
            elements,
            pos: width,
            full_rounds: FULL_ROUNDS,
            partial_rounds: PARTIAL_ROUNDS,
            constants,
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

        Ok(self.elements[1].clone())
    }

    fn full_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        // Apply the quintic S-Box to all elements
        for i in 0..self.elements.len() {
            let round_key = self.constants.round_constants[constants_offset];
            constants_offset += 1;

            self.elements[i] = quintic_s_box(
                cs.namespace(|| format!("quintic s-box {}", i)),
                &self.elements[i],
                Some(round_key),
            )?
        }
        self.constants_offset = constants_offset;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), false)?;
        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let round_key = self.constants.round_constants[self.constants_offset];
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            Some(round_key),
        )?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), true)?;

        Ok(())
    }

    fn product_mds<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
        add_round_keys: bool,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<AllocatedNum<E>> = Vec::with_capacity(WIDTH);

        for j in 0..WIDTH {
            let column = self.constants.mds_matrix[j].to_vec();
            // TODO: This could be cached per round to save synthesis time.
            let constant_term = if add_round_keys {
                let mut acc = E::Fr::zero();
                // Dot product of column and this round's keys.
                for k in 1..WIDTH {
                    let mut tmp = column[k];
                    let rk = self.constants.round_constants[self.constants_offset + k - 1];
                    tmp.mul_assign(&rk);
                    acc.add_assign(&tmp);
                }
                Some(acc)
            } else {
                None
            };

            let product = scalar_product(
                cs.namespace(|| format!("scalar product {}", j)),
                self.elements.as_slice(),
                &column,
                constant_term,
            )?;
            result.push(product);
        }
        if add_round_keys {
            self.constants_offset += WIDTH - 1;
        }
        self.elements = result;

        Ok(())
    }

    fn debug(&self) {
        let element_frs: Vec<_> = self
            .elements
            .iter()
            .map(|n| n.get_value().unwrap())
            .collect();
        dbg!(element_frs, self.constants_offset);
    }

    /// This works but is inefficient. Retained for reference.
    fn partial_round_with_explicit_round_constants<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        let round_key = self.constants.round_constants[self.constants_offset];
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            Some(round_key),
        )?;

        self.add_round_constants(cs.namespace(|| "add round keys"), true)?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), false)?;

        Ok(())
    }

    fn add_round_constants<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
        skip_first: bool,
    ) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;
        let start = if skip_first { 1 } else { 0 };

        for i in start..self.elements.len() {
            let constant = &self.constants.round_constants[constants_offset];
            constants_offset += 1;

            self.elements[i] = add(
                cs.namespace(|| format!("add round key {}", i)),
                &self.elements[i],
                constant,
            )?;
        }

        self.constants_offset = constants_offset;

        Ok(())
    }
}

// /// Create circuit for Poseidon hash.
// pub fn poseidon_hash_x<CS: ConstraintSystem<paired::bls12_381::Bls12>>(
//     mut cs: CS,
//     mut preimage: Vec<AllocatedNum<paired::bls12_381::Bls12>>,
// ) -> Result<AllocatedNum<paired::bls12_381::Bls12>, SynthesisError> {
//     let mut matrix: Vec<Vec<_>> = Vec::new();
//     for column in &*MDS_MATRIX {
//         matrix.push(column.to_vec());
//     }

//     // Add the arity tag to the front of the preimage.
//     let tag = *ARITY_TAG;
//     let tag_num = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(tag))?;
//     preimage.push(tag_num);
//     preimage.rotate_right(1);

//     let mut p = PoseidonCircuit::new(preimage, matrix, (&*ROUND_CONSTANTS).to_vec());

//     p.hash(cs)
// }

/// Create circuit for Poseidon hash.
pub fn poseidon_hash<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    mut preimage: Vec<AllocatedNum<E>>,
    constants: PoseidonConstants<E>,
) -> Result<AllocatedNum<E>, SynthesisError> {
    // Add the arity tag to the front of the preimage.
    let tag = arity_tag::<E>(ARITY);
    let tag_num = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(tag))?;
    preimage.push(tag_num);
    preimage.rotate_right(1);
    let mut p = PoseidonCircuit::new(preimage, constants);

    p.hash(cs)
}

pub fn poseidon_hash_simple<CS: ConstraintSystem<E>, E: Engine>(
    cs: CS,
    preimage: Vec<AllocatedNum<E>>,
) -> Result<AllocatedNum<E>, SynthesisError> {
    let round_constants = round_constants::<E>(WIDTH);
    let mds_matrix: Vec<Vec<_>> = generate_mds::<E>(WIDTH);

    let constants = PoseidonConstants::<E> {
        round_constants,
        mds_matrix,
    };

    poseidon_hash(cs, preimage, constants)
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    l: &AllocatedNum<E>,
    round_key: Option<E::Fr>,
) -> Result<AllocatedNum<E>, SynthesisError> {
    // If round_key was supplied, add it to l before squaring.
    let l2 = if let Some(rk) = round_key {
        square_sum(cs.namespace(|| "(l+rk)^2"), rk, l)?
    } else {
        l.square(cs.namespace(|| "l^2"))?
    };
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = if let Some(rk) = round_key {
        mul_sum(cs.namespace(|| "l4 * (l + rk)"), &l4, &l, rk)
    } else {
        l4.mul(cs.namespace(|| "l^5"), &l)
    };

    l5
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    to_add: E::Fr,
    num: &AllocatedNum<E>,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
{
    let res = AllocatedNum::alloc(cs.namespace(|| "squared sum"), || {
        let mut tmp = num
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp.square();

        Ok(tmp)
    })?;

    cs.enforce(
        || "squared sum constraint",
        |lc| lc + num.get_variable() + (to_add, CS::one()),
        |lc| lc + num.get_variable() + (to_add, CS::one()),
        |lc| lc + res.get_variable(),
    );
    Ok(res)
}

/// Calculates a * (b + to_add) — and enforces that constraint.
pub fn mul_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &AllocatedNum<E>,
    to_add: E::Fr,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
{
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp.mul_assign(
            &a.get_value()
                .ok_or_else(|| SynthesisError::AssignmentMissing)?,
        );

        Ok(tmp)
    })?;

    cs.enforce(
        || "mul sum constraint",
        |lc| lc + b.get_variable() + (to_add, CS::one()),
        |lc| lc + a.get_variable(),
        |lc| lc + res.get_variable(),
    );
    Ok(res)
}

/// Adds a constraint to CS, enforcing that a + b = sum.
///
/// a + b = sum
fn sum<E: Engine, A, AR, CS: ConstraintSystem<E>>(
    cs: &mut CS,
    annotation: A,
    a: &AllocatedNum<E>,
    b: &AllocatedNum<E>,
    sum: &AllocatedNum<E>,
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

/// Adds a constraint to CS, enforcing that sum is the sum of nums.
fn multi_sum<E: Engine, A, AR, CS: ConstraintSystem<E>>(
    cs: &mut CS,
    annotation: A,
    nums: &[AllocatedNum<E>],
    sum: &AllocatedNum<E>,
) where
    A: FnOnce() -> AR,
    AR: Into<String>,
{
    // (num[0] + num[1] + … + num[n]) * 1 = sum
    cs.enforce(
        annotation,
        |lc| nums.iter().fold(lc, |acc, num| acc + num.get_variable()),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
    );
}

fn add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &E::Fr,
) -> Result<AllocatedNum<E>, SynthesisError> {
    let sum = AllocatedNum::alloc(cs.namespace(|| "add"), || {
        let mut tmp = a
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(b);

        Ok(tmp)
    })?;

    // a + b = sum
    cs.enforce(
        || "sum constraint",
        |lc| lc + a.get_variable() + (*b, CS::one()),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
    );

    Ok(sum)
}

fn multi_add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    nums: &[AllocatedNum<E>],
) -> Result<AllocatedNum<E>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "multi_add"), || {
        Ok(nums.iter().fold(E::Fr::zero(), |mut acc, num| {
            acc.add_assign(
                &num.get_value()
                    .ok_or_else(|| SynthesisError::AssignmentMissing)
                    .unwrap(),
            );
            acc
        }))
    })?;

    // a + b = res
    multi_sum(&mut cs, || "sum constraint", nums, &res);

    Ok(res)
}

fn scalar_product<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    nums: &[AllocatedNum<E>],
    scalars: &[E::Fr],
    to_add: Option<E::Fr>,
) -> Result<AllocatedNum<E>, SynthesisError> {
    let product = AllocatedNum::alloc(cs.namespace(|| "scalar product"), || {
        let mut tmp = nums
            .iter()
            .zip(scalars)
            .fold(E::Fr::zero(), |mut acc, (num, scalar)| {
                let mut x = num
                    .get_value()
                    .ok_or_else(|| SynthesisError::AssignmentMissing)
                    .unwrap();
                x.mul_assign(scalar);
                acc.add_assign(&x);
                acc
            });
        if let Some(a) = to_add {
            tmp.add_assign(&a);
        }
        Ok(tmp)
    })?;

    cs.enforce(
        || "scalar product constraint",
        |lc| {
            let base = scalars
                .iter()
                .zip(nums)
                .fold(lc, |acc, (scalar, num)| acc + (*scalar, num.get_variable()));

            if let Some(a) = to_add {
                base + (a, CS::one())
            } else {
                base
            }
        },
        |lc| lc + CS::one(),
        |lc| lc + product.get_variable(),
    );

    Ok(product)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::TestConstraintSystem;
    use crate::{scalar_from_u64, Poseidon, ARITY};
    use bellperson::ConstraintSystem;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let cases = [(2, 426), (4, 608), (8, 972)];

        for (arity, constraints) in &cases {
            if *arity != ARITY {
                continue;
            }
            let mut cs = TestConstraintSystem::<Bls12>::new();
            let mut i = 0;
            let mut fr_data = [Fr::zero(); ARITY];
            let data: Vec<AllocatedNum<Bls12>> = (0..ARITY)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    fr_data[i] = fr;
                    i += 1;
                    AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
                })
                .collect::<Vec<_>>();

            let out = poseidon_hash_simple(&mut cs, data).expect("poseidon hashing failed");

            let mut p = Poseidon::<Bls12>::new(&fr_data);
            let expected: Fr = p.hash();

            assert!(cs.is_satisfied(), "constraints not satisfied");

            assert_eq!(
                expected,
                out.get_value().unwrap(),
                "circuit and non-circuit do not match"
            );

            assert_eq!(
                cs.num_constraints(),
                *constraints,
                "constraint size changed",
            );
        }
    }
    #[test]
    fn test_square_sum() {
        let mut cs = TestConstraintSystem::<Bls12>::new();

        let mut cs1 = cs.namespace(|| "square_sum");
        let two = scalar_from_u64::<Bls12>(2);
        let three = AllocatedNum::alloc(cs1.namespace(|| "three"), || {
            Ok(scalar_from_u64::<Bls12>(3))
        })
        .unwrap();
        let res = square_sum(cs1, two, &three).unwrap();

        let twenty_five: Fr = scalar_from_u64::<Bls12>(25);
        assert_eq!(twenty_five, res.get_value().unwrap());
    }

    #[test]
    fn test_scalar_product() {
        let mut cs = TestConstraintSystem::<Bls12>::new();
        let two = AllocatedNum::alloc(cs.namespace(|| "two"), || Ok(scalar_from_u64::<Bls12>(2)))
            .unwrap();
        let three =
            AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64::<Bls12>(3)))
                .unwrap();
        let four = AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64::<Bls12>(4)))
            .unwrap();

        let res = scalar_product(
            cs,
            &[two, three, four],
            &[
                scalar_from_u64::<Bls12>(5),
                scalar_from_u64::<Bls12>(6),
                scalar_from_u64::<Bls12>(7),
            ],
            None,
        )
        .unwrap();

        assert_eq!(scalar_from_u64::<Bls12>(56), res.get_value().unwrap());
    }
    #[test]
    fn test_scalar_product_with_add() {
        let mut cs = TestConstraintSystem::<Bls12>::new();
        let two = AllocatedNum::alloc(cs.namespace(|| "two"), || Ok(scalar_from_u64::<Bls12>(2)))
            .unwrap();
        let three =
            AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64::<Bls12>(3)))
                .unwrap();
        let four = AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64::<Bls12>(4)))
            .unwrap();

        let res = scalar_product(
            cs,
            &[two, three, four],
            &[
                scalar_from_u64::<Bls12>(5),
                scalar_from_u64::<Bls12>(6),
                scalar_from_u64::<Bls12>(7),
            ],
            Some(scalar_from_u64::<Bls12>(3)),
        )
        .unwrap();

        assert_eq!(scalar_from_u64::<Bls12>(59), res.get_value().unwrap());
    }
}
