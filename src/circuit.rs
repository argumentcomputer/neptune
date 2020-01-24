use crate::poseidon::ARITY_TAG;
use crate::{FULL_ROUNDS, MDS_MATRIX, PARTIAL_ROUNDS, ROUND_CONSTANTS, WIDTH};

use bellperson::gadgets::num;
use bellperson::gadgets::num::AllocatedNum;
use bellperson::{ConstraintSystem, SynthesisError};
use ff::{Field, ScalarEngine};
use paired::bls12_381::Bls12;
use paired::Engine;

#[derive(Clone)]
pub struct PoseidonCircuit<E: Engine> {
    constants_offset: usize,
    round_constants: Vec<AllocatedNum<E>>, // &'a [E::Fr],
    width: usize,
    elements: Vec<AllocatedNum<E>>,
    pos: usize,
    full_rounds: usize,
    partial_rounds: usize,
    mds_matrix: Vec<Vec<AllocatedNum<E>>>,
}

impl<E: Engine> PoseidonCircuit<E> {
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(
        elements: Vec<AllocatedNum<E>>,
        matrix: Vec<Vec<AllocatedNum<E>>>,
        round_constants: Vec<AllocatedNum<E>>,
    ) -> Self {
        let width = WIDTH;

        PoseidonCircuit {
            constants_offset: 0,
            round_constants,
            width,
            elements,
            pos: width,
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
        //panic!("xxx");

        for i in 0..self.full_rounds / 2 {
            self.full_round(cs.namespace(|| format!("final full round {}", i)))?;
        }

        Ok(self.elements[1].clone())
    }

    fn full_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        // Apply the quintic S-Box to all elements
        for i in 0..self.elements.len() {
            let round_key = self.round_constants[constants_offset].get_value();
            constants_offset += 1;

            self.elements[i] = quintic_s_box(
                cs.namespace(|| format!("quintic s-box {}", i)),
                &self.elements[i],
                round_key,
            )?
        }
        self.constants_offset = constants_offset;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), false)?;
        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let round_key = self.round_constants[self.constants_offset].get_value();
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            round_key,
        )?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), true)?;

        Ok(())
    }

    // This generates more constraints but is clearer, so keep for documentation.
    #[allow(dead_code)]
    fn full_round_old<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants(cs.namespace(|| "add r"))?;

        // Apply the quintic S-Box to all elements
        for i in 0..self.elements.len() {
            self.elements[i] = quintic_s_box(
                cs.namespace(|| format!("quintic s-box {}", i)),
                &self.elements[i],
                None,
            )?
        }

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), false)?;
        Ok(())
    }

    // This generates more constraints but is clearer, so keep for documentation.
    #[allow(dead_code)]
    fn partial_round_old<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants(cs.namespace(|| "add r"))?;

        // Apply the quintic S-Box to the first element.
        self.elements[0] =
            quintic_s_box(cs.namespace(|| "quintic s-box"), &self.elements[0], None)?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds(cs.namespace(|| "mds matrix product"), false)?;

        Ok(())
    }

    fn add_round_constants<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        for i in 0..self.elements.len() {
            let constant = &self.round_constants[constants_offset];
            constants_offset += 1;

            self.elements[i] = add(
                cs.namespace(|| format!("add round key {}", i)),
                &self.elements[i],
                &constant,
            )?;
        }

        self.constants_offset = constants_offset;

        Ok(())
    }

    fn product_mds<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
        add_round_keys: bool,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<AllocatedNum<E>> = Vec::with_capacity(WIDTH);
        let round_offset = self.constants_offset - 1; // Account for the constant already used.

        for j in 0..WIDTH {
            let mut to_add = Vec::new();

            for k in 0..WIDTH {
                let tmp = &self.mds_matrix[j][k];

                let product = if add_round_keys && k > 0 {
                    mul_sum(
                        cs.namespace(|| {
                            format!("multiply matrix element ({}, {}), adding round key", j, k)
                        }),
                        &tmp,
                        &self.elements[k],
                        self.round_constants[round_offset + k].get_value().unwrap(), // FIXME: Just make the constants be scalars.
                    )?
                } else {
                    tmp.mul(
                        cs.namespace(|| format!("multiply matrix element ({}, {})", j, k)),
                        &self.elements[k],
                    )?
                };

                to_add.push(product);
            }
            result.push(multi_add(
                cs.namespace(|| format!("sum row ({})", j)),
                to_add.as_slice(),
            )?);
        }
        self.elements = result;
        if add_round_keys {
            self.constants_offset += WIDTH - 1;
        }

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
}

fn poseidon_hash<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    mut preimage: Vec<AllocatedNum<Bls12>>,
) -> Result<AllocatedNum<Bls12>, SynthesisError> {
    let matrix = allocated_matrix(cs.namespace(|| "allocated matrix"), *MDS_MATRIX)?;
    let round_constants = allocated_round_constants(
        cs.namespace(|| "allocated round constants"),
        &*ROUND_CONSTANTS,
    )?;
    // Add the arity tag to the front of the preimage.
    let arity_tag = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(*ARITY_TAG))?;
    preimage.push(arity_tag);
    preimage.rotate_right(1);

    let mut p = PoseidonCircuit::new(preimage, matrix, round_constants);
    p.hash(cs)
}

fn allocated_matrix<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    fr_matrix: [[<paired::bls12_381::Bls12 as ScalarEngine>::Fr; WIDTH]; WIDTH],
) -> Result<Vec<Vec<AllocatedNum<Bls12>>>, SynthesisError> {
    let mut mat: Vec<Vec<AllocatedNum<Bls12>>> = Vec::new();
    // Should probably call these columns.
    for (i, row) in fr_matrix.iter().enumerate() {
        mat.push({
            let mut allocated_row = Vec::new();
            for (j, val) in row.iter().enumerate() {
                allocated_row.push(AllocatedNum::alloc(
                    cs.namespace(|| format!("mds matrix element ({},{})", i, j)),
                    || Ok(*val),
                )?)
            }
            allocated_row
        });
    }
    Ok(mat)
}

fn allocated_round_constants<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    fr_constants: &[<paired::bls12_381::Bls12 as ScalarEngine>::Fr],
) -> Result<Vec<AllocatedNum<Bls12>>, SynthesisError> {
    let mut allocated_constants: Vec<AllocatedNum<Bls12>> = Vec::new();
    for (i, val) in fr_constants.iter().enumerate() {
        allocated_constants.push(AllocatedNum::alloc(
            cs.namespace(|| format!("round constant {}", i)),
            || Ok(*val),
        )?)
    }
    Ok(allocated_constants)
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

/// Adds a constraint to CS, enforcing that sum is the sum of nums.
pub fn multi_sum<E: Engine, A, AR, CS: ConstraintSystem<E>>(
    cs: &mut CS,
    annotation: A,
    nums: &[num::AllocatedNum<E>],
    sum: &num::AllocatedNum<E>,
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

pub fn add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    a: &num::AllocatedNum<E>,
    b: &num::AllocatedNum<E>,
) -> Result<num::AllocatedNum<E>, SynthesisError> {
    let res = num::AllocatedNum::alloc(cs.namespace(|| "add"), || {
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

pub fn multi_add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    nums: &[num::AllocatedNum<E>],
) -> Result<num::AllocatedNum<E>, SynthesisError> {
    let res = num::AllocatedNum::alloc(cs.namespace(|| "multi_add"), || {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::TestConstraintSystem;
    use crate::{generate_mds, scalar_from_u64, Poseidon, ARITY, WIDTH};
    use bellperson::ConstraintSystem;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let t = WIDTH;
        let cases = [(2, 993), (4, 2175), (8, 6051)];

        let matrix = generate_mds(WIDTH);

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

            let out = poseidon_hash(&mut cs, data).expect("poseidon hashing failed");

            let mut p = Poseidon::new(fr_data);
            let expected = p.hash();

            assert_eq!(
                expected,
                out.get_value().unwrap(),
                "circuit and non-circuit do not match"
            );

            assert!(cs.is_satisfied(), "constraints not satisfied");
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
        let two = scalar_from_u64(2);
        let three =
            AllocatedNum::alloc(cs1.namespace(|| "three"), || Ok(scalar_from_u64(3))).unwrap();
        let res = square_sum(cs1, two, &three).unwrap();

        let twenty_five = scalar_from_u64(25);
        assert_eq!(twenty_five, res.get_value().unwrap());
    }
}
