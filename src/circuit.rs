/// Implement the legacy Poseidon hash circuit, retained for backwards compatibility with extant trusted setups.
use std::ops::{AddAssign, MulAssign};

use crate::circuit2::poseidon_hash_allocated;
use crate::hash_type::HashType;
use crate::matrix::Matrix;
use crate::mds::SparseMatrix;
use crate::poseidon::{Arity, PoseidonConstants};
use bellpepper_core::boolean::Boolean;
use bellpepper_core::num;
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError};
use ff::{Field, PrimeField};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub enum CircuitType {
    Legacy,
    OptimalAllocated,
}

impl CircuitType {
    pub fn label(&self) -> String {
        match self {
            Self::Legacy => "Legacy Poseidon Circuit".into(),
            Self::OptimalAllocated => "Optimal Allocated Poseidon Circuit,".into(),
        }
    }
}

/// Convenience function to potentially ease upgrade transition from legacy to optimal circuits.
pub fn poseidon_hash_circuit<CS, Scalar, A>(
    cs: CS,
    circuit_type: CircuitType,
    preimage: Vec<AllocatedNum<Scalar>>,
    constants: &PoseidonConstants<Scalar, A>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
    CS: ConstraintSystem<Scalar>,
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    match circuit_type {
        CircuitType::Legacy => poseidon_hash(cs, preimage, constants),
        CircuitType::OptimalAllocated => poseidon_hash_allocated(cs, preimage, constants),
    }
}

/// Similar to `num::Num`, we use `Elt` to accumulate both values and linear combinations, then eventually
/// extract into a `num::AllocatedNum`, enforcing that the linear combination corresponds to the result.
#[derive(Clone)]
enum Elt<Scalar: PrimeField> {
    Allocated(AllocatedNum<Scalar>),
    Num(num::Num<Scalar>),
}

impl<Scalar: PrimeField> Elt<Scalar> {
    const fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated(_))
    }

    const fn is_num(&self) -> bool {
        matches!(self, Self::Num(_))
    }

    fn num_from_fr<CS: ConstraintSystem<Scalar>>(fr: Scalar) -> Self {
        let num = num::Num::<Scalar>::zero();
        Self::Num(num.add_bool_with_coeff(CS::one(), &Boolean::Constant(true), fr))
    }

    fn ensure_allocated<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        enforce: bool,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        match self {
            Self::Allocated(v) => Ok(v.clone()),
            Self::Num(num) => {
                let v = AllocatedNum::alloc(cs.namespace(|| "allocate for Elt::Num"), || {
                    num.get_value().ok_or(SynthesisError::AssignmentMissing)
                })?;

                if enforce {
                    cs.enforce(
                        || "enforce num allocation preserves lc".to_string(),
                        |_| num.lc(Scalar::ONE),
                        |lc| lc + CS::one(),
                        |lc| lc + v.get_variable(),
                    );
                }
                Ok(v)
            }
        }
    }

    fn val(&self) -> Option<Scalar> {
        match self {
            Self::Allocated(v) => v.get_value(),
            Self::Num(num) => num.get_value(),
        }
    }

    fn lc(&self) -> LinearCombination<Scalar> {
        match self {
            Self::Num(num) => num.lc(Scalar::ONE),
            Self::Allocated(v) => LinearCombination::<Scalar>::zero() + v.get_variable(),
        }
    }

    /// Add two Nums and return a Num tracking the calculation. It is forbidden to invoke on an Allocated because the intended computation
    /// does not include that path.
    fn add(self, other: Elt<Scalar>) -> Elt<Scalar> {
        match (self, other) {
            (Elt::Num(a), Elt::Num(b)) => Elt::Num(a.add(&b)),
            _ => panic!("only two numbers may be added"),
        }
    }

    /// Scale
    fn scale<CS: ConstraintSystem<Scalar>>(self, scalar: Scalar) -> Elt<Scalar> {
        match self {
            Elt::Num(num) => Elt::Num(num.scale(scalar)),
            Elt::Allocated(a) => Elt::Num(a.into()).scale::<CS>(scalar),
        }
    }
}

/// Circuit for Poseidon hash.
pub struct PoseidonCircuit<'a, Scalar, A>
where
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    constants_offset: usize,
    width: usize,
    elements: Vec<Elt<Scalar>>,
    pos: usize,
    current_round: usize,
    constants: &'a PoseidonConstants<Scalar, A>,
    _w: PhantomData<A>,
}

/// PoseidonCircuit implementation.
impl<'a, Scalar, A> PoseidonCircuit<'a, Scalar, A>
where
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    /// Create a new Poseidon hasher for `preimage`.
    fn new(elements: Vec<Elt<Scalar>>, constants: &'a PoseidonConstants<Scalar, A>) -> Self {
        let width = constants.width();

        PoseidonCircuit {
            constants_offset: 0,
            width,
            elements,
            pos: width,
            current_round: 0,
            constants,
            _w: PhantomData::<A>,
        }
    }

    fn hash<CS: ConstraintSystem<Scalar>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        self.full_round(cs.namespace(|| "first round"), true, false)?;

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(
                cs.namespace(|| format!("initial full round {i}")),
                false,
                false,
            )?;
        }

        for i in 0..self.constants.partial_rounds {
            self.partial_round(cs.namespace(|| format!("partial round {i}")))?;
        }

        for i in 0..(self.constants.full_rounds / 2) - 1 {
            self.full_round(
                cs.namespace(|| format!("final full round {i}")),
                false,
                false,
            )?;
        }
        self.full_round(cs.namespace(|| "terminal full round"), false, true)?;

        self.elements[1].ensure_allocated(&mut cs.namespace(|| "hash result"), true)
    }

    fn full_round<CS: ConstraintSystem<Scalar>>(
        &mut self,
        mut cs: CS,
        first_round: bool,
        last_round: bool,
    ) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        let pre_round_keys = if first_round {
            (0..self.width)
                .map(|i| self.constants.compressed_round_constants[constants_offset + i])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        constants_offset += pre_round_keys.len();

        let post_round_keys = if first_round || !last_round {
            (0..self.width)
                .map(|i| self.constants.compressed_round_constants[constants_offset + i])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        constants_offset += post_round_keys.len();

        // Apply the quintic S-Box to all elements
        for i in 0..self.elements.len() {
            let pre_round_key = if first_round {
                let rk = pre_round_keys[i];
                Some(rk)
            } else {
                None
            };

            let post_round_key = if first_round || !last_round {
                let rk = post_round_keys[i];
                Some(rk)
            } else {
                None
            };

            if first_round {
                if i == 0 {
                    // The very first s-box for the constant arity tag can also be computed statically, as a constant.
                    self.elements[i] = constant_quintic_s_box_pre_add_tag::<CS, Scalar>(
                        &self.elements[i],
                        pre_round_key,
                        post_round_key,
                    );
                } else {
                    self.elements[i] = quintic_s_box_pre_add(
                        cs.namespace(|| format!("quintic s-box {i}")),
                        &self.elements[i],
                        pre_round_key,
                        post_round_key,
                    )?;
                }
            } else {
                self.elements[i] = quintic_s_box(
                    cs.namespace(|| format!("quintic s-box {i}")),
                    &self.elements[i],
                    post_round_key,
                )?;
            }
        }
        self.constants_offset = constants_offset;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>();
        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<Scalar>>(
        &mut self,
        mut cs: CS,
    ) -> Result<(), SynthesisError> {
        let round_key = self.constants.compressed_round_constants[self.constants_offset];
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            Some(round_key),
        )?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>();
        Ok(())
    }

    fn product_mds_m<CS: ConstraintSystem<Scalar>>(&mut self) {
        self.product_mds_with_matrix::<CS>(&self.constants.mds_matrices.m)
    }

    /// Set the provided elements with the result of the product between the elements and the appropriate
    /// MDS matrix.
    #[allow(clippy::collapsible_else_if)]
    fn product_mds<CS: ConstraintSystem<Scalar>>(&mut self) {
        let full_half = self.constants.half_full_rounds;
        let sparse_offset = full_half - 1;
        if self.current_round == sparse_offset {
            self.product_mds_with_matrix::<CS>(&self.constants.pre_sparse_matrix);
        } else {
            if (self.current_round > sparse_offset)
                && (self.current_round < full_half + self.constants.partial_rounds)
            {
                let index = self.current_round - sparse_offset - 1;
                let sparse_matrix = &self.constants.sparse_matrixes[index];

                self.product_mds_with_sparse_matrix::<CS>(sparse_matrix);
            } else {
                self.product_mds_m::<CS>();
            }
        };

        self.current_round += 1;
    }

    #[allow(clippy::ptr_arg)]
    fn product_mds_with_matrix<CS: ConstraintSystem<Scalar>>(&mut self, matrix: &Matrix<Scalar>) {
        let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

        for j in 0..self.constants.width() {
            let column = (0..self.constants.width())
                .map(|i| matrix[i][j])
                .collect::<Vec<_>>();

            let product = scalar_product::<Scalar, CS>(self.elements.as_slice(), &column);

            result.push(product);
        }

        self.elements = result;
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix<CS: ConstraintSystem<Scalar>>(
        &mut self,
        matrix: &SparseMatrix<Scalar>,
    ) {
        let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

        result.push(scalar_product::<Scalar, CS>(
            self.elements.as_slice(),
            &matrix.w_hat,
        ));

        for j in 1..self.width {
            result.push(
                self.elements[j].clone().add(
                    self.elements[0]
                        .clone() // First row is dense.
                        .scale::<CS>(matrix.v_rest[j - 1]), // Except for first row/column, diagonals are one.
                ),
            );
        }

        self.elements = result;
    }

    fn debug(&self) {
        let element_frs: Vec<_> = self.elements.iter().map(|n| n.val()).collect::<Vec<_>>();
        dbg!(element_frs, self.constants_offset);
    }
}

/// Create legacy circuit for Poseidon hash. If possible, prefer the equivalent 'optimal' alternatives.
pub fn poseidon_hash<CS, Scalar, A>(
    cs: CS,
    preimage: Vec<AllocatedNum<Scalar>>,
    constants: &PoseidonConstants<Scalar, A>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
    CS: ConstraintSystem<Scalar>,
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    let arity = A::to_usize();
    let tag_element = Elt::num_from_fr::<CS>(constants.domain_tag);
    let mut elements = Vec::with_capacity(arity + 1);
    elements.push(tag_element);
    elements.extend(preimage.into_iter().map(Elt::Allocated));

    if let HashType::ConstantLength(length) = constants.hash_type {
        assert!(length == arity, "Length must be equal to arity since no padding is provided. Check circuit2.rs for optimized and complete implementation");
    }

    let mut p = PoseidonCircuit::new(elements, constants);

    p.hash(cs)
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to result.
fn quintic_s_box<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    e: &Elt<Scalar>,
    post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
    let l = e.ensure_allocated(&mut cs.namespace(|| "S-box input"), true)?;

    // If round_key was supplied, add it after all exponentiation.
    let l2 = l.square(cs.namespace(|| "l^2"))?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(
        cs.namespace(|| "(l4 * l) + rk)"),
        &l4,
        &l,
        None,
        post_round_key,
        true,
    );

    Ok(Elt::Allocated(l5?))
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box_pre_add<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    e: &Elt<Scalar>,
    pre_round_key: Option<Scalar>,
    post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
    if let (Some(pre_round_key), Some(post_round_key)) = (pre_round_key, post_round_key) {
        let l = e.ensure_allocated(&mut cs.namespace(|| "S-box input"), true)?;

        // If round_key was supplied, add it to l before squaring.
        let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), pre_round_key, &l, true)?;
        let l4 = l2.square(cs.namespace(|| "l^4"))?;
        let l5 = mul_sum(
            cs.namespace(|| "l4 * (l + rk)"),
            &l4,
            &l,
            Some(pre_round_key),
            Some(post_round_key),
            true,
        );

        Ok(Elt::Allocated(l5?))
    } else {
        panic!("pre_round_key and post_round_key must both be provided.");
    }
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn constant_quintic_s_box_pre_add_tag<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    tag: &Elt<Scalar>,
    pre_round_key: Option<Scalar>,
    post_round_key: Option<Scalar>,
) -> Elt<Scalar> {
    let mut tag = tag.val().expect("missing tag val");
    pre_round_key.expect("pre_round_key must be provided");
    post_round_key.expect("post_round_key must be provided");

    crate::quintic_s_box::<Scalar>(&mut tag, pre_round_key.as_ref(), post_round_key.as_ref());

    Elt::num_from_fr::<CS>(tag)
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    to_add: Scalar,
    num: &AllocatedNum<Scalar>,
    enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "squared sum"), || {
        let mut tmp = num.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp = tmp.square();

        Ok(tmp)
    })?;

    if enforce {
        cs.enforce(
            || "squared sum constraint",
            |lc| lc + num.get_variable() + (to_add, CS::one()),
            |lc| lc + num.get_variable() + (to_add, CS::one()),
            |lc| lc + res.get_variable(),
        );
    }
    Ok(res)
}

/// Calculates (a * (pre_add + b)) + post_add — and enforces that constraint.
#[allow(clippy::collapsible_else_if)]
pub fn mul_sum<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    a: &AllocatedNum<Scalar>,
    b: &AllocatedNum<Scalar>,
    pre_add: Option<Scalar>,
    post_add: Option<Scalar>,
    enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        if let Some(x) = pre_add {
            tmp.add_assign(&x);
        }
        tmp.mul_assign(&a.get_value().ok_or(SynthesisError::AssignmentMissing)?);
        if let Some(x) = post_add {
            tmp.add_assign(&x);
        }

        Ok(tmp)
    })?;

    if enforce {
        if let Some(x) = post_add {
            let neg = -x;

            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul sum constraint pre-post-add",
                    |lc| lc + b.get_variable() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint post-add",
                    |lc| lc + b.get_variable(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            }
        } else {
            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul sum constraint pre-add",
                    |lc| lc + b.get_variable() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable(),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint",
                    |lc| lc + b.get_variable(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable(),
                );
            }
        }
    }
    Ok(res)
}

fn scalar_product_with_add<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    elts: &[Elt<Scalar>],
    scalars: &[Scalar],
    to_add: Scalar,
) -> Elt<Scalar> {
    let tmp = scalar_product::<Scalar, CS>(elts, scalars);
    tmp.add(Elt::<Scalar>::num_from_fr::<CS>(to_add))
}

fn scalar_product<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    elts: &[Elt<Scalar>],
    scalars: &[Scalar],
) -> Elt<Scalar> {
    elts.iter()
        .zip(scalars)
        .fold(Elt::Num(num::Num::zero()), |acc, (elt, &scalar)| {
            acc.add(elt.clone().scale::<CS>(scalar))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::HashMode;
    use crate::{Poseidon, Strength};
    use bellpepper_core::test_cs::TestConstraintSystem;
    use bellpepper_core::ConstraintSystem;
    use blstrs::Scalar as Fr;
    use generic_array::typenum;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        test_poseidon_hash_aux::<typenum::U2>(Strength::Standard, 311, false);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Standard, 377, false);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Standard, 505, false);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Standard, 761, false);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Standard, 1009, false);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Standard, 1385, false);

        test_poseidon_hash_aux::<typenum::U2>(Strength::Strengthened, 367, false);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Strengthened, 433, false);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Strengthened, 565, false);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Strengthened, 821, false);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Strengthened, 1069, false);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Strengthened, 1445, false);
    }

    fn test_poseidon_hash_aux<A>(
        strength: Strength,
        expected_constraints: usize,
        constant_length: bool,
    ) where
        A: Arity<Fr>,
    {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let arity = A::to_usize();
        let constants_x = if constant_length {
            PoseidonConstants::<Fr, A>::new_with_strength_and_type(
                strength,
                HashType::ConstantLength(arity),
            )
        } else {
            PoseidonConstants::<Fr, A>::new_with_strength(strength)
        };

        let range = if constant_length {
            1..=arity
        } else {
            arity..=arity
        };
        for preimage_length in range {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let constants = if constant_length {
                constants_x.with_length(preimage_length)
            } else {
                constants_x.clone()
            };
            let expected_constraints_calculated = {
                let arity_tag_constraints = 0;
                let width = 1 + arity;
                // The '- 1' term represents the first s-box for the arity tag, which is a constant and needs no constraint.
                let s_boxes = (width * constants.full_rounds) + constants.partial_rounds - 1;
                let s_box_constraints = 3 * s_boxes;
                let mds_constraints =
                    (width * constants.full_rounds) + constants.partial_rounds - arity;
                arity_tag_constraints + s_box_constraints + mds_constraints
            };
            let mut i = 0;

            let mut fr_data = vec![Fr::ZERO; preimage_length];
            let data: Vec<AllocatedNum<Fr>> = (0..preimage_length)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    fr_data[i] = fr;
                    i += 1;
                    AllocatedNum::alloc_infallible(cs.namespace(|| format!("data {}", i)), || fr)
                })
                .collect::<Vec<_>>();

            let out = poseidon_hash(&mut cs, data, &constants).expect("poseidon hashing failed");

            let mut p = Poseidon::<Fr, A>::new_with_preimage(&fr_data, &constants);
            let expected: Fr = p.hash_in_mode(HashMode::Correct);

            assert!(cs.is_satisfied(), "constraints not satisfied");

            assert_eq!(
                expected,
                out.get_value().unwrap(),
                "circuit and non-circuit do not match"
            );

            assert_eq!(
                expected_constraints_calculated,
                cs.num_constraints(),
                "constraint number miscalculated"
            );

            assert_eq!(
                expected_constraints,
                cs.num_constraints(),
                "constraint number changed",
            );
        }
    }

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    fn efr(n: u64) -> Elt<Fr> {
        Elt::num_from_fr::<TestConstraintSystem<Fr>>(fr(n))
    }

    #[test]
    fn test_square_sum() {
        let mut cs = TestConstraintSystem::<Fr>::new();

        let mut cs1 = cs.namespace(|| "square_sum");
        let two = fr(2);
        let three = AllocatedNum::alloc_infallible(cs1.namespace(|| "three"), || Fr::from(3));
        let res = square_sum(cs1, two, &three, true).unwrap();

        let twenty_five = Fr::from(25);
        assert_eq!(twenty_five, res.get_value().unwrap());
    }

    #[test]
    fn test_scalar_product() {
        {
            // Inputs are all linear combinations.
            let two = efr(2);
            let three = efr(3);
            let four = efr(4);

            let res = scalar_product::<Fr, TestConstraintSystem<Fr>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            );

            assert!(res.is_num());
            assert_eq!(Fr::from(56), res.val().unwrap());
        }
        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 = AllocatedNum::alloc_infallible(cs.namespace(|| "three"), || Fr::from(3));
            let three = Elt::Allocated(n3.clone());
            let n4 = AllocatedNum::alloc_infallible(cs.namespace(|| "four"), || Fr::from(4));
            let four = Elt::Allocated(n4.clone());

            let res = scalar_product::<Fr, TestConstraintSystem<Fr>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            );

            assert!(res.is_num());
            assert_eq!(Fr::from(56), res.val().unwrap());

            res.lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(6));
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(7));
                };
            });

            res.ensure_allocated(&mut cs, true).unwrap();
            assert!(cs.is_satisfied());
        }
        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 = AllocatedNum::alloc_infallible(cs.namespace(|| "three"), || Fr::from(3));
            let three = Elt::Allocated(n3.clone());
            let n4 = AllocatedNum::alloc_infallible(cs.namespace(|| "four"), || Fr::from(4));
            let four = Elt::Allocated(n4.clone());

            let mut res_vec = Vec::new();

            let res = scalar_product::<Fr, TestConstraintSystem<Fr>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            );

            res_vec.push(res);

            assert!(res_vec[0].is_num());
            assert_eq!(fr(56), res_vec[0].val().unwrap());

            res_vec[0].lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(6)); // 6 * three
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(7)); // 7 * four
                };
            });

            let four2 = Elt::Allocated(n4.clone());
            res_vec.push(efr(3));
            res_vec.push(four2);
            let res2 =
                scalar_product::<Fr, TestConstraintSystem<Fr>>(&res_vec, &[fr(7), fr(8), fr(9)]);

            res2.lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(42)); // 7 * 6 * three
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(58)); // (7 * 7 * four) + (9 * four)
                };
            });

            let allocated = res2.ensure_allocated(&mut cs, true).unwrap();

            let v = allocated.get_value().unwrap();
            assert_eq!(fr(452), v); // (7 * 56) + (8 * 3) + (9 * 4) = 448

            assert!(cs.is_satisfied());
        }
    }

    #[test]
    fn test_scalar_product_with_add() {
        let two = efr(2);
        let three = efr(3);
        let four = efr(4);

        let res = scalar_product_with_add::<Fr, TestConstraintSystem<Fr>>(
            &[two, three, four],
            &[fr(5), fr(6), fr(7)],
            fr(3),
        );

        assert!(res.is_num());
        assert_eq!(fr(59), res.val().unwrap());
    }
}
