/// The `circuit2` module implements the optimal Poseidon hash circuit.
use std::ops::{AddAssign, MulAssign};

use crate::circuit2_witness::poseidon_hash_allocated_witness;
use crate::hash_type::HashType;
use crate::matrix::Matrix;
use crate::mds::SparseMatrix;
use crate::poseidon::{Arity, PoseidonConstants};
use bellpepper_core::boolean::Boolean;
use bellpepper_core::num::{self, AllocatedNum};
use bellpepper_core::test_cs::TestConstraintSystem;
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError};
use ff::{Field, PrimeField};
use std::marker::PhantomData;

/// Similar to `num::Num`, we use `Elt` to accumulate both values and linear combinations, then eventually
/// extract into a `num::AllocatedNum`, enforcing that the linear combination corresponds to the result.
#[derive(Clone)]
pub enum Elt<Scalar: PrimeField> {
    Allocated(AllocatedNum<Scalar>),
    Num(num::Num<Scalar>),
}

impl<Scalar: PrimeField> From<AllocatedNum<Scalar>> for Elt<Scalar> {
    fn from(allocated: AllocatedNum<Scalar>) -> Self {
        Self::Allocated(allocated)
    }
}

impl<Scalar: PrimeField> Elt<Scalar> {
    pub const fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated(_))
    }

    pub const fn is_num(&self) -> bool {
        matches!(self, Self::Num(_))
    }

    pub fn num_from_fr<CS: ConstraintSystem<Scalar>>(fr: Scalar) -> Self {
        let num = num::Num::<Scalar>::zero();
        Self::Num(num.add_bool_with_coeff(CS::one(), &Boolean::Constant(true), fr))
    }

    pub fn ensure_allocated<CS: ConstraintSystem<Scalar>>(
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

    pub fn val(&self) -> Option<Scalar> {
        match self {
            Self::Allocated(v) => v.get_value(),
            Self::Num(num) => num.get_value(),
        }
    }

    pub fn lc(&self) -> LinearCombination<Scalar> {
        match self {
            Self::Num(num) => num.lc(Scalar::ONE),
            Self::Allocated(v) => LinearCombination::<Scalar>::zero() + v.get_variable(),
        }
    }

    /// Add two Elts and return Elt::Num tracking the calculation.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Elt<Scalar>) -> Result<Elt<Scalar>, SynthesisError> {
        match (self, other) {
            (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(&b))),
            (a, b) => Ok(Elt::Num(a.num().add(&b.num()))),
        }
    }

    pub fn add_ref(self, other: &Elt<Scalar>) -> Result<Elt<Scalar>, SynthesisError> {
        match (self, other) {
            (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(b))),
            (a, b) => Ok(Elt::Num(a.num().add(&b.num()))),
        }
    }

    /// Scale
    pub fn scale<CS: ConstraintSystem<Scalar>>(
        self,
        scalar: Scalar,
    ) -> Result<Elt<Scalar>, SynthesisError> {
        match self {
            Elt::Num(num) => Ok(Elt::Num(num.scale(scalar))),
            Elt::Allocated(a) => Elt::Num(a.into()).scale::<CS>(scalar),
        }
    }

    /// Square
    pub fn square<CS: ConstraintSystem<Scalar>>(
        &self,
        mut cs: CS,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        match self {
            Elt::Num(num) => {
                let allocated = AllocatedNum::alloc(&mut cs.namespace(|| "squared num"), || {
                    num.get_value()
                        .ok_or(SynthesisError::AssignmentMissing)
                        .map(|tmp| tmp * tmp)
                })?;
                cs.enforce(
                    || "squaring constraint",
                    |_| num.lc(Scalar::ONE),
                    |_| num.lc(Scalar::ONE),
                    |lc| lc + allocated.get_variable(),
                );
                Ok(allocated)
            }
            Elt::Allocated(a) => a.square(cs),
        }
    }

    pub fn num(&self) -> num::Num<Scalar> {
        match self {
            Elt::Num(num) => num.clone(),
            Elt::Allocated(a) => a.clone().into(),
        }
    }
}

/// Circuit for Poseidon hash.
pub struct PoseidonCircuit2<'a, Scalar, A>
where
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    constants_offset: usize,
    width: usize,
    pub(crate) elements: Vec<Elt<Scalar>>,
    pub(crate) pos: usize,
    current_round: usize,
    constants: &'a PoseidonConstants<Scalar, A>,
    _w: PhantomData<A>,
}

/// PoseidonCircuit2 implementation.
impl<'a, Scalar, A> PoseidonCircuit2<'a, Scalar, A>
where
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(elements: Vec<Elt<Scalar>>, constants: &'a PoseidonConstants<Scalar, A>) -> Self {
        let width = constants.width();

        PoseidonCircuit2 {
            constants_offset: 0,
            width,
            elements,
            pos: 1,
            current_round: 0,
            constants,
            _w: PhantomData::<A>,
        }
    }

    pub fn new_empty<CS: ConstraintSystem<Scalar>>(
        constants: &'a PoseidonConstants<Scalar, A>,
    ) -> Self {
        let elements = Self::initial_elements::<CS>();
        Self::new(elements, constants)
    }

    pub fn hash<CS: ConstraintSystem<Scalar>>(
        &mut self,
        cs: &mut CS,
    ) -> Result<Elt<Scalar>, SynthesisError> {
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

        let elt = self.elements[1].clone();
        self.reset_offsets();

        Ok(elt)
    }

    pub fn apply_padding<CS: ConstraintSystem<Scalar>>(&mut self) {
        if let HashType::ConstantLength(l) = self.constants.hash_type {
            let final_pos = 1 + (l % self.constants.arity());

            assert_eq!(
                self.pos, final_pos,
                "preimage length does not match constant length required for hash"
            );
        };
        match self.constants.hash_type {
            HashType::ConstantLength(_) | HashType::Encryption => {
                for elt in self.elements[self.pos..].iter_mut() {
                    *elt = Elt::num_from_fr::<CS>(Scalar::ZERO);
                }
                self.pos = self.elements.len();
            }
            HashType::VariableLength => todo!(),
            _ => (), // incl HashType::Sponge
        }
    }

    pub fn hash_to_allocated<CS: ConstraintSystem<Scalar>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        let elt = self.hash(&mut cs).unwrap();
        elt.ensure_allocated(&mut cs, true)
    }

    fn hash_to_num<CS: ConstraintSystem<Scalar>>(
        &mut self,
        mut cs: CS,
    ) -> Result<num::Num<Scalar>, SynthesisError> {
        self.hash(&mut cs).map(|elt| elt.num())
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
                {
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
        self.product_mds::<CS>()?;
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
        self.product_mds::<CS>()?;
        Ok(())
    }

    fn product_mds_m<CS: ConstraintSystem<Scalar>>(&mut self) -> Result<(), SynthesisError> {
        self.product_mds_with_matrix::<CS>(&self.constants.mds_matrices.m)
    }

    /// Set the provided elements with the result of the product between the elements and the appropriate
    /// MDS matrix.
    #[allow(clippy::collapsible_else_if)]
    fn product_mds<CS: ConstraintSystem<Scalar>>(&mut self) -> Result<(), SynthesisError> {
        let full_half = self.constants.half_full_rounds;
        let sparse_offset = full_half - 1;
        if self.current_round == sparse_offset {
            self.product_mds_with_matrix::<CS>(&self.constants.pre_sparse_matrix)?;
        } else {
            if (self.current_round > sparse_offset)
                && (self.current_round < full_half + self.constants.partial_rounds)
            {
                let index = self.current_round - sparse_offset - 1;
                let sparse_matrix = &self.constants.sparse_matrixes[index];

                self.product_mds_with_sparse_matrix::<CS>(sparse_matrix)?;
            } else {
                self.product_mds_m::<CS>()?;
            }
        };

        self.current_round += 1;
        Ok(())
    }

    #[allow(clippy::ptr_arg)]
    fn product_mds_with_matrix<CS: ConstraintSystem<Scalar>>(
        &mut self,
        matrix: &Matrix<Scalar>,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

        for j in 0..self.constants.width() {
            let column = (0..self.constants.width())
                .map(|i| matrix[i][j])
                .collect::<Vec<_>>();

            let product = scalar_product::<Scalar, CS>(self.elements.as_slice(), &column)?;

            result.push(product);
        }

        self.elements = result;

        Ok(())
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix<CS: ConstraintSystem<Scalar>>(
        &mut self,
        matrix: &SparseMatrix<Scalar>,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

        result.push(scalar_product::<Scalar, CS>(
            self.elements.as_slice(),
            &matrix.w_hat,
        )?);

        for j in 1..self.width {
            result.push(
                self.elements[j].clone().add(
                    self.elements[0]
                        .clone() // First row is dense.
                        .scale::<CS>(matrix.v_rest[j - 1])?, // Except for first row/column, diagonals are one.
                )?,
            );
        }

        self.elements = result;

        Ok(())
    }

    fn initial_elements<CS: ConstraintSystem<Scalar>>() -> Vec<Elt<Scalar>> {
        std::iter::repeat(Elt::num_from_fr::<CS>(Scalar::ZERO))
            .take(A::to_usize() + 1)
            .collect()
    }
    pub fn reset<CS: ConstraintSystem<Scalar>>(&mut self) {
        self.reset_offsets();
        self.elements = Self::initial_elements::<CS>();
    }

    pub fn reset_offsets(&mut self) {
        self.constants_offset = 0;
        self.current_round = 0;
        self.pos = 1;
    }

    pub(crate) fn debug(&self) {
        let element_frs: Vec<_> = self.elements.iter().map(|n| n.val()).collect::<Vec<_>>();
        dbg!(element_frs, self.constants_offset);
    }
}

/// Create circuit for Poseidon hash, returning an allocated `Num` at the cost of one constraint.
pub fn poseidon_hash_allocated<CS, Scalar, A>(
    cs: CS,
    preimage: Vec<AllocatedNum<Scalar>>,
    constants: &PoseidonConstants<Scalar, A>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
    CS: ConstraintSystem<Scalar>,
    Scalar: PrimeField,
    A: Arity<Scalar>,
{
    if cs.is_witness_generator() {
        let mut cs = cs;
        poseidon_hash_allocated_witness(&mut cs, &preimage, constants)
    } else {
        let arity = A::to_usize();
        let tag_element = Elt::num_from_fr::<CS>(constants.domain_tag);
        let mut elements = Vec::with_capacity(arity + 1);
        elements.push(tag_element);
        elements.extend(preimage.into_iter().map(Elt::Allocated));

        if let HashType::ConstantLength(length) = constants.hash_type {
            assert!(length <= arity, "illegal length: constants are malformed");
            // Add zero-padding.
            for _ in 0..(arity - length) {
                let elt = Elt::Num(num::Num::zero());
                elements.push(elt);
            }
        }
        let mut p = PoseidonCircuit2::new(elements, constants);

        p.hash_to_allocated(cs)
    }
}

/// Create circuit for Poseidon hash, minimizing constraints by returning an unallocated `Num`.
pub fn poseidon_hash_num<CS, Scalar, A>(
    cs: CS,
    preimage: Vec<AllocatedNum<Scalar>>,
    constants: &PoseidonConstants<Scalar, A>,
) -> Result<num::Num<Scalar>, SynthesisError>
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
        assert!(length <= arity, "illegal length: constants are malformed");
        // Add zero-padding.
        for _ in 0..(arity - length) {
            let elt = Elt::Num(num::Num::zero());
            elements.push(elt);
        }
    }

    let mut p = PoseidonCircuit2::new(elements, constants);

    p.hash_to_num(cs)
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to result.
fn quintic_s_box<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    l: &Elt<Scalar>,
    post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
    // If round_key was supplied, add it after all exponentiation.
    let l2 = l.square(cs.namespace(|| "l^2"))?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(
        cs.namespace(|| "(l4 * l) + rk)"),
        &l4,
        l,
        None,
        post_round_key,
        true,
    );

    Ok(Elt::Allocated(l5?))
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box_pre_add<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    l: &Elt<Scalar>,
    pre_round_key: Option<Scalar>,
    post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
    if let (Some(pre_round_key), Some(post_round_key)) = (pre_round_key, post_round_key) {
        // If round_key was supplied, add it to l before squaring.
        let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), pre_round_key, l, true)?;
        let l4 = l2.square(cs.namespace(|| "l^4"))?;
        let l5 = mul_sum(
            cs.namespace(|| "l4 * (l + rk)"),
            &l4,
            l,
            Some(pre_round_key),
            Some(post_round_key),
            true,
        );

        Ok(Elt::Allocated(l5?))
    } else {
        panic!("pre_round_key and post_round_key must both be provided.");
    }
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
    mut cs: CS,
    to_add: Scalar,
    elt: &Elt<Scalar>,
    enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "squared sum"), || {
        let mut tmp = elt.val().ok_or(SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp = tmp.square();
        Ok(tmp)
    })?;

    if enforce {
        cs.enforce(
            || "squared sum constraint",
            |_| elt.lc() + (to_add, CS::one()),
            |_| elt.lc() + (to_add, CS::one()),
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
    b: &Elt<Scalar>,
    pre_add: Option<Scalar>,
    post_add: Option<Scalar>,
    enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b.val().ok_or(SynthesisError::AssignmentMissing)?;
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
                    |_| b.lc() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint post-add",
                    |_| b.lc(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            }
        } else {
            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul sum constraint pre-add",
                    |_| b.lc() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable(),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint",
                    |_| b.lc(),
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
) -> Result<Elt<Scalar>, SynthesisError> {
    let tmp = scalar_product::<Scalar, CS>(elts, scalars)?;
    let tmp2 = tmp.add(Elt::<Scalar>::num_from_fr::<CS>(to_add))?;

    Ok(tmp2)
}

fn scalar_product<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    elts: &[Elt<Scalar>],
    scalars: &[Scalar],
) -> Result<Elt<Scalar>, SynthesisError> {
    elts.iter()
        .zip(scalars)
        .try_fold(Elt::Num(num::Num::zero()), |acc, (elt, &scalar)| {
            acc.add(elt.clone().scale::<CS>(scalar)?)
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
        test_poseidon_hash_aux::<typenum::U2>(Strength::Standard, 237, false);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Standard, 288, false);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Standard, 387, false);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Standard, 585, false);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Standard, 777, false);
        test_poseidon_hash_aux::<typenum::U32>(Strength::Standard, 972, false);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Standard, 1068, false);

        test_poseidon_hash_aux::<typenum::U2>(Strength::Strengthened, 279, false);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Strengthened, 330, false);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Strengthened, 432, false);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Strengthened, 630, false);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Strengthened, 822, false);
        test_poseidon_hash_aux::<typenum::U32>(Strength::Strengthened, 1017, false);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Strengthened, 1113, false);

        test_poseidon_hash_aux::<typenum::U15>(Strength::Standard, 561, true);
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
            let constants = if constant_length {
                constants_x.with_length(preimage_length)
            } else {
                constants_x.clone()
            };

            let expected_constraints_calculated = {
                let width = 1 + arity;
                let s_box_cost = 3;

                (width * s_box_cost * constants.full_rounds)
                    + (s_box_cost * constants.partial_rounds)
            };

            let mut data = |cs: &mut TestConstraintSystem<Fr>, fr_data: &mut [Fr]| {
                (0..preimage_length)
                    .map(|i| {
                        let fr = Fr::random(&mut rng);
                        fr_data[i] = fr;
                        AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr))
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            };

            {
                let mut cs = TestConstraintSystem::<Fr>::new();
                let mut fr_data = vec![Fr::ZERO; preimage_length];
                let data: Vec<AllocatedNum<Fr>> = data(&mut cs, &mut fr_data);

                let out = poseidon_hash_allocated(&mut cs, data.clone(), &constants)
                    .expect("poseidon hashing failed");

                let mut p = Poseidon::<Fr, A>::new_with_preimage(&fr_data, &constants);
                let expected: Fr = p.hash_in_mode(HashMode::Correct);

                let expected_constraints_calculated = expected_constraints_calculated + 1;
                let expected_constraints = expected_constraints + 1;

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

            {
                let mut cs = TestConstraintSystem::<Fr>::new();
                let mut fr_data = vec![Fr::ZERO; preimage_length];
                let data: Vec<AllocatedNum<Fr>> = data(&mut cs, &mut fr_data);

                let out =
                    poseidon_hash_num(&mut cs, data, &constants).expect("poseidon hashing failed");

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
        let three = Elt::Allocated(AllocatedNum::alloc_infallible(
            cs1.namespace(|| "three"),
            || Fr::from(3),
        ));
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
            )
            .unwrap();

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
            )
            .unwrap();

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
            )
            .unwrap();

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
                scalar_product::<Fr, TestConstraintSystem<Fr>>(&res_vec, &[fr(7), fr(8), fr(9)])
                    .unwrap();

            res2.lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(42)); // 7 * 6 * three
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(58)); // (7 * 7 * four) + (9 * four)
                };
            });

            let v = res2.val().unwrap();
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
        )
        .unwrap();

        assert!(res.is_num());
        assert_eq!(fr(59), res.val().unwrap());
    }
}
