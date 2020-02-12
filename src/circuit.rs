use crate::poseidon::PoseidonConstants;

use bellperson::gadgets::num::AllocatedNum;
use bellperson::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::Field;
use ff::ScalarEngine as Engine;
use generic_array::typenum;
use generic_array::ArrayLength;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Similar to `num::Num`, we use `Elt` to accumulate both values and linear combinations, then eventually
/// extract into a `num::Allocatednum`, enforcing that the linear combination corresponds to the result.
/// In this way, all intermediate calculations are accounted for, with the restriction that we can only
/// accumulate linear (not polynomial) constraints. The set of operations provided here ensure this invariant is maintained.
enum Elt<E: Engine> {
    Var(AllocatedNum<E>),
    Num(E::Fr, LinearCombination<E>),
}

impl<E: Engine> Elt<E> {
    fn num_from_fr<CS: ConstraintSystem<E>>(fr: E::Fr) -> Self {
        Elt::Num(fr, LinearCombination::zero() + (fr, CS::one()))
    }

    fn var<CS: ConstraintSystem<E>>(&self, cs: &mut CS) -> Result<AllocatedNum<E>, SynthesisError> {
        match self {
            Elt::Var(v) => Ok(v.clone()),
            Elt::Num(fr, lc) => {
                let v = AllocatedNum::alloc(cs.namespace(|| "allocate for Elt::Num"), || Ok(*fr))?;

                cs.enforce(
                    || format!("enforce num allocation preserves lc"),
                    |_| lc.clone(),
                    |lc| lc + CS::one(),
                    |lc| lc + v.get_variable(),
                );
                Ok(v)
            }
        }
    }

    fn val(&self) -> Option<E::Fr> {
        match self {
            Elt::Var(v) => v.get_value(),
            Elt::Num(fr, _lc) => Some(*fr),
        }
    }

    fn add<CS: ConstraintSystem<E>>(self, other: Elt<E>) -> Elt<E> {
        if let Elt::Num(fr, lc) = self {
            if let Elt::Num(fr2, lc2) = other {
                let mut new_fr = fr.clone();
                new_fr.add_assign(&fr2);

                // Coalesce like terms after adding, to prevent combinatorial explosion of successive multiplications.
                let new_lc = simplify_lc(lc + &lc2);

                Elt::Num(new_fr, new_lc)
            } else {
                unimplemented!("unsupported path")
            }
        } else {
            unimplemented!("unsupported path")
        }
    }

    fn scale<CS: ConstraintSystem<E>>(&self, scalar: E::Fr) -> Elt<E> {
        match self {
            Elt::Num(fr, lc) => {
                let mut tmp = fr.clone();
                tmp.mul_assign(&scalar);

                let new_lc = lc.as_ref().iter().fold(
                    LinearCombination::zero(),
                    |acc, (variable, mut fr)| {
                        fr.mul_assign(&scalar);
                        acc + (fr, *variable)
                    },
                );
                Elt::Num(tmp, new_lc)
            }
            Elt::Var(v) => {
                let lc = LinearCombination::<E>::zero() + v.get_variable();
                let val = v.get_value().expect("Element::Var(v) had no value.");
                Elt::Num(val, lc).scale::<CS>(scalar)
            }
        }
    }
}

fn add<E: Engine, CS: ConstraintSystem<E>>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &E::Fr,
    enforce: bool,
) -> Result<AllocatedNum<E>, SynthesisError> {
    let sum = AllocatedNum::alloc(cs.namespace(|| "add"), || {
        let mut tmp = a
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(b);

        Ok(tmp)
    })?;

    if enforce {
        // a + b = sum
        cs.enforce(
            || "sum constraint",
            |lc| lc + a.get_variable() + (*b, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + sum.get_variable(),
        );
    }

    Ok(sum)
}

#[derive(PartialEq, Eq, Debug, std::hash::Hash)]
enum Idx {
    Input(usize),
    Aux(usize),
}
fn simplify_lc<E: Engine>(lc: LinearCombination<E>) -> LinearCombination<E> {
    let mut map: HashMap<Idx, E::Fr> = HashMap::new();

    lc.as_ref().iter().for_each(|(var, fr)| {
        let key = match var.get_unchecked() {
            Index::Input(i) => Idx::Input(i),
            Index::Aux(i) => Idx::Aux(i),
        };

        let val = map.entry(key).or_insert(E::Fr::zero());
        val.add_assign(fr)
    });

    let simplified = map
        .iter()
        .fold(LinearCombination::<E>::zero(), |acc, (idx, &fr)| {
            let index = match idx {
                Idx::Input(i) => Index::Input(*i),
                Idx::Aux(i) => Index::Aux(*i),
            };
            acc + (fr, Variable::new_unchecked(index))
        });
    simplified
}

//#[derive(Clone)]
/// Circuit for Poseidon hash.
pub struct PoseidonCircuit<'a, E, Arity>
where
    E: Engine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    constants_offset: usize,
    width: usize,
    elements: Vec<Elt<E>>,
    pos: usize,
    constants: &'a PoseidonConstants<E, Arity>,
    _w: PhantomData<Arity>,
}

/// PoseidonCircuit implementation.
impl<'a, E, Arity> PoseidonCircuit<'a, E, Arity>
where
    E: Engine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(
        allocated_nums: Vec<AllocatedNum<E>>,
        constants: &'a PoseidonConstants<E, Arity>,
    ) -> Self {
        let width = constants.width();
        let elements = allocated_nums
            .iter()
            .map(|a| Elt::Var(a.to_owned()))
            .collect();
        PoseidonCircuit {
            constants_offset: 0,
            width,
            elements,
            pos: width,
            constants,
            _w: PhantomData::<Arity>,
        }
    }

    fn hash<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<E>, SynthesisError> {
        for i in 0..self.constants.full_rounds / 2 {
            self.full_round(cs.namespace(|| format!("initial full round {}", i)))?;
        }

        for i in 0..self.constants.partial_rounds {
            self.partial_round(cs.namespace(|| format!("partial round {}", i)))?;
        }

        for i in 0..self.constants.full_rounds / 2 {
            self.full_round(cs.namespace(|| format!("final full round {}", i)))?;
        }

        self.elements[1].var(&mut cs.namespace(|| "hash result"))
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
                round_key,
            )?
        }
        self.constants_offset = constants_offset;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>(false)?;
        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let round_key = self.constants.round_constants[self.constants_offset];
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            round_key,
        )?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>(true)?;

        Ok(())
    }

    fn product_mds<CS: ConstraintSystem<E>>(
        &mut self,
        partial_round: bool,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<Elt<E>> = Vec::with_capacity(self.constants.width());

        for j in 0..self.constants.width() {
            let column = self.constants.mds_matrices.m[j].to_vec();
            // TODO: This could be cached per round to save synthesis time.
            let constant_term = if partial_round {
                let mut acc = E::Fr::zero();
                // Dot product of column and this round's keys.
                for k in 1..self.constants.width() {
                    let mut tmp = column[k];
                    let rk = self.constants.round_constants[self.constants_offset + k - 1];
                    tmp.mul_assign(&rk);
                    acc.add_assign(&tmp);
                }
                Some(acc)
            } else {
                None
            };

            let product =
                scalar_product::<E, CS>(self.elements.as_slice(), &column, constant_term)?;
            result.push(product);
        }
        if partial_round {
            self.constants_offset += self.constants.width() - 1;
        }
        self.elements = result;

        Ok(())
    }

    fn debug(&self) {
        let element_frs: Vec<_> = self.elements.iter().map(|n| n.val()).collect::<Vec<_>>();
        dbg!(element_frs, self.constants_offset);
    }
}

/// Create circuit for Poseidon hash.
pub fn poseidon_hash<CS, E, Arity>(
    mut cs: CS,
    mut preimage: Vec<AllocatedNum<E>>,
    constants: &PoseidonConstants<E, Arity>,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
    E: Engine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    // Add the arity tag to the front of the preimage.
    let tag = constants.arity_tag; // This could be shared across hash invocations within a circuit. TODO: add a mechanism for any such shared allocations.
    let tag_num = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(tag))?;
    preimage.push(tag_num);
    preimage.rotate_right(1);
    let mut p = PoseidonCircuit::new(preimage, constants);

    p.hash(cs)
}

pub fn create_poseidon_parameters<'a, E, Arity>() -> PoseidonConstants<E, Arity>
where
    E: Engine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    PoseidonConstants::new()
}

pub fn poseidon_hash_simple<CS, E, Arity>(
    cs: CS,
    preimage: Vec<AllocatedNum<E>>,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
    E: Engine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    poseidon_hash(cs, preimage, &create_poseidon_parameters::<E, Arity>())
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    e: &Elt<E>,
    round_key: E::Fr,
) -> Result<Elt<E>, SynthesisError> {
    let l = e.var(&mut cs.namespace(|| "S-box input"))?;

    // If round_key was supplied, add it to l before squaring.
    let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), round_key, &l, true)?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(cs.namespace(|| "l4 * (l + rk)"), &l4, &l, round_key, true);

    Ok(Elt::Var(l5?))
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    to_add: E::Fr,
    num: &AllocatedNum<E>,
    enforce: bool,
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

/// Calculates a * (b + to_add) — and enforces that constraint.
pub fn mul_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &AllocatedNum<E>,
    to_add: E::Fr,
    enforce: bool,
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

    if enforce {
        cs.enforce(
            || "mul sum constraint",
            |lc| lc + b.get_variable() + (to_add, CS::one()),
            |lc| lc + a.get_variable(),
            |lc| lc + res.get_variable(),
        );
    }
    Ok(res)
}

fn scalar_product<E: Engine, CS: ConstraintSystem<E>>(
    elts: &[Elt<E>],
    scalars: &[E::Fr],
    to_add: Option<E::Fr>,
) -> Result<Elt<E>, SynthesisError> {
    let tmp: Elt<E> = elts.iter().zip(scalars).fold(
        Elt::Num(E::Fr::zero(), { LinearCombination::<E>::zero() }),
        |acc, (elt, &scalar)| acc.add::<CS>(elt.scale::<CS>(scalar)),
    );

    let tmp2 = if let Some(a) = to_add {
        tmp.add::<CS>(Elt::<E>::num_from_fr::<CS>(a))
    } else {
        tmp
    };

    Ok(tmp2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::HashMode;
    use crate::test::TestConstraintSystem;
    use crate::{scalar_from_u64, Poseidon};
    use bellperson::ConstraintSystem;
    use generic_array::typenum::U4;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let cases = [(2, 314), (4, 380), (8, 508), (16, 764)];

        // TODO: test multiple arities.
        let test_arity = 4;

        for (arity, constraints) in &cases {
            if *arity != test_arity {
                continue;
            }
            let mut cs = TestConstraintSystem::<Bls12>::new();
            let mut i = 0;

            let mut fr_data = vec![Fr::zero(); test_arity];
            let data: Vec<AllocatedNum<Bls12>> = (0..*arity)
                .enumerate()
                .map(|_| {
                    let fr = Fr::random(&mut rng);
                    fr_data[i] = fr;
                    i += 1;
                    AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
                })
                .collect::<Vec<_>>();

            let constants = PoseidonConstants::new();
            let out = poseidon_hash(&mut cs, data, &constants).expect("poseidon hashing failed");

            let expected_constraints_calculated = {
                let width = 1 + arity;
                let s_boxes = (width * constants.full_rounds) + constants.partial_rounds;
                let s_box_constraints = 3 * s_boxes;
                let mds_constraints =
                    (width * constants.full_rounds) + constants.partial_rounds - arity;
                let total_constraints = s_box_constraints + mds_constraints;

                total_constraints
            };

            let mut p = Poseidon::<Bls12, U4>::new_with_preimage(&fr_data, &constants);
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
                *constraints,
                cs.num_constraints(),
                "constraint number changed",
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
        let res = square_sum(cs1, two, &three, true).unwrap();

        let twenty_five: Fr = scalar_from_u64::<Bls12>(25);
        assert_eq!(twenty_five, res.get_value().unwrap());
    }

    #[test]
    fn test_scalar_product() {
        let two = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(2));
        let three = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(3));
        let four = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(4));

        let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
            &[two, three, four],
            &[
                scalar_from_u64::<Bls12>(5),
                scalar_from_u64::<Bls12>(6),
                scalar_from_u64::<Bls12>(7),
            ],
            None,
        )
        .unwrap();

        assert_eq!(scalar_from_u64::<Bls12>(56), res.val().unwrap());
    }
    #[test]
    fn test_scalar_product_with_add() {
        let two = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(2));
        let three = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(3));
        let four = Elt::num_from_fr::<TestConstraintSystem<Bls12>>(scalar_from_u64::<Bls12>(4));

        let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
            &[two, three, four],
            &[
                scalar_from_u64::<Bls12>(5),
                scalar_from_u64::<Bls12>(6),
                scalar_from_u64::<Bls12>(7),
            ],
            Some(scalar_from_u64::<Bls12>(3)),
        )
        .unwrap();

        assert_eq!(scalar_from_u64::<Bls12>(59), res.val().unwrap());
    }
}
