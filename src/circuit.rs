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
/// extract into a `num::AllocatedNum`, enforcing that the linear combination corresponds to the result.
/// In this way, all intermediate calculations are accounted for, with the restriction that we can only
/// accumulate linear (not polynomial) constraints. The set of operations provided here ensure this invariant is maintained.
enum Elt<E: Engine> {
    Allocated(AllocatedNum<E>),
    Num(Option<E::Fr>, LinearCombination<E>),
}

impl<E: Engine> Elt<E> {
    fn is_allocated(&self) -> bool {
        if let Self::Allocated(_) = self {
            true
        } else {
            false
        }
    }

    fn is_num(&self) -> bool {
        if let Self::Num(_, _) = self {
            true
        } else {
            false
        }
    }

    fn num_from_fr<CS: ConstraintSystem<E>>(fr: E::Fr) -> Self {
        let mut lc = LinearCombination::zero();
        lc = lc + (fr, CS::one());
        Self::Num(Some(fr), lc)
    }

    fn ensure_allocated<CS: ConstraintSystem<E>>(
        &self,
        cs: &mut CS,
        enforce: bool,
    ) -> Result<AllocatedNum<E>, SynthesisError> {
        match self {
            Self::Allocated(v) => Ok(v.clone()),
            Self::Num(fr, lc) => {
                let v = AllocatedNum::alloc(cs.namespace(|| "allocate for Elt::Num"), || {
                    fr.ok_or_else(|| SynthesisError::AssignmentMissing)
                })?;

                if enforce {
                    cs.enforce(
                        || format!("enforce num allocation preserves lc"),
                        |_| lc.clone(),
                        |lc| lc + CS::one(),
                        |lc| lc + v.get_variable(),
                    );
                }
                Ok(v)
            }
        }
    }

    fn val(&self) -> Option<E::Fr> {
        match self {
            Self::Allocated(v) => v.get_value(),
            Self::Num(fr, _lc) => *fr,
        }
    }

    fn lc(&self) -> LinearCombination<E> {
        match self {
            Self::Num(_fr, lc) => lc.clone(),
            Self::Allocated(v) => LinearCombination::<E>::zero() + v.get_variable(),
        }
    }

    /// Add two Nums and return a Num tracking the calculation. It is forbidden to invoke on an Allocated because the intended computation
    /// doe not include that path.
    fn add<CS: ConstraintSystem<E>>(self, other: Elt<E>) -> Result<Elt<E>, SynthesisError> {
        match (self, other) {
            (Elt::Num(Some(fr), lc), Elt::Num(Some(fr2), lc2)) => {
                let mut new_fr = fr;
                new_fr.add_assign(&fr2);

                // Coalesce like terms after adding, to prevent combinatorial
                // explosion of successive multiplications.
                let new_lc = simplify_lc(lc + &lc2);
                return Ok(Elt::Num(Some(new_fr), new_lc));
            }

            _ => panic!("only two numbers may be added"),
        }
    }

    /// Scale
    fn scale<CS: ConstraintSystem<E>>(&self, scalar: E::Fr) -> Result<Elt<E>, SynthesisError> {
        match self {
            Elt::Num(Some(fr), lc) => {
                let mut tmp = *fr;
                tmp.mul_assign(&scalar);

                let new_lc = lc.as_ref().iter().fold(
                    LinearCombination::zero(),
                    |acc, (variable, mut fr)| {
                        fr.mul_assign(&scalar);
                        acc + (fr, *variable)
                    },
                );
                Ok(Elt::Num(Some(tmp), new_lc))
            }
            Elt::Num(None, _) => Ok(Elt::Num(None, LinearCombination::zero())),
            Elt::Allocated(_) => Elt::Num(self.val(), self.lc()).scale::<CS>(scalar),
        }
    }
}

fn simplify_lc<E: Engine>(lc: LinearCombination<E>) -> LinearCombination<E> {
    #[derive(PartialEq, Eq, Debug, std::hash::Hash)]
    enum Idx {
        Input(usize),
        Aux(usize),
    }

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
        let elements = allocated_nums.into_iter().map(Elt::Allocated).collect();
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

        self.elements[1].ensure_allocated(&mut cs.namespace(|| "hash result"), true)
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
            let product = if partial_round {
                let mut acc = E::Fr::zero();
                // Dot product of column and this round's keys.
                for k in 1..self.constants.width() {
                    let mut tmp = column[k];
                    let rk = self.constants.round_constants[self.constants_offset + k - 1];
                    tmp.mul_assign(&rk);
                    acc.add_assign(&tmp);
                }

                scalar_product_with_add::<E, CS>(self.elements.as_slice(), &column, acc)
            } else {
                scalar_product::<E, CS>(self.elements.as_slice(), &column)
            }?;

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
    let tag = constants.arity_tag;
    let tag_num = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(tag))?;
    preimage.push(tag_num);
    preimage.rotate_right(1);

    let mut p = PoseidonCircuit::new(preimage, constants);

    p.hash(cs)
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    e: &Elt<E>,
    round_key: E::Fr,
) -> Result<Elt<E>, SynthesisError> {
    let l = e.ensure_allocated(&mut cs.namespace(|| "S-box input"), true)?;

    // If round_key was supplied, add it to l before squaring.
    let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), round_key, &l, true)?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(cs.namespace(|| "l4 * (l + rk)"), &l4, &l, round_key, true);

    Ok(Elt::Allocated(l5?))
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

fn scalar_product_with_add<E: Engine, CS: ConstraintSystem<E>>(
    elts: &[Elt<E>],
    scalars: &[E::Fr],
    to_add: E::Fr,
) -> Result<Elt<E>, SynthesisError> {
    let tmp = scalar_product::<E, CS>(elts, scalars)?;
    let tmp2 = tmp.add::<CS>(Elt::<E>::num_from_fr::<CS>(to_add))?;

    Ok(tmp2)
}

fn scalar_product<E: Engine, CS: ConstraintSystem<E>>(
    elts: &[Elt<E>],
    scalars: &[E::Fr],
) -> Result<Elt<E>, SynthesisError> {
    elts.iter().zip(scalars).try_fold(
        Elt::Num(Some(E::Fr::zero()), { LinearCombination::<E>::zero() }),
        |acc, (elt, &scalar)| acc.add::<CS>(elt.scale::<CS>(scalar)?),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::HashMode;
    use crate::test::TestConstraintSystem;
    use crate::{scalar_from_u64, Poseidon};
    use bellperson::ConstraintSystem;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use std::ops::Add;
    use typenum::bit::B1;
    use typenum::marker_traits::Unsigned;
    use typenum::uint::{UInt, UTerm};
    use typenum::Add1;

    #[test]
    fn test_poseidon_hash() {
        test_poseidon_hash_aux::<typenum::U2>(314);
        test_poseidon_hash_aux::<typenum::U4>(380);
        test_poseidon_hash_aux::<typenum::U8>(508);
        test_poseidon_hash_aux::<typenum::U16>(764);
        test_poseidon_hash_aux::<typenum::U24>(1012);
        test_poseidon_hash_aux::<typenum::U36>(1388);
    }
    fn test_poseidon_hash_aux<Arity>(expected_constraints: usize)
    where
        Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>>,
        Add1<Arity>: ArrayLength<<Bls12 as Engine>::Fr>,
    {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut cs = TestConstraintSystem::<Bls12>::new();
        let arity = Arity::to_usize();
        let constants = PoseidonConstants::<Bls12, Arity>::new();

        let expected_constraints_calculated = {
            let width = 1 + arity;
            let s_boxes = (width * constants.full_rounds) + constants.partial_rounds;
            let s_box_constraints = 3 * s_boxes;
            let mds_constraints =
                (width * constants.full_rounds) + constants.partial_rounds - arity;
            let total_constraints = s_box_constraints + mds_constraints;

            total_constraints
        };
        let mut i = 0;

        let mut fr_data = vec![Fr::zero(); arity];
        let data: Vec<AllocatedNum<Bls12>> = (0..arity)
            .enumerate()
            .map(|_| {
                let fr = Fr::random(&mut rng);
                fr_data[i] = fr;
                i += 1;
                AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
            })
            .collect::<Vec<_>>();

        let out = poseidon_hash(&mut cs, data, &constants).expect("poseidon hashing failed");

        let mut p = Poseidon::<Bls12, Arity>::new_with_preimage(&fr_data, &constants);
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

    fn fr(n: u64) -> <Bls12 as Engine>::Fr {
        scalar_from_u64::<Bls12>(n)
    }

    fn efr(n: u64) -> Elt<Bls12> {
        Elt::num_from_fr::<TestConstraintSystem<Bls12>>(fr(n))
    }

    #[test]
    fn test_square_sum() {
        let mut cs = TestConstraintSystem::<Bls12>::new();

        let mut cs1 = cs.namespace(|| "square_sum");
        let two = fr(2);
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
        {
            // Inputs are all linear combinations.
            let two = efr(2);
            let three = efr(3);
            let four = efr(4);

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            assert!(res.is_num());
            assert_eq!(scalar_from_u64::<Bls12>(56), res.val().unwrap());
        }
        {
            let mut cs = TestConstraintSystem::<Bls12>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 =
                AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64::<Bls12>(3)))
                    .unwrap();
            let three = Elt::Allocated(n3.clone());
            let n4 =
                AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64::<Bls12>(4)))
                    .unwrap();
            let four = Elt::Allocated(n4.clone());

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            assert!(res.is_num());
            assert_eq!(scalar_from_u64::<Bls12>(56), res.val().unwrap());

            res.lc().as_ref().iter().for_each(|(var, f)| {
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
            let mut cs = TestConstraintSystem::<Bls12>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 =
                AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64::<Bls12>(3)))
                    .unwrap();
            let three = Elt::Allocated(n3.clone());
            let n4 =
                AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64::<Bls12>(4)))
                    .unwrap();
            let four = Elt::Allocated(n4.clone());

            let mut res_vec = Vec::new();

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            res_vec.push(res);

            assert!(res_vec[0].is_num());
            assert_eq!(fr(56), res_vec[0].val().unwrap());

            res_vec[0].lc().as_ref().iter().for_each(|(var, f)| {
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
            let res2 = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &res_vec,
                &[fr(7), fr(8), fr(9)],
            )
            .unwrap();

            res2.lc().as_ref().iter().for_each(|(var, f)| {
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

        let res = scalar_product_with_add::<Bls12, TestConstraintSystem<Bls12>>(
            &[two, three, four],
            &[fr(5), fr(6), fr(7)],
            fr(3),
        )
        .unwrap();

        assert!(res.is_num());
        assert_eq!(fr(59), res.val().unwrap());
    }
}
