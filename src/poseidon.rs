use crate::matrix::Matrix;
use crate::mds::{create_mds_matrices, factor_to_sparse_matrixes, MDSMatrices, SparseMatrix};
use crate::poseidon_alt::{hash_correct, hash_optimized_dynamic};
use crate::preprocessing::compress_round_constants;
use crate::{matrix, quintic_s_box, BatchHasher, Strength, DEFAULT_STRENGTH};
use crate::{round_constants, round_numbers, scalar_from_u64, Error};
use ff::{Field, PrimeField, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use paired::bls12_381;
use paired::bls12_381::Bls12;
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::*;

/// The arity tag is the first element of a Poseidon permutation.
/// This extra element is necessary for 128-bit security.
pub fn arity_tag<Fr: PrimeField, A: Arity<Fr>>() -> Fr {
    A::tag()
}

/// Available arities for the Poseidon hasher.
pub trait Arity<T>: ArrayLength<T> {
    /// Must be Arity + 1.
    type ConstantsSize: ArrayLength<T>;

    fn tag() -> T;
}

macro_rules! impl_arity {
    ($($a:ty => $b:ty),*) => {
        $(
            impl<Fr: PrimeField> Arity<Fr> for $a {
                type ConstantsSize = $b;

                fn tag() -> Fr {
                    scalar_from_u64::<Fr>((1 << <$a as Unsigned>::to_usize()) - 1)
                }
            }
        )*
    };
}

// Dummy implementation to allow for an "optional" argument.
impl<Fr: PrimeField> Arity<Fr> for U0 {
    type ConstantsSize = U0;

    fn tag() -> Fr {
        unreachable!("dummy implementation for U0, should not be called")
    }
}

impl_arity!(
    U2 => U3,
    U3 => U4,
    U4 => U5,
    U5 => U6,
    U6 => U7,
    U7 => U8,
    U8 => U9,
    U9 => U10,
    U10 => U11,
    U11 => U12,
    U12 => U13,
    U13 => U14,
    U14 => U15,
    U15 => U16,
    U16 => U17,
    U17 => U18,
    U18 => U19,
    U19 => U20,
    U20 => U21,
    U21 => U22,
    U22 => U23,
    U23 => U24,
    U24 => U25,
    U25 => U26,
    U26 => U27,
    U27 => U28,
    U28 => U29,
    U29 => U30,
    U30 => U31,
    U31 => U32,
    U32 => U33,
    U33 => U34,
    U34 => U35,
    U35 => U36,
    U36 => U37
);

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
#[derive(Debug, Clone, PartialEq)]
pub struct Poseidon<'a, E, A = U2>
where
    E: ScalarEngine,
    A: Arity<E::Fr>,
{
    pub(crate) constants_offset: usize,
    pub(crate) current_round: usize, // Used in static optimization only for now.
    /// the elements to permute
    pub elements: GenericArray<E::Fr, A::ConstantsSize>,
    pos: usize,
    pub(crate) constants: &'a PoseidonConstants<E, A>,
    _e: PhantomData<E>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseidonConstants<E, A>
where
    E: ScalarEngine,
    A: Arity<E::Fr>,
{
    pub mds_matrices: MDSMatrices<E>,
    pub round_constants: Vec<E::Fr>,
    pub compressed_round_constants: Vec<E::Fr>,
    pub pre_sparse_matrix: Matrix<E::Fr>,
    pub sparse_matrixes: Vec<SparseMatrix<E>>,
    pub arity_tag: E::Fr,
    pub full_rounds: usize,
    pub half_full_rounds: usize,
    pub partial_rounds: usize,
    _a: PhantomData<A>,
}

#[derive(Debug, PartialEq)]
pub enum HashMode {
    // The initial and correct version of the algorithm. We should preserve the ability to hash this way for reference
    // and to preserve confidence in our tests along thew way.
    Correct,
    // This mode is meant to be mostly synchronized with `Correct` but may reduce or simplify the work performed by the algorithm, if not the code implementing.
    // Its purpose is for use during refactoring/development.
    OptimizedDynamic,
    // Consumes statically pre-processed constants for simplest operation.
    OptimizedStatic,
}
use HashMode::{Correct, OptimizedDynamic, OptimizedStatic};

pub const DEFAULT_HASH_MODE: HashMode = OptimizedStatic;

impl<'a, E, A> PoseidonConstants<E, A>
where
    E: ScalarEngine,
    A: Arity<E::Fr>,
{
    pub fn new() -> Self {
        Self::new_with_strength(DEFAULT_STRENGTH)
    }

    pub fn new_with_strength(strength: Strength) -> Self {
        let arity = A::to_usize();
        let width = arity + 1;

        let mds_matrices = create_mds_matrices::<E>(width);

        let (full_rounds, partial_rounds) = round_numbers(arity, &strength);
        let half_full_rounds = full_rounds / 2;
        let round_constants = round_constants::<E>(arity, &strength);
        let compressed_round_constants = compress_round_constants::<E>(
            width,
            full_rounds,
            partial_rounds,
            &round_constants,
            &mds_matrices,
            partial_rounds,
        );

        let (pre_sparse_matrix, sparse_matrixes) =
            factor_to_sparse_matrixes::<E>(mds_matrices.m.clone(), partial_rounds);

        // Ensure we have enough constants for the sbox rounds
        assert!(
            width * (full_rounds + partial_rounds) <= round_constants.len(),
            "Not enough round constants"
        );

        assert_eq!(
            full_rounds * width + partial_rounds,
            compressed_round_constants.len()
        );

        Self {
            mds_matrices,
            round_constants,
            compressed_round_constants,
            pre_sparse_matrix,
            sparse_matrixes,
            arity_tag: A::tag(),
            full_rounds,
            half_full_rounds,
            partial_rounds,
            _a: PhantomData::<A>,
        }
    }

    /// Returns the width.
    #[inline]
    pub fn arity(&self) -> usize {
        A::to_usize()
    }

    /// Returns the width.
    #[inline]
    pub fn width(&self) -> usize {
        A::ConstantsSize::to_usize()
    }
}

impl<'a, E, A> Poseidon<'a, E, A>
where
    E: ScalarEngine,
    A: Arity<E::Fr>,
{
    pub fn new(constants: &'a PoseidonConstants<E, A>) -> Self {
        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.arity_tag
            } else {
                E::Fr::zero()
            }
        });
        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: 1,
            constants,
            _e: PhantomData::<E>,
        }
    }
    pub fn new_with_preimage(preimage: &[E::Fr], constants: &'a PoseidonConstants<E, A>) -> Self {
        assert_eq!(preimage.len(), A::to_usize(), "Invalid preimage size");

        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.arity_tag
            } else {
                preimage[i - 1]
            }
        });

        let width = elements.len();

        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: width,
            constants,
            _e: PhantomData::<E>,
        }
    }

    /// Replace the elements with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is bigger than the arity.
    pub fn set_preimage(&mut self, preimage: &[E::Fr]) {
        self.reset();
        self.elements[1..].copy_from_slice(&preimage);
        self.pos = self.elements.len();
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.constants_offset = 0;
        self.current_round = 0;
        self.elements[1..]
            .iter_mut()
            .for_each(|l| *l = scalar_from_u64::<E::Fr>(0u64));
        self.elements[0] = self.constants.arity_tag;
        self.pos = 1;
    }

    /// The returned `usize` represents the element position (within arity) for the input operation
    pub fn input(&mut self, element: E::Fr) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        if self.pos >= self.constants.width() {
            return Err(Error::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.elements[self.pos] = element;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    pub fn hash_in_mode(&mut self, mode: HashMode) -> E::Fr {
        match mode {
            Correct => hash_correct(self),
            OptimizedDynamic => hash_optimized_dynamic(self),
            OptimizedStatic => self.hash_optimized_static(),
        }
    }

    pub fn hash(&mut self) -> E::Fr {
        self.hash_in_mode(DEFAULT_HASH_MODE)
    }

    pub fn hash_optimized_static(&mut self) -> E::Fr {
        // The first full round should use the initial constants.
        self.add_round_constants();

        for _ in 0..self.constants.half_full_rounds {
            self.full_round(false);
        }

        for _ in 0..self.constants.partial_rounds {
            self.partial_round();
        }

        // All but last full round.
        for _ in 1..self.constants.half_full_rounds {
            self.full_round(false);
        }
        self.full_round(true);

        assert_eq!(
            self.constants_offset,
            self.constants.compressed_round_constants.len(),
            "Constants consumed ({}) must equal preprocessed constants provided ({}).",
            self.constants_offset,
            self.constants.compressed_round_constants.len()
        );

        self.elements[1]
    }

    fn full_round(&mut self, last_round: bool) {
        let to_take = self.elements.len();
        let post_round_keys = self
            .constants
            .compressed_round_constants
            .iter()
            .skip(self.constants_offset)
            .take(to_take);

        if !last_round {
            let needed = self.constants_offset + to_take;
            assert!(
                needed <= self.constants.compressed_round_constants.len(),
                "Not enough preprocessed round constants ({}), need {}.",
                self.constants.compressed_round_constants.len(),
                needed
            );
        }
        self.elements
            .iter_mut()
            .zip(post_round_keys)
            .for_each(|(l, post)| {
                // Be explicit that no round key is added after last round of S-boxes.
                let post_key = if last_round {
                    panic!("Trying to skip last full round, but there is a key here! ({})");
                } else {
                    Some(post)
                };
                quintic_s_box::<E>(l, None, post_key);
            });
        // We need this because post_round_keys will have been empty, so it didn't happen in the for_each. :(
        if last_round {
            self.elements
                .iter_mut()
                .for_each(|l| quintic_s_box::<E>(l, None, None));
        } else {
            self.constants_offset += self.elements.len();
        }
        self.round_product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first (arity tag) poseidon leaf.
    fn partial_round(&mut self) {
        let post_round_key = self.constants.compressed_round_constants[self.constants_offset];

        // Apply the quintic S-Box to the first element
        quintic_s_box::<E>(&mut self.elements[0], None, Some(&post_round_key));
        self.constants_offset += 1;

        self.round_product_mds();
    }

    fn add_round_constants(&mut self) {
        for (element, round_constant) in self.elements.iter_mut().zip(
            self.constants
                .compressed_round_constants
                .iter()
                .skip(self.constants_offset),
        ) {
            element.add_assign(round_constant);
        }

        self.constants_offset += self.elements.len();
    }

    /// Set the provided elements with the result of the product between the elements and the appropriate
    /// MDS matrix.
    fn round_product_mds(&mut self) {
        let full_half = self.constants.half_full_rounds;
        let sparse_offset = full_half - 1;
        if self.current_round == sparse_offset {
            self.product_mds_with_matrix(&self.constants.pre_sparse_matrix);
        } else {
            if (self.current_round > sparse_offset)
                && (self.current_round < full_half + self.constants.partial_rounds)
            {
                let index = self.current_round - sparse_offset - 1;
                let sparse_matrix = &self.constants.sparse_matrixes[index];

                self.product_mds_with_sparse_matrix(&sparse_matrix);
            } else {
                self.product_mds();
            }
        };

        self.current_round += 1;
    }

    /// Set the provided elements with the result of the product between the elements and the constant
    /// MDS matrix.
    pub(crate) fn product_mds(&mut self) {
        self.product_mds_with_matrix(&self.constants.mds_matrices.m);
    }

    pub(crate) fn product_mds_with_matrix(&mut self, matrix: &Matrix<E::Fr>) {
        let mut result = GenericArray::<E::Fr, A::ConstantsSize>::generate(|_| E::Fr::zero());

        for (j, val) in result.iter_mut().enumerate() {
            for (i, row) in matrix.iter().enumerate() {
                let mut tmp = row[j];
                tmp.mul_assign(&self.elements[i]);
                val.add_assign(&tmp);
            }
        }

        let _ = std::mem::replace(&mut self.elements, result);
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix(&mut self, sparse_matrix: &SparseMatrix<E>) {
        let mut result = GenericArray::<E::Fr, A::ConstantsSize>::generate(|_| E::Fr::zero());

        // First column is dense.
        for (i, val) in sparse_matrix.w_hat.iter().enumerate() {
            let mut tmp = *val;
            tmp.mul_assign(&self.elements[i]);
            result[0].add_assign(&tmp);
        }

        for (j, val) in result.iter_mut().enumerate().skip(1) {
            // Except for first row/column, diagonals are one.
            val.add_assign(&self.elements[j]);

            // First row is dense.
            let mut tmp = sparse_matrix.v_rest[j - 1];
            tmp.mul_assign(&self.elements[0]);
            val.add_assign(&tmp);
        }

        let _ = std::mem::replace(&mut self.elements, result);
    }

    fn debug(&self, msg: &str) {
        dbg!(msg, &self.constants_offset, &self.elements);
    }
}

#[derive(Debug)]
pub struct SimplePoseidonBatchHasher<'a, A>
where
    A: Arity<bls12_381::Fr>,
{
    constants: PoseidonConstants<Bls12, A>,
    max_batch_size: usize,
    _s: PhantomData<Poseidon<'a, Bls12, A>>,
}

impl<'a, A> SimplePoseidonBatchHasher<'a, A>
where
    A: 'a + Arity<bls12_381::Fr>,
{
    pub(crate) fn new(max_batch_size: usize) -> Result<Self, Error> {
        Self::new_with_strength(DEFAULT_STRENGTH, max_batch_size)
    }

    pub(crate) fn new_with_strength(
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            constants: PoseidonConstants::<Bls12, A>::new_with_strength(strength),
            max_batch_size,
            _s: PhantomData::<Poseidon<'a, Bls12, A>>,
        })
    }
}
impl<'a, A> BatchHasher<A> for SimplePoseidonBatchHasher<'a, A>
where
    A: 'a + Arity<bls12_381::Fr>,
{
    fn hash(
        &mut self,
        preimages: &[GenericArray<bls12_381::Fr, A>],
    ) -> Result<Vec<bls12_381::Fr>, Error> {
        Ok(preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &self.constants).hash())
            .collect())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use ff::Field;
    use generic_array::typenum;
    use paired::bls12_381::{Bls12, Fr};

    #[test]
    fn reset() {
        let test_arity = 2;
        let preimage = vec![Scalar::one(); test_arity];
        let constants = PoseidonConstants::new();
        let mut h = Poseidon::<Bls12, U2>::new_with_preimage(&preimage, &constants);
        h.hash();
        h.reset();

        let default = Poseidon::<Bls12, U2>::new(&constants);
        assert_eq!(default.pos, h.pos);
        assert_eq!(default.elements, h.elements);
        assert_eq!(default.constants_offset, h.constants_offset);
    }

    #[test]
    fn hash_det() {
        let test_arity = 2;
        let mut preimage = vec![Scalar::zero(); test_arity];
        let constants = PoseidonConstants::new();
        preimage[0] = Scalar::one();

        let mut h = Poseidon::<Bls12, U2>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result: <Bls12 as ScalarEngine>::Fr = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_arity_3() {
        let mut preimage: [Scalar; 3] = [Scalar::zero(); 3];
        let constants = PoseidonConstants::new();
        preimage[0] = Scalar::one();

        let mut h = Poseidon::<Bls12, typenum::U3>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result: <Bls12 as ScalarEngine>::Fr = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_values() {
        hash_values_cases(Strength::Standard);
        hash_values_cases(Strength::Strengthened);
    }

    fn hash_values_cases(strength: Strength) {
        hash_values_aux::<typenum::U2>(strength);
        hash_values_aux::<typenum::U4>(strength);
        hash_values_aux::<typenum::U8>(strength);
        hash_values_aux::<typenum::U11>(strength);
        hash_values_aux::<typenum::U16>(strength);
        hash_values_aux::<typenum::U24>(strength);
        hash_values_aux::<typenum::U36>(strength);
    }

    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_values_aux<A>(strength: Strength)
    where
        A: Arity<Fr>,
    {
        let constants = PoseidonConstants::<Bls12, A>::new_with_strength(strength);
        let mut p = Poseidon::<Bls12, A>::new(&constants);
        let mut p2 = Poseidon::<Bls12, A>::new(&constants);
        let mut p3 = Poseidon::<Bls12, A>::new(&constants);
        let mut p4 = Poseidon::<Bls12, A>::new(&constants);

        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Fr>(n as u64);
            p.input(scalar).unwrap();
            p2.input(scalar).unwrap();
            p3.input(scalar).unwrap();
            p4.input(scalar).unwrap();

            preimage[n] = scalar;
        }

        let digest = p.hash();
        let digest2 = p2.hash_in_mode(Correct);
        let digest3 = p3.hash_in_mode(OptimizedStatic);
        let digest4 = p4.hash_in_mode(OptimizedDynamic);
        assert_eq!(digest, digest2);
        assert_eq!(digest, digest3);
        assert_eq!(digest, digest4);

        let expected = match strength {
            Strength::Standard => {
                // Currently secure round constants.
                match test_arity {
                    2 => scalar_from_u64s([
                        0x2e203c369a02e7ff,
                        0xa6fba9339d05a69d,
                        0x739e0fd902efe161,
                        0x396508d75e76a56b,
                    ]),
                    4 => scalar_from_u64s([
                        0x019814ff6662075d,
                        0xfb6b4605bf1327ec,
                        0x00db3c6579229399,
                        0x58a54b10a9e5848a,
                    ]),
                    8 => scalar_from_u64s([
                        0x2a9934f56d38a5e6,
                        0x4b682e9d9cc4aed9,
                        0x1201004211677077,
                        0x2394611da3a5de55,
                    ]),
                    11 => scalar_from_u64s([
                        0xcee3bbc32b693163,
                        0x09f3dcd8ccb08fc1,
                        0x6ca537e232ebe87a,
                        0x0c0fc1b2e5227f28,
                    ]),
                    16 => scalar_from_u64s([
                        0x1291c74060266d37,
                        0x5b8dbc6d30680a6f,
                        0xc1c2fb5a6f871e63,
                        0x2d3ae2663381ae8a,
                    ]),
                    24 => scalar_from_u64s([
                        0xd7ef3569f585b321,
                        0xc3e779f6468815b1,
                        0x066f39bf783f3d9f,
                        0x63beb8831f11ae15,
                    ]),
                    36 => scalar_from_u64s([
                        0x4473606dfa4e8140,
                        0x75cd368df8a8ac3c,
                        0x540a30e03c10bbaa,
                        0x699303082a6e5d5f,
                    ]),
                    _ => {
                        dbg!(digest, test_arity);
                        panic!("Arity lacks test vector: {}", test_arity)
                    }
                }
            }
            Strength::Strengthened =>
            // Strengthened round constants.
            {
                match test_arity {
                    2 => scalar_from_u64s([
                        0x793dbaf54552cd69,
                        0x5278ecbf17040ea6,
                        0xc48b36ecc4cab748,
                        0x33d28a753baee41b,
                    ]),
                    4 => scalar_from_u64s([
                        0x4650ee190212aa9a,
                        0xe5113a254d6f5c7e,
                        0x54013bdaf68ba4c2,
                        0x09d8207c51ca3f43,
                    ]),
                    8 => scalar_from_u64s([
                        0x9f0c3c93c3fc894e,
                        0xe843d4cfba662df1,
                        0xd69aae8fe1cb63e8,
                        0x69e61465981ae17e,
                    ]),
                    11 => scalar_from_u64s([
                        0x778af344d8f9e8b7,
                        0xc94fe2ca3f46d433,
                        0x07abbcf9b406e8d8,
                        0x28bb83ff439753c0,
                    ]),
                    16 => scalar_from_u64s([
                        0x3cc2664c5fd6ae07,
                        0xd7431eaaa5e43189,
                        0x43ba5f418c6ef01d,
                        0x68d7856395aa217e,
                    ]),
                    24 => scalar_from_u64s([
                        0x1df1da58827cb39d,
                        0x0566756b7b80fb10,
                        0x222eb82c6666be3d,
                        0x086e4e81a35bfd92,
                    ]),
                    36 => scalar_from_u64s([
                        0x636401e9371dc311,
                        0x8f69e35a702ed188,
                        0x64d73b2ddc03d43b,
                        0x609f8c6fe45cc054,
                    ]),
                    _ => {
                        dbg!(digest, test_arity);
                        panic!("Arity lacks test vector: {}", test_arity)
                    }
                }
            }
        };
        dbg!(test_arity);
        assert_eq!(expected, digest);
    }

    #[test]
    fn hash_compare_optimized() {
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Fr>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let mut p2 = p.clone();
        let mut p3 = p.clone();

        let digest_correct = p.hash_in_mode(Correct);

        let digest_optimized_dynamic = p2.hash_in_mode(OptimizedDynamic);
        let digest_optimized_static = p3.hash_in_mode(OptimizedStatic);

        assert_eq!(digest_correct, digest_optimized_dynamic);
        assert_eq!(digest_correct, digest_optimized_static);
    }

    #[test]
    fn default_is_standard() {
        let default_constants = PoseidonConstants::<Bls12, U8>::new();
        let standard_constants =
            PoseidonConstants::<Bls12, U8>::new_with_strength(Strength::Standard);

        assert_eq!(
            standard_constants.partial_rounds,
            default_constants.partial_rounds
        );
    }
}
