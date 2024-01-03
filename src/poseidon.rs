use crate::hash_type::HashType;
use crate::matrix::{apply_matrix, left_apply_matrix, transpose, Matrix};
use crate::mds::{
    create_mds_matrices, derive_mds_matrices, factor_to_sparse_matrixes, generate_mds, MdsMatrices,
    SparseMatrix,
};
use crate::poseidon_alt::{hash_correct, hash_optimized_dynamic};
use crate::preprocessing::compress_round_constants;
use crate::{matrix, quintic_s_box, BatchHasher, Strength, DEFAULT_STRENGTH};
use crate::{round_constants, round_numbers, Error};
#[cfg(feature = "abomonation")]
use abomonation::Abomonation;
#[cfg(feature = "abomonation")]
use abomonation_derive::Abomonation;
use ff::PrimeField;
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::*;

/// Available arities for the Poseidon hasher.
pub trait Arity<T>: ArrayLength {
    /// Must be Arity + 1.
    type ConstantsSize: ArrayLength;

    fn tag() -> T;
}

macro_rules! impl_arity {
    ($($a:ty),*) => {
        $(
            impl<F: PrimeField> Arity<F> for $a {
                type ConstantsSize = Add1<$a>;

                fn tag() -> F {
                    F::from((1 << <$a as Unsigned>::to_usize()) - 1)
                }
            }
        )*
    };
}

// Dummy implementation to allow for an "optional" argument.
impl<F: PrimeField> Arity<F> for U0 {
    type ConstantsSize = U0;

    fn tag() -> F {
        unreachable!("dummy implementation for U0, should not be called")
    }
}

impl_arity!(
    U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21,
    U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36
);

/// Holds preimage, some utility offsets and counters along with the reference
/// to [`PoseidonConstants`] required for hashing. [`Poseidon`] is parameterized
/// by [`ff::PrimeField`] and [`Arity`], which should be similar to [`PoseidonConstants`].
///
/// [`Poseidon`] accepts input `elements` set with length equal or less than [`Arity`].
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poseidon<'a, F, A = U2>
where
    F: PrimeField,
    A: Arity<F>,
{
    pub(crate) constants_offset: usize,
    pub(crate) current_round: usize, // Used in static optimization only for now.
    /// the elements to permute
    pub elements: GenericArray<F, A::ConstantsSize>,
    /// index of the next element of state to be absorbed
    pub(crate) pos: usize,
    pub(crate) constants: &'a PoseidonConstants<F, A>,
    _f: PhantomData<F>,
}

/// Holds constant values required for further [`Poseidon`] hashing. It contains MDS matrices,
/// round constants and numbers, parameters that specify security level ([`Strength`]) and
/// domain separation ([`HashType`]). Additional constants related to optimizations are also included.
///
/// For correct operation, [`PoseidonConstants`] instance should be parameterized with the same [`ff::PrimeField`]
/// and [`Arity`] as [`Poseidon`] instance that consumes it.
///
/// See original [Poseidon paper](https://eprint.iacr.org/2019/458.pdf) for more details.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "abomonation", derive(Abomonation))]
#[cfg_attr(feature = "abomonation", abomonation_bounds(where F::Repr: Abomonation))]
pub struct PoseidonConstants<F: PrimeField, A: Arity<F>> {
    pub mds_matrices: MdsMatrices<F>,
    #[cfg_attr(feature = "abomonation", abomonate_with(Option<Vec<F::Repr>>))]
    pub round_constants: Option<Vec<F>>, // TODO: figure out how to automatically allocate `None`
    #[cfg_attr(feature = "abomonation", abomonate_with(Vec<F::Repr>))]
    pub compressed_round_constants: Vec<F>,
    #[cfg_attr(feature = "abomonation", abomonate_with(Matrix<F::Repr>))]
    pub pre_sparse_matrix: Matrix<F>,
    pub sparse_matrixes: Vec<SparseMatrix<F>>,
    pub strength: Strength,
    /// The domain tag is the first element of a Poseidon permutation.
    /// This extra element is necessary for 128-bit security.
    #[cfg_attr(feature = "abomonation", abomonate_with(F::Repr))]
    pub domain_tag: F,
    pub full_rounds: usize,
    pub half_full_rounds: usize,
    pub partial_rounds: usize,
    pub hash_type: HashType<F, A>,
    pub(crate) _a: PhantomData<A>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum HashMode {
    // The initial and correct version of the algorithm. We should preserve the ability to hash this way for reference
    // and to preserve confidence in our tests along thew way.
    Correct,
    // This mode is meant to be mostly synchronized with `Correct` but may reduce or simplify the work performed by the
    // algorithm, if not the code implementing. Its purpose is for use during refactoring/development.
    OptimizedDynamic,
    // Consumes statically pre-processed constants for simplest operation.
    OptimizedStatic,
}
use HashMode::{Correct, OptimizedDynamic, OptimizedStatic};

pub const DEFAULT_HASH_MODE: HashMode = OptimizedStatic;

impl<F, A> PoseidonConstants<F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
    /// with following default parameters:
    /// - 128 bit of security;
    /// - Merkle Tree (where all leafs are presented) domain separation ([`HashType`]).
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::Strength;
    /// use neptune::hash_type::HashType;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new();
    ///
    /// assert_eq!(constants.strength, Strength::Standard);
    /// assert_eq!(constants.hash_type, HashType::MerkleTree);
    /// ```
    pub fn new() -> Self {
        Self::new_with_strength(DEFAULT_STRENGTH)
    }

    /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
    /// of constant-size preimages with following parameters:
    /// - 128 bit of security;
    /// - Constant-Input-Length Hashing domain separation ([`HashType`]).
    ///
    /// Instantiated [`PoseidonConstants`] still calculates internal constants based on [`Arity`], but calculation of
    /// [`HashType::domain_tag`] is based on input `length`.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::Strength;
    /// use neptune::hash_type::HashType;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U2;
    ///
    /// let preimage_length = 2usize;
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_constant_length(preimage_length);
    ///
    /// assert_eq!(constants.strength, Strength::Standard);
    /// assert_eq!(constants.hash_type, HashType::<Fp, U2>::ConstantLength(preimage_length));
    /// ```
    pub fn new_constant_length(length: usize) -> Self {
        Self::new_with_strength_and_type(DEFAULT_STRENGTH, HashType::ConstantLength(length))
    }

    /// Creates new instance of [`PoseidonConstants`] from already defined one with recomputed domain tag.
    ///
    /// It is assumed that input `length` is equal or less than [`Arity`].
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::Strength;
    /// use neptune::hash_type::HashType;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U8;
    ///
    /// let preimage_length = 2usize;
    /// let constants: PoseidonConstants<Fp, U8> = PoseidonConstants::new_constant_length(preimage_length);
    /// let constants: PoseidonConstants<Fp, U8> = constants.with_length(preimage_length - 2);
    ///
    /// assert_eq!(constants.strength, Strength::Standard);
    /// assert_eq!(constants.hash_type, HashType::<Fp, U8>::ConstantLength(preimage_length - 2));
    /// ```
    pub fn with_length(&self, length: usize) -> Self {
        let arity = A::to_usize();
        assert!(length <= arity);

        let hash_type = match self.hash_type {
            HashType::ConstantLength(_) => HashType::ConstantLength(length),
            _ => panic!("cannot set constant length of hash without type ConstantLength."),
        };

        let domain_tag = hash_type.domain_tag();

        Self {
            hash_type,
            domain_tag,
            ..self.clone()
        }
    }

    /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
    /// with Merkle Tree (where all leafs are presented) domain separation ([`HashType`]) custom security level ([`Strength`]).
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::Strength;
    /// use neptune::hash_type::HashType;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U2;
    ///
    /// let security_level = Strength::Strengthened;
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_with_strength(security_level);
    ///
    /// assert_eq!(constants.strength, Strength::Strengthened);
    /// assert_eq!(constants.hash_type, HashType::MerkleTree);
    /// ```
    pub fn new_with_strength(strength: Strength) -> Self {
        Self::new_with_strength_and_type(strength, HashType::MerkleTree)
    }

    /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
    /// with custom domain separation ([`HashType`]) and custom security level ([`Strength`]).
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::Strength;
    /// use neptune::hash_type::HashType;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U2;
    ///
    /// let domain_separation = HashType::Encryption;
    /// let security_level = Strength::Strengthened;
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_with_strength_and_type(security_level, domain_separation);
    ///
    /// assert_eq!(constants.strength, Strength::Strengthened);
    /// assert_eq!(constants.hash_type, HashType::Encryption);
    /// ```
    pub fn new_with_strength_and_type(strength: Strength, hash_type: HashType<F, A>) -> Self {
        assert!(hash_type.is_supported());
        let arity = A::to_usize();
        let width = arity + 1;
        let mds = generate_mds(width);
        let (full_rounds, partial_rounds) = round_numbers(arity, &strength);
        let round_constants = round_constants(arity, &strength);

        // Now call new_from_parameters with all the necessary parameters.
        Self::new_from_parameters(
            width,
            mds,
            round_constants,
            full_rounds,
            partial_rounds,
            hash_type,
            strength,
        )
    }

    /// Generates new instance of [`PoseidonConstants`] with matrix, constants and number of rounds.
    /// The matrix does not have to be symmetric.
    pub fn new_from_parameters(
        width: usize,
        m: Matrix<F>,
        round_constants: Vec<F>,
        full_rounds: usize,
        partial_rounds: usize,
        hash_type: HashType<F, A>,
        strength: Strength,
    ) -> Self {
        let mds_matrices = derive_mds_matrices(m);
        let half_full_rounds = full_rounds / 2;
        let compressed_round_constants = compress_round_constants(
            width,
            full_rounds,
            partial_rounds,
            &round_constants,
            &mds_matrices,
            partial_rounds,
        );

        let (pre_sparse_matrix, sparse_matrixes) =
            factor_to_sparse_matrixes(&transpose(&mds_matrices.m), partial_rounds);

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
            round_constants: Some(round_constants),
            compressed_round_constants,
            pre_sparse_matrix,
            sparse_matrixes,
            strength,
            domain_tag: hash_type.domain_tag(),
            full_rounds,
            half_full_rounds,
            partial_rounds,
            hash_type,
            _a: PhantomData::<A>,
        }
    }

    /// Returns the [`Arity`] value represented as `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U8;
    ///
    /// let constants: PoseidonConstants<Fp, U8> = PoseidonConstants::new();
    ///
    /// assert_eq!(constants.arity(), 8usize);
    /// ```
    #[inline]
    pub fn arity(&self) -> usize {
        A::to_usize()
    }

    /// Returns `width` value represented as `usize`. It equals to [`Arity`] + 1.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use pasta_curves::Fp;
    /// use generic_array::typenum::U8;
    ///
    /// let constants: PoseidonConstants<Fp, U8> = PoseidonConstants::new();
    ///
    /// assert_eq!(constants.width(), 8 + 1);
    /// ```
    #[inline]
    pub fn width(&self) -> usize {
        A::ConstantsSize::to_usize()
    }
}

impl<F, A> Default for PoseidonConstants<F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, F, A> Poseidon<'a, F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    /// Creates [`Poseidon`] instance using provided [`PoseidonConstants`] as input. Underlying set of
    /// elements are initialized and `domain_tag` from [`PoseidonConstants`] is used as zero element in the set.
    /// Therefore, hashing is eventually performed over [`Arity`] + 1 elements in fact, while [`Arity`] elements
    /// are occupied by preimage data.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U8;
    ///
    /// let constants: PoseidonConstants<Fp, U8> = PoseidonConstants::new();
    /// let poseidon = Poseidon::<Fp, U8>::new(&constants);
    ///
    /// assert_eq!(poseidon.elements.len(), 9);
    /// for index in 1..9 {
    ///     assert_eq!(poseidon.elements[index], Fp::ZERO);
    /// }
    /// ```
    pub fn new(constants: &'a PoseidonConstants<F, A>) -> Self {
        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.domain_tag
            } else {
                F::ZERO
            }
        });
        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: 1,
            constants,
            _f: PhantomData::<F>,
        }
    }

    /// Creates [`Poseidon`] instance using provided preimage and [`PoseidonConstants`] as input.
    /// Doesn't support [`PoseidonConstants`] with [`HashType::VariableLength`]. It is assumed that
    /// size of input preimage set can't be greater than [`Arity`].
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let preimage_set_length = 1;
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_constant_length(preimage_set_length);
    ///
    /// let preimage = vec![Fp::from(u64::MAX); preimage_set_length];
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new_with_preimage(&preimage, &constants);
    ///
    /// assert_eq!(constants.width(), 3);
    /// assert_eq!(poseidon.elements.len(), constants.width());
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MAX));
    /// assert_eq!(poseidon.elements[2], Fp::ZERO);
    /// ```
    pub fn new_with_preimage(preimage: &[F], constants: &'a PoseidonConstants<F, A>) -> Self {
        let elements = match constants.hash_type {
            HashType::ConstantLength(constant_len) => {
                assert_eq!(constant_len, preimage.len(), "Invalid preimage size");

                GenericArray::generate(|i| {
                    if i == 0 {
                        constants.domain_tag
                    } else if i > preimage.len() {
                        F::ZERO
                    } else {
                        preimage[i - 1]
                    }
                })
            }
            HashType::MerkleTreeSparse(_) => {
                panic!("Merkle Tree (with some empty leaves) hashes are not yet supported.")
            }
            HashType::VariableLength => panic!("variable-length hashes are not yet supported."),
            _ => {
                assert_eq!(preimage.len(), A::to_usize(), "Invalid preimage size");

                GenericArray::generate(|i| {
                    if i == 0 {
                        constants.domain_tag
                    } else {
                        preimage[i - 1]
                    }
                })
            }
        };
        let width = preimage.len() + 1;

        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: width,
            constants,
            _f: PhantomData::<F>,
        }
    }

    /// Replaces the elements with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is not equal to the arity.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_constant_length(1);
    /// let preimage = vec![Fp::from(u64::MAX)];
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new_with_preimage(&preimage, &constants);
    ///
    /// assert_eq!(poseidon.elements.len(), constants.width());
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MAX));
    /// assert_eq!(poseidon.elements[2], Fp::ZERO);
    ///
    /// let preimage = vec![Fp::from(u64::MIN), Fp::from(u64::MIN)];
    /// poseidon.set_preimage(&preimage);
    /// assert_eq!(poseidon.elements.len(), constants.width());
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MIN)); // Now it's u64::MIN
    /// assert_eq!(poseidon.elements[2], Fp::from(u64::MIN)); // Now it's u64::MIN
    /// ```
    pub fn set_preimage(&mut self, preimage: &[F]) {
        self.reset();
        self.elements[1..].copy_from_slice(preimage);
        self.pos = self.elements.len();
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.reset_offsets();
        self.elements[1..].iter_mut().for_each(|l| *l = F::ZERO);
        self.elements[0] = self.constants.domain_tag;
    }

    pub(crate) fn reset_offsets(&mut self) {
        self.constants_offset = 0;
        self.current_round = 0;
        self.pos = 1;
    }

    /// Adds one more field element of preimage to the underlying [`Poseidon`] buffer for further hashing.
    /// The returned `usize` represents the element position (within arity) for the input operation.
    /// Returns [`Error::FullBuffer`] if no more elements can be added for hashing.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new_constant_length(1);
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new(&constants);
    /// assert_eq!(poseidon.elements.len(), constants.width());
    /// assert_eq!(poseidon.elements[1], Fp::ZERO);
    /// assert_eq!(poseidon.elements[2], Fp::ZERO);
    ///
    /// let pos = poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// assert_eq!(pos, 1);
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MAX));
    /// assert_eq!(poseidon.elements[2], Fp::ZERO);
    ///
    /// let pos = poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// assert_eq!(pos, 2);
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MAX));
    /// assert_eq!(poseidon.elements[2], Fp::from(u64::MAX));
    ///
    /// // poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element"); // panic !!!
    /// ```
    pub fn input(&mut self, element: F) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        // To hash constant-length input greater than arity, use sponge explicitly.
        if self.pos >= self.constants.width() {
            return Err(Error::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.elements[self.pos] = element;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
    /// using provided [`HashMode`]. Always outputs digest expressed as a single field element
    /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::{HashMode, PoseidonConstants};
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new();
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new(&constants);
    ///
    /// poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// let digest = poseidon.hash_in_mode(HashMode::Correct);
    ///
    /// assert_ne!(digest, Fp::ZERO); // digest has `Fp` type
    /// ```
    pub fn hash_in_mode(&mut self, mode: HashMode) -> F {
        let res = match mode {
            Correct => hash_correct(self),
            OptimizedDynamic => hash_optimized_dynamic(self),
            OptimizedStatic => self.hash_optimized_static(),
        };
        self.reset_offsets();
        res
    }

    /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
    /// in default (optimized) mode. Always outputs digest expressed as a single field element
    /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new();
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new(&constants);
    ///
    /// poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// let digest = poseidon.hash();
    ///
    /// assert_ne!(digest, Fp::ZERO); // digest has `Fp` type
    /// ```
    pub fn hash(&mut self) -> F {
        self.hash_in_mode(DEFAULT_HASH_MODE)
    }

    pub(crate) fn apply_padding(&mut self) {
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
                    *elt = F::ZERO;
                }
                self.pos = self.elements.len();
            }
            HashType::VariableLength => todo!(),
            _ => (), // incl. HashType::Sponge
        }
    }

    /// Returns 1-th element from underlying [`Poseidon`] buffer. This function is important, since
    /// according to [`Poseidon`] design, after performing hashing, output digest will be stored at
    /// 1-st place of underlying buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new();
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new(&constants);
    ///
    /// poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// let output = poseidon.extract_output();
    ///
    /// assert_eq!(poseidon.elements[1], Fp::from(u64::MAX));
    /// assert_eq!(output, Fp::from(u64::MAX)); // output == input
    ///
    /// let digest = poseidon.hash();
    ///
    /// let output = poseidon.extract_output();
    ///
    /// assert_eq!(poseidon.elements[1], output);
    /// assert_eq!(digest, output); // output == digest
    /// ```
    #[inline]
    pub fn extract_output(&self) -> F {
        self.elements[1]
    }

    /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
    /// using [`HashMode::OptimizedStatic`] mode. Always outputs digest expressed as a single field element
    /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
    ///
    /// # Example
    ///
    /// ```
    /// use neptune::poseidon::PoseidonConstants;
    /// use neptune::poseidon::Poseidon;
    /// use pasta_curves::Fp;
    /// use ff::Field;
    /// use generic_array::typenum::U2;
    ///
    /// let constants: PoseidonConstants<Fp, U2> = PoseidonConstants::new();
    ///
    /// let mut poseidon = Poseidon::<Fp, U2>::new(&constants);
    ///
    /// poseidon.input(Fp::from(u64::MAX)).expect("can't add one more element");
    ///
    /// let digest = poseidon.hash_optimized_static();
    ///
    /// assert_ne!(digest, Fp::ZERO); // digest has `Fp` type
    /// ```
    pub fn hash_optimized_static(&mut self) -> F {
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

        self.extract_output()
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
                    panic!(
                        "Trying to skip last full round, but there is a key here! ({:?})",
                        post
                    );
                } else {
                    Some(post)
                };
                quintic_s_box(l, None, post_key);
            });
        // We need this because post_round_keys will have been empty, so it didn't happen in the for_each. :(
        if last_round {
            self.elements
                .iter_mut()
                .for_each(|l| quintic_s_box(l, None, None));
        } else {
            self.constants_offset += self.elements.len();
        }
        self.round_product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first (arity tag) poseidon leaf.
    fn partial_round(&mut self) {
        let post_round_key = self.constants.compressed_round_constants[self.constants_offset];

        // Apply the quintic S-Box to the first element
        quintic_s_box(&mut self.elements[0], None, Some(&post_round_key));
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
    #[allow(clippy::collapsible_else_if)]
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

                self.product_mds_with_sparse_matrix(sparse_matrix);
            } else {
                self.product_mds();
            }
        };

        self.current_round += 1;
    }

    /// Set the provided elements with the result of the product between the elements and the constant
    /// MDS matrix.
    pub(crate) fn product_mds(&mut self) {
        self.product_mds_with_matrix_left(&self.constants.mds_matrices.m);
    }

    /// NOTE: This calculates a vector-matrix product (`elements * matrix`) rather than the
    /// expected matrix-vector `(matrix * elements)`. This is a performance optimization which
    /// exploits the fact that our MDS matrices are symmetric by construction.
    #[allow(clippy::ptr_arg)]
    pub(crate) fn product_mds_with_matrix(&mut self, matrix: &Matrix<F>) {
        let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::ZERO);

        for (j, val) in result.iter_mut().enumerate() {
            for (i, row) in matrix.iter().enumerate() {
                let mut tmp = row[j];
                tmp.mul_assign(&self.elements[i]);
                val.add_assign(&tmp);
            }
        }

        let _ = std::mem::replace(&mut self.elements, result);
    }

    pub(crate) fn product_mds_with_matrix_left(&mut self, matrix: &Matrix<F>) {
        let result = left_apply_matrix(matrix, &self.elements);
        let _ = std::mem::replace(
            &mut self.elements,
            GenericArray::<F, A::ConstantsSize>::generate(|i| result[i]),
        );
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix(&mut self, sparse_matrix: &SparseMatrix<F>) {
        let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::ZERO);

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

    pub(crate) fn debug(&self, msg: &str) {
        dbg!(msg, &self.constants_offset, &self.elements);
    }
}

#[derive(Debug)]
pub struct SimplePoseidonBatchHasher<F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    constants: PoseidonConstants<F, A>,
    max_batch_size: usize,
}

impl<F, A> SimplePoseidonBatchHasher<F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    pub(crate) fn new(max_batch_size: usize) -> Self {
        Self::new_with_strength(DEFAULT_STRENGTH, max_batch_size)
    }

    pub(crate) fn new_with_strength(strength: Strength, max_batch_size: usize) -> Self {
        Self {
            constants: PoseidonConstants::<F, A>::new_with_strength(strength),
            max_batch_size,
        }
    }
}
impl<F, A> BatchHasher<F, A> for SimplePoseidonBatchHasher<F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    fn hash(&mut self, preimages: &[GenericArray<F, A>]) -> Result<Vec<F>, Error> {
        Ok(preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(preimage, &self.constants).hash())
            .collect())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sponge::vanilla::SpongeTrait;
    use crate::*;
    #[cfg(feature = "abomonation")]
    use abomonation::{decode, encode};
    use blstrs::Scalar as Fr;
    use ff::Field;
    use generic_array::typenum;
    use pasta_curves::pallas::Scalar as S1;

    #[test]
    fn reset() {
        let test_arity = 2;
        let preimage = vec![<Fr as Field>::ONE; test_arity];
        let constants = PoseidonConstants::new();
        let mut h = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);
        h.hash();
        h.reset();

        let default = Poseidon::<Fr, U2>::new(&constants);
        assert_eq!(default.pos, h.pos);
        assert_eq!(default.elements, h.elements);
        assert_eq!(default.constants_offset, h.constants_offset);
    }

    #[test]
    fn hash_det() {
        let test_arity = 2;
        let mut preimage = vec![Fr::ZERO; test_arity];
        let constants = PoseidonConstants::new();
        preimage[0] = <Fr as Field>::ONE;

        let mut h = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_arity_3() {
        let mut preimage = [Fr::ZERO; 3];
        let constants = PoseidonConstants::new();
        preimage[0] = <Fr as Field>::ONE;

        let mut h = Poseidon::<Fr, U3>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_values() {
        hash_values_cases(Strength::Standard);
        hash_values_cases(Strength::Strengthened);
    }

    fn hash_values_cases(strength: Strength) {
        hash_values_aux::<U2>(strength);
        hash_values_aux::<U4>(strength);
        hash_values_aux::<U8>(strength);
        hash_values_aux::<U11>(strength);
        hash_values_aux::<U16>(strength);
        hash_values_aux::<U24>(strength);
        hash_values_aux::<U36>(strength);
    }

    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_values_aux<A>(strength: Strength)
    where
        A: Arity<Fr>,
    {
        let merkle_constants = PoseidonConstants::<Fr, A>::new_with_strength(strength);
        let mut p = Poseidon::<Fr, A>::new(&merkle_constants);
        let mut p2 = Poseidon::<Fr, A>::new(&merkle_constants);
        let mut p3 = Poseidon::<Fr, A>::new(&merkle_constants);
        let mut p4 = Poseidon::<Fr, A>::new(&merkle_constants);

        // Constant-length hashing. Should be tested with arities above, below, and equal to the length.
        let constant_length = 4;
        let constant_constants = PoseidonConstants::<Fr, A>::new_with_strength_and_type(
            strength,
            HashType::ConstantLength(constant_length),
        );
        let mut pc = Poseidon::<Fr, A>::new(&constant_constants);
        let test_arity = A::to_usize();

        for n in 0..test_arity {
            let scalar = Fr::from(n as u64);
            p.input(scalar).unwrap();
            p2.input(scalar).unwrap();
            p3.input(scalar).unwrap();
            p4.input(scalar).unwrap();
        }

        use crate::sponge::vanilla::{Mode, Sponge};

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
                        // dbg!(digest, test_arity);
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
                        // dbg!(digest, test_arity);
                        panic!("Arity lacks test vector: {}", test_arity)
                    }
                }
            }
        };

        let mut constant_sponge = Sponge::new_with_constants(&constant_constants, Mode::Simplex);

        let check_simple = constant_length <= test_arity;
        for n in 0..constant_length {
            let scalar = Fr::from(n as u64);
            constant_sponge.absorb(&scalar, &mut ()).unwrap();
            if check_simple {
                pc.input(scalar).unwrap();
            }
        }

        let constant_sponge_digest = constant_sponge.squeeze(&mut ()).unwrap();
        if check_simple {
            let constant_simple_digest = pc.hash();
            assert_eq!(constant_simple_digest, constant_sponge_digest.unwrap());
        }

        let expected_constant = match strength {
            Strength::Standard =>
            // Currently secure round constants.
            {
                match test_arity {
                    2 => scalar_from_u64s([
                        0x1e12d20d3b71ec56,
                        0x7fb97ce0b8f66322,
                        0xc923003920c488d4,
                        0x19e8a3fe6c2df9ff,
                    ]),
                    4 => scalar_from_u64s([
                        0x8935b00a07909d45,
                        0x4984de08542c9977,
                        0x39443980077d7593,
                        0x3a21a6ae86754a29,
                    ]),
                    8 => scalar_from_u64s([
                        0x370a94532f818897,
                        0x203e3c7c4a85c1f9,
                        0xcad8b9f8aeb1578f,
                        0x5c6de4b69de9d792,
                    ]),
                    11 => scalar_from_u64s([
                        0xe9b0cb7d6496f73b,
                        0x7d2807d793af9582,
                        0xef841b6bf51a5a39,
                        0x02550c3a2113c7ca,
                    ]),
                    16 => scalar_from_u64s([
                        0x6a1e563d359c1bdd,
                        0x6b4493d5d40be9d3,
                        0x275a6eb04a0ecb37,
                        0x30ec6fa0fec08504,
                    ]),
                    24 => scalar_from_u64s([
                        0xc540772c5968a299,
                        0xe2e556352af20f97,
                        0x15ed0a6b8faba5aa,
                        0x327bdee6fa2b22b6,
                    ]),
                    36 => scalar_from_u64s([
                        0x7d89cddb70217dcd,
                        0x02ae71d3d04f0b32,
                        0xfe52151f29c50f99,
                        0x626bdae6cad79307,
                    ]),
                    _ => unimplemented!(),
                }
            }
            Strength::Strengthened => match test_arity {
                2 => scalar_from_u64s([
                    0xcbd4499072dcaff6,
                    0xdd21d8ebc5db51fb,
                    0x336c9c5c50e6a71e,
                    0x28156ad178f3a8fe,
                ]),
                4 => scalar_from_u64s([
                    0xa31d9dc66a42f972,
                    0xb5be830aae89db0d,
                    0xdff9a095d1d40420,
                    0x466e7819bb809c44,
                ]),
                8 => scalar_from_u64s([
                    0x6f2c393786312ee2,
                    0xadb6da339b87e590,
                    0xbf626c21fd6cb051,
                    0x0bb12009ab1fb62a,
                ]),
                11 => scalar_from_u64s([
                    0x6d14b130d0fc1ed5,
                    0x96e16aa48efc68a9,
                    0xf199e67d4e6e4bc7,
                    0x5ee31c86cd42e810,
                ]),
                16 => scalar_from_u64s([
                    0x7dbe8ac03eb7fb25,
                    0xeb53bd55f5095e4e,
                    0x5bc3390694ee8251,
                    0x4611720250274a29,
                ]),
                24 => scalar_from_u64s([
                    0xe2ed71355cbe9268,
                    0x2400a67b915b45fa,
                    0xaa5f37dd1685188e,
                    0x1075afe1e62be162,
                ]),
                36 => scalar_from_u64s([
                    0x8d6f2e5faf077152,
                    0xd3cd55eeec46751a,
                    0x4fc92a1baa0ee777,
                    0x4ed7c0e22446987f,
                ]),
                _ => unimplemented!(),
            },
        };
        assert_eq!(expected_constant, constant_sponge_digest.unwrap());

        assert_eq!(expected, digest);
    }

    #[test]
    fn hash_compare_optimized() {
        let constants = PoseidonConstants::<Fr, U2>::new();
        let mut p = Poseidon::<Fr, U2>::new(&constants);
        let test_arity = constants.arity();
        for n in 0..test_arity {
            let scalar = Fr::from(n as u64);
            p.input(scalar).unwrap();
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
        let default_constants = PoseidonConstants::<Fr, U8>::new();
        let standard_constants = PoseidonConstants::<Fr, U8>::new_with_strength(Strength::Standard);

        assert_eq!(
            standard_constants.partial_rounds,
            default_constants.partial_rounds
        );
    }

    #[cfg(feature = "abomonation")]
    #[test]
    fn roundtrip_abomonation() {
        let mut constants = PoseidonConstants::<Fr, U2>::new();
        constants.round_constants = None;
        let mut bytes = Vec::new();
        unsafe { encode(&constants, &mut bytes).unwrap() };

        if let Some((result, remaining)) =
            unsafe { decode::<PoseidonConstants<Fr, U2>>(&mut bytes) }
        {
            assert!(result == &constants);
            assert!(remaining.is_empty());
        }
    }
}
