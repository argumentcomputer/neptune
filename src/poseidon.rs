use crate::matrix;
use crate::{generate_mds, round_constants, round_numbers, scalar_from_u64, Error};
use ff::{Field, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;

/// The arity tag is the first element of a Poseidon permutation.
/// This extra element is necessary for 128-bit security.
pub fn arity_tag<E: ScalarEngine, Arity: typenum::Unsigned>() -> E::Fr {
    scalar_from_u64::<E>((1 << Arity::to_usize()) - 1)
}

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
#[derive(Debug, Clone, PartialEq)]
pub struct Poseidon<'a, E, Arity = typenum::U2>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    constants_offset: usize,
    /// the elements to permute
    pub elements: GenericArray<E::Fr, typenum::Add1<Arity>>,
    pos: usize,
    constants: &'a PoseidonConstants<E, Arity>,
    _e: PhantomData<E>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseidonConstants<E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub mds_matrix: Vec<Vec<E::Fr>>,
    pub inverse_mds_matrix: Vec<Vec<E::Fr>>,
    pub round_constants: Vec<E::Fr>,
    pub arity_tag: E::Fr,
    pub full_rounds: usize,
    pub partial_rounds: usize,
    _a: PhantomData<Arity>,
}

impl<E, Arity> PoseidonConstants<E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub fn new() -> Self {
        let arity = Arity::to_usize();
        let width = arity + 1;

        let mds_matrix = generate_mds::<E>(width);
        let inverse_mds_matrix = matrix::invert::<E>(&mds_matrix).unwrap();

        let (full_rounds, partial_rounds) = round_numbers(arity);
        let round_constants = round_constants::<E>(arity);

        // Ensure we have enough constants for the sbox rounds
        assert!(
            width * (full_rounds + partial_rounds) <= round_constants.len(),
            "Not enough round constants"
        );

        Self {
            mds_matrix,
            inverse_mds_matrix,
            round_constants,
            arity_tag: arity_tag::<E, Arity>(),
            full_rounds,
            partial_rounds,
            _a: PhantomData::<Arity>,
        }
    }

    /// Returns the width.
    #[inline]
    pub fn arity(&self) -> usize {
        Arity::to_usize()
    }

    /// Returns the width.
    #[inline]
    pub fn width(&self) -> usize {
        use typenum::Unsigned;
        typenum::Add1::<Arity>::to_usize()
    }
}

impl<'a, E, Arity> Poseidon<'a, E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub fn new(constants: &'a PoseidonConstants<E, Arity>) -> Self {
        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.arity_tag
            } else {
                E::Fr::zero()
            }
        });

        Poseidon {
            constants_offset: 0,
            elements,
            pos: 1,
            constants,
            _e: PhantomData::<E>,
        }
    }
    pub fn new_with_preimage(
        preimage: &[E::Fr],
        constants: &'a PoseidonConstants<E, Arity>,
    ) -> Self {
        assert_eq!(preimage.len(), Arity::to_usize(), "Invalid preimage size");

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
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.constants_offset = 0;
        self.elements[1..]
            .iter_mut()
            .for_each(|l| *l = scalar_from_u64::<E>(0u64));
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

    pub fn hash(&mut self) -> E::Fr {
        self.hash_optimized()
        // let funky = true;

        // if funky {
        //     self.hash_funky()
        // } else {
        //     self.hash_simple()
        // }
    }

    /// The number of rounds is divided into two equal parts for the full rounds, plus the partial rounds.
    ///
    /// The returned element is the second poseidon element, the first is the arity tag.
    pub fn hash_correct(&mut self) -> E::Fr {
        self.debug("Hash simple");
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        // The first full round should use the initial constants.
        self.full_round(true, false);

        self.debug("After first full round");

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(true, false);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        self.debug("Before first partial round");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(true, false);

        self.debug("After first partial round");

        for _ in 1..self.constants.partial_rounds {
            self.partial_round(true, false);
        }

        self.debug("After last partial round");

        for _ in 0..self.constants.full_rounds / 2 {
            self.full_round(true, false);
        }

        self.debug("After last full round");

        self.elements[1]
    }

    pub fn hash_simple(&mut self) -> E::Fr {
        self.debug("Hash simple");
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        // The first full round should use the initial constants.
        self.full_round(true, false);
        self.debug("After first full round");

        // => M(B)

        // M(B) + S
        //self.add_round_constants();

        // Verify that the round constants here (S), really are the S from full_round (post_vec).

        self.debug("After adding constants");

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(true, false);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        // self.add_round_constants();

        self.debug("Before first partial round");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(true, false);

        // self.debug("After first partial round");

        // for _ in 1..self.constants.partial_rounds {
        //     self.partial_round(true, false);
        // }

        // self.debug("After last partial round");

        // for _ in 0..self.constants.full_rounds / 2 {
        //     self.full_round(true, false);
        // }

        // self.debug("After last full round");

        // B + S
        self.elements[1]
    }

    pub fn hash_funky(&mut self) -> E::Fr {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        self.debug("Hash funky");
        // The first full round should use the initial constants.
        self.full_round(true, true);
        self.debug("After first full round");

        // => M(B + M^-1(S))

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(false, true);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        self.debug("Before first partial round");

        // // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(false, false);

        // self.debug("After first partial round");

        // for _ in 1..self.constants.partial_rounds {
        //     self.partial_round(true, false);
        // }

        // self.debug("After last partial round");

        // for _ in 0..self.constants.full_rounds / 2 {
        //     self.full_round(true, false);
        // }

        // self.debug("After last full round");

        // M(B + M^-1(S))
        self.elements[1]
    }

    pub fn hash_optimized(&mut self) -> E::Fr {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat

        self.debug("Hash funky");

        // The first full round should use the initial constants.
        self.full_round(true, true);

        self.debug("After first full round");

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(false, true);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        self.debug("Before first partial round");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(false, false);

        self.debug("After first partial round");

        for _ in 1..self.constants.partial_rounds {
            self.partial_round(true, false);
        }

        self.debug("After last partial round");

        for _ in 0..self.constants.full_rounds / 2 {
            self.full_round(true, false);
        }

        self.debug("After last full round");

        self.elements[1]
    }

    /// The full round function will add the round constants and apply the S-Box to all poseidon elements, including the bitflags first element.
    ///
    /// After that, the poseidon elements will be set to the result of the product between the poseidon elements and the constant MDS matrix.
    pub fn full_round(&mut self, add_current_round_keys: bool, absorb_next_round_keys: bool) {
        // NOTE: decrease in performance is expected during this refactoring.
        // We seek to preserve correctness while transforming the algorithm to an eventually more performant one.

        // Apply the quintic S-Box to all elements, after adding the round key.
        // Round keys are added in the S-box to match circuits (where the addition is free)
        // and in preparation for the shift to adding round keys after (rather than before) applying the S-box.

        let pre_round_keys = self
            .constants
            .round_constants
            .iter()
            .skip(self.constants_offset)
            .map(|x| {
                if add_current_round_keys {
                    Some(x)
                } else {
                    None
                }
            });

        if absorb_next_round_keys {
            // Using the notation from `test_inverse` in matrix.rs:

            // S
            let post_vec = self
                .constants
                .round_constants
                .iter()
                .skip(
                    self.constants_offset
                        + if add_current_round_keys {
                            self.elements.len()
                        } else {
                            0
                        },
                )
                .take(self.elements.len())
                .map(|x| *x)
                .collect::<Vec<_>>();

            // Compute the constants which should be added *before* the next `product_mds`.
            // in order to have the same effect as adding the given constants *after* the next `product_mds`.

            // M^-1(S)
            let inverted_vec =
                matrix::apply_matrix::<E>(&self.constants.inverse_mds_matrix, &post_vec);

            // M(M^-1(S))
            let original = matrix::apply_matrix::<E>(&self.constants.mds_matrix, &inverted_vec);

            dbg!(
                "post_vec (should be same as some round constants)",
                &post_vec,
                &inverted_vec,
                &original
            );
            // S = M(M^-1(S))
            assert_eq!(&post_vec, &original, "Oh no, the inversion trick failed.");

            let post_round_keys = inverted_vec.iter();

            // S-Box Output = B.
            // With post-add, result is B + M^-1(S).
            self.elements
                .iter_mut()
                .zip(pre_round_keys.zip(post_round_keys))
                .for_each(|(l, (pre, post))| {
                    dbg!("a");
                    quintic_s_box::<E>(l, pre, Some(post));
                });
        } else {
            self.elements
                .iter_mut()
                .zip(pre_round_keys)
                .for_each(|(l, pre)| {
                    dbg!("b");
                    quintic_s_box::<E>(l, pre, None);
                });
        }
        let mut consumed = 0;
        if add_current_round_keys {
            consumed += self.elements.len()
        };
        if absorb_next_round_keys {
            consumed += self.elements.len()
        };
        self.constants_offset += consumed;

        let stashed = self.elements.clone();

        // If absorb_next_round_keys
        //   M(B + M^-1(S)
        // else
        //   M(B)
        // Multiply the elements by the constant MDS matrix
        self.product_mds();

        let applied = matrix::apply_matrix::<E>(&self.constants.mds_matrix, &stashed.to_vec());
        assert_eq!(
            applied[..],
            self.elements[..],
            "product_mds gives different result than matrix application",
        );
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
    pub fn partial_round(&mut self, add_current_round_keys: bool, absorb_next_round_keys: bool) {
        assert!(!absorb_next_round_keys); // Not yet implemented.

        if add_current_round_keys {
            // Every element of the hash buffer is incremented by the round constants
            self.add_round_constants();
        }

        // Apply the quintic S-Box to the first element
        quintic_s_box::<E>(&mut self.elements[0], None, None);

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// For every leaf, add the round constants with index defined by the constants offset, and increment the
    /// offset
    fn add_round_constants(&mut self) {
        for (element, round_constant) in self.elements.iter_mut().zip(
            self.constants
                .round_constants
                .iter()
                .skip(self.constants_offset),
        ) {
            dbg!("adding round constant: {}", &round_constant);
            element.add_assign(round_constant);
        }

        self.constants_offset += self.elements.len();
    }

    /// Set the provided elements with the result of the product between the elements and the constant
    /// MDS matrix
    fn product_mds(&mut self) {
        let mut result = GenericArray::<E::Fr, typenum::Add1<Arity>>::generate(|_| E::Fr::zero());

        for (result, mds_row) in result.iter_mut().zip(self.constants.mds_matrix.iter()) {
            for (mds, element) in mds_row.iter().zip(self.elements.iter()) {
                let mut tmp = *mds;
                tmp.mul_assign(element);
                result.add_assign(&tmp);
            }
        }

        std::mem::replace(&mut self.elements, result);
    }

    fn debug(&self, msg: &str) {
        dbg!(msg, &self.constants_offset, &self.elements);
    }
}

/// Apply the quintic S-Box (s^5) to a given item
fn quintic_s_box<E: ScalarEngine>(
    l: &mut E::Fr,
    pre_add: Option<&E::Fr>,
    post_add: Option<&E::Fr>,
) {
    if let Some(x) = pre_add {
        l.add_assign(x);
    }
    dbg!("S-box input", &l);
    let c = *l;
    let mut tmp = l.clone();
    tmp.mul_assign(&c);
    tmp.mul_assign(&tmp.clone());
    l.mul_assign(&tmp);
    dbg!("S-box output", &l);
    if let Some(x) = post_add {
        l.add_assign(x);
        dbg!("After S-box post-add", &l);
    }
}

/// Poseidon convenience hash function.
/// NOTE: this is expensive, since it computes all constants when initializing hasher struct.
pub fn poseidon<E, Arity>(preimage: &[E::Fr]) -> E::Fr
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    let constants = PoseidonConstants::<E, Arity>::new();
    Poseidon::<E, Arity>::new_with_preimage(preimage, &constants).hash()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use ff::Field;
    //    use generic_array::typenum::{U2, U4, U8};
    use generic_array::typenum::U2;
    use paired::bls12_381::Bls12;

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
    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_values() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let digest = p.hash();
        let expected = match test_arity {
            2 => scalar_from_u64s([
                0x7179d3495ac25e92,
                0x81052897659f7762,
                0x316a6d20e4a55d6c,
                0x409e8342edab687b,
            ]),
            4 => scalar_from_u64s([
                0xf53a7d58aacf0621,
                0x42d3a014639efdcf,
                0xe1a3fddb08c13a46,
                0x43f94dbd0abd1c99,
            ]),
            8 => scalar_from_u64s([
                0xa6a3e7a6b2cc7b85,
                0xfb1eb8f641dd9dc3,
                0xfd2a373272ebf604,
                0x433c1e9e8de226e5,
            ]),
            _ => {
                dbg!(digest);
                panic!("Arity lacks test vector: {}", test_arity)
            }
        };
        dbg!(test_arity);
        assert_eq!(expected, digest);

        assert_eq!(
            digest,
            poseidon::<Bls12, U2>(&preimage),
            "Poseidon wrapper disagrees with element-at-a-time invocation."
        );
    }

    #[test]
    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_compare_funky() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let mut p2 = p.clone();
        // M(B) + S
        let digest_simple = p.hash_simple();

        // M(B + M^-1(S))
        let digest_funky = p2.hash_funky();

        dbg!(&p.constants.round_constants[0..10]);
        // M(B) + S = M(B + M^-1(S))

        p.debug("hash simple");
        p2.debug("hash funky");
        assert_eq!(digest_simple, digest_funky);
    }

    #[test]
    fn hash_compare_optimized() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let mut p2 = p.clone();

        let digest_correct = p.hash_correct();

        let digest_optimized = p2.hash_optimized();

        dbg!(&p.constants.round_constants[0..10]);

        assert_eq!(digest_correct, digest_optimized);
    }
}
