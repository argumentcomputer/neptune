use crate::matrix::Matrix;
use crate::mds::{create_mds_matrices, factor_to_sparse_matrixes, MDSMatrices, SparseMatrix};
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

/// The number of rounds is divided into two equal parts for the full rounds, plus the partial rounds.
///
/// The returned element is the second poseidon element, the first is the arity tag.
pub fn hash_correct(&mut self) -> E::Fr {
    // This counter is incremented when a round constants is read. Therefore, the round constants never repeat.
    // The first full round should use the initial constants.
    self.full_round();

    for _ in 1..self.constants.half_full_rounds {
        self.full_round();
    }

    self.partial_round();

    for _ in 1..self.constants.partial_rounds {
        self.partial_round();
    }

    for _ in 0..self.constants.half_full_rounds {
        self.full_round();
    }

    self.elements[1]
}

pub fn full_round(&mut self) {
    // Apply the quintic S-Box to all elements, after adding the round key.
    // Round keys are added in the S-box to match circuits (where the addition is free)
    // and in preparation for the shift to adding round keys after (rather than before) applying the S-box.

    let pre_round_keys = self
        .constants
        .round_constants
        .iter()
        .skip(self.constants_offset)
        .map(|x| Some(x));

    self.elements
        .iter_mut()
        .zip(pre_round_keys)
        .for_each(|(l, pre)| {
            quintic_s_box::<E>(l, pre, None);
        });

    self.constants_offset += self.elements.len();

    // M(B)
    // Multiply the elements by the constant MDS matrix
    self.product_mds();
}

/// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
pub fn partial_round(&mut self) {
    // Every element of the hash buffer is incremented by the round constants
    self.add_round_constants();

    // Apply the quintic S-Box to the first element
    quintic_s_box::<E>(&mut self.elements[0], None, None);

    // Multiply the elements by the constant MDS matrix
    self.product_mds();
}

pub fn hash_optimized_dynamic(&mut self) -> E::Fr {
    // The first full round should use the initial constants.
    self.full_round_dynamic(true, true);

    for _ in 1..(self.constants.half_full_rounds) {
        self.full_round_dynamic(false, true);
    }

    self.partial_round_dynamic();

    for _ in 1..self.constants.partial_rounds {
        self.partial_round();
    }

    for _ in 0..self.constants.half_full_rounds {
        self.full_round_dynamic(true, false);
    }

    self.elements[1]
}

pub fn full_round_dynamic(&mut self, add_current_round_keys: bool, absorb_next_round_keys: bool) {
    // NOTE: decrease in performance is expected when using this pathway.
    // We seek to preserve correctness while transforming the algorithm to an eventually more performant one.

    // Round keys are added in the S-box to match circuits (where the addition is free).
    // If requested, add round keys synthesized from following round after (rather than before) applying the S-box.
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
        let inverted_vec = matrix::apply_matrix::<E>(&self.constants.mds_matrices.m_inv, &post_vec);

        // M(M^-1(S))
        let original = matrix::apply_matrix::<E>(&self.constants.mds_matrices.m, &inverted_vec);

        // S = M(M^-1(S))
        assert_eq!(&post_vec, &original, "Oh no, the inversion trick failed.");

        let post_round_keys = inverted_vec.iter();

        // S-Box Output = B.
        // With post-add, result is B + M^-1(S).
        self.elements
            .iter_mut()
            .zip(pre_round_keys.zip(post_round_keys))
            .for_each(|(l, (pre, post))| {
                quintic_s_box::<E>(l, pre, Some(post));
            });
    } else {
        self.elements
            .iter_mut()
            .zip(pre_round_keys)
            .for_each(|(l, pre)| {
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

    // If absorb_next_round_keys
    //   M(B + M^-1(S)
    // else
    //   M(B)
    // Multiply the elements by the constant MDS matrix
    self.product_mds();
}

pub fn partial_round_dynamic(&mut self) {
    // Apply the quintic S-Box to the first element
    quintic_s_box::<E>(&mut self.elements[0], None, None);

    // Multiply the elements by the constant MDS matrix
    self.product_mds();
}

/// For every leaf, add the round constants with index defined by the constants offset, and increment the
/// offset.
fn add_round_constants(&mut self) {
    for (element, round_constant) in self.elements.iter_mut().zip(
        self.constants
            .round_constants
            .iter()
            .skip(self.constants_offset),
    ) {
        element.add_assign(round_constant);
    }

    self.constants_offset += self.elements.len();
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
