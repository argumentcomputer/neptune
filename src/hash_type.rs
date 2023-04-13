/// `HashType` provides support for domain separation tags.
/// For 128-bit security, we need to reserve one (~256-bit) field element per Poseidon permutation.
/// This element cannot be used for hash preimage data — but can be assigned a constant value designating
/// the hash function built on top of the underlying permutation.
///
/// `neptune` implements a variation of the domain separation tag scheme suggested in the updated Poseidon paper. This
/// allows for a variety of modes. This ensures that digest values produced using one hash function cannot be reused
/// where another is required.
///
/// Because `neptune` also supports a first-class notion of `Strength`, we include a mechanism for composing
/// `Strength` with `HashType` so that hashes with `Strength` other than `Standard` (currently only `Strengthened`)
/// may still express the full range of hash function types.
use crate::{Arity, Strength};
use ff::PrimeField;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HashType<F: PrimeField, A: Arity<F>> {
    MerkleTree,
    MerkleTreeSparse(u64),
    VariableLength,
    ConstantLength(usize),
    Encryption,
    #[serde(skip)]
    Custom(CType<F, A>),
    Sponge,
}

impl<F: PrimeField, A: Arity<F>> HashType<F, A> {
    /// Implements domain separation defined in original [Poseidon paper](https://eprint.iacr.org/2019/458.pdf).
    /// Calculates field element used as a zero element in underlying [`crate::poseidon::Poseidon`] buffer that holds preimage.
    pub fn domain_tag(&self) -> F {
        match self {
            // 2^arity - 1
            HashType::MerkleTree => A::tag(),
            // bitmask
            HashType::MerkleTreeSparse(bitmask) => F::from(*bitmask),
            // 2^64
            HashType::VariableLength => pow2::<F>(64),
            // length * 2^64
            // length of 0 denotes a duplex sponge
            HashType::ConstantLength(length) => x_pow2::<F>(*length as u64, 64),
            // 2^32 or (2^32 + 2^32 = 2^33) with strength tag
            HashType::Encryption => pow2::<F>(32),
            // identifier * 2^40
            // identifier must be in range [1..=256]
            // If identifier == 0 then the strengthened version collides with Encryption with standard strength.
            // NOTE: in order to leave room for future `Strength` tags,
            // we make identifier a multiple of 2^40 rather than 2^32.
            HashType::Custom(ref ctype) => ctype.domain_tag(),
            HashType::Sponge => F::zero(),
        }
    }

    /// Some HashTypes require more testing so are not yet supported, since they are not yet needed.
    /// As and when needed, support can be added, along with tests to ensure the initial implementation
    /// is sound.
    pub const fn is_supported(&self) -> bool {
        match self {
            HashType::MerkleTree => true,
            HashType::MerkleTreeSparse(_) => false,
            HashType::VariableLength => false,
            HashType::ConstantLength(_) => true,
            HashType::Encryption => true,
            HashType::Custom(_) => true,
            HashType::Sponge => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CType<F: PrimeField, A: Arity<F>> {
    Arbitrary(u64),
    _Phantom((F, A)),
}

impl<F: PrimeField, A: Arity<F>> CType<F, A> {
    const fn identifier(&self) -> u64 {
        match self {
            CType::Arbitrary(id) => *id,
            CType::_Phantom(_) => panic!("_Phantom is not a real custom tag type."),
        }
    }

    fn domain_tag(&self) -> F {
        let id = self.identifier();
        assert!(id > 0, "custom domain tag id out of range");
        assert!(id <= 256, "custom domain tag id out of range");

        x_pow2::<F>(id, 40)
    }
}

/// pow2(n) = 2^n
fn pow2<F: PrimeField>(n: u64) -> F {
    F::from(2).pow_vartime([n])
}

/// x_pow2(x, n) = x * 2^n
fn x_pow2<F: PrimeField>(coeff: u64, n: u64) -> F {
    let mut tmp = pow2::<F>(n);
    tmp.mul_assign(F::from(coeff));
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scalar_from_u64s, Strength};
    use blstrs::Scalar as Fr;
    use generic_array::typenum::{U15, U8};
    use std::collections::HashSet;

    #[test]
    fn test_domain_tags() {
        let merkle_standard = HashType::MerkleTree::<Fr, U8>.domain_tag();
        let expected_merkle_standard = scalar_from_u64s([
            0x00000000000000ff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ]);

        assert_eq!(expected_merkle_standard, merkle_standard);

        // TODO: tests for
        // MerkleTreeSparse(u64),
        // VariableLength,
        // Custom(CType<Fr, A>),

        let mut all_tags = Vec::new();

        for length in 1..15 {
            let constant_standard = HashType::ConstantLength::<Fr, U15>(length).domain_tag();

            all_tags.push(constant_standard);

            if length <= 8 {
                let constant_standard_alt_arity =
                    HashType::ConstantLength::<Fr, U8>(length).domain_tag();

                // Constant-length tag is independent of arity.
                assert_eq!(constant_standard, constant_standard_alt_arity);
            }

            assert_eq!(
                constant_standard,
                scalar_from_u64s([
                    0x0000000000000000,
                    length as u64,
                    0x0000000000000000,
                    0x0000000000000000
                ])
            );
        }

        let encryption_standard = HashType::Encryption::<Fr, U8>.domain_tag();
        let expected_encryption_standard = scalar_from_u64s([
            0x0000000100000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ]);
        assert_eq!(expected_encryption_standard, encryption_standard,);

        for index in 1..=256 {
            let custom = HashType::Custom::<Fr, U8>(CType::Arbitrary(index));
            let standard_custom = custom.domain_tag();

            let expected_standard_custom = scalar_from_u64s([
                0x0000010000000000 * index,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ]);

            all_tags.push(expected_standard_custom);

            assert_eq!(expected_standard_custom, standard_custom);
        }

        all_tags.extend(&[expected_merkle_standard, expected_encryption_standard]);

        let standard_sponge = HashType::Sponge::<Fr, U8>.domain_tag();
        let expected_standard_sponge = scalar_from_u64s([0, 0, 0, 0]);

        all_tags.push(standard_sponge);

        assert_eq!(expected_standard_sponge, standard_sponge);

        let mut all_tags_set = HashSet::new();
        all_tags.iter().for_each(|x| {
            let _ = all_tags_set.insert(x.to_repr());
        });

        // Cardinality of set and vector are the same,
        // hence no tag is duplicated.
        assert_eq!(all_tags.len(), all_tags_set.len(), "duplicate tag produced");
    }
}
