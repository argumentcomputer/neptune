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
use serde::{Deserialize, Serialize, Serializer, __private as private, de, ser, Deserializer};

#[derive(Clone, Debug, PartialEq)]
pub enum HashType<F: PrimeField, A: Arity<F>> {
    MerkleTree,
    MerkleTreeSparse(u64),
    VariableLength,
    ConstantLength(usize),
    Encryption,
    // #[serde(skip)] // See: https://github.com/bincode-org/bincode/issues/424
    Custom(CType<F, A>),
    Sponge,
}

#[doc(hidden)]
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
            HashType::Sponge => F::ZERO,
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

impl<F: PrimeField, A: Arity<F>> Serialize for HashType<F, A> {
    fn serialize<S>(&self, serializer: S) -> private::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            HashType::MerkleTree => {
                Serializer::serialize_unit_variant(serializer, "HashType", 0u32, "MerkleTree")
            }
            HashType::MerkleTreeSparse(ref field0) => Serializer::serialize_newtype_variant(
                serializer,
                "HashType",
                1u32,
                "MerkleTreeSparse",
                field0,
            ),
            HashType::VariableLength => {
                Serializer::serialize_unit_variant(serializer, "HashType", 2u32, "VariableLength")
            }
            HashType::ConstantLength(ref field0) => Serializer::serialize_newtype_variant(
                serializer,
                "HashType",
                3u32,
                "ConstantLength",
                field0,
            ),
            HashType::Encryption => {
                Serializer::serialize_unit_variant(serializer, "HashType", 4u32, "Encryption")
            }
            HashType::Custom(..) => private::Err(ser::Error::custom(
                "the enum variant HashType::Custom cannot be serialized",
            )),
            HashType::Sponge => {
                Serializer::serialize_unit_variant(serializer, "HashType", 5u32, "Sponge")
            }
        }
    }
}

#[doc(hidden)]
impl<'de, F: PrimeField, A: Arity<F>> Deserialize<'de> for HashType<F, A> {
    fn deserialize<D>(deserializer: D) -> private::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[allow(non_camel_case_types)]
        #[doc(hidden)]
        enum Field {
            field0,
            field1,
            field2,
            field3,
            field4,
            field6,
        }
        #[doc(hidden)]
        struct FieldVisitor;
        impl<'de> de::Visitor<'de> for FieldVisitor {
            type Value = Field;
            fn expecting(&self, formatter: &mut private::Formatter) -> private::fmt::Result {
                private::Formatter::write_str(formatter, "variant identifier")
            }
            fn visit_u64<E>(self, value: u64) -> private::Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    0u64 => private::Ok(Field::field0),
                    1u64 => private::Ok(Field::field1),
                    2u64 => private::Ok(Field::field2),
                    3u64 => private::Ok(Field::field3),
                    4u64 => private::Ok(Field::field4),
                    5u64 => private::Ok(Field::field6),
                    _ => private::Err(de::Error::invalid_value(
                        de::Unexpected::Unsigned(value),
                        &"variant index 0 <= i < 6",
                    )),
                }
            }
            fn visit_str<E>(self, value: &str) -> private::Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    "MerkleTree" => private::Ok(Field::field0),
                    "MerkleTreeSparse" => private::Ok(Field::field1),
                    "VariableLength" => private::Ok(Field::field2),
                    "ConstantLength" => private::Ok(Field::field3),
                    "Encryption" => private::Ok(Field::field4),
                    "Sponge" => private::Ok(Field::field6),
                    _ => private::Err(de::Error::unknown_variant(value, VARIANTS)),
                }
            }
            fn visit_bytes<E>(self, value: &[u8]) -> private::Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    b"MerkleTree" => private::Ok(Field::field0),
                    b"MerkleTreeSparse" => private::Ok(Field::field1),
                    b"VariableLength" => private::Ok(Field::field2),
                    b"ConstantLength" => private::Ok(Field::field3),
                    b"Encryption" => private::Ok(Field::field4),
                    b"Sponge" => private::Ok(Field::field6),
                    _ => {
                        let value = &private::from_utf8_lossy(value);
                        private::Err(de::Error::unknown_variant(value, VARIANTS))
                    }
                }
            }
        }
        impl<'de> Deserialize<'de> for Field {
            #[inline]
            fn deserialize<D>(deserializer: D) -> private::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Deserializer::deserialize_identifier(deserializer, FieldVisitor)
            }
        }
        #[doc(hidden)]
        struct Visitor<'de, F: PrimeField, A: Arity<F>> {
            marker: private::PhantomData<HashType<F, A>>,
            lifetime: private::PhantomData<&'de ()>,
        }
        impl<'de, F: PrimeField, A: Arity<F>> de::Visitor<'de> for Visitor<'de, F, A> {
            type Value = HashType<F, A>;
            fn expecting(&self, formatter: &mut private::Formatter) -> private::fmt::Result {
                private::Formatter::write_str(formatter, "enum HashType")
            }
            fn visit_enum<__A>(self, data: __A) -> private::Result<Self::Value, __A::Error>
            where
                __A: de::EnumAccess<'de>,
            {
                match match de::EnumAccess::variant(data) {
                    private::Ok(val) => val,
                    private::Err(err) => {
                        return private::Err(err);
                    }
                } {
                    (Field::field0, variant) => {
                        match de::VariantAccess::unit_variant(variant) {
                            private::Ok(val) => val,
                            private::Err(err) => {
                                return private::Err(err);
                            }
                        };
                        private::Ok(HashType::MerkleTree)
                    }
                    (Field::field1, variant) => private::Result::map(
                        de::VariantAccess::newtype_variant::<u64>(variant),
                        HashType::MerkleTreeSparse,
                    ),
                    (Field::field2, variant) => {
                        match de::VariantAccess::unit_variant(variant) {
                            private::Ok(val) => val,
                            private::Err(err) => {
                                return private::Err(err);
                            }
                        };
                        private::Ok(HashType::VariableLength)
                    }
                    (Field::field3, variant) => private::Result::map(
                        de::VariantAccess::newtype_variant::<usize>(variant),
                        HashType::ConstantLength,
                    ),
                    (Field::field4, variant) => {
                        match de::VariantAccess::unit_variant(variant) {
                            private::Ok(val) => val,
                            private::Err(err) => {
                                return private::Err(err);
                            }
                        };
                        private::Ok(HashType::Encryption)
                    }
                    (Field::field6, variant) => {
                        match de::VariantAccess::unit_variant(variant) {
                            private::Ok(val) => val,
                            private::Err(err) => {
                                return private::Err(err);
                            }
                        };
                        private::Ok(HashType::Sponge)
                    }
                }
            }
        }
        #[doc(hidden)]
        const VARIANTS: &'static [&'static str] = &[
            "MerkleTree",
            "MerkleTreeSparse",
            "VariableLength",
            "ConstantLength",
            "Encryption",
            "Sponge",
        ];
        Deserializer::deserialize_enum(
            deserializer,
            "HashType",
            VARIANTS,
            Visitor {
                marker: private::PhantomData::<HashType<F, A>>,
                lifetime: private::PhantomData,
            },
        )
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
