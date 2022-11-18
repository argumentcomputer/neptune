use ff::PrimeField;
use serde::{
    de::{self, Deserializer, MapAccess, SeqAccess, Visitor},
    ser::{SerializeStruct, Serializer},
    Deserialize, Serialize,
};
use std::fmt;
use std::marker::PhantomData;

use crate::hash_type::HashType;
use crate::poseidon::PoseidonConstants;
use crate::Arity;

impl<F, A> Serialize for PoseidonConstants<F, A>
where
    F: PrimeField + Serialize,
    A: Arity<F>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PoseidonConstants", 8)?;
        state.serialize_field("mds", &self.mds_matrices)?;
        state.serialize_field("crc", &self.compressed_round_constants)?;
        state.serialize_field("psm", &self.pre_sparse_matrix)?;
        state.serialize_field("sm", &self.sparse_matrixes)?;
        state.serialize_field("s", &self.strength)?;
        state.serialize_field("rf", &self.full_rounds)?;
        state.serialize_field("rp", &self.partial_rounds)?;
        state.serialize_field("ht", &self.hash_type)?;
        state.end()
    }
}

impl<'de, F, A> Deserialize<'de> for PoseidonConstants<F, A>
where
    F: PrimeField + Deserialize<'de>,
    A: Arity<F>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Mds,
            Crc,
            Psm,
            Sm,
            S,
            Rf,
            Rp,
            Ht,
        }

        struct PoseidonConstantsVisitor<F, A>
        where
            F: PrimeField,
            A: Arity<F>,
        {
            _f: PhantomData<F>,
            _a: PhantomData<A>,
        }

        impl<'de, F, A> Visitor<'de> for PoseidonConstantsVisitor<F, A>
        where
            F: PrimeField + Deserialize<'de>,
            A: Arity<F>,
        {
            type Value = PoseidonConstants<F, A>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct PoseidonConstants")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<PoseidonConstants<F, A>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let mds_matrices = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let compressed_round_constants = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let pre_sparse_matrix = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let sparse_matrixes = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let strength = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                let full_rounds = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(5, &self))?;
                let partial_rounds = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(6, &self))?;
                let hash_type: HashType<F, A> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(7, &self))?;

                Ok(PoseidonConstants {
                    mds_matrices,
                    round_constants: None,
                    compressed_round_constants,
                    pre_sparse_matrix,
                    sparse_matrixes,
                    strength,
                    domain_tag: hash_type.domain_tag(),
                    full_rounds,
                    half_full_rounds: full_rounds / 2,
                    partial_rounds,
                    hash_type,
                    _a: PhantomData::<A>,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<PoseidonConstants<F, A>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut mds_matrices = None;
                let mut compressed_round_constants = None;
                let mut pre_sparse_matrix = None;
                let mut sparse_matrixes = None;
                let mut strength = None;
                let mut full_rounds = None;
                let mut partial_rounds = None;
                let mut hash_type = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Mds => {
                            if mds_matrices.is_some() {
                                return Err(de::Error::duplicate_field("mds_matrices"));
                            }
                            mds_matrices = Some(map.next_value()?);
                        }
                        Field::Crc => {
                            if compressed_round_constants.is_some() {
                                return Err(de::Error::duplicate_field(
                                    "compressed_round_constants",
                                ));
                            }
                            compressed_round_constants = Some(map.next_value()?);
                        }
                        Field::Psm => {
                            if pre_sparse_matrix.is_some() {
                                return Err(de::Error::duplicate_field("pre_sparse_matrix"));
                            }
                            pre_sparse_matrix = Some(map.next_value()?);
                        }
                        Field::Sm => {
                            if sparse_matrixes.is_some() {
                                return Err(de::Error::duplicate_field("sparse_matrixes"));
                            }
                            sparse_matrixes = Some(map.next_value()?);
                        }
                        Field::S => {
                            if strength.is_some() {
                                return Err(de::Error::duplicate_field("strength"));
                            }
                            strength = Some(map.next_value()?);
                        }
                        Field::Rf => {
                            if full_rounds.is_some() {
                                return Err(de::Error::duplicate_field("full_rounds"));
                            }
                            full_rounds = Some(map.next_value()?);
                        }
                        Field::Rp => {
                            if partial_rounds.is_some() {
                                return Err(de::Error::duplicate_field("partial_rounds"));
                            }
                            partial_rounds = Some(map.next_value()?);
                        }
                        Field::Ht => {
                            if hash_type.is_some() {
                                return Err(de::Error::duplicate_field("hash_type"));
                            }
                            hash_type = Some(map.next_value()?);
                        }
                    }
                }

                let mds_matrices =
                    mds_matrices.ok_or_else(|| de::Error::missing_field("mds_matrices"))?;
                let compressed_round_constants = compressed_round_constants
                    .ok_or_else(|| de::Error::missing_field("compressed_round_constants"))?;
                let pre_sparse_matrix = pre_sparse_matrix
                    .ok_or_else(|| de::Error::missing_field("pre_sparse_matrix"))?;
                let sparse_matrixes =
                    sparse_matrixes.ok_or_else(|| de::Error::missing_field("sparse_matrixes"))?;
                let strength = strength.ok_or_else(|| de::Error::missing_field("strength"))?;
                let full_rounds =
                    full_rounds.ok_or_else(|| de::Error::missing_field("full_rounds"))?;
                let partial_rounds =
                    partial_rounds.ok_or_else(|| de::Error::missing_field("partial_rounds"))?;
                let hash_type: HashType<F, A> =
                    hash_type.ok_or_else(|| de::Error::missing_field("hash_type"))?;
                Ok(PoseidonConstants {
                    mds_matrices,
                    round_constants: None,
                    compressed_round_constants,
                    pre_sparse_matrix,
                    sparse_matrixes,
                    strength,
                    domain_tag: hash_type.domain_tag(),
                    full_rounds,
                    half_full_rounds: full_rounds / 2,
                    partial_rounds,
                    hash_type,
                    _a: PhantomData::<A>,
                })
            }
        }

        const FIELDS: &[&str] = &["
	  mds_matrices,
	  compressed_round_constants,
	  pre_sparse_matrix,
	  sparse_matrixes,
	  strength,
	  full_rounds,
	  partial_rounds,
	  hash_type,
"];
        deserializer.deserialize_struct(
            "PoseidonConstants",
            FIELDS,
            PoseidonConstantsVisitor {
                _f: PhantomData,
                _a: PhantomData,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Poseidon;
    use blstrs::Scalar as Fr;
    use ff::Field;
    use generic_array::typenum;
    use pasta_curves::pallas::Scalar as S1;
    use typenum::{U1, U2};

    #[test]
    fn serde_roundtrip() {
        let mut constants = PoseidonConstants::<S1, U1>::new();
        constants.round_constants = None;

        assert_eq!(
            constants,
            serde_json::from_slice(&serde_json::to_vec(&constants).unwrap()).unwrap()
        );
    }
    #[test]
    fn serde_hash_blstrs() {
        let constants = PoseidonConstants::<Fr, U2>::new();
        let constants2 = serde_json::from_slice(&serde_json::to_vec(&constants).unwrap()).unwrap();
        let test_arity = 2;
        let preimage = vec![<Fr as Field>::one(); test_arity];
        let mut h1 = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);
        let mut h2 = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants2);

        assert_eq!(h1.hash(), h2.hash())
    }

    #[test]
    fn serde_hash_pallas() {
        let constants = PoseidonConstants::<S1, U2>::new();
        let constants2 = serde_json::from_slice(&serde_json::to_vec(&constants).unwrap()).unwrap();
        let test_arity = 2;
        let preimage = vec![<S1 as Field>::one(); test_arity];
        let mut h1 = Poseidon::<S1, U2>::new_with_preimage(&preimage, &constants);
        let mut h2 = Poseidon::<S1, U2>::new_with_preimage(&preimage, &constants2);

        assert_eq!(h1.hash(), h2.hash())
    }
}
