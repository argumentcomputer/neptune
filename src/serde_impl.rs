use std::fmt;
use serde::{
  ser::{Serializer, SerializeStruct},
  Serialize,
  Deserialize,
  de::{self, Deserializer, Visitor, SeqAccess, MapAccess},
};
use generic_array::typenum;
use typenum::Unsigned;
use ff::PrimeField;
use std::marker::PhantomData;
use pasta_curves::pallas::Scalar as S1;

use crate::Arity;
use crate::poseidon::PoseidonConstants;

impl<F, A> Serialize for PoseidonConstants<F, A>
  where
  F: PrimeField + Serialize,
  A: Arity<F>,
{
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut state = serializer.serialize_struct("PoseidonConstants", 11)?;
    state.serialize_field("mdsmatrices", &self.mds_matrices)?;
    state.serialize_field("roundconstants", &self.round_constants)?;
    state.serialize_field("compressedroundconstants", &self.compressed_round_constants)?;
    state.serialize_field("presparsematrix", &self.pre_sparse_matrix)?;
    state.serialize_field("sparsematrixes", &self.sparse_matrixes)?;
    state.serialize_field("strength", &self.strength)?;
    state.serialize_field("domaintag", &self.domain_tag)?;
    state.serialize_field("fullrounds", &self.full_rounds)?;
    state.serialize_field("halffullrounds", &self.half_full_rounds)?;
    state.serialize_field("partialrounds", &self.partial_rounds)?;
    state.serialize_field("hashtype", &self.hash_type)?;
    state.end()
  }
}

impl<'de, F, A> Deserialize<'de> for PoseidonConstants<F, A>
where
  F: PrimeField + Deserialize<'de>,
  A: Arity<F>,
{
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where D: Deserializer<'de>,
  {
    #[derive(Deserialize)]
    #[serde(field_identifier, rename_all = "lowercase")]
    enum Field {
      MdsMatrices,
      RoundConstants,
      CompressedRoundConstants,
      PreSparseMatrix,
      SparseMatrixes,
      Strength,
      DomainTag,
      FullRounds,
      HalfFullRounds,
      PartialRounds,
      HashType,
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
        let mds_matrices = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let round_constants = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let compressed_round_constants = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let pre_sparse_matrix = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let sparse_matrixes = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let strength = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let domain_tag = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        let full_rounds = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(6, &self))?;
        let half_full_rounds = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(7, &self))?;
        let partial_rounds = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(8, &self))?;
        let hash_type = seq.next_element()?
          .ok_or_else(|| de::Error::invalid_length(9, &self))?;
        Ok(PoseidonConstants {
	  mds_matrices,
	  round_constants,
	  compressed_round_constants,
	  pre_sparse_matrix,
	  sparse_matrixes,
	  strength,
	  domain_tag,
	  full_rounds,
	  half_full_rounds,
	  partial_rounds,
	  hash_type,
	  _a: PhantomData
	})
      }

      fn visit_map<V>(self, mut map: V) -> Result<PoseidonConstants<F, A>, V::Error>
      where
        V: MapAccess<'de>,
      {
	let mut mds_matrices = None;
	let mut round_constants = None;
	let mut compressed_round_constants = None;
	let mut pre_sparse_matrix = None;
	let mut sparse_matrixes = None;
	let mut strength = None;
	let mut domain_tag = None;
	let mut full_rounds = None;
	let mut half_full_rounds = None;
	let mut partial_rounds = None;
	let mut hash_type = None;
        while let Some(key) = map.next_key()? {
          match key {
            Field::MdsMatrices => {
              if mds_matrices.is_some() {
                return Err(de::Error::duplicate_field("mds_matrices"));
              }
              mds_matrices = Some(map.next_value()?);
            }
            Field::RoundConstants => {
              if round_constants.is_some() {
                return Err(de::Error::duplicate_field("round_constants"));
              }
              round_constants = Some(map.next_value()?);
            }
            Field::CompressedRoundConstants => {
              if compressed_round_constants.is_some() {
                return Err(de::Error::duplicate_field("compressed_round_constants"));
              }
              compressed_round_constants = Some(map.next_value()?);
            }
            Field::PreSparseMatrix => {
              if pre_sparse_matrix.is_some() {
                return Err(de::Error::duplicate_field("pre_sparse_matrix"));
              }
              pre_sparse_matrix = Some(map.next_value()?);
            }
            Field::SparseMatrixes => {
              if sparse_matrixes.is_some() {
                return Err(de::Error::duplicate_field("sparse_matrixes"));
              }
              sparse_matrixes = Some(map.next_value()?);
            }
            Field::Strength => {
              if strength.is_some() {
                return Err(de::Error::duplicate_field("strength"));
              }
              strength = Some(map.next_value()?);
            }
            Field::DomainTag => {
              if domain_tag.is_some() {
                return Err(de::Error::duplicate_field("domain_tag"));
              }
              domain_tag = Some(map.next_value()?);
            }
            Field::FullRounds => {
              if full_rounds.is_some() {
                return Err(de::Error::duplicate_field("full_rounds"));
              }
              full_rounds = Some(map.next_value()?);
            }
            Field::HalfFullRounds => {
              if half_full_rounds.is_some() {
                return Err(de::Error::duplicate_field("half_full_rounds"));
              }
              half_full_rounds = Some(map.next_value()?);
            }
            Field::PartialRounds => {
              if partial_rounds.is_some() {
                return Err(de::Error::duplicate_field("partial_rounds"));
              }
              partial_rounds = Some(map.next_value()?);
            }
            Field::HashType => {
              if hash_type.is_some() {
                return Err(de::Error::duplicate_field("hash_type"));
              }
              hash_type = Some(map.next_value()?);
            }
          }
        }
        let mds_matrices = mds_matrices.ok_or_else(|| de::Error::missing_field("mds_matrices"))?;
        let round_constants = round_constants.ok_or_else(|| de::Error::missing_field("round_constants"))?;
        let compressed_round_constants = compressed_round_constants.ok_or_else(|| de::Error::missing_field("compressed_round_constants"))?;
        let pre_sparse_matrix = pre_sparse_matrix.ok_or_else(|| de::Error::missing_field("pre_sparse_matrix"))?;
        let sparse_matrixes = sparse_matrixes.ok_or_else(|| de::Error::missing_field("sparse_matrixes"))?;
        let strength = strength.ok_or_else(|| de::Error::missing_field("strength"))?;
        let domain_tag = domain_tag.ok_or_else(|| de::Error::missing_field("domain_tag"))?;
        let full_rounds = full_rounds.ok_or_else(|| de::Error::missing_field("full_rounds"))?;
        let half_full_rounds = half_full_rounds.ok_or_else(|| de::Error::missing_field("half_full_rounds"))?;
        let partial_rounds = partial_rounds.ok_or_else(|| de::Error::missing_field("partial_rounds"))?;
        let hash_type = hash_type.ok_or_else(|| de::Error::missing_field("hash_type"))?;
        Ok(PoseidonConstants {
	  mds_matrices,
	  round_constants,
	  compressed_round_constants,
	  pre_sparse_matrix,
	  sparse_matrixes,
	  strength,
	  domain_tag,
	  full_rounds,
	  half_full_rounds,
	  partial_rounds,
	  hash_type,
	  _a: PhantomData
	})
      }
    }

      const FIELDS: &'static [&'static str] = &["
	  mds_matrices,
	  round_constants,
	  compressed_round_constants,
	  pre_sparse_matrix,
	  sparse_matrixes,
	  strength,
	  domain_tag,
	  full_rounds,
	  half_full_rounds,
	  partial_rounds,
	  hash_type,
"];
    deserializer.deserialize_struct("PoseidonConstants", FIELDS, PoseidonConstantsVisitor { _f: PhantomData, _a: PhantomData })

  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use typenum::U1;

  #[test]
  fn serde_poseidon_constants() {
    let mystruct = PoseidonConstants::<S1, U1>::new();

    assert_eq!(mystruct, serde_json::from_slice(&serde_json::to_vec(&mystruct).unwrap()).unwrap());
  } 
}
