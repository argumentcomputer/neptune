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
            bincode::deserialize(&bincode::serialize(&constants).unwrap()).unwrap()
        );
    }

    #[test]
    fn serde_hash_blstrs() {
        let constants = PoseidonConstants::<Fr, U2>::new();
        let constants2 = bincode::deserialize(&bincode::serialize(&constants).unwrap()).unwrap();
        let test_arity = 2;
        let preimage = vec![<Fr as Field>::ONE; test_arity];
        let mut h1 = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);
        let mut h2 = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants2);

        assert_eq!(h1.hash(), h2.hash())
    }

    #[test]
    fn serde_hash_pallas() {
        let constants = PoseidonConstants::<S1, U2>::new();
        let constants2 = bincode::deserialize(&bincode::serialize(&constants).unwrap()).unwrap();
        let test_arity = 2;
        let preimage = vec![<S1 as Field>::ONE; test_arity];
        let mut h1 = Poseidon::<S1, U2>::new_with_preimage(&preimage, &constants);
        let mut h2 = Poseidon::<S1, U2>::new_with_preimage(&preimage, &constants2);

        assert_eq!(h1.hash(), h2.hash())
    }
}
