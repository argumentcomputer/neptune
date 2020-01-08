#![feature(external_doc)]
#![deny(missing_docs)]
#![doc(include = "../README.md")]

use std::ops;

use lazy_static::*;
#[cfg(feature = "big-merkle")]
use serde::{Deserialize, Serialize};

pub use crate::poseidon::Poseidon;
pub use paired::bls12_381::Fr as Scalar;
// pub use curve25519_dalek::scalar::Scalar;
pub use error::Error;
// pub use merkle::MerkleTree;
// pub use proof::Proof;

#[cfg(feature = "big-merkle")]
pub use big_merkle::{BigMerkleTree, BigProof, MerkleCoord, MerkleRange};

mod error;
// mod merkle;
mod poseidon;
// mod proof;

#[cfg(feature = "big-merkle")]
mod big_merkle;

include!("constants.rs");

lazy_static! {
    static ref ROUND_CONSTANTS: [Scalar; 960] = {
        let bytes = include_bytes!("../assets/ark.bin");
        unsafe { std::ptr::read(bytes.as_ptr() as *const _) }
    };
    static ref MDS_MATRIX: [[Scalar; WIDTH]; WIDTH] = {
        let bytes = include_bytes!("../assets/mds.bin");
        // assert_eq!(bytes.len(), (WIDTH * WIDTH) << 5);
        unsafe { std::ptr::read(bytes.as_ptr() as *const _) }
    };
}

/// The items for the [`MerkleTree`] and [`Poseidon`] must implement this trait
///
/// The implementation must be serializable for the [`BigMerkleTree`] storage
#[cfg(feature = "big-merkle")]
pub trait PoseidonLeaf:
    Copy + From<Scalar> + From<[u8; 32]> + PartialEq + Serialize + for<'d> Deserialize<'d> + Send + Sync
{
}

/// The items for the [`MerkleTree`] and [`Poseidon`] must implement this trait
#[cfg(not(feature = "big-merkle"))]
pub trait PoseidonLeaf:
    Copy + From<u64> + From<Scalar> + PartialEq + ops::MulAssign + ops::AddAssign
{
}

/// convert
pub fn scalar_from_u64(i: u64) -> Scalar {
    use ff::PrimeField;
    Scalar::from_repr(paired::bls12_381::FrRepr::from(i)).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn constants_consistency() {
        // Grant we have enough constants for the sbox rounds
        assert!(WIDTH * (FULL_ROUNDS + PARTIAL_ROUNDS) <= ROUND_CONSTANTS.len());

        // Sanity check for the arity
        assert!(MERKLE_ARITY > 1);

        // // Sanity check for the height
        // assert!(MERKLE_HEIGHT > 1);

        // Enforce a relation between the provided MDS matrix and the arity of the merkle tree
        assert_eq!(WIDTH, MERKLE_ARITY + 1);

        // Enforce at least one level for the merkle tree
        assert!(MERKLE_WIDTH > MERKLE_ARITY);

        // // Grant the defined arity is consistent with the defined width
        // assert_eq!(
        //     MERKLE_ARITY.pow(std::cmp::max(2, MERKLE_HEIGHT as u32)),
        //     MERKLE_WIDTH
        // );
    }
}
