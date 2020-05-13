#![allow(dead_code)]
#[macro_use]
extern crate lazy_static;

pub use crate::poseidon::{Arity, Poseidon};
use crate::round_constants::generate_constants;
pub use error::Error;
use ff::{Field, PrimeField, ScalarEngine};
use generic_array::GenericArray;
pub use paired::bls12_381::Fr as Scalar;
use paired::bls12_381::FrRepr;

/// Poseidon circuit
pub mod circuit;
pub mod error;
mod matrix;
mod mds;

/// Poseidon hash
pub mod poseidon;
mod preprocessing;
mod round_constants;
mod test;

/// Column Tree Builder
#[cfg(feature = "gpu")]
pub mod column_tree_builder;
#[cfg(feature = "gpu")]
mod gpu;

/// Batch Hasher
#[cfg(feature = "gpu")]
pub mod batch_hasher;

pub mod bench_cs;

pub(crate) const TEST_SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

pub trait BatchHasher<A>
where
    A: Arity<Scalar>,
{
    // type State;

    fn hash(&mut self, preimages: &[GenericArray<Scalar, A>]) -> Result<Vec<Scalar>, Error>;

    fn hash_into_slice(
        &mut self,
        target_slice: &mut [Scalar],
        preimages: &[GenericArray<Scalar, A>],
    ) -> Result<(), Error> {
        assert_eq!(target_slice.len(), preimages.len());
        // FIXME: Account for max batch size.

        Ok(target_slice.copy_from_slice(self.hash(preimages)?.as_slice()))
    }

    /// How many leaves must be provided as input to the supported tree builder, intended for building a final small tree
    /// rather than require repeated batches of dwindling size at the top of a tree whose rows are built in batches.
    fn tree_leaf_count(&self) -> Option<usize> {
        None
    }

    /// Build a merkle tree from provided leaves in a single call to the GPU.
    /// Leaf count must match the value returned by `self.tree_leaf_count()`.
    fn build_tree(&mut self, _leaves: &[Scalar]) -> Result<Vec<Scalar>, Error> {
        unimplemented!();
    }

    /// `max_batch_size` is advisory. Implenters of `BatchHasher` should ensure that up to the returned max hashes can
    /// be safely performed on the target GPU (currently 2080Ti). The max returned should represent a safe batch size
    /// optimized for performance.
    /// `BatchHasher` users are responsible for not attempting to hash batches larger than the advised maximum.
    fn max_batch_size(&self) -> usize {
        700000
    }
}

pub fn round_numbers(arity: usize) -> (usize, usize) {
    let width = arity + 1;

    let full_rounds = 8;
    let partial_rounds = match width {
        2 | 3 => 55,
        4 | 5 | 6 | 7 => 56,
        8 | 9 | 10 | 11 | 12 => 57,
        17 | 25 => 59,
        37 => 60,
        65 => 61,
        //24 => 42, // Just for a comparative benchmark — don't use this.
        _ => panic!(format!("unsupported arity {}", arity)),
    };

    (full_rounds, partial_rounds)
}

/// convert
pub fn scalar_from_u64<Fr: PrimeField>(i: u64) -> Fr {
    Fr::from_repr(<Fr::Repr as From<u64>>::from(i)).unwrap()
}

/// create field element from four u64
pub fn scalar_from_u64s(parts: [u64; 4]) -> Scalar {
    Scalar::from_repr(FrRepr(parts)).unwrap()
}

const SBOX: u8 = 1; // x^5
const FIELD: u8 = 1; // Gf(p)
const FIELD_SIZE: usize = 255; // n  Maybe Get this from Scalar.

fn round_constants<E: ScalarEngine>(arity: usize) -> Vec<E::Fr> {
    let t = arity + 1;
    let n = t * FIELD_SIZE;

    let (full_rounds, partial_rounds) = round_numbers(arity);

    let r_f = full_rounds as u16;
    let r_p = partial_rounds as u16;
    generate_constants::<E>(FIELD, SBOX, n as u16, t as u16, r_f, r_p)
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
    let c = *l;
    let mut tmp = l.clone();
    tmp.mul_assign(&c);
    tmp.mul_assign(&tmp.clone());
    l.mul_assign(&tmp);
    if let Some(x) = post_add {
        l.add_assign(x);
    }
}
