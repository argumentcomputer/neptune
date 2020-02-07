#![allow(dead_code)]

pub use crate::poseidon::Poseidon;
use crate::round_constants::generate_constants;
pub use error::Error;
use ff::{PrimeField, ScalarEngine};
pub use paired::bls12_381::Fr as Scalar;
use paired::bls12_381::FrRepr;

/// Poseidon circuit
pub mod circuit;
mod error;
mod matrix;
mod mds;
/// Poseidon hash
pub mod poseidon;
mod round_constants;
mod test;

pub(crate) const TEST_SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

pub fn round_numbers(arity: usize) -> (usize, usize) {
    let width = arity + 1;

    let full_rounds = 8;
    let partial_rounds = match width {
        2 | 3 => 55,
        4 | 5 | 6 | 7 => 56,
        8 | 9 | 10 | 11 | 12 => 57,
        _ => panic!("unsupoorted arity"),
    };

    (full_rounds, partial_rounds)
}

/// convert
pub fn scalar_from_u64<E: ScalarEngine>(i: u64) -> E::Fr {
    <E::Fr as PrimeField>::from_repr(<<E::Fr as PrimeField>::Repr as From<u64>>::from(i)).unwrap()
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
