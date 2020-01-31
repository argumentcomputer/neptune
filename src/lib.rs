#![feature(external_doc)]
#![feature(associated_type_bounds)]
#![allow(dead_code)]
#![doc(include = "../README.md")]

pub use crate::poseidon::Poseidon;
use crate::round_constants::generate_constants;
pub use error::Error;
use ff::ScalarEngine as Engine;
use ff::{Field, PrimeField, ScalarEngine};
pub use paired::bls12_381::Fr as Scalar;
use paired::bls12_381::FrRepr;

/// Poseidon circuit
pub mod circuit;
mod error;
/// Poseidon hash
pub mod poseidon;
mod round_constants;
mod test;

include!("constants.rs");

pub(crate) const TEST_SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

/// Maximum width for which we will pre-generate MDS matrices.
pub const MAX_SUPPORTED_WIDTH: usize = 9;

/// convert
pub fn scalar_from_u64<E: ScalarEngine>(i: u64) -> E::Fr {
    <E::Fr as PrimeField>::from_repr(<<E::Fr as PrimeField>::Repr as From<u64>>::from(i)).unwrap()
}

/// create field element from four u64
pub fn scalar_from_u64s(parts: [u64; 4]) -> Scalar {
    Scalar::from_repr(FrRepr(parts)).unwrap()
}

fn generate_mds<E: Engine>(t: usize) -> Vec<Vec<E::Fr>> {
    let mut matrix: Vec<Vec<E::Fr>> = Vec::with_capacity(t);
    let mut xs: Vec<E::Fr> = Vec::with_capacity(t);
    let mut ys: Vec<E::Fr> = Vec::with_capacity(t);

    // Generate x and y values deterministically for the cauchy matrix
    // where x[i] != y[i] to allow the values to be inverted
    // and there are no duplicates in the x vector or y vector, so that the determinant is always non-zero
    // [a b]
    // [c d]
    // det(M) = (ad - bc) ; if a == b and c == d => det(M) =0
    // For an MDS matrix, every possible mxm submatrix, must have det(M) != 0
    for i in 0..t {
        let x = scalar_from_u64::<E>((i) as u64);
        let y = scalar_from_u64::<E>((i + t) as u64);
        xs.push(x);
        ys.push(y);
    }

    for i in 0..t {
        let mut row: Vec<E::Fr> = Vec::with_capacity(t);
        for j in 0..t {
            // Generate the entry at (i,j)
            let mut tmp = xs[i];
            tmp.add_assign(&ys[j]);
            let entry = tmp.inverse().unwrap();
            row.insert(j, entry);
        }
        matrix.push(row);
    }

    matrix
}

/// From the paper ():
/// The round constants are generated using the Grain LFSR [23] in a self-shrinking
/// mode:
/// 1. Initialize the state with 80 bits b0, b1, . . . , b79, where
/// (a) b0, b1 describe the field,
/// (b) bi for 2 ≤ i ≤ 5 describe the S-Box,
/// (c) bi for 6 ≤ i ≤ 17 are the binary representation of n,
/// (d) bi for 18 ≤ i ≤ 29 are the binary representation of t,
/// (e) bi for 30 ≤ i ≤ 39 are the binary representation of RF ,
/// (f) bi for 40 ≤ i ≤ 49 are the binary representation of RP , and
/// (g) bi for 50 ≤ i ≤ 79 are set to 1.
/// 2. Update the bits using bi+80 = bi+62 ⊕ bi+51 ⊕ bi+38 ⊕ bi+23 ⊕ bi+13 ⊕ bi
/// .
/// 3. Discard the first 160 bits.
/// 4. Evaluate bits in pairs: If the first bit is a 1, output the second bit. If it is a
/// 0, discard the second bit.
/// Using this method, the generation of round constants depends on the specific
/// instance, and thus different round constants are used even if some of the chosen
/// parameters (e.g., n and t) are the same.
/// If a randomly sampled integer is not in Fp, we discard this value and take the
/// next one. Note that cryptographically strong randomness is not needed for the
/// round constants, and other methods can also be used.

const SBOX: u8 = 1; // x^5
const FIELD: u8 = 1; // Gf(p)
const FIELD_SIZE: usize = 255; // n  Maybe Get this from Scalar.
const R_F_FIXED: u16 = FULL_ROUNDS as u16;
const R_P_FIXED: u16 = PARTIAL_ROUNDS as u16;

fn round_constants<E: ScalarEngine>(arity: usize) -> Vec<E::Fr> {
    let t = arity + 1;
    let n = t * FIELD_SIZE;

    // TODO: These need to be derived from t
    let r_f = R_F_FIXED;
    let r_p = R_P_FIXED;
    generate_constants::<E>(FIELD, SBOX, n as u16, t as u16, r_f, r_p)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use paired::bls12_381::Bls12;

    #[test]
    fn constants_consistency() {
        // Grant we have enough constants for the sbox rounds
        assert!(WIDTH * (FULL_ROUNDS + PARTIAL_ROUNDS) <= round_constants::<Bls12>(ARITY).len());
    }
}
