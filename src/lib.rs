#![feature(external_doc)]
#![deny(missing_docs)]
#![doc(include = "../README.md")]

use lazy_static::*;

pub use crate::poseidon::Poseidon;
pub use error::Error;
use ff::{Field, PrimeField};
pub use paired::bls12_381::Fr as Scalar;
use paired::bls12_381::FrRepr;

mod circuit;
mod error;
mod poseidon;
mod test;

include!("constants.rs");

pub(crate) const TEST_SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

/// Maximum width for which we will pre-generate MDS matrices.
pub const MAX_SUPPORTED_WIDTH: usize = 9;

lazy_static! {
    static ref ROUND_CONSTANTS: [Scalar; 960] = {
        // FIXME: These ark constants will not be correct, since we have changed curves. We need to figure out how to generate them ourselves.
        // UPDATE: Note that the paper says, ' Note that cryptographically strong randomness is not needed for the round constants, and other methods can also be used.'
        // So this may not matter for correctness, though we should still formally define these rather than use the initial magic binary still present now.
        let bytes = include_bytes!("../assets/ark.bin");
        unsafe { std::ptr::read(bytes.as_ptr() as *const _) }
    };
    static ref MDS_MATRIX: [[Scalar; WIDTH]; WIDTH] = {
        let mut matrix: [[Scalar; WIDTH]; WIDTH] = [[Scalar::one(); WIDTH]; WIDTH];
        for (i, row) in generate_mds(WIDTH).iter_mut().enumerate() {
            matrix[i].copy_from_slice(&row[..]);
        }
        matrix
    };
}

/// convert
pub fn scalar_from_u64(i: u64) -> Scalar {
    Scalar::from_repr(paired::bls12_381::FrRepr::from(i)).unwrap()
}

/// create field element from four u64
pub fn scalar_from_u64s(parts: [u64; 4]) -> Scalar {
    Scalar::from_repr(FrRepr(parts)).unwrap()
}

fn generate_mds(t: usize) -> Vec<Vec<Scalar>> {
    let mut matrix: Vec<Vec<Scalar>> = Vec::with_capacity(t);
    let mut xs: Vec<Scalar> = Vec::with_capacity(t);
    let mut ys: Vec<Scalar> = Vec::with_capacity(t);

    // Generate x and y values deterministically for the cauchy matrix
    // where x[i] != y[i] to allow the values to be inverted
    // and there are no duplicates in the x vector or y vector, so that the determinant is always non-zero
    // [a b]
    // [c d]
    // det(M) = (ad - bc) ; if a == b and c == d => det(M) =0
    // For an MDS matrix, every possible mxm submatrix, must have det(M) != 0
    for i in 0..t {
        let x = scalar_from_u64((i) as u64);
        let y = scalar_from_u64((i + t) as u64);
        xs.push(x);
        ys.push(y);
    }

    for i in 0..t {
        let mut row: Vec<Scalar> = Vec::with_capacity(t);
        for j in 0..t {
            // Generate the entry at (i,j)
            let mut entry = xs[i];
            entry.add_assign(&ys[i]);
            entry = entry.inverse().unwrap();
            row.insert(j, entry);
        }
        matrix.push(row);
    }

    matrix
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn constants_consistency() {
        // Grant we have enough constants for the sbox rounds
        assert!(WIDTH * (FULL_ROUNDS + PARTIAL_ROUNDS) <= ROUND_CONSTANTS.len());
    }
}
