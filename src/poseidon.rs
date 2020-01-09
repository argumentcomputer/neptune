use crate::{
    scalar_from_u64, Error, Scalar, FULL_ROUNDS, MDS_MATRIX, PARTIAL_ROUNDS, ROUND_CONSTANTS, WIDTH,
};
use ff::Field;

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
///
/// The leaves must implement [`ops::Mul`] against a [`Scalar`], because the MDS matrix and the
/// round constants are set, by default, as scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poseidon {
    constants_offset: usize,
    leaves: [Scalar; WIDTH],
    pos: usize,
}

impl Default for Poseidon {
    fn default() -> Self {
        Poseidon {
            constants_offset: 0,
            leaves: [Scalar::zero(); WIDTH],
            pos: 0,
        }
    }
}

impl Poseidon {
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(preimage: [Scalar; WIDTH]) -> Self {
        Poseidon {
            constants_offset: 0,
            leaves: preimage,
            pos: WIDTH,
        }
    }

    /// Replace the leaves with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is bigger than the arity.
    pub fn set_preimage(&mut self, preimage: [Scalar; WIDTH]) {
        self.reset();
        self.leaves = preimage
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.constants_offset = 0;
        self.leaves
            .iter_mut()
            .for_each(|l| *l = scalar_from_u64(0u64));
    }

    /// The returned `usize` represents the leaf position for the insert operation
    pub fn push(&mut self, leaf: Scalar) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        if self.pos > WIDTH {
            return Err(Error::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.leaves[self.pos] = leaf;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    /// The absent elements will be considered as zeroes in the permutation.
    ///
    /// The number of rounds is divided into two equal parts for the full rounds, plus the partial rounds.
    ///
    /// The returned element is the second poseidon leaf, for the first is initially the bitflags scheme.
    pub fn hash(&mut self) -> Scalar {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        for _ in 0..FULL_ROUNDS / 2 {
            self.full_round();
        }

        for _ in 0..PARTIAL_ROUNDS {
            self.partial_round();
        }

        for _ in 0..FULL_ROUNDS / 2 {
            self.full_round();
        }

        // The first bitflags element is discarded, so we can use the first actual leaf as a result
        // of the hash
        self.leaves[1]
    }

    /// The full round function will add the round constants and apply the S-Box to all poseidon leaves, including the bitflags first element.
    ///
    /// After that, the poseidon elements will be set to the result of the product between the poseidon leaves and the constant MDS matrix.
    pub fn full_round(&mut self) {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants();

        // Apply the quintic S-Box to all elements
        self.leaves.iter_mut().for_each(|l| quintic_s_box(l));

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
    pub fn partial_round(&mut self) {
        // Every element of the hash buffer is incremented by the round constants
        self.add_round_constants();

        // Apply the quintic S-Box to the bitflags element
        quintic_s_box(&mut self.leaves[0]);

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// For every leaf, add the round constants with index defined by the constants offset, and increment the
    /// offset
    fn add_round_constants(&mut self) {
        let mut constants_offset = self.constants_offset;

        self.leaves.iter_mut().for_each(|l| {
            l.add_assign(&ROUND_CONSTANTS[constants_offset]);
            constants_offset += 1;
        });

        self.constants_offset = constants_offset;
    }

    /// Set the provided leaves with the result of the product between the leaves and the constant
    /// MDS matrix
    fn product_mds(&mut self) {
        let mut result = [scalar_from_u64(0u64); WIDTH];

        for j in 0..WIDTH {
            for k in 0..WIDTH {
                let mut tmp = MDS_MATRIX[j][k];
                tmp.mul_assign(&self.leaves[k]);
                result[j].add_assign(&tmp);
            }
        }

        self.leaves.copy_from_slice(&result);
    }
}

/// Apply the quintic S-Box (s^5) to a given item
fn quintic_s_box(l: &mut Scalar) {
    let c = *l;
    for _ in 0..4 {
        l.mul_assign(&c);
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ff::Field;

    #[test]
    fn reset() {
        let preimage: [Scalar; WIDTH] = [Scalar::one(); WIDTH];
        let mut h = Poseidon::new(preimage);
        h.hash();
        h.reset();

        assert_eq!(Poseidon::default(), h);
    }

    #[test]
    fn hash_det() {
        let mut preimage: [Scalar; WIDTH] = [Scalar::zero(); WIDTH];
        preimage[0] = Scalar::one();

        let mut h = Poseidon::new(preimage);

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }
}
