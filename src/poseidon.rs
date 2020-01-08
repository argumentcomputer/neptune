use crate::{
    scalar_from_u64, Error, Scalar, FULL_ROUNDS, MDS_MATRIX, MERKLE_ARITY, PARTIAL_ROUNDS,
    ROUND_CONSTANTS, WIDTH,
};
use ff::Field;

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
///
/// The leaves must implement [`ops::Mul`] against a [`Scalar`], because the MDS matrix and the
/// round constants are set, by default, as scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poseidon {
    constants_offset: usize,
    present_elements: u64,
    pos: usize,
    leaves: [Scalar; WIDTH],
}

impl Default for Poseidon {
    fn default() -> Self {
        Poseidon {
            present_elements: 0u64,
            constants_offset: 0,
            pos: 1,
            leaves: [Scalar::zero(); WIDTH],
        }
    }
}

impl Poseidon {
    /// The poseidon width will be defined by `arity + 1`, because the first element will be a set of bitflags defining which element is present or absent. The absent elements will be represented by `0`, and the present ones by `1`, considering inverse order.
    ///
    /// For example: given we have an arity of `8`, and  if we have two present elements, three absent, and three present, we will have the first element as `0xe3`, or `(11100011)`.
    ///
    /// Every time we push an element, we set the related bitflag with the proper state.
    ///
    /// The returned `usize` represents the leaf position for the insert operation
    pub fn push(&mut self, leaf: Scalar) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        if self.pos > MERKLE_ARITY {
            return Err(Error::FullBuffer);
        }

        self.insert_unchecked(self.pos - 1, leaf);
        self.pos += 1;

        Ok(self.pos - 2)
    }

    /// Insert the provided leaf in the defined position.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub(crate) fn insert_unchecked(&mut self, index: usize, leaf: Scalar) {
        let mut mask = 1u64;
        mask <<= index;
        self.present_elements |= mask;

        // Set current element, and increase the pointer
        self.leaves[index + 1] = leaf;
    }

    /// Removes an item that is indexed by `index`.
    ///
    /// Internally, the buffer is stored in form of `index + 1`, because of the leading bitflags
    /// element.
    ///
    /// # Example
    ///
    /// use dusk_poseidon_merkle::*;
    ///
    /// let mut h = Poseidon::default();
    ///
    /// assert!(h.remove(0).is_err());
    ///
    /// let idx = h.push(Scalar::one()).unwrap();
    /// assert_eq!(0, idx);
    ///
    /// h.remove(0).unwrap();
    ///
    pub fn remove(&mut self, index: usize) -> Result<Scalar, Error> {
        let index = index + 1;
        if index >= self.pos {
            return Err(Error::IndexOutOfBounds);
        }

        Ok(self.remove_unchecked(index))
    }

    /// Removes the first equivalence of the item from the leafs set and returns it.
    pub fn remove_item(&mut self, item: &Scalar) -> Option<Scalar> {
        self.leaves
            .iter()
            .enumerate()
            .fold(None, |mut acc, (i, s)| {
                if acc.is_none() && i > 0 && s == item {
                    acc.replace(i);
                }

                acc
            })
            .map(|idx| self.remove_unchecked(idx))
    }

    /// Set the provided index as absent for the hash calculation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove_unchecked(&mut self, index: usize) -> Scalar {
        let leaf = self.leaves[index];
        self.leaves[index] = scalar_from_u64(0u64);

        let mut mask = 1u64;
        mask <<= index;
        self.present_elements &= !mask;

        leaf
    }

    /// Replace the leaves with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is bigger than the arity.
    pub fn replace(&mut self, buf: &[Option<Scalar>]) {
        self.reset();
        buf.iter().enumerate().for_each(|(i, scalar)| {
            if let Some(s) = scalar {
                self.insert_unchecked(i, *s);
            }
        });
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.present_elements = 0;
        self.constants_offset = 0;
        self.pos = 1;
        self.leaves
            .iter_mut()
            .for_each(|l| *l = scalar_from_u64(0u64));
    }

    /// The absent elements will be considered as zeroes in the permutation.
    ///
    /// The number of rounds is divided into two equal parts for the full rounds, plus the partial rounds.
    ///
    /// The returned element is the second poseidon leaf, for the first is initially the bitflags scheme.
    pub fn hash(&mut self) -> Scalar {
        // The first element is a set of bitflags to differentiate zeroed leaves from absent
        // ones
        //
        // This avoids collisions
        self.leaves[0] = scalar_from_u64(self.present_elements);

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
        // Every element of the merkle tree, plus the bitflag, is incremented by the round constants
        self.add_round_constants();

        // Apply the quintic S-Box to all elements
        self.leaves.iter_mut().for_each(|l| quintic_s_box(l));

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
    pub fn partial_round(&mut self) {
        // Every element of the merkle tree, plus the bitflag, is incremented by the round constants
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
        let mut h = Poseidon::default();
        for _ in 0..MERKLE_ARITY {
            h.push(Scalar::one()).unwrap();
        }
        h.hash();
        h.reset();

        assert_eq!(Poseidon::default(), h);
    }

    #[test]
    fn hash_det() {
        let mut h = Poseidon::default();
        h.push(Scalar::one()).unwrap();

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }
}
