use crate::error::Error;
use crate::poseidon::poseidon;
use ff::{Field, ScalarEngine};
use generic_array::{typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;
use std::ops::Add;
use typenum::bit::B1;
use typenum::marker_traits::Unsigned;
use typenum::uint::{UInt, UTerm};

pub struct ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    TreeArity: Unsigned,
{
    leaf_count: usize,
    data: Vec<E::Fr>,
    /// Index of the first unfilled datum.
    fill_index: usize,
    _c: PhantomData<ColumnArity>,
    _t: PhantomData<TreeArity>,
}

impl<E, ColumnArity, TreeArity> ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<E::Fr>,
    TreeArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<E::Fr>,
{
    pub fn new(leaf_count: usize) -> Self {
        let builder = Self {
            leaf_count,
            data: vec![E::Fr::zero(); leaf_count],
            fill_index: 0,
            _c: PhantomData::<ColumnArity>,
            _t: PhantomData::<TreeArity>,
        };

        // This will panic if leaf_count is not compatible with tree arity.
        // That is the desired behavior so such a programmer error is caught at development time.
        let _ = builder.tree_size();

        builder
    }

    pub fn add_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<usize, Error> {
        let start = self.fill_index;
        let column_count = columns.len();
        let end = start + column_count;

        if end > self.leaf_count {
            return Err(Error::Other("too many columns".to_string()));
        }

        columns
            .iter()
            .zip(self.data[start..end].iter_mut())
            .for_each(
                |(column, place)| *place = poseidon::<E, ColumnArity>(&column), // FIXME: create and use a hasher!
            );

        self.fill_index += column_count;

        Ok(self.leaf_count - self.fill_index)
    }

    pub fn add_final_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<Vec<E::Fr>, Error> {
        let columns_remaining = self.add_columns(columns)?;

        if columns_remaining != 0 {
            // We could make this an error, but as long as data is initialized to zero at each reset,
            // early finalization is equivalent to zero-padding.
        }

        let tree = self.build_tree();
        self.reset();

        tree
    }

    pub fn reset(&mut self) {
        self.fill_index = 0;
        self.data
            .iter_mut()
            .for_each(|place| *place = E::Fr::zero());
    }

    fn build_tree(&self) -> Result<Vec<E::Fr>, Error> {
        let tree_size = self.tree_size();
        let arity = TreeArity::to_usize();

        let mut tree_data = vec![E::Fr::zero(); tree_size];

        tree_data[0..self.leaf_count].copy_from_slice(&self.data);

        let (mut start, mut end) = (0, arity);

        for i in self.leaf_count..tree_size {
            tree_data[i] = poseidon::<E, TreeArity>(&tree_data[start..end]);
            start += arity;
            end += arity;
        }

        Ok(tree_data)
    }

    fn tree_size(&self) -> usize {
        let arity = TreeArity::to_usize();

        let mut tree_size = 0;
        let mut current_row_size = self.leaf_count;

        while current_row_size >= 1 {
            tree_size += current_row_size;
            assert_eq!(
                0,
                current_row_size % arity,
                "Tree leaf count does not have a power of arity as a factor."
            );
            current_row_size /= arity;
        }

        assert_eq!(
            1, current_row_size,
            "Final row of tree was not the root: tree leaf count was not a power of arity."
        );

        tree_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_tree_builder() {
        // panic!("FIXME");
    }
}
