use crate::error::Error;
use crate::gpu;
use crate::poseidon::{Poseidon, PoseidonConstants};

use ff::{Field, ScalarEngine};
use generic_array::{typenum, ArrayLength, GenericArray};
use std::ops::Add;
use typenum::bit::B1;
use typenum::uint::{UInt, UTerm};

pub type GPUColumnTreeBuilder<ColumnArity, TreeArity> =
    gpu::ColumnTreeBuilder2k<ColumnArity, TreeArity>;

pub trait ColumnTreeBuilderTrait<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<E::Fr>,
    TreeArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<E::Fr>,
{
    fn new(leaf_count: usize) -> Self;
    fn add_columns(&mut self, columns: &[GenericArray<E::Fr, ColumnArity>]) -> Result<(), Error>;
    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<Vec<E::Fr>, Error>;
    fn reset(&mut self);
}

pub struct ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    TreeArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
{
    leaf_count: usize,
    data: Vec<E::Fr>,
    /// Index of the first unfilled datum.
    fill_index: usize,
    column_constants: PoseidonConstants<E, ColumnArity>,
    tree_constants: PoseidonConstants<E, TreeArity>,
}

impl<E, ColumnArity, TreeArity> ColumnTreeBuilderTrait<E, ColumnArity, TreeArity>
    for ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<E::Fr>,
    TreeArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<E::Fr>,
{
    fn new(leaf_count: usize) -> Self {
        let builder = Self {
            leaf_count,
            data: vec![E::Fr::zero(); leaf_count],
            fill_index: 0,
            column_constants: PoseidonConstants::<E, ColumnArity>::new(),
            tree_constants: PoseidonConstants::<E, TreeArity>::new(),
        };

        // This will panic if leaf_count is not compatible with tree arity.
        // That is the desired behavior so such a programmer error is caught at development time.
        let _ = builder.tree_size();

        builder
    }

    fn add_columns(&mut self, columns: &[GenericArray<E::Fr, ColumnArity>]) -> Result<(), Error> {
        let start = self.fill_index;
        let column_count = columns.len();
        let end = start + column_count;

        if end > self.leaf_count {
            return Err(Error::Other("too many columns".to_string()));
        }

        columns.iter().enumerate().for_each(|(i, column)| {
            self.data[start + i] =
                Poseidon::new_with_preimage(&column, &self.column_constants).hash();
        });

        self.fill_index += column_count;

        Ok(())
    }

    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<Vec<E::Fr>, Error> {
        self.add_columns(columns)?;

        let tree = self.build_tree();
        self.reset();

        tree
    }

    fn reset(&mut self) {
        self.fill_index = 0;
        self.data
            .iter_mut()
            .for_each(|place| *place = E::Fr::zero());
    }
}

impl<E, ColumnArity, TreeArity> ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<E::Fr>,
    TreeArity: ArrayLength<E::Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<E::Fr>,
{
    fn build_tree(&self) -> Result<Vec<E::Fr>, Error> {
        let tree_size = self.tree_size();
        let arity = TreeArity::to_usize();

        let mut tree_data = vec![E::Fr::zero(); tree_size];

        tree_data[0..self.leaf_count].copy_from_slice(&self.data);

        let (mut start, mut end) = (0, arity);

        for i in self.leaf_count..tree_size {
            tree_data[i] =
                Poseidon::new_with_preimage(&tree_data[start..end], &self.tree_constants).hash();
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
            if current_row_size != 1 {
                assert_eq!(
                    0,
                    current_row_size % arity,
                    "Tree leaf count does not have a power of arity as a factor."
                );
            }
            current_row_size /= arity;
        }

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
