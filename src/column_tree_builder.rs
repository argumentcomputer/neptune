use crate::error::Error;
use ff::{Field, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;

pub struct ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: ArrayLength<E::Fr>,
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
    ColumnArity: ArrayLength<E::Fr>,
    TreeArity: Unsigned,
{
    pub fn new(leaf_count: usize) -> Self {
        Self {
            leaf_count,
            data: vec![E::Fr::zero(); leaf_count],
            fill_index: 0,
            _c: PhantomData::<ColumnArity>,
            _t: PhantomData::<TreeArity>,
        }
    }

    pub fn add_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<usize, Error> {
        let column_count = columns.len();

        // FIXME: add implementation.

        self.fill_index += column_count;

        Ok(self.leaf_count - self.fill_index)
    }

    pub fn add_final_columns(
        &mut self,
        columns: &[GenericArray<E::Fr, ColumnArity>],
    ) -> Result<Vec<E::Fr>, Error> {
        let columns_remaining = self.add_columns(columns)?;

        if columns_remaining == 0 {
        } else {
        }

        self.reset();

        unimplemented!();
    }

    pub fn reset(&mut self) {
        self.fill_index = 0;
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
