use ff::{Field, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;

pub struct ColumnTreeBuilder<E, ColumnArity, TreeArity>
where
    E: ScalarEngine,
    ColumnArity: Unsigned,
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
    ColumnArity: Unsigned,
    TreeArity: Unsigned,
{
    pub fn new(leaf_count: usize) -> Self {
        Self {
            leaf_count,
            data: Vec::with_capacity(leaf_count),
            fill_index: 0,
            _c: PhantomData::<ColumnArity>,
            _t: PhantomData::<TreeArity>,
        }
    }

    pub fn add_columns(columns: &[E::Fr]) {
        unimplemented!();
    }

    pub fn add_final_columns(columns: &[E::Fr]) -> Vec<E::Fr> {
        unimplemented!();
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
