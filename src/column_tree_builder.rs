use crate::batch_hasher::{Batcher, BatcherType};
use crate::error::Error;
use crate::poseidon::{Poseidon, PoseidonConstants};
use crate::BatchHasher;
use ff::Field;
use generic_array::{typenum, ArrayLength, GenericArray};
use paired::bls12_381::{Bls12, Fr};
use std::ops::Add;
use typenum::bit::B1;
use typenum::uint::{UInt, UTerm};

pub trait ColumnTreeBuilderTrait<ColumnArity, TreeArity>
where
    ColumnArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<Fr>,
    TreeArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<Fr>,
{
    fn add_columns(&mut self, columns: &[GenericArray<Fr, ColumnArity>]) -> Result<(), Error>;
    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<Fr, ColumnArity>],
    ) -> Result<Vec<Fr>, Error>;
    fn reset(&mut self);
}

pub struct ColumnTreeBuilder<'a, ColumnArity, TreeArity>
where
    ColumnArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<Fr>,
    TreeArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<Fr>,
{
    pub leaf_count: usize,
    data: Vec<Fr>,
    /// Index of the first unfilled datum.
    fill_index: usize,
    column_constants: PoseidonConstants<Bls12, ColumnArity>,
    tree_constants: PoseidonConstants<Bls12, TreeArity>,
    pub column_batcher: Option<Batcher<'a, ColumnArity>>,
    pub tree_batcher: Option<Batcher<'a, TreeArity>>,
}

impl<ColumnArity, TreeArity> ColumnTreeBuilderTrait<ColumnArity, TreeArity>
    for ColumnTreeBuilder<'_, ColumnArity, TreeArity>
where
    ColumnArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<Fr>,
    TreeArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<Fr>,
{
    fn add_columns(&mut self, columns: &[GenericArray<Fr, ColumnArity>]) -> Result<(), Error> {
        let start = self.fill_index;
        let column_count = columns.len();
        let end = start + column_count;

        if end > self.leaf_count {
            return Err(Error::Other("too many columns".to_string()));
        }

        match &mut self.column_batcher {
            Some(batcher) => {
                batcher.hash_into_slice(&mut self.data[start..start + column_count], columns)?;
            }
            None => columns.iter().enumerate().for_each(|(i, column)| {
                self.data[start + i] =
                    Poseidon::new_with_preimage(&column, &self.column_constants).hash();
            }),
        };

        self.fill_index += column_count;

        Ok(())
    }

    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<Fr, ColumnArity>],
    ) -> Result<Vec<Fr>, Error> {
        self.add_columns(columns)?;

        let tree = self.build_tree();
        self.reset();

        tree
    }

    fn reset(&mut self) {
        self.fill_index = 0;
        self.data.iter_mut().for_each(|place| *place = Fr::zero());
    }
}
fn as_generic_arrays<'a, Arity: ArrayLength<Fr>>(vec: &'a [Fr]) -> &'a [GenericArray<Fr, Arity>] {
    assert_eq!(0, vec.len() % Arity::to_usize());

    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const () as *const GenericArray<Fr, Arity>,
            vec.len() / Arity::to_usize(),
        )
    }
}

impl<ColumnArity, TreeArity> ColumnTreeBuilder<'_, ColumnArity, TreeArity>
where
    ColumnArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <ColumnArity as Add<B1>>::Output: ArrayLength<Fr>,
    TreeArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
    <TreeArity as Add<B1>>::Output: ArrayLength<Fr>,
{
    pub fn new(
        t: Option<BatcherType>,
        leaf_count: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
    ) -> Result<Self, Error> {
        let builder = Self {
            leaf_count,
            data: vec![Fr::zero(); leaf_count],
            fill_index: 0,
            column_constants: PoseidonConstants::<Bls12, ColumnArity>::new(),
            tree_constants: PoseidonConstants::<Bls12, TreeArity>::new(),
            column_batcher: if let Some(t) = &t {
                Some(Batcher::<ColumnArity>::new(t, max_column_batch_size)?)
            } else {
                None
            },
            tree_batcher: if let Some(t) = &t {
                Some(Batcher::<TreeArity>::new(t, max_tree_batch_size)?)
            } else {
                None
            },
        };

        // This will panic if leaf_count is not compatible with tree arity.
        // That is the desired behavior so such a programmer error is caught at development time.
        let _ = builder.tree_size();

        Ok(builder)
    }

    pub fn build_tree(&mut self) -> Result<Vec<Fr>, Error> {
        let tree_size = self.tree_size();
        let arity = TreeArity::to_usize();

        let mut tree_data = vec![Fr::zero(); tree_size];

        tree_data[0..self.leaf_count].copy_from_slice(&self.data);

        let (mut start, mut end) = (0, arity);

        match &mut self.tree_batcher {
            Some(batcher) => {
                let max_batch_size = batcher.max_batch_size();

                let (mut row_start, mut row_end) = (0, self.leaf_count);
                while row_end < tree_size {
                    let row_size = row_end - row_start;
                    assert_eq!(0, row_size % arity);
                    let new_row_size = row_size / arity;
                    let (new_row_start, new_row_end) = (row_end, row_end + new_row_size);

                    if let Some(leaf_count) = batcher.tree_leaf_count() {
                        if row_size == leaf_count {
                            let remaining_tree =
                                batcher.build_tree(&tree_data[row_start..row_end])?;
                            tree_data[new_row_start..tree_size].copy_from_slice(&remaining_tree);
                            return Ok(tree_data);
                        };
                    }

                    let mut total_hashed = 0;
                    let mut batch_start = row_start;
                    while total_hashed < new_row_size {
                        let batch_end = usize::min(batch_start + (max_batch_size * arity), row_end);
                        let batch_size = (batch_end - batch_start) / arity;
                        let preimages =
                            as_generic_arrays::<TreeArity>(&tree_data[batch_start..batch_end]);
                        let hashed = batcher.hash(&preimages)?;

                        // Poor-man's copy_from_slice avoids a mutable second borrow of tree_data.
                        for i in 0..hashed.len() {
                            tree_data[new_row_start + total_hashed + i] = hashed[i]
                        }
                        total_hashed += batch_size;
                        batch_start = batch_end;
                    }

                    row_start = new_row_start;
                    row_end = new_row_end;
                }
            }
            None => {
                for i in self.leaf_count..tree_size {
                    tree_data[i] =
                        Poseidon::new_with_preimage(&tree_data[start..end], &self.tree_constants)
                            .hash();
                    start += arity;
                    end += arity;
                }
            }
        }

        Ok(tree_data)
    }

    pub fn tree_size(&self) -> usize {
        let arity = TreeArity::to_usize();

        let mut tree_size = 0;
        let mut current_row_size = self.leaf_count;

        while current_row_size >= 1 {
            tree_size += current_row_size;
            if current_row_size != 1 {
                assert_eq!(
                    0,
                    current_row_size % arity,
                    "Tree leaf count ({}) does not have a power of tree arity ({}) as a factor.",
                    self.leaf_count,
                    arity
                )
            }
            current_row_size /= arity;
        }

        tree_size
    }

    pub fn tree_height(&self) -> usize {
        let arity = TreeArity::to_usize();

        let mut tree_height = 0;
        let mut current_row_size = self.leaf_count;

        // Could also just calculate log base arity directly.
        while current_row_size >= 1 {
            if current_row_size != 1 {
                tree_height += 1;
                assert_eq!(
                    0,
                    current_row_size % arity,
                    "Tree leaf count does not have a power of arity as a factor."
                );
            }
            current_row_size /= arity;
        }
        tree_height
    }

    // Compute root of tree composed of all identical columns. For use in checking correctness of GPU column tree-building
    // without the cost of generating a full column tree.
    pub fn compute_uniform_tree_root(
        &mut self,
        column: GenericArray<Fr, ColumnArity>,
    ) -> Result<Fr, Error> {
        let arity = TreeArity::to_usize();
        // All the leaves will be the same.
        let mut element = Poseidon::new_with_preimage(&column, &self.column_constants).hash();

        for _ in 0..self.tree_height() {
            let preimage = vec![element; arity];
            // Each row is the hash of the identical elements in the previous row.
            element = Poseidon::new_with_preimage(&preimage, &self.tree_constants).hash();
        }

        // The last element computed is the root.
        Ok(element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::{poseidon, Poseidon, SimplePoseidonBatchHasher};
    use crate::BatchHasher;
    use ff::{Field, ScalarEngine};
    use generic_array::sequence::GenericSequence;
    use generic_array::typenum::{U11, U8};
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_column_tree_builder() {
        // 16KiB tree has 512 leaves.
        test_column_tree_builder_aux(None, 512, 32, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::CPU), 512, 32, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::GPU), 512, 32, 512, 512);

        // 128KiB tree has 4096 leaves.
        test_column_tree_builder_aux(None, 512, 19, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::CPU), 512, 32, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::GPU), 512, 32, 512, 512);

        // 512MiB
        // test_column_tree_builder_aux(Some(BatcherType::CPU), 16777216, 32);
        // test_column_tree_builder_aux(Some(BatcherType::CPU), 134217728, 100);
    }

    #[test]
    #[ignore] // FIXME: add a feature flag. Very expensive test without actual GPU.
    fn test_column_tree_builder_512m() {
        // 512MiB
        test_column_tree_builder_aux(Some(BatcherType::GPU), 16777216, 32, 400000, 700000);
    }

    #[test]
    #[ignore] // FIXME: add a feature flag. Very expensive test without actual GPU.
    fn test_column_tree_builder_4g() {
        //4GiB
        test_column_tree_builder_aux(Some(BatcherType::GPU), 134217728, 100, 400000, 700000);
    }

    fn test_column_tree_builder_aux(
        batcher_type: Option<BatcherType>,
        leaves: usize,
        num_batches: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
    ) -> Fr {
        let batch_size = leaves / num_batches;

        let mut builder = ColumnTreeBuilder::<U11, U8>::new(
            batcher_type,
            leaves,
            max_column_batch_size,
            max_tree_batch_size,
        )
        .unwrap();

        // Simplify computing the expected root.
        let constant_element = Fr::zero();
        let constant_column = GenericArray::<Fr, U11>::generate(|i| constant_element);

        let max_batch_size = if let Some(batcher) = &builder.column_batcher {
            batcher.max_batch_size()
        } else {
            leaves
        };

        let effective_batch_size = usize::min(batch_size, max_batch_size);

        let mut total_columns = 0;
        while total_columns + effective_batch_size < leaves {
            let columns: Vec<GenericArray<Fr, U11>> =
                (0..effective_batch_size).map(|_| constant_column).collect();

            let res = builder.add_columns(columns.as_slice()).unwrap();
            total_columns += columns.len();
        }

        let final_columns: Vec<_> = (0..leaves - total_columns)
            .map(|_| GenericArray::<Fr, U11>::generate(|i| constant_element))
            .collect();

        let res = builder.add_final_columns(final_columns.as_slice()).unwrap();
        total_columns += final_columns.len();

        let computed_root = res[res.len() - 1];

        let expected_root = builder.compute_uniform_tree_root(final_columns[0]).unwrap();
        let expected_size = builder.tree_size();

        assert_eq!(expected_size, res.len());
        assert_eq!(expected_root, computed_root);

        res[res.len() - 1]
    }
}
