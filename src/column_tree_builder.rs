use crate::batch_hasher::{Batcher, BatcherType};
use crate::error::Error;
use crate::poseidon::{Poseidon, PoseidonConstants};
use crate::tree_builder::{TreeBuilder, TreeBuilderTrait};
use crate::BatchHasher;
use bellperson::bls::{Bls12, Fr};
use ff::Field;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use rust_gpu_tools::opencl::GPUSelector;

pub trait ColumnTreeBuilderTrait<ColumnArity, TreeArity>
where
    ColumnArity: typenum::Unsigned,
    TreeArity: typenum::Unsigned,
{
    fn add_columns(&mut self, columns: &[Fr]) -> Result<(), Error>;
    fn add_final_columns(&mut self, columns: &[Fr]) -> Result<(Vec<Fr>, Vec<Fr>), Error>;

    fn reset(&mut self);
}

pub struct ColumnTreeBuilder<ColumnArity, TreeArity>
where
    ColumnArity: typenum::Unsigned,
    TreeArity: typenum::Unsigned,
{
    pub leaf_count: usize,
    data: Vec<Fr>,
    /// Index of the first unfilled datum.
    fill_index: usize,
    column_constants: PoseidonConstants<Bls12, ColumnArity>,
    pub column_batcher: Option<Batcher<ColumnArity>>,
    tree_builder: TreeBuilder<TreeArity>,
}

impl<ColumnArity, TreeArity> ColumnTreeBuilderTrait<ColumnArity, TreeArity>
    for ColumnTreeBuilder<ColumnArity, TreeArity>
where
    ColumnArity: typenum::Unsigned,
    TreeArity: typenum::Unsigned,
{
    fn add_columns(&mut self, columns: &[Fr]) -> Result<(), Error> {
        let start = self.fill_index;
        let column_count = columns.len() / ColumnArity::to_usize();
        let end = start + column_count;

        if end > self.leaf_count {
            return Err(Error::Other("too many columns".to_string()));
        }

        match self.column_batcher {
            Some(ref mut batcher) => {
                batcher.hash_into_slice(&mut self.data[start..start + column_count], columns)?;
            }
            None => columns
                .chunks(ColumnArity::to_usize())
                .enumerate()
                .for_each(|(i, column)| {
                    self.data[start + i] =
                        Poseidon::new_with_preimage(&column, &self.column_constants).hash();
                }),
        };

        self.fill_index += column_count;

        Ok(())
    }

    fn add_final_columns(&mut self, columns: &[Fr]) -> Result<(Vec<Fr>, Vec<Fr>), Error> {
        self.add_columns(columns)?;

        let (base, tree) = self.tree_builder.add_final_leaves(&self.data)?;
        self.reset();

        Ok((base, tree))
    }

    fn reset(&mut self) {
        self.fill_index = 0;
        self.data.iter_mut().for_each(|place| *place = Fr::zero());
    }
}

impl<ColumnArity, TreeArity> ColumnTreeBuilder<ColumnArity, TreeArity>
where
    ColumnArity: typenum::Unsigned,
    TreeArity: typenum::Unsigned,
{
    pub fn new(
        t: Option<BatcherType>,
        leaf_count: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
    ) -> Result<Self, Error> {
        let column_batcher = match &t {
            Some(t) => Some(Batcher::<ColumnArity>::new(t, max_column_batch_size)?),
            None => None,
        };

        let tree_builder = match {
            match &column_batcher {
                Some(b) => b.futhark_context(),
                None => None,
            }
        } {
            Some(ctx) => TreeBuilder::<TreeArity>::new(
                Some(BatcherType::FromFutharkContext(ctx)),
                leaf_count,
                max_tree_batch_size,
                0,
            )?,
            None => TreeBuilder::<TreeArity>::new(t, leaf_count, max_tree_batch_size, 0)?,
        };

        let builder = Self {
            leaf_count,
            data: vec![Fr::zero(); leaf_count],
            fill_index: 0,
            column_constants: PoseidonConstants::<Bls12, ColumnArity>::new(),
            column_batcher,
            tree_builder,
        };

        Ok(builder)
    }

    pub fn tree_size(&self) -> usize {
        self.tree_builder.tree_size(0)
    }

    // Compute root of tree composed of all identical columns. For use in checking correctness of GPU column tree-building
    // without the cost of generating a full column tree.
    pub fn compute_uniform_tree_root(&mut self, column: &[Fr]) -> Result<Fr, Error> {
        // All the leaves will be the same.
        let element = Poseidon::new_with_preimage(&column, &self.column_constants).hash();

        self.tree_builder.compute_uniform_tree_root(element)
    }
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::Poseidon;
    use crate::BatchHasher;
    use bellperson::bls::Fr;
    use ff::Field;
    use typenum::{U11, U8};

    #[test]
    fn test_column_tree_builder() {
        // 16KiB tree has 512 leaves.
        test_column_tree_builder_aux(None, 512, 32, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::CPU), 512, 32, 512, 512);

        #[cfg(feature = "gpu")]
        test_column_tree_builder_aux(Some(BatcherType::GPU), 512, 32, 512, 512);
    }

    fn test_column_tree_builder_aux(
        batcher_type: Option<BatcherType>,
        leaves: usize,
        num_batches: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
    ) {
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
        let constant_column = vec![constant_element; 11];

        let max_batch_size = if let Some(batcher) = &builder.column_batcher {
            batcher.max_batch_size()
        } else {
            leaves
        };

        let effective_batch_size = usize::min(batch_size, max_batch_size);

        let mut total_columns = 0;
        while total_columns + effective_batch_size < leaves {
            let columns: Vec<Fr> = (0..effective_batch_size)
                .map(|_| constant_column.as_slice())
                .flatten()
                .cloned()
                .collect();

            let _ = builder.add_columns(&columns).unwrap();
            total_columns += effective_batch_size;
        }

        let x = vec![constant_element; 11];
        let final_columns: Vec<Fr> = (0..leaves - total_columns)
            .map(|_| x.as_slice())
            .flatten()
            .cloned()
            .collect();

        let (base, res) = builder.add_final_columns(&final_columns).unwrap();

        let column_hash =
            Poseidon::new_with_preimage(&constant_column, &builder.column_constants).hash();
        assert!(base.iter().all(|x| *x == column_hash));

        let computed_root = res[res.len() - 1];

        let expected_root = builder.compute_uniform_tree_root(&x).unwrap();
        let expected_size = builder.tree_builder.tree_size(0);

        assert_eq!(leaves, base.len());
        assert_eq!(expected_size, res.len());
        assert_eq!(expected_root, computed_root);
    }
}
