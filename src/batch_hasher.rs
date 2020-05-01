use crate::error::Error;
use crate::gpu::GPUBatchHasher;
use crate::poseidon::SimplePoseidonBatchHasher;
use crate::{Arity, BatchHasher};
use generic_array::GenericArray;
use paired::bls12_381::Fr;

#[derive(Clone, Debug)]
pub enum BatcherType {
    GPU,
    CPU,
}

pub enum Batcher<'a, A>
where
    A: Arity<Fr>,
{
    GPU(GPUBatchHasher<A>),
    CPU(SimplePoseidonBatchHasher<'a, A>),
}

impl<A> Batcher<'_, A>
where
    A: Arity<Fr>,
{
    pub(crate) fn t(&self) -> BatcherType {
        match self {
            Batcher::GPU(_) => BatcherType::GPU,
            Batcher::CPU(_) => BatcherType::CPU,
        }
    }

    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub(crate) fn new(t: &BatcherType, max_batch_size: usize) -> Result<Self, Error> {
        match t {
            BatcherType::GPU => Ok(Batcher::GPU(GPUBatchHasher::<A>::new(max_batch_size)?)),

            BatcherType::CPU => Ok(Batcher::CPU(SimplePoseidonBatchHasher::<A>::new(
                max_batch_size,
            )?)),
        }
    }
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    pub(crate) fn new(t: &BatcherType, max_batch_size: usize) -> Result<Self, Error> {
        match t {
            BatcherType::GPU => Err(Error::Other("GPU not configured".to_string())),
            BatcherType::CPU => Ok(Batcher::CPU(SimplePoseidonBatchHasher::<A>::new(
                max_batch_size,
            )?)),
        }
    }
}

impl<A> BatchHasher<A> for Batcher<'_, A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        match self {
            Batcher::GPU(batcher) => batcher.hash(preimages),
            Batcher::CPU(batcher) => batcher.hash(preimages),
        }
    }

    fn tree_leaf_count(&self) -> Option<usize> {
        match self {
            Batcher::GPU(batcher) => batcher.tree_leaf_count(),
            Batcher::CPU(batcher) => batcher.tree_leaf_count(),
        }
    }

    fn build_tree(&mut self, leaves: &[Fr]) -> Result<Vec<Fr>, Error> {
        match self {
            Batcher::GPU(batcher) => batcher.build_tree(leaves),
            Batcher::CPU(batcher) => batcher.build_tree(leaves),
        }
    }

    fn max_batch_size(&self) -> usize {
        match self {
            Batcher::GPU(batcher) => batcher.max_batch_size(),
            Batcher::CPU(batcher) => batcher.max_batch_size(),
        }
    }
}
