use crate::error::Error;
use crate::gpu::GPUBatchHasher;
use crate::poseidon::SimplePoseidonBatchHasher;
use crate::BatchHasher;
use generic_array::{typenum, ArrayLength, GenericArray};
use paired::bls12_381::Fr;
use std::ops::Add;
use typenum::bit::B1;
use typenum::uint::{UInt, UTerm, Unsigned};
//use typenum::{UInt, UTerm, Unsigned, U11, U2, U8};

#[derive(Clone, Debug)]
pub enum BatcherType {
    GPU,
    CPU,
}

pub enum Batcher<'a, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    GPU(GPUBatchHasher<Arity>),
    CPU(SimplePoseidonBatchHasher<'a, Arity>),
}

impl<Arity> Batcher<'_, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    pub(crate) fn t(&self) -> BatcherType {
        match self {
            Batcher::GPU(_) => BatcherType::GPU,
            Batcher::CPU(_) => BatcherType::CPU,
        }
    }

    pub(crate) fn new(t: &BatcherType, max_batch_size: usize) -> Result<Self, Error> {
        match t {
            BatcherType::GPU => Ok(Batcher::GPU(GPUBatchHasher::<Arity>::new(max_batch_size)?)),
            BatcherType::CPU => Ok(Batcher::CPU(SimplePoseidonBatchHasher::<Arity>::new(
                max_batch_size,
            )?)),
        }
    }
}

impl<Arity> BatchHasher<Arity> for Batcher<'_, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, Arity>]) -> Result<Vec<Fr>, Error> {
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
