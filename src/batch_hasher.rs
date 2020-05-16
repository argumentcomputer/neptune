use crate::error::Error;
use crate::poseidon::SimplePoseidonBatchHasher;
use crate::{Arity, BatchHasher};
use generic_array::GenericArray;
use paired::bls12_381::Fr;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub enum BatcherType {
    GPU,
    CPU,
}

#[cfg(not(target_os = "macos"))]
use crate::gpu::GPUBatchHasher;

pub enum Batcher<'a, A>
where
    A: Arity<Fr>,
{
    #[cfg(not(target_os = "macos"))]
    GPU(GPUBatchHasher<'a, A>),
    #[cfg(target_os = "macos")]
    GPU(NoGPUBatchHasher<A>),
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

    fn max_batch_size(&self) -> usize {
        match self {
            Batcher::GPU(batcher) => batcher.max_batch_size(),
            Batcher::CPU(batcher) => batcher.max_batch_size(),
        }
    }
}

// /// NoGPUBatchHasher is a dummy required so we can build with the gpu flag even on platforms on which we cannot currently
// /// run with GPU.
pub struct NoGPUBatchHasher<A>(PhantomData<A>);

impl<A> BatchHasher<A> for NoGPUBatchHasher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, _preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        unimplemented!();
    }

    fn max_batch_size(&self) -> usize {
        unimplemented!();
    }
}
