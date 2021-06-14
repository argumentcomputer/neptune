use rust_gpu_tools::opencl;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::error::{ClError, Error};
use crate::poseidon::SimplePoseidonBatchHasher;
#[cfg(feature = "opencl")]
use crate::proteus::gpu::CLBatchHasher;
#[cfg(feature = "gpu")]
use crate::triton::cl;
use crate::{Arity, BatchHasher, Strength, DEFAULT_STRENGTH};
use bellperson::bls::Fr;
use generic_array::GenericArray;

#[cfg(feature = "gpu")]
use triton::FutharkContext;

#[derive(Clone)]
pub enum BatcherType {
    #[cfg(feature = "gpu")]
    FromDevice(opencl::Device),
    #[cfg(feature = "opencl")]
    FromDevice(opencl::Device),
    #[cfg(feature = "gpu")]
    GPU,
    CPU,
    #[cfg(feature = "opencl")]
    OpenCL,
}

impl Debug for BatcherType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("BatcherType::"))?;
        match self {
            #[cfg(feature = "gpu")]
            BatcherType::FromDevice(_) => f.write_fmt(format_args!("FromDevice")),
            #[cfg(feature = "opencl")]
            BatcherType::FromDevice(_) => f.write_fmt(format_args!("FromDevice")),
            BatcherType::CPU => f.write_fmt(format_args!("CPU")),
            #[cfg(feature = "gpu")]
            BatcherType::GPU => f.write_fmt(format_args!("GPU")),
            #[cfg(feature = "opencl")]
            BatcherType::OpenCL => f.write_fmt(format_args!("OpenCL")),
        }
    }
}

#[cfg(feature = "gpu")]
use crate::triton::gpu::GPUBatchHasher;

pub enum Batcher<A>
where
    A: Arity<Fr>,
{
    #[cfg(feature = "gpu")]
    GPU(GPUBatchHasher<A>),
    CPU(SimplePoseidonBatchHasher<A>),
    #[cfg(feature = "opencl")]
    OpenCL(CLBatchHasher<A>),
}

impl<A> Batcher<A>
where
    A: Arity<Fr>,
{
    pub(crate) fn t(&self) -> BatcherType {
        match self {
            #[cfg(feature = "gpu")]
            Batcher::GPU(_) => BatcherType::GPU,
            Batcher::CPU(_) => BatcherType::CPU,
            #[cfg(feature = "opencl")]
            Batcher::OpenCL(_) => BatcherType::OpenCL,
        }
    }

    pub(crate) fn new(t: &BatcherType, max_batch_size: usize) -> Result<Self, Error> {
        Self::new_with_strength(DEFAULT_STRENGTH, t, max_batch_size)
    }

    pub(crate) fn new_with_strength(
        strength: Strength,
        t: &BatcherType,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        match t {
            BatcherType::CPU => Ok(Batcher::CPU(
                SimplePoseidonBatchHasher::<A>::new_with_strength(strength, max_batch_size)?,
            )),
            #[cfg(feature = "gpu")]
            BatcherType::GPU => Ok(Batcher::GPU(GPUBatchHasher::<A>::new_with_strength(
                cl::default_futhark_context()?,
                strength,
                max_batch_size,
            )?)),
            #[cfg(feature = "gpu")]
            BatcherType::FromDevice(device) => {
                let futhark_context = cl::futhark_context(&device)?;
                Ok(Batcher::GPU(GPUBatchHasher::<A>::new_with_strength(
                    futhark_context,
                    strength,
                    max_batch_size,
                )?))
            }
            #[cfg(feature = "opencl")]
            BatcherType::OpenCL => {
                let all = opencl::Device::all();
                let device = all
                    .first()
                    .ok_or_else(|| Error::ClError(ClError::DeviceNotFound))?;

                Ok(Batcher::OpenCL(CLBatchHasher::<A>::new_with_strength(
                    device,
                    strength,
                    max_batch_size,
                )?))
            }
            #[cfg(feature = "opencl")]
            BatcherType::FromDevice(device) => Ok(Batcher::OpenCL(
                CLBatchHasher::<A>::new_with_strength(&device, strength, max_batch_size)?,
            )),
        }
    }
}

impl<A> BatchHasher<A> for Batcher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        match self {
            Batcher::CPU(batcher) => batcher.hash(preimages),
            #[cfg(feature = "gpu")]
            Batcher::GPU(batcher) => batcher.hash(preimages),
            #[cfg(feature = "opencl")]
            Batcher::OpenCL(batcher) => batcher.hash(preimages),
        }
    }

    fn max_batch_size(&self) -> usize {
        match self {
            Batcher::CPU(batcher) => batcher.max_batch_size(),
            #[cfg(feature = "gpu")]
            Batcher::GPU(batcher) => batcher.max_batch_size(),
            #[cfg(feature = "opencl")]
            Batcher::OpenCL(batcher) => batcher.max_batch_size(),
        }
    }
}
