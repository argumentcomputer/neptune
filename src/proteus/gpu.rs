use super::program;
use super::sources::{generate_program, DerivedConstants};
use crate::error::{ClError, Error};
use crate::hash_type::HashType;
use crate::poseidon::PoseidonConstants;
use crate::{Arity, BatchHasher, Strength, DEFAULT_STRENGTH};
use blstrs::Scalar as Fr;
use ff::{Field, PrimeField};
use generic_array::{typenum, ArrayLength, GenericArray};
use log::info;
use rust_gpu_tools::{program_closures, Device, Program};
use std::collections::HashMap;
use std::marker::PhantomData;
use typenum::{U11, U2, U8};

#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;
use std::ffi::c_void;

#[derive(Debug)]
enum Buffer<T> {
    #[cfg(feature = "cuda")]
    Cuda(cuda::Buffer<T>),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Buffer<T>),
}

#[cfg(feature = "cuda")]
impl<T> cuda::KernelArgument for Buffer<T> {
    fn as_c_void(&self) -> *mut c_void {
        match self {
            Self::Cuda(buffer) => buffer.as_c_void(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => unreachable!(),
        }
    }
}

#[cfg(feature = "opencl")]
impl<T> opencl::KernelArgument for Buffer<T> {
    fn push(&self, kernel: &mut opencl::Kernel) {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => unreachable!(),
            Self::OpenCl(buffer) => buffer.push(kernel),
        };
    }
}

#[derive(Debug)]
struct GpuConstants<A>(PoseidonConstants<Fr, A>)
where
    A: Arity<Fr>;

pub struct ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    device: Device,
    constants: GpuConstants<A>,
    constants_buffer: Buffer<Fr>,
    max_batch_size: usize,
    program: Program,
}

impl<A> GpuConstants<A>
where
    A: Arity<Fr>,
{
    fn strength(&self) -> Strength {
        self.0.strength
    }

    fn derived_constants(&self) -> DerivedConstants {
        DerivedConstants::new(self.0.arity(), self.0.full_rounds, self.0.partial_rounds)
    }

    fn to_vec(&self) -> Vec<Fr> {
        let constants_elements = self.derived_constants().constants_elements;

        let constants = &self.0;
        let mut data = Vec::with_capacity(constants_elements);
        data.push(constants.domain_tag);
        data.extend(&constants.compressed_round_constants);
        data.extend(constants.mds_matrices.m.iter().flatten());
        data.extend(constants.pre_sparse_matrix.iter().flatten());
        for sm in &constants.sparse_matrixes {
            data.extend(&sm.w_hat);
            data.extend(&sm.v_rest);
        }
        data
    }
}

impl<A> ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    /// Create a new `GPUBatchHasher` and initialize it with state corresponding with its `A`.
    pub(crate) fn new(device: &Device, max_batch_size: usize) -> Result<Self, Error> {
        Self::new_with_strength(device, DEFAULT_STRENGTH, max_batch_size)
    }

    pub(crate) fn new_with_strength(
        device: &Device,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        let constants = GpuConstants(PoseidonConstants::<Fr, A>::new_with_strength(strength));
        let program = program::program::<Fr>(device)?;

        // Allocate the buffer only once and re-use it in the hashing steps
        let constants_buffer = match program {
            #[cfg(feature = "cuda")]
            Program::Cuda(ref cuda_program) => cuda_program.run(
                |prog, _| -> Result<Buffer<Fr>, Error> {
                    let buffer = prog.create_buffer_from_slice(&constants.to_vec())?;
                    Ok(Buffer::Cuda(buffer))
                },
                (),
            )?,
            #[cfg(feature = "opencl")]
            Program::Opencl(ref opencl_program) => opencl_program.run(
                |prog, _| -> Result<Buffer<Fr>, Error> {
                    let buffer = prog.create_buffer_from_slice(&constants.to_vec())?;
                    Ok(Buffer::OpenCl(buffer))
                },
                (),
            )?,
        };

        Ok(Self {
            device: device.clone(),
            constants,
            constants_buffer,
            max_batch_size,
            program,
        })
    }

    pub(crate) fn device(&self) -> Device {
        self.device.clone()
    }
}

const LOCAL_WORK_SIZE: usize = 256;
impl<A> BatchHasher<A> for ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        let local_work_size = LOCAL_WORK_SIZE;
        let max_batch_size = self.max_batch_size;
        let batch_size = preimages.len();
        assert!(batch_size <= max_batch_size);

        let global_work_size = calc_global_work_size(batch_size, local_work_size);
        let num_hashes = preimages.len();

        let kernel_name = match (A::to_usize(), self.constants.strength()) {
            #[cfg(feature = "arity2")]
            (2, Strength::Standard) => "hash_preimages_2_standard",
            #[cfg(all(feature = "arity2", feature = "strengthened"))]
            (2, Strength::Strengthened) => "hash_preimages_2_strengthened",
            #[cfg(feature = "arity4")]
            (4, Strength::Standard) => "hash_preimages_4_standard",
            #[cfg(all(feature = "arity4", feature = "strengthened"))]
            (4, Strength::Strengthened) => "hash_preimages_4_strengthened",
            #[cfg(feature = "arity8")]
            (8, Strength::Standard) => "hash_preimages_8_standard",
            #[cfg(all(feature = "arity8", feature = "strengthened"))]
            (8, Strength::Strengthened) => "hash_preimages_8_strengthened",
            #[cfg(feature = "arity11")]
            (11, Strength::Standard) => "hash_preimages_11_standard",
            #[cfg(all(feature = "arity11", feature = "strengthened"))]
            (11, Strength::Strengthened) => "hash_preimages_11_strengthened",
            #[cfg(feature = "arity16")]
            (16, Strength::Standard) => "hash_preimages_16_standard",
            #[cfg(all(feature = "arity16", feature = "strengthened"))]
            (16, Strength::Strengthened) => "hash_preimages_16_strengthened",
            #[cfg(feature = "arity24")]
            (24, Strength::Standard) => "hash_preimages_24_standard",
            #[cfg(all(feature = "arity24", feature = "strengthened"))]
            (24, Strength::Strengthened) => "hash_preimages_24_strengthened",
            #[cfg(feature = "arity36")]
            (36, Strength::Standard) => "hash_preimages_36_standard",
            #[cfg(all(feature = "arity36", feature = "strengthened"))]
            (36, Strength::Strengthened) => "hash_preimages_36_strengthened",
            (arity, strength) => return Err(Error::GpuError(format!("No kernel for arity {} and strength {:?} available. Try to enable the `arity{}` feature flag.", arity, strength, arity))),
        };

        let closures = program_closures!(|program, _args| -> Result<Vec<Fr>, Error> {
            let kernel = program.create_kernel(kernel_name, global_work_size, local_work_size)?;
            let preimages_buffer = program.create_buffer_from_slice(&preimages)?;
            let result_buffer = unsafe { program.create_buffer::<Fr>(num_hashes)? };

            kernel
                .arg(&self.constants_buffer)
                .arg(&preimages_buffer)
                .arg(&result_buffer)
                .arg(&(preimages.len() as i32))
                .run()?;

            let mut frs = vec![<Fr as Field>::zero(); num_hashes];
            program.read_into_buffer(&result_buffer, &mut frs)?;
            Ok(frs.to_vec())
        });

        let results = self.program.run(closures, ())?;
        Ok(results)
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

/// Set `global_work_size` to the smallest value possible, so that the
/// total number of threads is >= `batch-size`.
fn calc_global_work_size(batch_size: usize, local_work_size: usize) -> usize {
    (batch_size / local_work_size) + (batch_size % local_work_size != 0) as usize
}

#[cfg(test)]
#[cfg(all(feature = "opencl", not(target_os = "macos")))]
mod test {
    use super::*;
    use crate::poseidon::{Poseidon, SimplePoseidonBatchHasher};
    use generic_array::sequence::GenericSequence;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_batch_hash_2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let device = *Device::all().first().expect("Cannot get a device");

        // NOTE: `batch_size` is not a multiple of `LOCAL_WORK_SIZE`.
        let batch_size = 1025;

        let mut cl_hasher =
            ClBatchHasher::<U2>::new_with_strength(device, Strength::Standard, batch_size).unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U2>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let cl_hashes = cl_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        dbg!(
            &cl_hashes,
            &expected_hashes,
            &cl_hashes.len(),
            &expected_hashes.len()
        );

        assert_eq!(expected_hashes, cl_hashes);
    }

    #[test]
    fn test_calc_global_work_size() {
        let inputs = vec![
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (1, 1),
            (1, 4),
            (37, 11),
            (37, 57),
            (32, 4),
            (32, 16),
        ];

        for (batch_size, local_work_size) in inputs {
            let global_work_size = calc_global_work_size(batch_size, local_work_size);
            // Make sure the total number of threads is bigger than the batch size.
            assert!(
                global_work_size * local_work_size >= batch_size,
                "global work size is not greater than or equal to the batch size:: {} * {} is not >= {}",
                global_work_size,
                local_work_size,
                batch_size);
            // Make also sure that it's the minimum value.
            assert!(
                (global_work_size - 1) * local_work_size < batch_size,
                "global work size is not minimal: ({} - 1) * {} is not < {}",
                global_work_size,
                local_work_size,
                batch_size
            );
        }
    }
}
