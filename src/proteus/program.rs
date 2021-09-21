use std::env;

use ec_gpu::GpuField;
use log::info;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;
use rust_gpu_tools::{Device, Framework, Program};

use crate::error::{ClError, Error};
use crate::proteus::sources;

/// Returns the program for the preferred [`rust_gpu_tools::device::Framework`].
///
/// If the device supports CUDA, then CUDA is used, else OpenCL. You can force a selection with
/// the environment variable `NEPTUNE_GPU_FRAMEWORK`, which can be set either to `cuda` or `opencl`.
pub fn program<Fr>(device: &Device) -> Result<Program, Error>
where
    Fr: GpuField,
{
    let framework = match env::var("NEPTUNE_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Framework::Cuda
                }

                #[cfg(not(feature = "cuda"))]
                return Err(Error::GpuError("CUDA framework is not supported, please compile with the `cuda` feature enabled.".to_string()));
            }
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Framework::Opencl
                }

                #[cfg(not(feature = "opencl"))]
                return Err(Error::GpuError("OpenCL framework is not supported, please compile with the `opencl` feature enabled.".to_string()));
            }
            _ => device.framework(),
        },
        Err(_) => device.framework(),
    };
    program_use_framework::<Fr>(device, &framework)
}

/// Returns the program for the specified [`rust_gpu_tools::device::Framework`].
pub fn program_use_framework<Fr>(device: &Device, framework: &Framework) -> Result<Program, Error>
where
    Fr: GpuField,
{
    match framework {
        #[cfg(feature = "cuda")]
        Framework::Cuda => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_FATBIN"));
            let cuda_device = device
                .cuda_device()
                .ok_or(Error::ClError(ClError::DeviceNotFound))?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
            Ok(Program::Cuda(program))
        }
        #[cfg(feature = "opencl")]
        Framework::Opencl => {
            info!("Using kernel on OpenCL.");
            let src = sources::generate_program::<Fr, ec_gpu_gen::Limb64>();
            let opencl_device = device
                .opencl_device()
                .ok_or(Error::ClError(ClError::DeviceNotFound))?;
            let program = opencl::Program::from_opencl(opencl_device, &src)
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
            Ok(Program::Opencl(program))
        }
    }
}
