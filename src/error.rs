use std::{error, fmt};

#[derive(Debug, Clone)]
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub enum ClError {
    DeviceNotFound,
    PlatformNotFound,
    BusIdNotAvailable,
    NvidiaBusIdNotAvailable,
    AmdTopologyNotAvailable,
    PlatformNameNotAvailable,
    CannotCreateContext,
    CannotCreateQueue,
    GetDeviceError,
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub type ClResult<T> = Result<T, ClError>;

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            ClError::DeviceNotFound => write!(f, "Device not found."),
            ClError::PlatformNotFound => write!(f, "Platform not found."),
            ClError::BusIdNotAvailable => write!(f, "Cannot extract bus-id for the given device."),
            ClError::NvidiaBusIdNotAvailable => {
                write!(f, "Cannot extract bus-id for the given Nvidia device.")
            }
            ClError::AmdTopologyNotAvailable => {
                write!(f, "Cannot extract bus-id for the given AMD device.")
            }
            ClError::PlatformNameNotAvailable => {
                write!(f, "Cannot extract platform name for the given platform.")
            }
            ClError::CannotCreateContext => write!(f, "Cannot create cl_context."),
            ClError::CannotCreateQueue => write!(f, "Cannot create cl_command_queue."),
            ClError::GetDeviceError => write!(f, "Cannot get Device"),
        }
    }
}

#[derive(Debug, Clone)]
/// Possible error states for the hashing.
pub enum Error {
    /// The allowed number of leaves cannot be greater than the arity of the tree.
    FullBuffer,
    /// Attempt to reference an index element that is out of bounds
    IndexOutOfBounds,
    GpuError(String),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    ClError(ClError),
    Other(String),
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl From<ec_gpu_gen::rust_gpu_tools::GPUError> for Error {
    fn from(e: ec_gpu_gen::rust_gpu_tools::GPUError) -> Self {
        Self::GpuError(format!("GPU tools error: {e}"))
    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl From<ec_gpu_gen::EcError> for Error {
    fn from(e: ec_gpu_gen::EcError) -> Self {
        Self::GpuError(format!("EC-GPU error: {e}"))
    }
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Error::FullBuffer => write!(
                f,
                "The size of the buffer cannot be greater than the hash arity."
            ),
            Error::IndexOutOfBounds => write!(f, "The referenced index is outs of bounds."),
            Error::GpuError(s) => write!(f, "GPU Error: {s}"),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            Error::ClError(e) => write!(f, "OpenCL Error: {e}"),
            Error::Other(s) => write!(f, "{s}"),
        }
    }
}
