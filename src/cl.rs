use log::*;
use rust_gpu_tools::opencl::{cl_device_id, Device, GPUSelector};
use std::collections::HashMap;
use std::fmt;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use triton::bindings;
use triton::FutharkContext;
const MAX_LEN: usize = 128;

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

lazy_static! {
    pub static ref FUTHARK_CONTEXT_MAP: RwLock<HashMap<String, Arc<Mutex<FutharkContext>>>> =
        RwLock::new(HashMap::new());
}

#[derive(Debug, Clone)]
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
pub type ClResult<T> = std::result::Result<T, ClError>;

impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
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

fn create_context(device: bindings::cl_device_id) -> ClResult<bindings::cl_context> {
    let mut res = 0i32;
    let context = unsafe {
        bindings::clCreateContext(
            ptr::null(),
            1,
            [device].as_mut_ptr(),
            None,
            ptr::null_mut(),
            &mut res,
        )
    };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(context)
    } else {
        Err(ClError::CannotCreateContext)
    }
}

fn create_queue(
    context: bindings::cl_context,
    device: bindings::cl_device_id,
) -> ClResult<bindings::cl_command_queue> {
    let mut res = 0i32;
    let context = unsafe { bindings::clCreateCommandQueue(context, device, 0, &mut res) };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(context)
    } else {
        Err(ClError::CannotCreateQueue)
    }
}

fn create_futhark_context(device: bindings::cl_device_id) -> ClResult<FutharkContext> {
    unsafe {
        let context = create_context(device)?;
        let queue = create_queue(context, device)?;

        let ctx_config = bindings::futhark_context_config_new();
        let ctx = bindings::futhark_context_new_with_command_queue(ctx_config, queue);
        Ok(FutharkContext {
            context: ctx,
            config: ctx_config,
        })
    }
}

pub fn futhark_context(selector: GPUSelector) -> ClResult<Arc<Mutex<FutharkContext>>> {
    info!("getting context for ~{:?}", selector);
    let mut map = FUTHARK_CONTEXT_MAP.write().unwrap();

    let key = selector.get_key();
    info!("device key: {}", key);
    if !map.contains_key(&key) {
        if let Some(device) = selector.get_device() {
            info!("device: {:?}", device);
            let cl_device_id = unsafe {
                std::mem::transmute::<cl_device_id, bindings::cl_device_id>(device.cl_device_id())
            };
            let context = create_futhark_context(cl_device_id)?;
            map.insert(key.clone(), Arc::new(Mutex::new(context)));
        } else {
            unimplemented!();
        }
    }
    Ok(Arc::clone(&map[&key]))
}

pub fn default_futhark_context() -> ClResult<Arc<Mutex<FutharkContext>>> {
    info!("getting default futhark context");
    let bus_id = std::env::var("NEPTUNE_DEFAULT_GPU")
        .ok()
        .and_then(|v| match v.parse::<u32>() {
            Ok(bus_id) => Some(bus_id),
            Err(_) => {
                error!("Bus-id '{}' is given in wrong format!", v);
                None
            }
        });
    match bus_id {
        Some(bus_id) => {
            info!(
                "Using device with bus-id {} for creating the FutharkContext...",
                bus_id
            );
            futhark_context(GPUSelector::BusId(bus_id))
        }
        .or_else(|_| {
            error!(
                "A device with the given bus-id doesn't exist! Defaulting to the first device..."
            );
            futhark_context(GPUSelector::Index(0))
        }),
        None => futhark_context(GPUSelector::Index(0)),
    }
}

fn to_u32(inp: &[u8]) -> u32 {
    (inp[0] as u32) + ((inp[1] as u32) << 8) + ((inp[2] as u32) << 16) + ((inp[3] as u32) << 24)
}
