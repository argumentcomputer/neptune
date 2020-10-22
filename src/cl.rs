use log::*;
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
    pub static ref FUTHARK_CONTEXT_MAP: RwLock<HashMap<u32, Arc<Mutex<FutharkContext>>>> =
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
        }
    }
}

fn get_platforms() -> ClResult<Vec<bindings::cl_platform_id>> {
    let mut platforms = [ptr::null_mut(); MAX_LEN];
    let mut num_platforms = 0u32;
    let res = unsafe {
        bindings::clGetPlatformIDs(MAX_LEN as u32, platforms.as_mut_ptr(), &mut num_platforms)
    };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(platforms[..num_platforms as usize].to_vec())
    } else {
        Err(ClError::PlatformNotFound)
    }
}

fn get_platform_name(platform_id: bindings::cl_platform_id) -> ClResult<String> {
    let mut name = [0u8; MAX_LEN];
    let mut len = 0u64;
    let res = unsafe {
        bindings::clGetPlatformInfo(
            platform_id,
            bindings::CL_PLATFORM_NAME as u32,
            MAX_LEN as u64,
            name.as_mut_ptr() as *mut std::ffi::c_void,
            &mut len,
        )
    };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(String::from_utf8(name[..len as usize - 1].to_vec()).unwrap())
    } else {
        Err(ClError::PlatformNameNotAvailable)
    }
}

fn get_platform_by_name(name: &str) -> ClResult<bindings::cl_platform_id> {
    for plat in get_platforms()? {
        if get_platform_name(plat)? == name {
            return Ok(plat);
        }
    }
    Err(ClError::PlatformNotFound)
}

fn get_devices(platform_id: bindings::cl_platform_id) -> ClResult<Vec<bindings::cl_device_id>> {
    let mut devs = [ptr::null_mut(); MAX_LEN];
    let mut num_devs = 0u32;
    let res = unsafe {
        bindings::clGetDeviceIDs(
            platform_id,
            bindings::CL_DEVICE_TYPE_GPU as u64,
            MAX_LEN as u32,
            devs.as_mut_ptr(),
            &mut num_devs,
        )
    };
    if res == bindings::CL_SUCCESS as i32 {
        Ok(devs[..num_devs as usize].to_vec())
    } else {
        Err(ClError::DeviceNotFound)
    }
}

fn get_bus_id(device: bindings::cl_device_id) -> ClResult<u32> {
    get_nvidia_bus_id(device)
        .or_else(|_| Ok(get_amd_topology(device)?.bus as u32))
        .map_err(|_: ClError| ClError::BusIdNotAvailable)
}

fn get_nvidia_bus_id(device: bindings::cl_device_id) -> ClResult<u32> {
    let mut ret = [0u8; MAX_LEN];
    let mut len = 0u64;
    let res = unsafe {
        bindings::clGetDeviceInfo(
            device,
            0x4008 as u32,
            MAX_LEN as u64,
            ret.as_mut_ptr() as *mut std::ffi::c_void,
            &mut len,
        )
    };
    if res == bindings::CL_SUCCESS as i32 && len == 4 {
        Ok(to_u32(&ret[..4]))
    } else {
        Err(ClError::NvidiaBusIdNotAvailable)
    }
}

fn get_amd_topology(device: bindings::cl_device_id) -> ClResult<cl_amd_device_topology> {
    let mut ret = cl_amd_device_topology::default();
    let size = std::mem::size_of::<cl_amd_device_topology>() as u64;
    let mut len = 0u64;
    let res = unsafe {
        bindings::clGetDeviceInfo(
            device,
            0x4037 as u32,
            size,
            &mut ret as *mut cl_amd_device_topology as *mut std::ffi::c_void,
            &mut len,
        )
    };
    if res == bindings::CL_SUCCESS as i32 && len == size {
        Ok(ret)
    } else {
        Err(ClError::AmdTopologyNotAvailable)
    }
}

fn get_device_by_bus_id(bus_id: u32) -> ClResult<bindings::cl_device_id> {
    for dev in get_all_devices()? {
        if get_bus_id(dev)? == bus_id {
            return Ok(dev);
        }
    }

    Err(ClError::DeviceNotFound)
}

fn get_all_devices() -> ClResult<Vec<bindings::cl_device_id>> {
    let mut devices = Vec::new();

    let mut platforms = get_platforms()?;

    if let Ok(platform) = get_platform_by_name("NVIDIA CUDA") {
        // If there is an Nvidia platform, make it the first, so any Nvidia card will be the default.
        platforms = platforms
            .iter()
            .filter(|x| **x != platform)
            .map(|x| *x)
            .collect::<Vec<_>>();
        platforms.insert(0, platform);
    }

    for platform in platforms {
        if let Ok(devs) = get_devices(platform) {
            for dev in devs {
                devices.push(dev);
            }
        } else {
            warn!(
                "Cannot get device list for platform: {}!",
                get_platform_name(platform)?
            );
        }
    }
    Ok(devices)
}

fn get_first_device() -> ClResult<bindings::cl_device_id> {
    let devs = get_all_devices()?;
    devs.first().map(|d| *d).ok_or(ClError::DeviceNotFound)
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

#[derive(Debug, Clone, Copy)]
pub enum GPUSelector {
    BusId(u32),
    Index(usize),
}

impl GPUSelector {
    pub fn get_bus_id(&self) -> ClResult<u32> {
        match self {
            GPUSelector::BusId(bus_id) => Ok(*bus_id),
            GPUSelector::Index(index) => Ok(get_bus_id(get_all_devices()?[*index])?),
        }
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

pub fn get_all_nvidia_devices() -> ClResult<Vec<bindings::cl_device_id>> {
    Ok(get_devices(get_platform_by_name("NVIDIA CUDA")?)?)
}

pub fn get_all_bus_ids() -> ClResult<Vec<u32>> {
    let mut bus_ids = Vec::new();
    for dev in get_all_devices()? {
        match get_bus_id(dev) {
            Ok(bus_id) => bus_ids.push(bus_id),
            Err(_) => (),
        }
    }
    bus_ids.sort_unstable();
    Ok(bus_ids)
}

pub fn futhark_context(selector: GPUSelector) -> ClResult<Arc<Mutex<FutharkContext>>> {
    info!("getting context for ~{:?}", selector);
    let mut map = FUTHARK_CONTEXT_MAP.write().unwrap();
    let bus_id = selector.get_bus_id()?;
    if !map.contains_key(&bus_id) {
        let device = get_device_by_bus_id(bus_id)?;
        let context = create_futhark_context(device)?;
        map.insert(bus_id, Arc::new(Mutex::new(context)));
    }
    Ok(Arc::clone(&map[&bus_id]))
}

pub fn default_futhark_context() -> ClResult<Arc<Mutex<FutharkContext>>> {
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

#[cfg(test)]
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
mod tests {
    use super::*;

    #[test]
    fn test_bus_id_uniqueness() {
        let mut bus_ids = get_all_bus_ids().unwrap();
        let count = bus_ids.len();

        bus_ids.dedup();
        assert_eq!(
            count,
            bus_ids.len(),
            "get_all_bus_ids() returned duplicates"
        );
    }
}
