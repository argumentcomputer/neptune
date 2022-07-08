/// The build script is used to generate the CUDA kernel and OpenCL source at compile-time, if the
/// `cuda` and/or `opencl` feature is enabled.
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    #[path = "src/proteus/sources.rs"]
    mod sources;

    let source_builder = sources::generate_program();
    ec_gpu_gen::generate(&source_builder);
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}
