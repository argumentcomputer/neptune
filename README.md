# Neptune ![crates.io](https://img.shields.io/crates/v/neptune.svg) [![CircleCI](https://circleci.com/gh/filecoin-project/neptune.svg?style=svg)](https://circleci.com/gh/filecoin-project/neptune)

## About
Neptune is a Rust implementation of the [Poseidon hash function](https://www.poseidon-hash.info/) tuned for
[Filecoin](https://filecoin.io/).

Neptune has been [audited by ADBK Consulting](poseidon-in-filecoin-final-report.pdf) and deemed fully compliant with the
paper ([Starkad and Poseidon: New Hash Functions for Zero Knowledge Proof
Systems](https://eprint.iacr.org/2019/458.pdf)).

Neptune is specialized to the [BLS12-381 curve](https://electriccoin.co/blog/new-snark-curve/). Although the API allows
for type specialization to other fields, the round numbers, constants, and s-box selection may not be correct. Do not do
this.

Hashes of arbitrary arities are generally supported — but secure round numbers have only been calculated for a
selection (including especially 2, 4, and 8 — which are explicitly, rather than incidentally, supported). [Filecoin
Proofs](https://github.com/filecoin-project/rust-fil-proofs) make heavy use of 8-ary merkle trees and merkle inclusion
proofs (in SNARKs).

At the time of the 1.0.0 release, Neptune on RTX 2080Ti GPU can build 8-ary Merkle trees for 4GiB of input in 16 seconds.

## Implementation Specification

Filecoin's Poseidon specification is published in the Filecoin specification document [here](https://spec.filecoin.io/#section-algorithms.crypto.poseidon). Additionally, a PDF version is mirrored in this repo [here](poseidon_spec.pdf).

## Environment variables

 - `NEPTUNE_DEFAULT_GPU=<unique-id>` allows you to select the default GPU that tree-builder is going to run on given its unique ID.

(The unique ID is the UUID or the hexadecimal Bus-ID that can be found through `nvidia-smi`, `rocm-smi`, `lspci` and etc.)

 - `NEPTUNE_GPU_FRAMEWORK=<cuda | opencl>` allows to select whether the CUDA or OpenCL implementation should be used. If not set, `cuda` will be used if available.

 - `NEPTUNE_CUDA_NVCC_ARGS`

By default the CUDA kernel is compiled for several architectures, which may take a long time. `BELLMAN_CUDA_NVCC_ARGS` can be used to override those arguments. The input and output file will still be automatically set.

    // Example for compiling the kernel for only the Turing architecture
    NEPTUNE_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_75 --generate-code=arch=compute_75,code=sm_75"

## Rust feature flags

Neptune also supports batch hashing and tree building, which can be performed on a GPU. The underlying GPU
implementation, [neptune-triton](https://github.com/filecoin-project/neptune-triton) is implemented in the [Futhark
Programming Language](https://futhark-lang.org/). To use `neptune-triton` GPU batch hashing, compile `neptune` with the
`futhark` feature.

Neptune now implements GPU batch hashing in pure CUDA/OpenCL. The initial implementation is a bit less than 2x faster than
the Futhark implementation, so once stabilized this will likely be the preferred option. The pure CUDA/OpenCL batch hashing
is provided by the internal `proteus` module. To use `proteus`, compile `neptune` with the `opencl` and/or `cuda` feature.

The `futhark` and `cuda/opencl` features are mutually exclusive. The `cuda` and `opencl` feature can be used independently or together. If both `cuda` and `opencl` are used, you can also select which implementation to use via the `NEPTUNE_GPU_FRAMEWORK` environment variable.

### Arities

The CUDA/OpenCL kernel (enabled with the `cuda/opencl` feature) is generated with specific arities. Those arities need to be specified at compile-time via Rust feature flags. Available features are `arity2`, `arity4`, `arity8`, `arity11`, `arity16`, `arity24`, `arity36`. When the `strengthened` feature is enables, there will be an additional strengthened version available for each arity.

When using the `cuda` feature, the kernel is generated at compile-time. The more arities are used, the longer is the compile time. Hence, by default there are no specific arities enabled. You need to set at least one yourself.

## Running the tests

As the compile-time of the kernel depends on how many arities are used, there are no arities enabled by default. In order to run the test, all arities need to explicitly be enabled. To run all tests on e.g. the CUDA implementation, run:

    cargo test --no-default-features --features blst,cuda,arity2,arity4,arity8,arity11,arity16,arity24,arity36

## Future Work

The following are likely areas of future work:

- [x] Support for multiple GPUs.
- [x] Support domain separation tag.
- [x] Improve throughput (?) by using OpenCL directly.

## History

Neptune was originally bootstrapped from [Dusk's reference implementation](https://github.com/dusk-network/dusk-poseidon-merkle).

## Changes
[CHANGELOG](CHANGELOG.md)

## License

MIT or Apache 2.0
