# Changelog

All notable changes to neptune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://book.async.rs/overview/stability-guarantees.html).

## Unreleased

## 5.0.0 - 2021-9-30

- Remove pairing requirement for circuits (https://github.com/filecoin-project/neptune/pull/111)
- Add support for CUDA (https://github.com/filecoin-project/neptune/pull/109)
- Use correct global work size (https://github.com/filecoin-project/neptune/pull/108)
- Properly call ec_gpu_gen::common() (https://github.com/filecoin-project/neptune/pull/110)
- Use upstream group, ff and pairing dependencies (https://github.com/filecoin-project/neptune/pull/103)
- Remove unnecessary to_vec (https://github.com/filecoin-project/neptune/pull/107)
- Pass GPU data from a pre-populated vector (https://github.com/filecoin-project/neptune/pull/106)

## 4.0.0 - 2021-8-2
- Upgrade to latest rust-gpu-tools (https://github.com/filecoin-project/neptune/pull/91)
- Rename GPU feature to Futhark (https://github.com/filecoin-project/neptune/pull/100)
- Improve Clippy on CI (https://github.com/filecoin-project/neptune/pull/92)
- Fix Clippy warnings (https://github.com/filecoin-project/neptune/pull/98)
- Remove BatcherType (https://github.com/filecoin-project/neptune/pull/97)
- Remove GPUSelector (https://github.com/filecoin-project/neptune/pull/96)

## 3.0.0 - 2021-6-1
- Breaking update of `bellperson` to `0.14` and associated dependency upgrades.

## 2.7 - 2021-3-9
- Use bellperson 0.13.

## 2.6 - 2021-1-21
- Pure OpenCL implementation of batch hashing. (https://github.com/filecoin-project/neptune/pull/78)

## 2.5 [release commited but never published to crates.io, due to authentication glitch]

## 2.4.0 - 2020-11-17

- Customize batch-sizes of gbench through cli args. (https://github.com/filecoin-project/neptune/pull/50)
- Remove [most] macos conditional code. (https://github.com/filecoin-project/neptune/pull/72)
- Refactor, moving device selection to rust-gpu-tools. (https://github.com/filecoin-project/neptune/pull/70)
- Only clear cache when no hashers are active for futhark context. (https://github.com/filecoin-project/neptune/pull/68)

## 2.2.0 - 2020-11-01

- Update `bellperson` to `0.12.0`
  [67](https://github.com/filecoin-project/neptune/pull/67)

## 2.1.1 - 2020-10-30

- Fix `GPUBatchHasher` not clearing GPU caches
  [66](https://github.com/filecoin-project/neptune/pull/66)

## 2.1.0 - 2020-10-29

- Enable `blst` backend.
  [63](https://github.com/filecoin-project/neptune/pull/63)
- Explicitly reuse FutharkContext in related Batchers.
  [62](https://github.com/filecoin-project/neptune/pull/62)
- Make GPUSelector accessible from gbench.
  [59](https://github.com/filecoin-project/neptune/pull/59)
- Create SECURITY.MD.
  [57](https://github.com/filecoin-project/neptune/pull/57)
- Avoid compiling any OpenCL on macos.
  [56](https://github.com/filecoin-project/neptune/pull/56)
- Use latest neptune-triton.
  [55](https://github.com/filecoin-project/neptune/pull/55)

## 2.0.0 - 2020-08-4

- Add support for domain separation tags. In addition to support for new hash functions built on the Poseidon permutation,
  this introduces a breaking change to the DST used for Strengthened Poseidon.
  [43](https://github.com/filecoin-project/neptune/pull/43).

