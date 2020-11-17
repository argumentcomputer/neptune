# Changelog

All notable changes to neptune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://book.async.rs/overview/stability-guarantees.html).

## Unreleased

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

