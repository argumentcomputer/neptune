# gbench

Benchmarking tool for Neptune.

## Usage

Running gbench with customized batch sizes on the default GPU:

`RUST_LOG=info cargo run -- --max-tree-batch-size 700000 --max-column-batch-size 400000`


## Environment variables

 - `NEPTUNE_GBENCH_GPUS=<pci-id1>,<pci-id2>,...` allows you to select the GPUs you want to run gbench on given a comma-separated list of pci-ids.

(Pci-id is a decimal integer that can be found through `nvidia-smi`, `rocm-smi`, `lspci` and etc.)
