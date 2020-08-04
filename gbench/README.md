# gbench

Benchmarking tool for Neptune.

## Environment variables

 - `NEPTUNE_GBENCH_GPUS=<bus-id1>,<bus-id2>,...` allows you to select the GPUs you want to run gbench on given a comma-separated list of bus-ids.
 
(Bus-id is a decimal integer that can be found through `nvidia-smi`, `rocm-smi`, `lspci` and etc.)
