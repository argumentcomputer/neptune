use blstrs::Scalar as Fr;
use ec_gpu::GpuName;
use ec_gpu_gen::rust_gpu_tools::{Device, UniqueId};
use ff::PrimeField;
use generic_array::sequence::GenericSequence;
use generic_array::typenum::{U11, U8};
use generic_array::GenericArray;
use log::info;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use neptune::{batch_hasher::Batcher, BatchHasher};
use pasta_curves::{Fp, Fq as Fv};
use std::convert::TryFrom;
use std::str::FromStr;
use std::thread;
use std::time::Instant;
use structopt::StructOpt;

fn bench_column_building<F: PrimeField + GpuName>(
    device: &Device,
    log_prefix: &str,
    max_column_batch_size: usize,
    max_tree_batch_size: usize,
    leaves: usize,
) -> F {
    let column_batcher = Batcher::<F, U11>::new(device, max_column_batch_size).unwrap();
    let tree_batcher = Batcher::<F, U8>::new(device, max_tree_batch_size).unwrap();
    info!("{}: Creating ColumnTreeBuilder", log_prefix);
    let mut builder =
        ColumnTreeBuilder::<F, U11, U8>::new(Some(column_batcher), Some(tree_batcher), leaves)
            .unwrap();
    info!("{}: ColumnTreeBuilder created", log_prefix);

    // Simplify computing the expected root.
    let constant_element = F::zero();
    let constant_column = GenericArray::<F, U11>::generate(|_| constant_element);

    let max_batch_size = if let Some(batcher) = &builder.column_batcher {
        batcher.max_batch_size()
    } else {
        leaves
    };

    let effective_batch_size = usize::min(leaves, max_batch_size);
    info!(
        "{}: Using effective batch size {} to build columns",
        log_prefix, effective_batch_size
    );

    info!("{}: adding column batches", log_prefix);
    info!("{}: start commitment", log_prefix);
    let start = Instant::now();
    let mut total_columns = 0;
    while total_columns + effective_batch_size < leaves {
        print!(".");
        let columns: Vec<GenericArray<F, U11>> =
            (0..effective_batch_size).map(|_| constant_column).collect();

        builder.add_columns(columns.as_slice()).unwrap();
        total_columns += columns.len();
    }
    println!();

    let final_columns: Vec<_> = (0..leaves - total_columns)
        .map(|_| GenericArray::<F, U11>::generate(|_| constant_element))
        .collect();

    info!(
        "{}: adding final column batch and building tree",
        log_prefix
    );
    let (_, res) = builder.add_final_columns(final_columns.as_slice()).unwrap();
    info!("{}: end commitment", log_prefix);
    let elapsed = start.elapsed();
    info!("{}: commitment time: {:?}", log_prefix, elapsed);

    total_columns += final_columns.len();
    assert_eq!(total_columns, leaves);

    let computed_root = res[res.len() - 1];

    let expected_root = builder.compute_uniform_tree_root(final_columns[0]).unwrap();
    let expected_size = builder.tree_size();

    assert_eq!(
        expected_size,
        res.len(),
        "{log_prefix}: result tree was not expected size"
    );
    assert_eq!(
        expected_root, computed_root,
        "{log_prefix}: computed root was not the expected one"
    );

    res[res.len() - 1]
}

#[derive(Clone, Copy, Debug)]
enum Field {
    Bls,
    Pallas,
    Vesta,
}

impl FromStr for Field {
    type Err = &'static str;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "bls" => Ok(Self::Bls),
            "pallas" => Ok(Self::Pallas),
            "vesta" => Ok(Self::Vesta),
            _ => Err("Unknown field, use `bls`, `pallas` or `vesta`."),
        }
    }
}

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "Neptune gbench", about = "Neptune benchmarking program")]
struct Opts {
    #[structopt(long = "max-tree-batch-size", default_value = "700000")]
    max_tree_batch_size: usize,
    #[structopt(long = "max-column-batch-size", default_value = "400000")]
    max_column_batch_size: usize,
    #[structopt(long = "field", default_value = "bls", possible_values = &["bls", "pallas", "vesta"])]
    field: Field,
}

fn main() {
    #[cfg(all(any(feature = "cuda", feature = "opencl"), target_os = "macos"))]
    unimplemented!("Running on macos is not recommended and may have bad consequences -- experiment at your own risk.");
    env_logger::init();

    let opts = Opts::from_args();

    let field = opts.field;
    let kib = 1024 * 1024 * 4; // 4GiB
                               // let kib = 1024 * 512; // 512MiB
    let bytes = kib * 1024;
    let leaves = bytes / 32;
    let max_column_batch_size = opts.max_column_batch_size;
    let max_tree_batch_size = opts.max_tree_batch_size;

    info!("field: {:?}", field);
    info!("KiB: {}", kib);
    info!("leaves: {}", leaves);
    info!("max column batch size: {}", max_column_batch_size);
    info!("max tree batch size: {}", max_tree_batch_size);

    // Comma separated list of GPU bus-ids
    let gpus = std::env::var("NEPTUNE_GBENCH_GPUS");

    let default_device = *Device::all().first().expect("Cannot get a default device");

    let devices = gpus
        .map(|v| {
            v.split(',')
                .map(|s| UniqueId::try_from(s).expect("Invalid unique ID!"))
                .map(|unique_id| {
                    Device::by_unique_id(unique_id)
                        .unwrap_or_else(|| panic!("No device with unique ID {} found!", unique_id))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|_| vec![default_device]);

    let mut threads = Vec::new();
    for device in devices {
        threads.push(thread::spawn(move || {
            let log_prefix = format!("GPU[{device:?}]");
            for i in 0..3 {
                info!("{} --> Run {}", log_prefix, i);
                match field {
                    Field::Bls => {
                        bench_column_building::<Fr>(
                            device,
                            &log_prefix,
                            max_column_batch_size,
                            max_tree_batch_size,
                            leaves,
                        );
                    }
                    Field::Pallas => {
                        bench_column_building::<Fp>(
                            device,
                            &log_prefix,
                            max_column_batch_size,
                            max_tree_batch_size,
                            leaves,
                        );
                    }
                    Field::Vesta => {
                        bench_column_building::<Fv>(
                            device,
                            &log_prefix,
                            max_column_batch_size,
                            max_tree_batch_size,
                            leaves,
                        );
                    }
                }
            }
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }
    info!("end");
    // Leave time to verify GPU memory usage goes to zero before exiting.
    std::thread::sleep(std::time::Duration::from_millis(15000));
}
