use ff::Field;
use generic_array::sequence::GenericSequence;
use generic_array::typenum::{U11, U8};
use generic_array::GenericArray;
use log::info;
use neptune::batch_hasher::BatcherType;
use neptune::cl;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use neptune::error::Error;
use neptune::BatchHasher;
use paired::bls12_381::Fr;
use std::result::Result;
use std::thread;
use std::time::Instant;

fn bench_column_building(
    log_prefix: &str,
    batcher_type: Option<BatcherType>,
    leaves: usize,
    max_column_batch_size: usize,
    max_tree_batch_size: usize,
) -> Fr {
    info!("{}: Creating ColumnTreeBuilder", log_prefix);
    let mut builder = ColumnTreeBuilder::<U11, U8>::new(
        batcher_type,
        leaves,
        max_column_batch_size,
        max_tree_batch_size,
    )
    .unwrap();
    info!("{}: ColumnTreeBuilder created", log_prefix);

    // Simplify computing the expected root.
    let constant_element = Fr::zero();
    let constant_column = GenericArray::<Fr, U11>::generate(|_| constant_element);

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
        let columns: Vec<GenericArray<Fr, U11>> =
            (0..effective_batch_size).map(|_| constant_column).collect();

        let _ = builder.add_columns(columns.as_slice()).unwrap();
        total_columns += columns.len();
    }
    println!("");

    let final_columns: Vec<_> = (0..leaves - total_columns)
        .map(|_| GenericArray::<Fr, U11>::generate(|_| constant_element))
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
        "{}: result tree was not expected size",
        log_prefix
    );
    assert_eq!(
        expected_root, computed_root,
        "{}: computed root was not the expected one",
        log_prefix
    );

    res[res.len() - 1]
}

fn main() -> Result<(), Error> {
    env_logger::init();

    let kib = 1024 * 1024 * 4; // 4GiB
                               // let kib = 1024 * 512; // 512MiB
    let bytes = kib * 1024;
    let leaves = bytes / 32;
    let max_column_batch_size = 400000;
    let max_tree_batch_size = 700000;

    info!("KiB: {}", kib);
    info!("leaves: {}", leaves);
    info!("max column batch size: {}", max_column_batch_size);
    info!("max tree batch size: {}", max_tree_batch_size);

    // Comma separated list of GPU bus-ids
    let bus_ids = std::env::var("NEPTUNE_GBENCH_GPUS")
        .map(|v| {
            v.split(",")
                .map(|s| s.parse::<u32>().expect("Invalid Bus-Id number!"))
                .collect::<Vec<u32>>()
        })
        .unwrap_or(vec![cl::get_all_bus_ids().unwrap()[0]]);

    let mut threads = Vec::new();
    for bus_id in bus_ids {
        threads.push(thread::spawn(move || {
            let log_prefix = format!("GPU[Bus-id: {}]", bus_id);
            for i in 0..3 {
                info!("{} --> Run {}", log_prefix, i);
                bench_column_building(
                    &log_prefix,
                    Some(BatcherType::CustomGPU(cl::GPUSelector::BusId(bus_id))),
                    leaves,
                    max_column_batch_size,
                    max_tree_batch_size,
                );
            }
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }
    info!("end");
    // Leave time to verify GPU memory usage goes to zero before exiting.
    std::thread::sleep(std::time::Duration::from_millis(15000));
    Ok(())
}
