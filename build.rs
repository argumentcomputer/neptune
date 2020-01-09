use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("CARGO_MANIFEST_DIR").expect("No out dir");
    let dest_path = Path::new(&out_dir).join("src").join("constants.rs");
    let mut f = File::create(&dest_path).expect("Could not create file");

    let width = env::var("POSEIDON_WIDTH")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_WIDTH"))
        .unwrap_or(4);

    let full_rounds = env::var("POSEIDON_FULL_ROUNDS")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_FULL_ROUNDS"))
        .unwrap_or(8);

    let partial_rounds = env::var("POSEIDON_PARTIAL_ROUNDS")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_PARTIAL_ROUNDS"))
        .unwrap_or(59);

    write!(
        &mut f,
        r#"// Poseidon constants
/// Width of a Poseidon hash, in elemen
pub const WIDTH: usize = {};
pub(crate) const FULL_ROUNDS: usize = {};
pub(crate) const PARTIAL_ROUNDS: usize = {};

"#,
        width, full_rounds, partial_rounds
    )
    .expect("Could not write file");
}
