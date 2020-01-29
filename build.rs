use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("CARGO_MANIFEST_DIR").expect("No out dir");
    let dest_path = Path::new(&out_dir).join("src").join("constants.rs");
    let mut f = File::create(&dest_path).expect("Could not create file");

    let default_arity = 2;

    let arity = env::var("POSEIDON_ARITY")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_ARITY"))
        .unwrap_or(default_arity);

    let width = arity + 1;

    let default_full_rounds = 8;
    let default_partial_rounds = match width {
        2 | 3 => 55,
        4 | 5 | 6 | 7 => 56,
        8 | 9 => 57,
        _ => panic!("unsupoorted arity"),
    };

    let full_rounds = env::var("POSEIDON_FULL_ROUNDS")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_FULL_ROUNDS"))
        .unwrap_or(default_full_rounds);

    let partial_rounds = env::var("POSEIDON_PARTIAL_ROUNDS")
        .map(|s| s.parse().expect("Failed to parse POSEIDON_PARTIAL_ROUNDS"))
        .unwrap_or(default_partial_rounds); // Conservative value (for arity 8) until this adapts to arity.

    write!(
        &mut f,
        r#"// Poseidon constants
/// Width of a Poseidon permutation, in elements
pub const WIDTH: usize = {};
/// Arity of a Poseidon hash, in elements
pub const ARITY: usize = {};
pub(crate) const FULL_ROUNDS: usize = {};
pub(crate) const PARTIAL_ROUNDS: usize = {};

"#,
        width, arity, full_rounds, partial_rounds
    )
    .expect("Could not write file");
}
