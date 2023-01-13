// This is a hack to make it possible to include this file also in build.rs.
#[path = "../round_numbers.rs"]
mod round_numbers;

#[cfg(feature = "bls")]
use blstrs::Scalar as Fr;
use ec_gpu::GpuName;
use ec_gpu_gen::SourceBuilder;
#[cfg(feature = "pasta")]
use pasta_curves::{Fp, Fq as Fv};

use round_numbers::{round_numbers_base, round_numbers_even_partial, round_numbers_strengthened};

pub struct DerivedConstants {
    pub arity: usize,
    pub partial_rounds: usize,
    pub width: usize,
    pub sparse_matrix_size: usize,
    pub full_half: usize,
    pub sparse_offset: usize,
    pub constants_elements: usize,

    // Offsets
    pub domain_tag_offset: usize,
    pub round_keys_offset: usize,
    pub mds_matrix_offset: usize,
    pub pre_sparse_matrix_offset: usize,
    pub sparse_matrixes_offset: usize,
    pub w_hat_offset: usize,
    pub v_rest_offset: usize,
}

#[allow(clippy::suspicious_operation_groupings)]
impl DerivedConstants {
    pub fn new(arity: usize, full_rounds: usize, partial_rounds: usize) -> Self {
        let sparse_count = partial_rounds;
        let width = arity + 1;
        let sparse_matrix_size = 2 * width - 1;
        let rk_count = width * full_rounds + partial_rounds;
        let full_half = full_rounds / 2;
        let sparse_offset = full_half - 1;
        let constants_elements =
            1 + rk_count + (width * width) + (width * width) + (sparse_count * sparse_matrix_size);

        let matrix_size = width * width;
        let mut offset = 0;
        let domain_tag_offset = offset;
        offset += 1;
        let round_keys_offset = offset;
        offset += rk_count;
        let mds_matrix_offset = offset;
        offset += matrix_size;
        let pre_sparse_matrix_offset = offset;
        offset += matrix_size;
        let sparse_matrixes_offset = offset;

        let w_hat_offset = 0;
        let v_rest_offset = width;

        Self {
            arity,
            partial_rounds,
            width,
            sparse_matrix_size,
            full_half,
            sparse_offset,
            constants_elements,
            domain_tag_offset,
            round_keys_offset,
            mds_matrix_offset,
            pre_sparse_matrix_offset,
            sparse_matrixes_offset,
            w_hat_offset,
            v_rest_offset,
        }
    }
}

/// Code that is the same, independent of the arity.
fn shared(field: &str) -> String {
    format!(include_str!("cl/shared.cl"), field = field)
}

fn poseidon_source(field: &str, strength: &str, derived_constants: &DerivedConstants) -> String {
    let DerivedConstants {
        arity,
        partial_rounds,
        width,
        sparse_matrix_size,
        full_half,
        sparse_offset,
        constants_elements,
        domain_tag_offset,
        round_keys_offset,
        mds_matrix_offset,
        pre_sparse_matrix_offset,
        sparse_matrixes_offset,
        w_hat_offset,
        v_rest_offset,
    } = derived_constants;

    format!(
        include_str!("cl/poseidon.cl"),
        arity = arity,
        field = field,
        partial_rounds = partial_rounds,
        width = width,
        sparse_matrix_size = sparse_matrix_size,
        full_half = full_half,
        sparse_offset = sparse_offset,
        constants_elements = constants_elements,
        domain_tag_offset = domain_tag_offset,
        round_keys_offset = round_keys_offset,
        mds_matrix_offset = mds_matrix_offset,
        pre_sparse_matrix_offset = pre_sparse_matrix_offset,
        w_hat_offset = w_hat_offset,
        v_rest_offset = v_rest_offset,
        sparse_matrixes_offset = sparse_matrixes_offset,
        strength = strength,
    )
}

/// Returns the kernels source code for the given constants.
///
/// The constants can be generated based on the the arity and the strength. The `derived_constants`
/// parameter is a list of tuples, where the first element contains the standard strength
/// parameters, the second element is the strengthed, and the third is the "even-partial" strength.
/// parameters, the second element is the strengthed one.
fn generate_program_from_constants<F>(
    derived_constants: &[(DerivedConstants, DerivedConstants, DerivedConstants)],
) -> String
    where
        F: GpuName + 'static,
{
    let mut source = vec![shared(&F::name())];
    for (standard, _strengthened, _even_partial) in derived_constants {
        source.push(poseidon_source(&F::name(), "standard", standard));
        #[cfg(feature = "strengthened")]
        source.push(poseidon_source(&F::name(), "strengthened", &_strengthened));
        #[cfg(feature = "even-partial")]
        source.push(poseidon_source(&F::name(), "even_partial", _even_partial));
    }
    source.join("\n")
}

/// Returns derived constants based on the arity.
///
/// It returns all three Poseidon constants: standard, strengthened, and even-partial.
fn derive_constants(arity: usize) -> (DerivedConstants, DerivedConstants, DerivedConstants) {
    let (full_standard, partial_standard) = round_numbers_base(arity);
    let (full_strengthened, partial_strengthened) = round_numbers_strengthened(arity);
    let (full_even, partial_even) = round_numbers_even_partial(arity);
    (
        DerivedConstants::new(arity, full_standard, partial_standard),
        DerivedConstants::new(arity, full_strengthened, partial_strengthened),
        DerivedConstants::new(arity, full_even, partial_even),
    )
}

/// Returns the kernels source based on the set feature flags.
///
/// Kernels for certain arities are enabled by feature flags.
pub fn generate_program() -> SourceBuilder {
    #[cfg(any(feature = "bls", feature = "pasta"))]
        let derived_constants = vec![
        #[cfg(feature = "arity2")]
            derive_constants(2),
        #[cfg(feature = "arity4")]
            derive_constants(4),
        #[cfg(feature = "arity8")]
            derive_constants(8),
        #[cfg(feature = "arity11")]
            derive_constants(11),
        #[cfg(feature = "arity16")]
            derive_constants(16),
        #[cfg(feature = "arity24")]
            derive_constants(24),
        #[cfg(feature = "arity36")]
            derive_constants(36),
    ];

    let source_builder = SourceBuilder::new();

    #[cfg(feature = "bls")]
        let source_builder = source_builder
        .add_field::<Fr>()
        .append_source(generate_program_from_constants::<Fr>(&derived_constants));
    #[cfg(feature = "pasta")]
        let source_builder = source_builder
        .add_field::<Fp>()
        .add_field::<Fv>()
        .append_source(generate_program_from_constants::<Fp>(&derived_constants))
        .append_source(generate_program_from_constants::<Fv>(&derived_constants));

    source_builder
}
