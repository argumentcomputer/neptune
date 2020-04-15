use crate::error::Error;
use crate::poseidon::PoseidonConstants;
use ff::{PrimeField, PrimeFieldDecodingError};
use generic_array::typenum;
use paired::bls12_381::{Bls12, Fr, FrRepr};
use std::ops::Add;
use triton::FutharkContext;
use triton::{Array_u64_1d, Array_u64_2d, Array_u64_3d};
use typenum::bit::B1;
use typenum::{UInt, UTerm, Unsigned, U2};

struct GPUConstants<Arity>(PoseidonConstants<Bls12, Arity>)
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>>;

impl<Arity> GPUConstants<Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>>,
{
    fn arity_tag(&self, ctx: &FutharkContext) -> Result<Array_u64_1d, Error> {
        let arity_tag = self.0.arity_tag;
        array_u64_1d_from_fr(ctx, arity_tag)
    }

    fn round_keys(&self, ctx: &FutharkContext) -> Result<Array_u64_2d, Error> {
        let round_keys = &self.0.compressed_round_constants;
        array_u64_2d_from_frs(ctx, &round_keys)
    }

    fn mds_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let matrix = &self.0.mds_matrices.m;

        array_u64_3d_from_frs_2d(ctx, matrix)
    }

    fn pre_sparse_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let pre_sparse_matrix = &self.0.pre_sparse_matrix;

        array_u64_3d_from_frs_2d(ctx, pre_sparse_matrix)
    }

    fn sparse_matrixes(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let sparse_matrixes = &self.0.sparse_matrixes;

        let frs_2d: Vec<Vec<Fr>> = sparse_matrixes
            .iter()
            .map(|m| {
                let mut x = m.w_hat.clone();
                x.extend(m.v_rest.clone());
                x.into_iter().collect()
            })
            .collect();

        array_u64_3d_from_frs_2d(ctx, &frs_2d)
    }
}

fn frs_to_u64s(frs: &[Fr]) -> Vec<u64> {
    let mut res = vec![u64::default(); frs.len() * 4];
    for (src, dest) in frs.iter().zip(res.chunks_mut(4)) {
        dest.copy_from_slice(&src.into_repr().0);
    }
    res
}

fn frs_2d_to_u64s(frs_2d: &[Vec<Fr>]) -> Vec<u64> {
    frs_2d
        .iter()
        .flat_map(|row| frs_to_u64s(row).into_iter())
        .collect()
}

fn array_u64_1d_from_fr(ctx: &FutharkContext, fr: Fr) -> Result<Array_u64_1d, Error> {
    Array_u64_1d::from_vec(*ctx, &fr.into_repr().0, &[4, 1])
        .map_err(|e| Error::Other(format!("error converting Fr: {:?}", e).to_string()))
}

fn array_u64_2d_from_frs(ctx: &FutharkContext, frs: &[Fr]) -> Result<Array_u64_2d, Error> {
    let u64s = frs_to_u64s(frs);

    let d2 = 4;
    let d1 = u64s.len() as i64 / d2;
    let dim = [d1, d2];

    Array_u64_2d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs: {:?}", e).to_string()))
}

fn array_u64_3d_from_frs_2d(
    ctx: &FutharkContext,
    frs_2d: &[Vec<Fr>],
) -> Result<Array_u64_3d, Error> {
    let u64s = frs_2d_to_u64s(frs_2d);

    let mut l = u64s.len() as i64;
    let d1 = 4; // One Fr is 4 x u64.
    l /= d1;

    let d2 = frs_2d[0].len() as i64;
    assert!(
        frs_2d.iter().all(|x| x.len() == d2 as usize),
        "Frs must be grouped uniformly"
    );
    l /= d2;

    let d3 = l as i64;
    let dim = [d3, d2, d1];

    Array_u64_3d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs 2d: {:?}", e).to_string()))
}

pub fn exercise_gpu() -> Result<(), triton::Error> {
    let mut ctx = FutharkContext::new();

    let res_arr = ctx.simple11(5)?;
    let (vec, shape) = &res_arr.to_vec();
    let n = shape[0];
    let chunk_size = shape[1] as usize;

    assert_eq!(2, shape.len());

    for (i, chunk) in vec.chunks(chunk_size).enumerate() {
        print!("res {} of {}: ", i, n);
        print!("[");
        for elt in chunk.iter() {
            print!("{}, ", elt);
        }
        println!("]");
    }

    Ok(())
}

pub fn u64s_into_fr(limbs: &[u64]) -> Result<Fr, PrimeFieldDecodingError> {
    assert_eq!(limbs.len(), 4);
    let mut limb_arr = [0; 4];
    limb_arr.copy_from_slice(&limbs[..]);
    let repr = FrRepr(limb_arr);
    let fr = Fr::from_repr(repr);

    fr
}

fn unpack_fr_array(vec_shape: (Vec<u64>, &[i64])) -> Result<Vec<Fr>, Error> {
    let (vec, shape) = vec_shape;
    let chunk_size = shape[0] as usize;

    vec.chunks(chunk_size)
        .map(|x| u64s_into_fr(x))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::DecodingError)
}

fn simple11(n: i32) -> Result<Vec<Fr>, Error> {
    let mut ctx = FutharkContext::new();

    let res_arr = ctx
        .simple11(n)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;
    let (vec, shape) = &res_arr.to_vec();
    unpack_fr_array((vec.to_vec(), shape.as_slice()))
}

pub fn hash_binary(preimage: [Fr; 2]) -> Result<Fr, Error>
where
{
    let mut ctx = FutharkContext::new();
    let constants = GPUConstants(PoseidonConstants::<Bls12, U2>::new());
    let state = ctx
        .init2(
            constants.arity_tag(&ctx)?,
            constants.round_keys(&ctx)?,
            constants.mds_matrix(&ctx)?,
            constants.pre_sparse_matrix(&ctx)?,
            constants.sparse_matrixes(&ctx)?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let preimage_u64s = frs_to_u64s(&preimage);

    let (vec, shape) = ctx
        .hash2(
            state,
            Array_u64_1d::from_vec(ctx, &preimage_u64s, &[8, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?
        .to_vec();

    unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])
}

type p2_state = triton::FutharkOpaque9B6Df2Ea;

fn test_binary_get_state(ctx: &mut FutharkContext) -> Result<p2_state, Error> {
    let constants = GPUConstants(PoseidonConstants::<Bls12, U2>::new());
    let state = ctx
        .init2(
            constants.arity_tag(&ctx)?,
            constants.round_keys(&ctx)?,
            constants.mds_matrix(&ctx)?,
            constants.pre_sparse_matrix(&ctx)?,
            constants.sparse_matrixes(&ctx)?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    Ok(state)
}

fn test_binary(ctx: &mut FutharkContext, preimage: [Fr; 3]) -> Result<Fr, Error>
where
{
    let preimage_u64s = frs_to_u64s(&preimage);

    let (vec, shape) = ctx
        .test2(
            Array_u64_1d::from_vec(*ctx, &preimage_u64s, &[preimage.len() as i64 * 4, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?
        .to_vec();

    unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::poseidon;
    use ff::Field;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_simple11() {
        let res = simple11(5).unwrap();
        // This just tests that the expected number of values were returned, for now.
        // TODO: verify the results.
        assert_eq!(5, res.len());
    }

    #[test]
    fn test_hash_binary() {
        let a = Fr::zero();
        let b = Fr::zero();
        let preimage = [a, b];
        let cpu_res = poseidon::<Bls12, U2>(&preimage);
        let gpu_res = hash_binary(preimage).unwrap();

        assert_eq!(
            cpu_res, gpu_res,
            "GPU result ({:?}) differed from CPU ({:?}) result).",
            gpu_res, cpu_res
        );
    }

    #[test]
    fn test_test_binary() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let mut ctx = FutharkContext::new();

        for i in 0..100000 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let c = Fr::random(&mut rng);
            let input = [a, b, c];
            let gpu_res = test_binary(&mut ctx, input).unwrap();

            let mut cpu_res = a.clone();
            cpu_res.mul_assign(&b);
            let cpu_mul_res = cpu_res.clone();
            cpu_res.add_assign(&c);

            dbg!(&a, &b, &c, &cpu_mul_res, &cpu_res, &gpu_res);

            assert_eq!(
                cpu_res, gpu_res,
                "GPU result ({:?}) differed from CPU ({:?}) result).",
                gpu_res, cpu_res
            );
        }
    }

    #[test]
    fn test_debug_init2() {
        let mut ctx = FutharkContext::new();
        let constants = GPUConstants(PoseidonConstants::<Bls12, U2>::new()); // TODO: make this a method.
        let state = ctx
            .debug_init2(
                constants.arity_tag(&ctx).unwrap(),
                constants.round_keys(&ctx).unwrap(),
                constants.mds_matrix(&ctx).unwrap(),
                constants.pre_sparse_matrix(&ctx).unwrap(),
                constants.sparse_matrixes(&ctx).unwrap(),
            )
            .map_err(|e| Error::GPUError(format!("{:?}", e)))
            .unwrap();
    }
}
