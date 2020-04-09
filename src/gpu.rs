use crate::error::Error;
use ff::{Field, PrimeField, PrimeFieldDecodingError, PrimeFieldRepr, ScalarEngine};
use paired::bls12_381::{Fr, FrRepr};
use triton::FutharkContext;

pub fn test_gpu() -> Result<(), triton::Error> {
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

    dbg!(&fr);

    fr
}

fn simple11(n: i32) -> Result<Vec<Fr>, Error> {
    let mut ctx = FutharkContext::new();

    let res_arr = ctx.simple11(n).map_err(|_| Error::GPUError)?;
    let (vec, shape) = &res_arr.to_vec();
    let n = shape[0];
    let chunk_size = shape[1] as usize;

    dbg!(&vec, &chunk_size);

    let hashes = vec
        .chunks(chunk_size)
        .map(|x| u64s_into_fr(dbg!(x)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::DecodingError);

    hashes
}

// pub fn hash_preimage<Arity>(&[E::Fr]) -> E::Fr where Arity: Unsigned, E: ScalarEngine {
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asdf() {
        let res = simple11(5).unwrap();

        // TODO: This is getting a DecodingError because the last element returned is not in the field.
        // This is probably because we need to perform a final Montgomery reduction before returning from triton.
        // When adding that, also add the initial conversion to Montgomery representation.

        assert_eq!(5, res.len());
    }
}
