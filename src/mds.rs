use ff::{Field, ScalarEngine};

use crate::matrix::{apply_matrix, invert, is_invertible, mat_mul, minor, Matrix, Scalar};
use crate::scalar_from_u64;

#[derive(Clone, Debug, PartialEq)]
pub struct MDSMatrices<E: ScalarEngine> {
    pub m: Matrix<Scalar<E>>,
    pub m_inv: Matrix<Scalar<E>>,
    pub m_hat: Matrix<Scalar<E>>,
    pub m_hat_inv: Matrix<Scalar<E>>,
    pub m_prime: Matrix<Scalar<E>>,
    pub m_double_prime: Matrix<Scalar<E>>,
    pub m_prime_inv: Matrix<Scalar<E>>,
    pub m_double_prime_inv: Matrix<Scalar<E>>,
}

pub fn create_mds_matrices<'a, E: ScalarEngine>(t: usize) -> MDSMatrices<E> {
    let m = generate_mds::<E>(t);
    derive_mds_matrices(m)
}

pub fn derive_mds_matrices<'a, E: ScalarEngine>(m: Matrix<Scalar<E>>) -> MDSMatrices<E> {
    let m_inv = invert::<E>(&m).unwrap(); // m is MDS so invertible.
    let m_hat = minor::<E>(&m, 0, 0);
    let m_hat_inv = invert::<E>(&m_hat).unwrap(); // If this returns None, then `mds_matrix` was not correctly generated.
    let m_prime = make_prime::<E>(&m);
    let m_double_prime = make_double_prime::<E>(&m, &m_hat_inv);
    let m_prime_inv = invert::<E>(&m_prime).unwrap();
    let m_double_prime_inv = invert::<E>(&m_double_prime).unwrap();

    MDSMatrices {
        m,
        m_inv,
        m_hat,
        m_hat_inv,
        m_prime,
        m_double_prime,
        m_prime_inv,
        m_double_prime_inv,
    }
}

pub fn factor_to_sparse_matrices<E: ScalarEngine>(
    base_matrix: Matrix<Scalar<E>>,
    n: usize,
) -> Vec<Matrix<Scalar<E>>> {
    let (last, mut all) = (0..n).fold((base_matrix.clone(), Vec::new()), |(curr, mut acc), _| {
        let derived = derive_mds_matrices::<E>(curr);
        acc.push(derived.m_double_prime);
        let new = mat_mul::<E>(&base_matrix, &derived.m_prime).unwrap();
        (new, acc)
    });
    all.push(last);
    all.reverse();
    all
}

fn generate_mds<E: ScalarEngine>(t: usize) -> Matrix<Scalar<E>> {
    // Source: https://github.com/dusk-network/dusk-poseidon-merkle/commit/776c37734ea2e71bb608ce4bc58fdb5f208112a7#diff-2eee9b20fb23edcc0bf84b14167cbfdc
    let mut matrix: Vec<Vec<E::Fr>> = Vec::with_capacity(t);
    let mut xs: Vec<E::Fr> = Vec::with_capacity(t);
    let mut ys: Vec<E::Fr> = Vec::with_capacity(t);

    // Generate x and y values deterministically for the cauchy matrix
    // where x[i] != y[i] to allow the values to be inverted
    // and there are no duplicates in the x vector or y vector, so that the determinant is always non-zero
    // [a b]
    // [c d]
    // det(M) = (ad - bc) ; if a == b and c == d => det(M) =0
    // For an MDS matrix, every possible mxm submatrix, must have det(M) != 0
    for i in 0..t {
        let x = scalar_from_u64::<E>((i) as u64);
        let y = scalar_from_u64::<E>((i + t) as u64);
        xs.push(x);
        ys.push(y);
    }

    // for i in 0..t {
    //     let mut row: Vec<Scalar> = Vec::with_capacity(t);
    //     for j in 0..t {
    //         // Generate the entry at (i,j)
    //         let entry = (xs[i] + ys[j]).invert();
    //         row.insert(j, entry);
    //     }
    //     matrix.push(row);
    // }
    // Adapted from source above to use E::Fr.
    for i in 0..t {
        let mut row: Vec<E::Fr> = Vec::with_capacity(t);
        for j in 0..t {
            // Generate the entry at (i,j)
            let mut tmp = xs[i];
            tmp.add_assign(&ys[j]);
            let entry = tmp.inverse().unwrap();
            row.insert(j, entry);
        }
        matrix.push(row);
    }

    assert!(is_invertible::<E>(&matrix));
    // To ensure correctness, we would check all sub-matrices for invertibility. Meanwhile, this is a simple sanity check.
    matrix
}

fn make_prime<E: ScalarEngine>(m: &Matrix<Scalar<E>>) -> Matrix<Scalar<E>> {
    m.iter()
        .enumerate()
        .map(|(i, row)| match i {
            0 => {
                let mut new_row = vec![Scalar::<E>::zero(); row.len()];
                new_row[0] = Scalar::<E>::one();
                new_row
            }
            _ => {
                let mut new_row = vec![Scalar::<E>::zero(); row.len()];
                new_row[1..].copy_from_slice(&row[1..]);
                new_row
            }
        })
        .collect()
}

fn make_double_prime<E: ScalarEngine>(
    m: &Matrix<Scalar<E>>,
    m_hat_inv: &Matrix<Scalar<E>>,
) -> Matrix<Scalar<E>> {
    let (v, w) = make_v_w::<E>(m);
    let w_hat = apply_matrix::<E>(m_hat_inv, &w);

    m.iter()
        .enumerate()
        .map(|(i, row)| match i {
            0 => {
                let mut new_row = Vec::with_capacity(row.len());
                new_row.push(row[0]);
                new_row.extend(&v);
                new_row
            }
            _ => {
                let mut new_row = vec![Scalar::<E>::zero(); row.len()];
                new_row[0] = w_hat[i - 1];
                new_row[i] = Scalar::<E>::one();
                new_row
            }
        })
        .collect()
}

fn make_v_w<E: ScalarEngine>(m: &Matrix<Scalar<E>>) -> (Vec<Scalar<E>>, Vec<Scalar<E>>) {
    let v = m[0][1..].to_vec();
    let mut w = Vec::new();
    for i in 1..m.len() {
        w.push(m[i][0]);
    }

    (v, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use matrix::left_apply_matrix;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_mds_matrices_creation() {
        for i in 2..5 {
            test_mds_matrices_creation_aux(i);
        }
    }

    fn test_mds_matrices_creation_aux(width: usize) {
        let MDSMatrices {
            m,
            m_inv,
            m_hat,
            m_hat_inv: _,
            m_prime,
            m_double_prime,
            m_prime_inv,
            m_double_prime_inv: _,
        } = create_mds_matrices::<Bls12>(width);

        for i in 0..m_hat.len() {
            for j in 0..m_hat[i].len() {
                assert_eq!(m[i + 1][j + 1], m_hat[i][j], "MDS minor has wrong value.");
            }
        }

        // M^-1 x M = I
        assert!(matrix::is_identity::<Bls12>(
            &matrix::mat_mul::<Bls12>(&m_inv, &m).unwrap()
        ));

        // M' x M'' = I
        assert_eq!(
            m,
            matrix::mat_mul::<Bls12>(&m_prime, &m_double_prime).unwrap()
        );

        //
        assert!(matrix::is_identity::<Bls12>(
            &matrix::mat_mul::<Bls12>(&m_prime_inv, &m_prime).unwrap()
        ));
    }

    #[test]
    fn test_swapping() {
        test_swapping_aux(3);
    }

    fn test_swapping_aux(width: usize) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let MDSMatrices {
            m,
            m_inv: _,
            m_hat: _,
            m_hat_inv: _,
            m_prime,
            m_double_prime,
            m_prime_inv,
            m_double_prime_inv: _,
        } = create_mds_matrices::<Bls12>(width);

        let mut base = Vec::with_capacity(width);
        for _ in 0..width {
            base.push(Fr::random(&mut rng));
        }

        let mut x = base.clone();
        x[0] = Fr::random(&mut rng);

        let mut y = base.clone();
        y[0] = Fr::random(&mut rng);

        let qx = apply_matrix::<Bls12>(&m_prime, &x);
        let qy = apply_matrix::<Bls12>(&m_prime, &y);
        assert_eq!(qx[0], x[0]);
        assert_eq!(qy[0], y[0]);
        assert_eq!(qx[1..], qy[1..]);

        let zx = apply_matrix::<Bls12>(&m_prime_inv, &x);
        let zy = apply_matrix::<Bls12>(&m_prime_inv, &y);

        assert_eq!(zx[0], x[0]);
        assert_eq!(zy[0], y[0]);
        assert_eq!(zx[1..], zy[1..]);

        let mx = left_apply_matrix::<Bls12>(&m, &x);
        let m1_m2_x =
            left_apply_matrix::<Bls12>(&m_prime, &left_apply_matrix::<Bls12>(&m_double_prime, &x));
        assert_eq!(mx, m1_m2_x);

        let xm = apply_matrix::<Bls12>(&m, &x);
        let x_m1_m2 = apply_matrix::<Bls12>(&m_double_prime, &apply_matrix::<Bls12>(&m_prime, &x));
        assert_eq!(xm, x_m1_m2);

        let mut rk = Vec::with_capacity(width);
        for _ in 0..width {
            rk.push(Fr::random(&mut rng));
        }

        // let simple = vec_add::<Bls12>(&apply_matrix::<Bls12>(&m, &base), &rk);
        // let rk_inv = apply_matrix::<Bls12>(&m_prime_inv, &rk);
        // let shifted = vec_add::<Bls12>(&base, &rk_inv);
        // let alt = apply_matrix::<Bls12>(&m_double_prime, &shifted);

        dbg!(&m);
        //        assert_eq!(simple, alt);
    }

    #[test]
    fn test_factor_to_sparse_matrices() {
        test_factor_to_sparse_matrices_aux(3, 3);
        test_factor_to_sparse_matrices_aux(4, 3);
        test_factor_to_sparse_matrices_aux(5, 3);
    }

    fn test_factor_to_sparse_matrices_aux(width: usize, n: usize) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let m = generate_mds::<Bls12>(width);
        let m2 = m.clone();

        let sparse = factor_to_sparse_matrices::<Bls12>(m, n);
        assert_eq!(n + 1, sparse.len());

        let mut initial = Vec::with_capacity(width);
        for _ in 0..width {
            initial.push(Fr::random(&mut rng));
        }

        let mut round_keys = Vec::with_capacity(width);
        for _ in 0..n {
            round_keys.push(Fr::random(&mut rng));
        }

        let expected = std::iter::repeat(m2).take(n).zip(&round_keys).fold(
            initial.clone(),
            |mut acc, (m, rk)| {
                apply_matrix::<Bls12>(&m, &acc);
                quintic_s_box::<Bls12>(&mut acc[0], None, Some(&rk));
                acc
            },
        );

        let actual = sparse
            .iter()
            .zip(&round_keys)
            .fold(initial.clone(), |mut acc, (m, rk)| {
                apply_matrix::<Bls12>(&m, &acc);
                quintic_s_box::<Bls12>(&mut acc[0], None, Some(&rk));
                acc
            });
        assert_eq!(expected, actual);
    }
}
