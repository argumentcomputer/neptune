use ff::{Field, ScalarEngine};

use crate::matrix;
use crate::matrix::{
    apply_matrix, invert, is_identity, is_invertible, is_square, mat_mul, minor, Matrix, Scalar,
};
use crate::scalar_from_u64;

#[derive(Clone, Debug, PartialEq)]
pub struct MDSMatrices<E: ScalarEngine> {
    pub m: Matrix<Scalar<E>>,
    pub m_inv: Matrix<Scalar<E>>,
    pub m_hat: Matrix<Scalar<E>>,
    pub m_hat_inv: Matrix<Scalar<E>>,
    pub m_prime: Matrix<Scalar<E>>,
    pub m_double_prime: Matrix<Scalar<E>>,
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

    MDSMatrices {
        m,
        m_inv,
        m_hat,
        m_hat_inv,
        m_prime,
        m_double_prime,
    }
}

/// A `SparseMatrix` is specifically one of the form of M''.
/// This means its first row and column are each dense, and the interior matrix
/// (minor to the element in both the row and column) is the identity.
/// We will pluralize this compact structure `sparse_matrixes` to distinguish from `sparse_matrices` from which they are created.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix<E: ScalarEngine> {
    /// `w_hat` is the first column of the M'' matrix. It will be directly multiplied (scalar product) with a row of state elements.
    pub w_hat: Vec<Scalar<E>>,
    /// `v_rest` contains all but the first (already included in `w_hat`).
    pub v_rest: Vec<Scalar<E>>,
}

impl<E: ScalarEngine> SparseMatrix<E> {
    pub fn new(m_double_prime: Matrix<Scalar<E>>) -> Self {
        assert!(Self::is_sparse_matrix(&m_double_prime));
        let size = matrix::rows(&m_double_prime);

        let w_hat = (0..size).map(|i| m_double_prime[i][0]).collect::<Vec<_>>();
        let v_rest = m_double_prime[0][1..].to_vec();

        Self { w_hat, v_rest }
    }

    pub fn is_sparse_matrix(m: &Matrix<Scalar<E>>) -> bool {
        is_square(&m) && is_identity::<E>(&minor::<E>(&m, 0, 0))
    }

    pub fn size(&self) -> usize {
        self.w_hat.len()
    }

    pub fn to_matrix(&self) -> Matrix<Scalar<E>> {
        let mut m = matrix::make_identity::<E>(self.size());
        for (j, elt) in self.w_hat.iter().enumerate() {
            m[j][0] = *elt;
        }
        for (i, elt) in self.v_rest.iter().enumerate() {
            m[0][i + 1] = *elt;
        }
        m
    }
}

// - Having effectively moved the round-key additions into the S-boxes, refactor MDS matrices used for partial-round mix layer to use sparse matrices.
// - This requires using a different (sparse) matrix at each partial round, rather than the same dense matrix at each.
//   - The MDS matrix, M, for each such round, starting from the last, is factored into two components, such that M' x M'' = M.
//   - M'' is sparse and replaces M for the round.
//   - The previous layer's M is then replaced by M x M' = M*.
//   - M* is likewise factored into M*' and M*'', and the process continues.
pub fn factor_to_sparse_matrixes<E: ScalarEngine>(
    base_matrix: Matrix<Scalar<E>>,
    n: usize,
) -> (Matrix<Scalar<E>>, Vec<SparseMatrix<E>>) {
    let (pre_sparse, sparse_matrices) = factor_to_sparse_matrices::<E>(base_matrix, n);
    let sparse_matrixes = sparse_matrices
        .iter()
        .map(|m| SparseMatrix::<E>::new(m.to_vec()))
        .collect::<Vec<_>>();

    (pre_sparse, sparse_matrixes)
}

pub fn factor_to_sparse_matrices<E: ScalarEngine>(
    base_matrix: Matrix<Scalar<E>>,
    n: usize,
) -> (Matrix<Scalar<E>>, Vec<Matrix<Scalar<E>>>) {
    let (pre_sparse, mut all) =
        (0..n).fold((base_matrix.clone(), Vec::new()), |(curr, mut acc), _| {
            let derived = derive_mds_matrices::<E>(curr);
            acc.push(derived.m_double_prime);
            let new = mat_mul::<E>(&base_matrix, &derived.m_prime).unwrap();
            (new, acc)
        });
    all.reverse();
    (pre_sparse, all)
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
        let x = scalar_from_u64((i) as u64);
        let y = scalar_from_u64((i + t) as u64);
        xs.push(x);
        ys.push(y);
    }

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

    // To ensure correctness, we would check all sub-matrices for invertibility. Meanwhile, this is a simple sanity check.
    assert!(is_invertible::<E>(&matrix));

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
    use bellperson::bls::{Bls12, Fr};
    use matrix::left_apply_matrix;
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
    }

    #[test]
    fn test_factor_to_sparse_matrices() {
        for width in 3..9 {
            test_factor_to_sparse_matrices_aux(width, 3);
        }
    }

    fn test_factor_to_sparse_matrices_aux(width: usize, n: usize) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let m = generate_mds::<Bls12>(width);
        let m2 = m.clone();

        let (pre_sparse, sparse) = factor_to_sparse_matrices::<Bls12>(m, n);
        assert_eq!(n, sparse.len());

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

        let actual = sparse.iter().chain(&[pre_sparse]).zip(&round_keys).fold(
            initial.clone(),
            |mut acc, (m, rk)| {
                apply_matrix::<Bls12>(&m, &acc);
                quintic_s_box::<Bls12>(&mut acc[0], None, Some(&rk));
                acc
            },
        );
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_factor_to_sparse_matrixes() {
        for width in 3..9 {
            test_factor_to_sparse_matrixes_aux(width, 3);
        }
    }

    fn test_factor_to_sparse_matrixes_aux(width: usize, n: usize) {
        let m = generate_mds::<Bls12>(width);
        let m2 = m.clone();

        let (pre_sparse, sparse_matrices) = factor_to_sparse_matrices::<Bls12>(m, n);
        assert_eq!(n, sparse_matrices.len());

        let (pre_sparse2, sparse_matrixes) = factor_to_sparse_matrixes::<Bls12>(m2, n);

        assert_eq!(pre_sparse, pre_sparse2);

        let matrices_again = sparse_matrixes
            .iter()
            .map(|m| m.to_matrix())
            .collect::<Vec<_>>();
        dbg!(&sparse_matrixes, &sparse_matrices);

        let _ = sparse_matrices
            .iter()
            .zip(matrices_again.iter())
            .map(|(a, b)| {
                dbg!(&a, &b);
                assert_eq!(a, b)
            })
            .collect::<Vec<_>>();

        assert_eq!(sparse_matrices, matrices_again);
    }
}
