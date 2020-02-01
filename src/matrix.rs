use ff::{Field, ScalarEngine};

/// Matrix functions here are, at least for now, quick and dirty — intended only to support precomputation of poseidon optimization.

/// Matrix represented as a Vec of rows, so that m[i][j] represents the jth column of the ith row in Matrix, m.
type Matrix<T> = Vec<Vec<T>>;
type Scalar<E> = <E as ScalarEngine>::Fr;

fn rows<T>(matrix: &Matrix<T>) -> usize {
    matrix.len()
}

/// Panics if `matrix` is not actually a matrix. So only use any of these functions on well-formed data.
/// Only use during constant calculation on matrices known to have been constructed correctly.
fn columns<T>(matrix: &Matrix<T>) -> usize {
    if matrix.len() > 0 {
        let length = matrix[0].len();
        for i in 1..rows(matrix) {
            if matrix[i].len() != length {
                panic!("not a matrix");
            }
        }
        length
    } else {
        0
    }
}

/// This is very inefficient as matrices grow. However, we only need it for preprocessing constants,
/// and it is (for now) sufficient for the relatively small widths we need to support.
/// TODO: Use a more efficient method.
pub(crate) fn invert<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Option<Matrix<Scalar<E>>> {
    let cofactor_matrix = cofactor_matrix::<E>(matrix);
    let determinant = determinant_with_cofactor_matrix::<E>(matrix, &cofactor_matrix);
    let adjugate = transpose::<E>(&cofactor_matrix);

    Some(scalar_mul::<E>(determinant.inverse()?, &adjugate))
}

pub(crate) fn is_invertible<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> bool {
    is_square(matrix) && determinant::<E>(matrix) != Scalar::<E>::zero()
}

fn scalar_mul<E: ScalarEngine>(scalar: Scalar<E>, matrix: &Matrix<Scalar<E>>) -> Matrix<Scalar<E>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|val| {
                    let mut prod = scalar.clone();
                    prod.mul_assign(val);
                    prod
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn mat_mul<E: ScalarEngine>(
    a: &Matrix<Scalar<E>>,
    b: &Matrix<Scalar<E>>,
) -> Option<Matrix<Scalar<E>>> {
    if rows(a) != columns(b) {
        return None;
    };

    let b_t = transpose::<E>(b);

    let mut res = Vec::with_capacity(rows(a));
    for i in 0..rows(a) {
        let mut row = Vec::with_capacity(columns(b));
        for j in 0..columns(b) {
            row.push(vec_mul::<E>(&a[i], &b_t[j]));
        }
        res.push(row);
    }

    Some(res)
}

fn vec_mul<E: ScalarEngine>(a: &Vec<Scalar<E>>, b: &Vec<Scalar<E>>) -> Scalar<E> {
    a.iter()
        .zip(b)
        .fold(Scalar::<E>::zero(), |mut acc, (v1, v2)| {
            let mut tmp = v1.clone();
            tmp.mul_assign(&v2);
            acc.add_assign(&tmp);
            acc
        })
}

fn vec_add<E: ScalarEngine>(a: &Vec<Scalar<E>>, b: &Vec<Scalar<E>>) -> Vec<Scalar<E>> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let mut res = a.clone();
            res.add_assign(b);
            res
        })
        .collect::<Vec<_>>()
}

/// Multiply a square matrix by a vector of same size: MV where V is considered a row vector.
pub fn apply_matrix<E: ScalarEngine>(m: &Matrix<Scalar<E>>, v: &Vec<Scalar<E>>) -> Vec<Scalar<E>> {
    assert!(is_square(m), "Only square matrix can be applied to vector.");
    assert_eq!(
        rows(m),
        v.len(),
        "Matrix can only be applied to vector of same size."
    );

    let mut result: Vec<Scalar<E>> = vec![Scalar::<E>::zero(); v.len()];

    for (result, row) in result.iter_mut().zip(m.iter()) {
        for (mat_val, vec_val) in row.iter().zip(v) {
            let mut tmp = *mat_val;
            tmp.mul_assign(vec_val);
            result.add_assign(&tmp);
        }
    }
    result
}

fn transpose<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Matrix<Scalar<E>> {
    let size = rows(matrix);
    let mut new = Vec::with_capacity(size);
    for j in 0..size {
        let mut row = Vec::with_capacity(size);
        for i in 0..size {
            row.push(matrix[i][j])
        }
        new.push(row);
    }
    new
}

fn is_identity<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> bool {
    let one = Scalar::<E>::one();
    let zero = Scalar::<E>::zero();

    for i in 0..rows(matrix) {
        for j in 0..columns(matrix) {
            let kronecker = matrix[i][j] == if i == j { one } else { zero };
            if !kronecker {
                return false;
            };
        }
    }
    true
}

fn is_square<T>(matrix: &Matrix<T>) -> bool {
    rows(matrix) == columns(matrix)
}

pub fn determinant<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Scalar<E> {
    let mut acc = Scalar::<E>::zero();

    for j in 0..columns(matrix) {
        let mut tmp = matrix[0][j];
        let cofactor = cofactor::<E>(&matrix, 0, j);
        tmp.mul_assign(&cofactor);
        acc.add_assign(&tmp);
    }
    acc
}

fn determinant_with_cofactor_matrix<E: ScalarEngine>(
    matrix: &Matrix<Scalar<E>>,
    cofactor_matrix: &Matrix<Scalar<E>>,
) -> Scalar<E> {
    matrix[0]
        .iter()
        .zip(&cofactor_matrix[0])
        .fold(Scalar::<E>::zero(), |mut acc, (a, b)| {
            let mut tmp = a.clone();
            tmp.mul_assign(&b);
            acc.add_assign(&tmp);
            acc
        })
}

fn cofactor_matrix<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Matrix<Scalar<E>> {
    assert!(is_square(matrix));
    let size = rows(matrix);
    let mut m = Vec::with_capacity(size);
    for i in 0..size {
        let mut row = Vec::with_capacity(size);
        for j in 0..size {
            row.push(cofactor::<E>(matrix, i, j));
        }
        m.push(row);
    }
    m
}

fn cofactor<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>, i: usize, j: usize) -> Scalar<E> {
    let minor_det = if rows(matrix) == 1 {
        Scalar::<E>::one()
    } else {
        let m = minor::<E>(matrix, i, j);
        determinant::<E>(&m)
    };

    let mut acc = Scalar::<E>::zero();
    if (i + j) % 2 == 0 {
        acc.add_assign(&minor_det);
    } else {
        acc.sub_assign(&minor_det);
    }
    acc
}

fn minor<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>, i: usize, j: usize) -> Matrix<Scalar<E>> {
    assert!(is_square(matrix));
    let size = rows(matrix);
    assert!(size > 0);
    let new_size = size - 1;
    let mut new: Matrix<Scalar<E>> = Vec::with_capacity(new_size);

    for ii in 0..size {
        if ii != i {
            let mut row = Vec::with_capacity(new_size);
            for jj in 0..size {
                if jj != j {
                    row.push(matrix[ii][jj]);
                }
            }
            new.push(row);
        }
    }
    assert!(is_square(&new));
    new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_from_u64;
    use paired::bls12_381::Bls12;

    #[test]
    fn test_minor() {
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let five = scalar_from_u64::<Bls12>(5);
        let six = scalar_from_u64::<Bls12>(6);
        let seven = scalar_from_u64::<Bls12>(7);
        let eight = scalar_from_u64::<Bls12>(8);
        let nine = scalar_from_u64::<Bls12>(9);

        let m = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        let cases = [
            (0, 0, vec![vec![five, six], vec![eight, nine]]),
            (0, 1, vec![vec![four, six], vec![seven, nine]]),
            (0, 2, vec![vec![four, five], vec![seven, eight]]),
            (1, 0, vec![vec![two, three], vec![eight, nine]]),
            (1, 1, vec![vec![one, three], vec![seven, nine]]),
            (1, 2, vec![vec![one, two], vec![seven, eight]]),
            (2, 0, vec![vec![two, three], vec![five, six]]),
            (2, 1, vec![vec![one, three], vec![four, six]]),
            (2, 2, vec![vec![one, two], vec![four, five]]),
        ];
        for (i, j, expected) in &cases {
            let result = minor::<Bls12>(&m, *i, *j);

            assert_eq!(*expected, result);
        }
        //        assert_eq!(, minor::<Bls12>(&vec![vec![one]], 0, 0));
    }

    #[test]
    fn test_determinant() {
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let five = scalar_from_u64::<Bls12>(5);
        let six = scalar_from_u64::<Bls12>(6);
        let seven = scalar_from_u64::<Bls12>(7);
        let eight = scalar_from_u64::<Bls12>(8);

        let m1 = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        let res1 = determinant::<Bls12>(&m1);
        // + 1 * (40 - 48)
        // - 2 * (32 - 42)
        // + 3 * (32 - 35)

        // + 1 * -8
        // - 2 * -10
        // + 3 * -3

        // = -8 + 20 - 9 = 3
        assert_eq!(three, res1);

        let m2 = vec![vec![one, two], vec![three, eight]];
        let res2 = determinant::<Bls12>(&m2);
        // 1 * 8 - 2 * 3
        // = 8 - 6 = 2

        assert_eq!(two, res2);
    }

    #[test]
    fn test_scalar_mul() {
        let zero = scalar_from_u64::<Bls12>(0);
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let six = scalar_from_u64::<Bls12>(6);

        let m = vec![vec![zero, one], vec![two, three]];
        let res = scalar_mul::<Bls12>(two, &m);

        let expected = vec![vec![zero, two], vec![four, six]];

        assert_eq!(expected, res);
    }

    #[test]
    fn test_vec_mul() {
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let five = scalar_from_u64::<Bls12>(5);
        let six = scalar_from_u64::<Bls12>(6);

        let a = vec![one, two, three];
        let b = vec![four, five, six];
        let res = vec_mul::<Bls12>(&a, &b);

        let expected = scalar_from_u64::<Bls12>(32);

        assert_eq!(expected, res);
    }

    #[test]
    fn test_transpose() {
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let five = scalar_from_u64::<Bls12>(5);
        let six = scalar_from_u64::<Bls12>(6);
        let seven = scalar_from_u64::<Bls12>(7);
        let eight = scalar_from_u64::<Bls12>(8);
        let nine = scalar_from_u64::<Bls12>(9);

        let m = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        let expected = vec![
            vec![one, four, seven],
            vec![two, five, eight],
            vec![three, six, nine],
        ];

        let res = transpose::<Bls12>(&m);
        assert_eq!(expected, res);
    }

    #[test]
    fn test_inverse() {
        let one = scalar_from_u64::<Bls12>(1);
        let two = scalar_from_u64::<Bls12>(2);
        let three = scalar_from_u64::<Bls12>(3);
        let four = scalar_from_u64::<Bls12>(4);
        let five = scalar_from_u64::<Bls12>(5);
        let six = scalar_from_u64::<Bls12>(6);
        let seven = scalar_from_u64::<Bls12>(7);
        let eight = scalar_from_u64::<Bls12>(8);
        let nine = scalar_from_u64::<Bls12>(9);

        let m = vec![
            vec![one, two, three],
            vec![four, three, six],
            vec![five, eight, seven],
        ];

        let m1 = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        assert!(!is_invertible::<Bls12>(&m1));
        assert!(is_invertible::<Bls12>(&m));

        let m_inv = invert::<Bls12>(&m).unwrap();

        let computed_identity = mat_mul::<Bls12>(&m, &m_inv).unwrap();

        assert!(is_identity::<Bls12>(&computed_identity));

        // S
        let some_vec = vec![six, five, four];

        // M^-1(S)
        let inverse_applied = super::apply_matrix::<Bls12>(&m_inv, &some_vec);

        // M(M^-1(S))
        let m_applied_after_inverse = super::apply_matrix::<Bls12>(&m, &inverse_applied);

        // S = M(M^-1(S))
        assert_eq!(
            some_vec, m_applied_after_inverse,
            "M(M^-1(V))) = V did not hold"
        );

        //panic!();
        // B
        let base_vec = vec![eight, two, five];

        // S + M(B)
        let add_after_apply = vec_add::<Bls12>(&some_vec, &apply_matrix::<Bls12>(&m, &base_vec));

        // M(B + M^-1(S))
        let apply_after_add =
            apply_matrix::<Bls12>(&m, &vec_add::<Bls12>(&base_vec, &inverse_applied));

        // S + M(B) = M(B + M^-1(S))
        assert_eq!(add_after_apply, apply_after_add, "breakin' the law");
    }
}
