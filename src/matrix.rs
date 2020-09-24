use ff::{Field, ScalarEngine};

/// Matrix functions here are, at least for now, quick and dirty — intended only to support precomputation of poseidon optimization.

/// Matrix represented as a Vec of rows, so that m[i][j] represents the jth column of the ith row in Matrix, m.
pub type Matrix<T> = Vec<Vec<T>>;
pub type Scalar<E> = <E as ScalarEngine>::Fr;

pub fn rows<T>(matrix: &Matrix<T>) -> usize {
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

// This wastefully discards the actual inverse, if it exists, so in general callers should
// just call `invert` if that result will be needed.
pub(crate) fn is_invertible<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> bool {
    is_square(matrix) && invert::<E>(matrix).is_some()
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

fn scalar_vec_mul<E: ScalarEngine>(scalar: Scalar<E>, vec: &[Scalar<E>]) -> Vec<Scalar<E>> {
    vec.iter()
        .map(|val| {
            let mut prod = scalar.clone();
            prod.mul_assign(val);
            prod
        })
        .collect::<Vec<_>>()
}

pub fn mat_mul<E: ScalarEngine>(
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

fn vec_mul<E: ScalarEngine>(a: &[Scalar<E>], b: &[Scalar<E>]) -> Scalar<E> {
    a.iter()
        .zip(b)
        .fold(Scalar::<E>::zero(), |mut acc, (v1, v2)| {
            let mut tmp = v1.clone();
            tmp.mul_assign(&v2);
            acc.add_assign(&tmp);
            acc
        })
}

pub fn vec_add<E: ScalarEngine>(a: &[Scalar<E>], b: &[Scalar<E>]) -> Vec<Scalar<E>> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let mut res = a.clone();
            res.add_assign(b);
            res
        })
        .collect::<Vec<_>>()
}

pub fn vec_sub<E: ScalarEngine>(a: &[Scalar<E>], b: &[Scalar<E>]) -> Vec<Scalar<E>> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let mut res = a.clone();
            res.sub_assign(b);
            res
        })
        .collect::<Vec<_>>()
}

/// Left-multiply a vector by a square matrix of same size: MV where V is considered a column vector.
pub fn left_apply_matrix<E: ScalarEngine>(
    m: &Matrix<Scalar<E>>,
    v: &[Scalar<E>],
) -> Vec<Scalar<E>> {
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

/// Right-multiply a vector by a square matrix  of same size: VM where V is considered a row vector.
pub fn apply_matrix<E: ScalarEngine>(m: &Matrix<Scalar<E>>, v: &[Scalar<E>]) -> Vec<Scalar<E>> {
    assert!(is_square(m), "Only square matrix can be applied to vector.");
    assert_eq!(
        rows(m),
        v.len(),
        "Matrix can only be applied to vector of same size."
    );

    let mut result: Vec<Scalar<E>> = vec![Scalar::<E>::zero(); v.len()];
    for (j, val) in result.iter_mut().enumerate() {
        for (i, row) in m.iter().enumerate() {
            let mut tmp = row[j];
            tmp.mul_assign(&v[i]);
            val.add_assign(&tmp);
        }
    }

    result
}

pub fn transpose<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Matrix<Scalar<E>> {
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

pub fn make_identity<E: ScalarEngine>(size: usize) -> Matrix<Scalar<E>> {
    let mut result = vec![vec![Scalar::<E>::zero(); size]; size];
    for i in 0..size {
        result[i][i] = Scalar::<E>::one();
    }
    result
}

pub fn kronecker_delta<E: ScalarEngine>(i: usize, j: usize) -> Scalar<E> {
    if i == j {
        Scalar::<E>::one()
    } else {
        Scalar::<E>::zero()
    }
}

pub fn is_identity<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> bool {
    for i in 0..rows(matrix) {
        for j in 0..columns(matrix) {
            if matrix[i][j] != kronecker_delta::<E>(i, j) {
                return false;
            }
        }
    }
    true
}

pub fn is_square<T>(matrix: &Matrix<T>) -> bool {
    rows(matrix) == columns(matrix)
}

pub fn minor<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>, i: usize, j: usize) -> Matrix<Scalar<E>> {
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

// Assumes matrix is partially reduced to upper triangular. `column` is the column to eliminate from all rows.
// Returns `None` if either:
//   - no non-zero pivot can be found for `column`
//   - `column` is not the first
fn eliminate<E: ScalarEngine>(
    matrix: &Matrix<Scalar<E>>,
    column: usize,
    shadow: &mut Matrix<Scalar<E>>,
) -> Option<Matrix<Scalar<E>>> {
    let zero = Scalar::<E>::zero();
    let pivot_index = (0..rows(matrix))
        .find(|&i| matrix[i][column] != zero && (0..column).all(|j| matrix[i][j] == zero))?;

    let pivot = &matrix[pivot_index];
    let pivot_val = pivot[column];

    let inv_pivot = pivot_val.inverse()?; // This should never fail since we have a non-zero `pivot_val` if we got here.
    let mut result = Vec::with_capacity(matrix.len());
    result.push(pivot.clone());

    for (i, row) in matrix.iter().enumerate() {
        if i == pivot_index {
            continue;
        };
        let val = row[column];
        if val == zero {
            // Value is already eliminated.
            result.push(row.to_vec());
        } else {
            let mut factor = val.clone();
            factor.mul_assign(&inv_pivot);

            let scaled_pivot = scalar_vec_mul::<E>(factor, &pivot);
            let eliminated = vec_sub::<E>(row, &scaled_pivot);
            result.push(eliminated);

            let shadow_pivot = &shadow[pivot_index];
            let scaled_shadow_pivot = scalar_vec_mul::<E>(factor, shadow_pivot);
            let shadow_row = &shadow[i];
            shadow[i] = vec_sub::<E>(shadow_row, &scaled_shadow_pivot);
        }
    }
    Some(result)
}

// `matrix` must be square.
fn upper_triangular<E: ScalarEngine>(
    matrix: &Matrix<Scalar<E>>,
    mut shadow: &mut Matrix<Scalar<E>>,
) -> Option<Matrix<Scalar<E>>> {
    assert!(is_square(matrix));
    let mut result = Vec::with_capacity(matrix.len());
    let mut shadow_result = Vec::with_capacity(matrix.len());

    let mut curr = matrix.clone();
    let mut column = 0;
    while curr.len() > 1 {
        let initial_rows = curr.len();

        curr = eliminate::<E>(&curr, column, &mut shadow)?;
        result.push(curr[0].clone());
        shadow_result.push(shadow[0].clone());
        column += 1;

        curr = curr[1..].to_vec();
        *shadow = shadow[1..].to_vec();
        assert_eq!(curr.len(), initial_rows - 1);
    }
    result.push(curr[0].clone());
    shadow_result.push(shadow[0].clone());

    *shadow = shadow_result;

    Some(result)
}

// `matrix` must be upper triangular.
fn reduce_to_identity<E: ScalarEngine>(
    matrix: &Matrix<Scalar<E>>,
    shadow: &mut Matrix<Scalar<E>>,
) -> Option<Matrix<Scalar<E>>> {
    let size = rows(matrix);
    let mut result: Matrix<Scalar<E>> = Vec::new();
    let mut shadow_result: Matrix<Scalar<E>> = Vec::new();

    for i in 0..size {
        let idx = size - i - 1;
        let row = &matrix[idx];
        let shadow_row = &shadow[idx];

        let val = row[idx];
        let inv = val.inverse()?; // If `val` is zero, then there is no inverse, and we cannot compute a result.

        let mut normalized = scalar_vec_mul::<E>(inv, &row);
        let mut shadow_normalized = scalar_vec_mul::<E>(inv, &shadow_row);

        for j in 0..i {
            let idx = size - j - 1;
            let val = normalized[idx];
            let subtracted = scalar_vec_mul::<E>(val, &result[j]);
            let result_subtracted = scalar_vec_mul::<E>(val, &shadow_result[j]);

            normalized = vec_sub::<E>(&normalized, &subtracted);
            shadow_normalized = vec_sub::<E>(&shadow_normalized, &result_subtracted);
        }

        result.push(normalized);
        shadow_result.push(shadow_normalized);
    }

    result.reverse();
    shadow_result.reverse();

    *shadow = shadow_result;
    Some(result)
}

//
pub(crate) fn invert<E: ScalarEngine>(matrix: &Matrix<Scalar<E>>) -> Option<Matrix<Scalar<E>>> {
    let mut shadow = make_identity::<E>(columns(matrix));
    let ut = upper_triangular::<E>(&matrix, &mut shadow);

    ut.and_then(|x| reduce_to_identity::<E>(&x, &mut shadow))
        .and(Some(shadow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_from_u64;
    use bellperson::bls::{Bls12, Fr};

    #[test]
    fn test_minor() {
        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        let nine = scalar_from_u64::<Fr>(9);

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
    }

    #[test]
    fn test_scalar_mul() {
        let zero = scalar_from_u64::<Fr>(0);
        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let six = scalar_from_u64::<Fr>(6);

        let m = vec![vec![zero, one], vec![two, three]];
        let res = scalar_mul::<Bls12>(two, &m);

        let expected = vec![vec![zero, two], vec![four, six]];

        assert_eq!(expected, res);
    }

    #[test]
    fn test_vec_mul() {
        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);

        let a = vec![one, two, three];
        let b = vec![four, five, six];
        let res = vec_mul::<Bls12>(&a, &b);

        let expected = scalar_from_u64::<Fr>(32);

        assert_eq!(expected, res);
    }

    #[test]
    fn test_transpose() {
        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        let nine = scalar_from_u64::<Fr>(9);

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
        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        let nine = scalar_from_u64::<Fr>(9);

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

    #[test]
    fn test_eliminate() {
        //let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        //        let nine = scalar_from_u64::<Fr>(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        for i in 0..rows(&m) {
            let mut shadow = make_identity::<Bls12>(columns(&m));
            let res = eliminate::<Bls12>(&m, i, &mut shadow);
            if i > 0 {
                assert!(res.is_none());
                continue;
            } else {
                assert!(res.is_some());
            }

            assert_eq!(
                1,
                res.unwrap()
                    .iter()
                    .filter(|&row| row[i] != <Bls12 as ScalarEngine>::Fr::zero())
                    .count()
            );
        }
    }
    #[test]
    fn test_upper_triangular() {
        //        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        //        let nine = scalar_from_u64::<Fr>(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        let mut shadow = make_identity::<Bls12>(columns(&m));
        let _res = upper_triangular::<Bls12>(&m, &mut shadow);

        // Actually assert things.
    }

    #[test]
    fn test_reduce_to_identity() {
        //        let one = scalar_from_u64::<Fr>(1);
        let two = scalar_from_u64::<Fr>(2);
        let three = scalar_from_u64::<Fr>(3);
        let four = scalar_from_u64::<Fr>(4);
        let five = scalar_from_u64::<Fr>(5);
        let six = scalar_from_u64::<Fr>(6);
        let seven = scalar_from_u64::<Fr>(7);
        let eight = scalar_from_u64::<Fr>(8);
        //        let nine = scalar_from_u64::<Fr>(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        let mut shadow = make_identity::<Bls12>(columns(&m));
        let ut = upper_triangular::<Bls12>(&m, &mut shadow);

        let res = ut
            .and_then(|x| reduce_to_identity::<Bls12>(&x, &mut shadow))
            .unwrap();

        assert!(is_identity::<Bls12>(&res));
        let prod = mat_mul::<Bls12>(&m, &shadow).unwrap();

        assert!(is_identity::<Bls12>(&prod));
    }
}
