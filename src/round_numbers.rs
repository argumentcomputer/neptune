//! A port of `calc_round_numbers.py`
//! https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/9d80ec0473ad7cde5a12f3aac46439ad0da68c0a/code/scripts/calc_round_numbers.py
//! from Python2 to Rust for a (roughly) 256-bit prime field (e.g. BLS12-381's scalar field) and
//! 128-bit security level.

// The number of bits of the Poseidon prime field modulus. Denoted `n` in the Poseidon paper
// (where `n = ceil(log2(p))`). Note that BLS12-381's scalar field modulus is 255 bits, however we
// use 256 bits for simplicity when operating on bytes as the single bit difference does not affect
// the round number security properties.
const PRIME_BITLEN: usize = 256;

// Security level (in bits), denoted `M` in the Poseidon paper.
const M: usize = 128;

// The number of S-boxes (also called the "cost") given by equation (14) in the Poseidon paper:
// `cost = t * R_F + R_P`.
#[inline]
fn n_sboxes(t: usize, rf: usize, rp: usize) -> usize {
    t * rf + rp
}

// Returns the round numbers for a given width `t`. Here, the `security_margin` parameter does not
// indicate that we are calculating `R_F` and `R_P` for the "strengthened" round numbers, done in
// the function `round_numbers_strengthened()`.
pub(crate) fn calc_round_numbers(t: usize, security_margin: bool) -> (usize, usize) {
    let mut rf = 0;
    let mut rp = 0;
    let mut n_sboxes_min = usize::MAX;

    for mut rf_test in (2..=1000).step_by(2) {
        for mut rp_test in 4..200 {
            if round_numbers_are_secure(t, rf_test, rp_test) {
                if security_margin {
                    rf_test += 2;
                    rp_test = (1.075 * rp_test as f32).ceil() as usize;
                }
                let n_sboxes = n_sboxes(t, rf_test, rp_test);
                if n_sboxes < n_sboxes_min || (n_sboxes == n_sboxes_min && rf_test < rf) {
                    rf = rf_test;
                    rp = rp_test;
                    n_sboxes_min = n_sboxes;
                }
            }
        }
    }

    (rf, rp)
}

// Returns `true` if the provided round numbers satisfy the security inequalities specified in the
// Poseidon paper.
fn round_numbers_are_secure(t: usize, rf: usize, rp: usize) -> bool {
    let (rp, t, n, m) = (rp as f32, t as f32, PRIME_BITLEN as f32, M as f32);
    let rf_stat = if m <= (n - 3.0) * (t + 1.0) {
        6.0
    } else {
        10.0
    };
    let rf_interp = 0.43 * m + t.log2() - rp;
    let rf_grob_1 = 0.21 * n - rp;
    let rf_grob_2 = (0.14 * n - 1.0 - rp) / (t - 1.0);
    let rf_max = [rf_stat, rf_interp, rf_grob_1, rf_grob_2]
        .iter()
        .map(|rf| rf.ceil() as usize)
        .max()
        .unwrap();
    rf >= rf_max
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    #[test]
    fn test_round_numbers_against_known_values() {
        // Each case contains a `t` (where `t = arity + 1`) and the `R_P` expected for that `t`.
        let cases = [
            (2usize, 55usize),
            (3, 55),
            (4, 56),
            (5, 56),
            (6, 56),
            (7, 56),
            (8, 57),
            (9, 57),
            (10, 57),
            (11, 57),
            (12, 57),
            (13, 57),
            (14, 57),
            (15, 57),
            (16, 59),
            (17, 59),
            (25, 59),
            (37, 60),
            (65, 61),
        ];
        for (t, rp_expected) in cases.iter() {
            let (rf, rp) = calc_round_numbers(*t, true);
            assert_eq!(rf, 8);
            assert_eq!(rp, *rp_expected);
        }
    }

    #[ignore]
    #[test]
    fn test_round_numbers_against_python_script() {
        // A parsed line from `parameters/round_numbers.txt`.
        struct Line {
            t: usize,
            rf: usize,
            rp: usize,
            sbox_cost: usize,
            size_cost: usize,
        }

        let lines: Vec<Line> = fs::read_to_string("parameters/round_numbers.txt")
            .expect("failed to read round numbers file: `parameters/round_numbers.txt`")
            .lines()
            .skip_while(|line| line.starts_with('#'))
            .map(|line| {
                let nums: Vec<usize> = line
                    .split(' ')
                    .map(|s| {
                        s.parse().unwrap_or_else(|_| {
                            panic!("failed to parse line as `usize`s: {}", line)
                        })
                    })
                    .collect();
                assert_eq!(nums.len(), 5, "line in does not contain 5 values: {}", line);
                Line {
                    t: nums[0],
                    rf: nums[1],
                    rp: nums[2],
                    sbox_cost: nums[3],
                    size_cost: nums[4],
                }
            })
            .collect();

        assert!(
            !lines.is_empty(),
            "no lines were parsed from `round_numbers.txt`",
        );

        for line in lines {
            let (rf, rp) = calc_round_numbers(line.t, true);
            let sbox_cost = n_sboxes(line.t, rf, rp);
            let size_cost = sbox_cost * PRIME_BITLEN;

            assert_eq!(rf, line.rf, "full rounds differ from script");
            assert_eq!(rp, line.rp, "partial rounds differ from script");
            assert_eq!(sbox_cost, line.sbox_cost, "cost differs from script");
            assert_eq!(size_cost, line.size_cost, "size-cost differs from script");
        }
    }
}
