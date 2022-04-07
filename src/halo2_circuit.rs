use std::iter;
use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{self, AssignedCell, Layouter, Region, Value},
    plonk::{
        Advice, Column, ConstraintSystem, Constraints, Error, Expression, Fixed, Selector,
        VirtualCells,
    },
    poly::Rotation,
};

use crate::{
    hash_type::HashType,
    matrix,
    mds::generate_mds,
    poseidon::{Arity, PoseidonConstants},
    round_numbers, Strength,
};

const ALPHA: u64 = 5;

#[derive(Clone, Debug)]
pub struct PoseidonConfig<F, A>
where
    F: FieldExt,
    A: Arity<F>,
{
    state: Vec<Column<Advice>>,
    rc_a: Vec<Column<Fixed>>,
    // If there is an even number of partial rounds for arity `A` (and `Strength::Standard`), we use
    // a row optimized poseidon circuit (which performs two partial rounds per row) at the cost of
    // an additional advice column `partial_sbox` and `width` number of fixed columns `rc_b`.
    partial_sbox: Option<Column<Advice>>,
    rc_b: Option<Vec<Column<Fixed>>>,
    s_full: Selector,
    s_partial: Selector,
    _f: PhantomData<F>,
    _a: PhantomData<A>,
}

impl<F, A> PoseidonConfig<F, A>
where
    F: FieldExt,
    A: Arity<F>,
{
    // Input-output columns; equality-enabled advice where a chip caller can allocate preimages and
    // copy digests from.
    pub fn io_cols(&self) -> &[Column<Advice>] {
        &self.state
    }

    // An equality and constant-enabled fixed column.
    pub fn consts_col(&self) -> &Column<Fixed> {
        &self.rc_a[0]
    }

    // If you have two arities `A` and `B` which you know are the same type (but where the
    // compiler doesn't) `change_arity` can be used to convert the `A` config into the `B` config
    // without having to call `PoseidonConfig::<F, B>::configure` (which duplicates the `A`/`B`
    // configuration in the constraint system).
    pub fn change_arity<B: Arity<F>>(self) -> PoseidonConfig<F, B> {
        assert_eq!(A::to_usize(), B::to_usize());
        PoseidonConfig {
            state: self.state,
            partial_sbox: self.partial_sbox,
            rc_a: self.rc_a,
            rc_b: self.rc_b,
            s_full: self.s_full,
            s_partial: self.s_partial,
            _f: PhantomData,
            _a: PhantomData,
        }
    }
}

pub struct PoseidonChip<F, A>
where
    F: FieldExt,
    A: Arity<F>,
{
    config: PoseidonConfig<F, A>,
}

impl<F, A> PoseidonChip<F, A>
where
    F: FieldExt,
    A: Arity<F>,
{
    pub fn construct(config: PoseidonConfig<F, A>) -> Self {
        PoseidonChip { config }
    }

    // # Side Effects
    //
    // - `advice[..width]` will be equality constrained.
    // - `fixed[0]` will be equality constrained.
    #[allow(clippy::needless_collect)]
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: &[Column<Advice>],
        fixed: &[Column<Fixed>],
    ) -> PoseidonConfig<F, A> {
        let arity = A::to_usize();
        let width = arity + 1;

        assert!(arity > 0);
        assert!(advice.len() >= width);
        assert!(fixed.len() >= width);

        let state = advice[..width].to_vec();
        let rc_a = fixed[..width].to_vec();

        // Allows the preimage and digest to be copied into and out of the hash function region.
        for col in &state {
            meta.enable_equality(*col);
        }
        // Allows the constant domain tag (stored in a fixed column) to be copied into the first
        // initial state word (advice column).
        meta.enable_constant(rc_a[0]);

        let s_full = meta.selector();
        let s_partial = meta.selector();

        let mds = generate_mds::<F>(width);

        let pow_5 = |v: Expression<F>| {
            let v2 = v.clone() * v.clone();
            v2.clone() * v2 * v
        };

        meta.create_gate("full round", |meta| {
            let s_full = meta.query_selector(s_full);

            // Apply sbox to each input state word.
            let sbox_out: Vec<Expression<F>> = (0..width)
                .map(|i| {
                    let word = meta.query_advice(state[i], Rotation::cur());
                    let rc = meta.query_fixed(rc_a[i], Rotation::cur());
                    pow_5(word + rc)
                })
                .collect();

            // Mds mixing.
            let mds_out: Vec<Expression<F>> = (0..width)
                .map(|col| {
                    let mut dot_prod = sbox_out[0].clone() * mds[0][col];
                    for row in 1..width {
                        dot_prod = dot_prod + sbox_out[row].clone() * mds[row][col];
                    }
                    dot_prod
                })
                .collect();

            // Next round's input state.
            let state_next: Vec<Expression<F>> = state
                .iter()
                .map(|col| meta.query_advice(*col, Rotation::next()))
                .collect();

            // Assert that this round's output state is equal to the next round's input state.
            Constraints::with_selector(
                s_full,
                mds_out
                    .into_iter()
                    .zip(state_next.into_iter())
                    .map(|(out, next)| out - next),
            )
        });

        // If there is an odd number of partial rounds, perform one partial round per row, otherwise
        // perform two partial rounds per row.
        if Self::rp_is_odd() {
            assert_eq!(advice.len(), width);
            assert_eq!(fixed.len(), width);

            // Perform one partial round.
            meta.create_gate("partial round", |meta| {
                let s_partial = meta.query_selector(s_partial);

                let sbox_out: Vec<Expression<F>> = (0..width)
                    .map(|i| {
                        let word = meta.query_advice(state[i], Rotation::cur());
                        let rc = meta.query_fixed(rc_a[i], Rotation::cur());
                        if i == 0 {
                            pow_5(word + rc)
                        } else {
                            word + rc
                        }
                    })
                    .collect();

                let mds_out: Vec<Expression<F>> = (0..width)
                    .map(|col| {
                        let mut dot_prod = sbox_out[0].clone() * mds[0][col];
                        for row in 1..width {
                            dot_prod = dot_prod + sbox_out[row].clone() * mds[row][col];
                        }
                        dot_prod
                    })
                    .collect();

                let state_next: Vec<Expression<F>> = state
                    .iter()
                    .map(|col| meta.query_advice(*col, Rotation::next()))
                    .collect();

                Constraints::with_selector(
                    s_partial,
                    mds_out
                        .into_iter()
                        .zip(state_next.into_iter())
                        .map(|(out, next)| out - next),
                )
            });

            PoseidonConfig {
                state,
                partial_sbox: None,
                rc_a,
                rc_b: None,
                s_full,
                s_partial,
                _f: PhantomData,
                _a: PhantomData,
            }
        } else {
            // Add one advice column for partial round A's sbox output.
            assert_eq!(advice.len(), width + 1);
            // Two partial rounds' (A and B) round constants.
            assert_eq!(fixed.len(), 2 * width);

            let partial_sbox = advice[width];
            let rc_b = fixed[width..].to_vec();

            let mds_inv = matrix::invert(&mds).expect("mds matrix in non-invertible");

            // Perform two partial rounds (A and B).
            meta.create_gate("partial rounds", |meta| {
                let s_partial = meta.query_selector(s_partial);

                // The first element of round A's input state
                let a_in_0 = meta.query_advice(state[0], Rotation::cur());
                let a_sbox_out_0 = meta.query_advice(partial_sbox, Rotation::cur());

                // Compute the `i`-th state element output by round A (equivalent to round B's `i`-th
                // input state element.
                let a_out = |i: usize, meta: &mut VirtualCells<F>| {
                    let dot_prod = a_sbox_out_0.clone() * mds[i][0];
                    (1..width).fold(dot_prod, |dot_prod, j| {
                        let a_in_j = meta.query_advice(state[j], Rotation::cur());
                        let c = meta.query_fixed(rc_a[j], Rotation::cur());
                        dot_prod + (a_in_j + c) * mds[i][j]
                    })
                };

                // Compute the `i`-th sbox output for round B by computing the dot product of B's output
                // state with the `i`-th column of the inverse MDS matrix.
                let b_sbox_out = |i: usize, meta: &mut VirtualCells<F>| {
                    (0..width)
                        .map(|j| {
                            let b_out_j = meta.query_advice(state[j], Rotation::next());
                            b_out_j * mds_inv[i][j]
                        })
                        .reduce(|dot_prod, next| dot_prod + next)
                        .unwrap()
                };

                // Apply the sbox to the first elemet of round A's input state and assert that it is
                // equal to the value in the `partial_sbox` column of the current row.
                let a_sbox_out_0 = {
                    let c = meta.query_fixed(rc_a[0], Rotation::cur());
                    pow_5(a_in_0 + c) - a_sbox_out_0.clone()
                };

                // Compute the first state element output by round A (i.e. B's first input state
                // element) and assert that it is equal to the dot product of round B's output state and
                // the first column of the inverse MDS matrix.
                let b_sbox_out_0 = {
                    let b_in_0 = a_out(0, meta);
                    let c = meta.query_fixed(rc_b[0], Rotation::cur());
                    pow_5(b_in_0 + c) - b_sbox_out(0, meta)
                };

                // For each B input state element `i > 0`, add the corresponding round constant and
                // assert that the sum is equal to B's `i`-th` sbox output (computed by taking the
                // dot product of B's output state with the `i`-th column of MDS^-1).
                let b_out_linear = (1..width).map(|i| {
                    let b_in_i = a_out(i, meta);
                    let c = meta.query_fixed(rc_b[i], Rotation::cur());
                    b_in_i + c - b_sbox_out(i, meta)
                });

                Constraints::with_selector(
                    s_partial,
                    iter::once(a_sbox_out_0)
                        .chain(iter::once(b_sbox_out_0))
                        .chain(b_out_linear)
                        .collect::<Vec<_>>(),
                )
            });

            PoseidonConfig {
                state,
                partial_sbox: Some(partial_sbox),
                rc_a,
                rc_b: Some(rc_b),
                s_full,
                s_partial,
                _f: PhantomData,
                _a: PhantomData,
            }
        }
    }

    // Assign the initial state `domain tag || preimage` into the state vector of the provided row.
    fn assign_initial_state(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        preimage: &[AssignedCell<F, F>],
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = Self::width();

        let mut state = Vec::<AssignedCell<F, F>>::with_capacity(width);

        // Assign the domain tag into a fixed column and copy the assigned value into the
        // first initial state element.
        let domain_tag = region.assign_advice_from_constant(
            || "initial_state[0] = domain tag",
            self.config.state[0],
            offset,
            consts.domain_tag,
        )?;
        state.push(domain_tag);

        // Copy the preimage into the remaining initial state elements.
        for (i, (word, col)) in preimage.iter().zip(&self.config.state[1..]).enumerate() {
            state.push(word.copy_advice(
                || format!("copy preimage[{}] into initial_state[{}]", i, i + 1),
                region,
                *col,
                offset,
            )?);
        }

        Ok(state)
    }

    fn assign_round_constants(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        round: usize,
        rc_cols: &[Column<Fixed>],
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = Self::width();
        consts
            .round_constants
            .iter()
            .skip(round * width)
            .take(width)
            .zip(rc_cols)
            .enumerate()
            .map(|(i, (rc, col))| {
                region.assign_fixed(
                    || format!("round_{} rc[{}]", round, i),
                    *col,
                    offset,
                    || Value::known(*rc),
                )
            })
            .collect()
    }

    fn assign_state(
        &self,
        region: &mut Region<F>,
        state: &[Value<F>],
        round: usize,
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        state
            .iter()
            .zip(&self.config.state)
            .enumerate()
            .map(|(i, (word, col))| {
                region.assign_advice(
                    || format!("round_{} state[{}]", round, i),
                    *col,
                    offset,
                    || *word,
                )
            })
            .collect()
    }

    // Perform a full round on `state` and assign the output state in the next row.
    fn full_round(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        state: &[AssignedCell<F, F>],
        round: usize,
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = Self::width();

        self.config.s_full.enable(region, offset)?;
        self.assign_round_constants(region, consts, round, &self.config.rc_a, offset)?;

        let round_consts = consts
            .round_constants
            .iter()
            .skip(round * width)
            .take(width);

        let mds = &consts.mds_matrices.m;
        let mds_cols = (0..width).map(|col| (0..width).map(move |row| mds[row][col]));

        // Add a round constant to each state elememt, then apply the sbox to each sum.
        let sbox_out: Vec<Value<F>> = state
            .iter()
            .zip(round_consts)
            .map(|(word, rc)| word.value().map(|word| (*word + rc).pow_vartime([ALPHA])))
            .collect();

        let next_state: Vec<Value<F>> = mds_cols
            .map(|mds_col| {
                let mut dot_prod = Value::known(F::zero());
                for (sbox_out, m) in sbox_out.iter().zip(mds_col) {
                    let sbox_out_times_m = sbox_out.map(|sbox_out| sbox_out * m);
                    dot_prod = dot_prod + sbox_out_times_m;
                }
                dot_prod
            })
            .collect();

        self.assign_state(region, &next_state, round + 1, offset + 1)
    }

    // Perform one partial round on `state` and assign the output state in the next row.
    fn one_partial_round(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        state: &[AssignedCell<F, F>],
        round: usize,
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = Self::width();

        self.config.s_partial.enable(region, offset)?;
        self.assign_round_constants(region, consts, round, &self.config.rc_a, offset)?;

        let round_consts = consts
            .round_constants
            .iter()
            .skip(round * width)
            .take(width);

        // Add a round constant to each state elememt, then apply the sbox to each sum.
        let sbox_out: Vec<Value<F>> = state
            .iter()
            .zip(round_consts)
            .enumerate()
            .map(|(i, (word, rc))| {
                word.value().map(|word| {
                    if i == 0 {
                        (*word + rc).pow_vartime([ALPHA])
                    } else {
                        *word + rc
                    }
                })
            })
            .collect();

        let mds = &consts.mds_matrices.m;
        let mds_cols = (0..width).map(|col| (0..width).map(move |row| mds[row][col]));

        let next_state: Vec<Value<F>> = mds_cols
            .map(|mds_col| {
                let mut dot_prod = Value::known(F::zero());
                for (sbox_out, m) in sbox_out.iter().zip(mds_col) {
                    let sbox_out_times_m = sbox_out.map(|sbox_out| sbox_out * m);
                    dot_prod = dot_prod + sbox_out_times_m;
                }
                dot_prod
            })
            .collect();

        self.assign_state(region, &next_state, round + 1, offset + 1)
    }

    // Perform 2 partial rounds (A and B) on `state` and assign the output state in the next row.
    fn two_partial_rounds(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        state: &[AssignedCell<F, F>],
        round: usize,
        offset: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = Self::width();

        let round_a = round;
        let round_b = round + 1;

        self.config.s_partial.enable(region, offset)?;

        // Assign the first and second rounds' round constants in the `rc_a` and `rc_b` columns
        // respectively.
        self.assign_round_constants(region, consts, round_a, &self.config.rc_a, offset)?;
        self.assign_round_constants(
            region,
            consts,
            round_b,
            self.config
                .rc_b
                .as_ref()
                .expect("chip config missing `rc_b` fixed columns"),
            offset,
        )?;

        let round_consts_a = consts
            .round_constants
            .iter()
            .skip(round_a * width)
            .take(width);

        let round_consts_b = consts
            .round_constants
            .iter()
            .skip(round_b * width)
            .take(width);

        let mds = &consts.mds_matrices.m;
        let mds_cols_a = (0..width).map(|col| (0..width).map(move |row| mds[row][col]));
        let mds_cols_b = mds_cols_a.clone();

        // Add a round constant to each state elememt, then apply the sbox to the first sum.
        let sbox_out_a: Vec<Value<F>> = state
            .iter()
            .zip(round_consts_a)
            .enumerate()
            .map(|(i, (word, rc))| {
                if i == 0 {
                    word.value().map(|word| (*word + rc).pow_vartime([ALPHA]))
                } else {
                    word.value().map(|word| *word + rc)
                }
            })
            .collect();

        // Assign the first state element's sbox output in the `partial_sbox` column.
        region.assign_advice(
            || format!("round_{} sbox output A", round_a),
            self.config
                .partial_sbox
                .expect("chip config missing `partial_sbox` advice column"),
            offset,
            || sbox_out_a[0],
        )?;

        let input_state_b: Vec<Value<F>> = mds_cols_a
            .map(|mds_col| {
                let mut dot_prod = Value::known(F::zero());
                for (sbox_out, m) in sbox_out_a.iter().zip(mds_col) {
                    let sbox_out_times_m = sbox_out.map(|sbox_out| sbox_out * m);
                    dot_prod = dot_prod + sbox_out_times_m;
                }
                dot_prod
            })
            .collect();

        // Add the associated round constant to each state elememt, then apply the sbox to the first
        // element.
        let sbox_out_b: Vec<Value<F>> = input_state_b
            .iter()
            .zip(round_consts_b)
            .enumerate()
            .map(|(i, (word, rc))| {
                if i == 0 {
                    word.as_ref().map(|word| (*word + rc).pow_vartime([ALPHA]))
                } else {
                    word.as_ref().map(|word| *word + rc)
                }
            })
            .collect();

        // Multiply the sbox outputs by the MDS matrix.
        let output_state_b: Vec<Value<F>> = mds_cols_b
            .map(|mds_col| {
                let mut dot_prod = Value::known(F::zero());
                for (sbox_out, m) in sbox_out_b.iter().zip(mds_col) {
                    let sbox_out_times_m = sbox_out.map(|sbox_out| sbox_out * m);
                    dot_prod = dot_prod + sbox_out_times_m;
                }
                dot_prod
            })
            .collect();

        self.assign_state(region, &output_state_b, round_b + 1, offset + 1)
    }

    pub fn hash(
        &self,
        mut layouter: impl Layouter<F>,
        preimage: &[AssignedCell<F, F>],
        consts: &PoseidonConstants<F, A>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let arity = A::to_usize();
        let width = arity + 1;

        assert!(arity > 0);
        assert_eq!(preimage.len(), arity);

        // This chip does not support preimage padding (i.e. `preimage.len()` must equal `arity`) or
        // the sponge construction.
        match &consts.hash_type {
            HashType::ConstantLength(n) => {
                if *n != width {
                    unimplemented!("halo2 circuit does not support constant-length padding");
                }
            }
            HashType::Sponge => unimplemented!("halo2 circuit does not support sponge"),
            hash_type => {
                if !hash_type.is_supported() {
                    unimplemented!("halo2 circuit does not support hash type");
                }
            }
        };

        let rp_is_odd = Self::rp_is_odd();
        assert_eq!(consts.partial_rounds & 1 == 1, rp_is_odd);

        layouter.assign_region(
            || "poseidon",
            |mut region| {
                let mut round = 0;
                let mut offset = 0;

                let mut state = self.assign_initial_state(&mut region, consts, preimage, offset)?;

                for _ in 0..consts.half_full_rounds {
                    state = self.full_round(&mut region, consts, &state, round, offset)?;
                    round += 1;
                    offset += 1;
                }

                if rp_is_odd {
                    for _ in 0..consts.partial_rounds {
                        state =
                            self.one_partial_round(&mut region, consts, &state, round, offset)?;
                        round += 1;
                        offset += 1;
                    }
                } else {
                    for _ in 0..consts.partial_rounds / 2 {
                        state =
                            self.two_partial_rounds(&mut region, consts, &state, round, offset)?;
                        round += 2;
                        offset += 1;
                    }
                }

                for _ in 0..consts.half_full_rounds {
                    state = self.full_round(&mut region, consts, &state, round, offset)?;
                    round += 1;
                    offset += 1;
                }

                Ok(state[1].clone())
            },
        )
    }

    // Input-output columns; equality-enabled advice where a chip caller can allocate preimages and
    // copy digests from.
    pub fn io_cols(&self) -> &[Column<Advice>] {
        self.config.io_cols()
    }

    // An equality and constant-enabled fixed column.
    pub fn consts_col(&self) -> &Column<Fixed> {
        self.config.consts_col()
    }

    fn width() -> usize {
        A::to_usize() + 1
    }

    // Returns `true` if this arity (for strength `Strength::Standard`) has an odd number of partial
    // rounds.
    #[inline]
    fn rp_is_odd() -> bool {
        // These partial round numbers were taken from the file `parameters/round_numbers.txt`.
        // These values are hardcoded here to avoid repeated calls to `round_numbers(...)` when
        // calculating this chips constraint system shape (i.e. row and column counts).
        match Self::width() {
            2 | 3 | 8..=31 | 63..=124 => true,
            width if width <= 125 => false,
            _ => unimplemented!("arity exceeds the maximum supported arity (125)"),
        }
    }

    // The number of constraint system used per call to `self.hash()`.
    #[inline]
    pub fn num_rows() -> usize {
        let arity = A::to_usize();
        if arity == 0 {
            return 0;
        }
        let (rf, rp) = round_numbers(arity, &Strength::Standard);
        let rp_is_odd = rp & 1 == 1;
        if rp_is_odd {
            rf + rp
        } else {
            rf + rp / 2
        }
    }

    // The number of advice columns used by this chip.
    #[inline]
    pub fn num_advice_total() -> usize {
        if A::to_usize() == 0 {
            0
        } else if Self::rp_is_odd() {
            Self::width()
        } else {
            Self::width() + 1
        }
    }

    // The number of equality-enabled advice columns used by this chip.
    #[inline]
    pub fn num_advice_eq() -> usize {
        if A::to_usize() == 0 {
            0
        } else {
            Self::width()
        }
    }

    // The number of non-equality-enabled advice columns used by this chip.
    #[inline]
    pub fn num_advice_neq() -> usize {
        Self::num_advice_total() - Self::num_advice_eq()
    }

    // The number of fixed columns used by this chip.
    #[inline]
    pub fn num_fixed_total() -> usize {
        if A::to_usize() == 0 {
            0
        } else if Self::rp_is_odd() {
            Self::width()
        } else {
            2 * Self::width()
        }
    }

    // The number of equality-enabled fixed columns used by this chip.
    #[inline]
    pub fn num_fixed_eq() -> usize {
        if A::to_usize() == 0 {
            0
        } else {
            1
        }
    }

    // The number of non-equality-enabled fixed columns used by this chip.
    #[inline]
    pub fn num_fixed_neq() -> usize {
        Self::num_fixed_total() - Self::num_fixed_eq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use generic_array::typenum::{Unsigned, U1, U11, U2, U3, U4, U8};
    use halo2_proofs::{
        arithmetic::{CurveAffine, CurveExt, FieldExt},
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        pasta::{EqAffine, Fp, Fq},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Error, Instance,
            SingleVerifier,
        },
        poly::commitment::Params,
        transcript::{Blake2bRead, Blake2bWrite, Challenge255},
    };
    use rand::rngs::OsRng;

    use crate::{round_numbers, Poseidon, Strength};

    #[test]
    fn test_poseidon_chip() {
        struct MyCircuit<A: Arity<Fp>> {
            preimage: Vec<Value<Fp>>,
            expected_digest: Value<Fp>,
            _a: PhantomData<A>,
        }

        impl<A: Arity<Fp>> Circuit<Fp> for MyCircuit<A> {
            type Config = PoseidonConfig<Fp, A>;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MyCircuit {
                    preimage: vec![Value::unknown(); A::to_usize()],
                    expected_digest: Value::unknown(),
                    _a: PhantomData,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                let num_advice = PoseidonChip::<Fp, A>::num_advice_total();
                let num_fixed = PoseidonChip::<Fp, A>::num_fixed_total();

                let advice: Vec<Column<Advice>> =
                    (0..num_advice).map(|_| meta.advice_column()).collect();
                let fixed: Vec<Column<Fixed>> =
                    (0..num_fixed).map(|_| meta.fixed_column()).collect();

                PoseidonChip::configure(meta, &advice, &fixed)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                let poseidon_chip = PoseidonChip::<Fp, A>::construct(config);

                let preimage = layouter.assign_region(
                    || "assign preimage",
                    |mut region| {
                        let offset = 0;
                        self.preimage
                            .iter()
                            .zip(poseidon_chip.io_cols())
                            .enumerate()
                            .map(|(i, (word, col))| {
                                region.assign_advice(
                                    || format!("preimage[{}]", i),
                                    *col,
                                    offset,
                                    || *word,
                                )
                            })
                            .collect::<Result<Vec<AssignedCell<Fp, Fp>>, Error>>()
                    },
                )?;

                let consts = PoseidonConstants::<Fp, A>::new();

                poseidon_chip
                    .hash(layouter, &preimage, &consts)?
                    .value()
                    .zip(self.expected_digest.as_ref())
                    .assert_if_known(|(digest, expected_digest)| digest == expected_digest);

                Ok(())
            }
        }

        impl<A: Arity<Fp>> MyCircuit<A> {
            fn k() -> u32 {
                // Add one row for preimage allocation.
                let rows = PoseidonChip::<Fp, A>::num_rows() + 1;
                // Adding one to `k` ensures that we have enough rows.
                (rows as f32).log2().ceil() as u32 + 1
            }
        }

        type TranscriptReader<'proof> = Blake2bRead<&'proof [u8], EqAffine, Challenge255<EqAffine>>;
        type TranscriptWriter = Blake2bWrite<Vec<u8>, EqAffine, Challenge255<EqAffine>>;

        fn test_poseidon_chip_inner<A: Arity<Fp>>(gen_proof: bool) {
            let arity = A::to_usize();
            let preimage: Vec<Fp> = (0..arity as u64).map(Fp::from).collect();

            let consts = PoseidonConstants::<Fp, A>::new();
            let digest = Poseidon::new_with_preimage(&preimage, &consts).hash();

            let circ = MyCircuit::<A> {
                preimage: preimage.into_iter().map(Value::known).collect(),
                expected_digest: Value::known(digest),
                _a: PhantomData,
            };

            let k = MyCircuit::<A>::k();
            let prover = MockProver::run(k, &circ, vec![]).unwrap();
            assert!(prover.verify().is_ok());

            if gen_proof {
                let params = Params::<EqAffine>::new(k);
                let pk = {
                    let vk = keygen_vk(&params, &circ).expect("failed to create verifying key");
                    keygen_pk(&params, vk, &circ).expect("failed to create proving key")
                };
                let vk = pk.get_vk();

                let mut transcript = TranscriptWriter::init(vec![]);
                create_proof(&params, &pk, &[circ], &[&[]], &mut OsRng, &mut transcript)
                    .expect("failed to create halo2 proof");
                let proof_bytes: Vec<u8> = transcript.finalize();

                let mut transcript = TranscriptReader::init(&proof_bytes);
                let verifier_strategy = SingleVerifier::new(&params);
                verify_proof(&params, vk, verifier_strategy, &[&[]], &mut transcript)
                    .expect("failed to verify halo2 proof");
            }
        }

        test_poseidon_chip_inner::<U1>(true);
        test_poseidon_chip_inner::<U2>(true);
        test_poseidon_chip_inner::<U4>(true);
        test_poseidon_chip_inner::<U8>(false);
        test_poseidon_chip_inner::<U11>(false);
    }
}
