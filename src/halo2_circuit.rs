use std::iter;
use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector, VirtualCells},
    poly::Rotation,
};

use crate::{
    hash_type::HashType,
    matrix,
    mds::generate_mds,
    poseidon::{Arity, PoseidonConstants},
};

const ALPHA: u64 = 5;

#[derive(Clone, Debug)]
pub struct PoseidonConfig<F, A>
where
    F: FieldExt,
    A: Arity<F>,
{
    state: Vec<Column<Advice>>,
    partial_sbox: Column<Advice>,
    rc_a: Vec<Column<Fixed>>,
    rc_b: Vec<Column<Fixed>>,
    s_full: Selector,
    s_partial: Selector,
    _f: PhantomData<F>,
    _a: PhantomData<A>,
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
    // # Side Effects
    //
    // - All `io` columns will be equality constrained.
    // - The first `fixed` column will be equality constrained.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        io: Vec<Column<Advice>>,
        extra: Column<Advice>,
        fixed: Vec<Column<Fixed>>,
    ) -> PoseidonConfig<F, A> {
        let width = A::to_usize() + 1;

        assert_eq!(io.len(), width);
        assert_eq!(fixed.len(), 2 * width);

        // Rename columns.
        let state = io;
        let partial_sbox = extra;
        let rc_b = fixed[..width].to_vec();
        let rc_a = fixed[width..].to_vec();

        // Allows the preimage to be copied into the hash function's region.
        for col in state.iter() {
            meta.enable_equality(*col);
        }
        // Allows the domain tag to be copied into the first state column.
        meta.enable_equality(rc_b[0]);

        let s_full = meta.selector();
        let s_partial = meta.selector();

        let mds = generate_mds::<F>(width);
        let mds_inv = matrix::invert(&mds).expect("mds matrix in non-invertible");

        let pow_5 = |v: Expression<F>| {
            let v2 = v.clone() * v.clone();
            v2.clone() * v2 * v
        };

        meta.create_gate("full round", |meta| {
            let s_full = meta.query_selector(s_full);

            // Assert that the dot product of the current round's (i.e. rows's) state with the MDS
            // matrix is equal to the next round's state.
            (0..width)
                .map(|i| {
                    let next_elem = meta.query_advice(state[i], Rotation::next());
                    let dot_prod = (0..width)
                        .map(|j| {
                            let elem = meta.query_advice(state[j], Rotation::cur());
                            let c = meta.query_fixed(rc_a[j], Rotation::cur());
                            pow_5(elem + c) * mds[j][i]
                        })
                        .reduce(|dot_prod, next| dot_prod + next)
                        .unwrap();
                    s_full.clone() * (dot_prod - next_elem)
                })
                .collect::<Vec<_>>()
        });

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
                s_partial.clone() * (pow_5(a_in_0 + c) - a_sbox_out_0.clone())
            };

            // Compute the first state element output by round A (i.e. B's first input state
            // element) and assert that it is equal to the dot product of round B's output state and
            // the first column of the inverse MDS matrix.
            let b_sbox_out_0 = {
                let b_in_0 = a_out(0, meta);
                let c = meta.query_fixed(rc_b[0], Rotation::cur());
                s_partial.clone() * (pow_5(b_in_0 + c) - b_sbox_out(0, meta))
            };

            // For each element `i > 0`, compute the `i`-th input state element for round B and
            // assert that its sum, with the corresponding round constant, is equal to round B's
            // `i`-th output state element (computed by taking the dot product of B's output state
            // with the `i`-th column of the inverse MDS matrix).
            let b_out_linear = (1..width).map(|i| {
                let b_in_i = a_out(i, meta);
                let c = meta.query_fixed(rc_b[i], Rotation::cur());
                s_partial.clone() * (b_in_i + c - b_sbox_out(i, meta))
            });

            iter::once(a_sbox_out_0)
                .chain(iter::once(b_sbox_out_0))
                .chain(b_out_linear)
                .collect::<Vec<_>>()
        });

        PoseidonConfig {
            state,
            partial_sbox,
            rc_a,
            rc_b,
            s_full,
            s_partial,
            _f: PhantomData,
            _a: PhantomData,
        }
    }

    fn construct(config: PoseidonConfig<F, A>) -> Self {
        PoseidonChip { config }
    }

    // Assign the initial state `domain tag || preimage` into the state vector of the provided row.
    fn assign_initial_state(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        preimage: &[AssignedCell<F, F>],
        row: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = A::to_usize() + 1;

        let mut state = Vec::<AssignedCell<F, F>>::with_capacity(width);

        // Assign the domain tag into a fixed column and copy the assigned value into the
        // first initial state element.
        let domain_tag = region.assign_fixed(
            || "domain tag",
            self.config.rc_b[0],
            row,
            || Ok(consts.domain_tag),
        )?;
        state.push(domain_tag.copy_advice(
            || "initial state 0 (domain tag)",
            region,
            self.config.state[0],
            row,
        )?);

        // Copy the preimage into the remaining initial state elements.
        for (i, elem) in preimage.iter().enumerate() {
            state.push(elem.copy_advice(
                || format!("initial state {} (preimage {})", i + 1, i),
                region,
                self.config.state[i + 1],
                row,
            )?);
        }

        Ok(state)
    }

    fn assign_round_constants(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        cols: &[Column<Fixed>],
        round: usize,
        row: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = A::to_usize() + 1;
        consts
            .round_constants
            .iter()
            .skip(round * width)
            .take(width)
            .zip(cols.iter())
            .enumerate()
            .map(|(i, (c, col))| {
                region.assign_fixed(
                    || format!("round constant {} (round {})", i, round),
                    *col,
                    row,
                    || Ok(*c),
                )
            })
            .collect()
    }

    fn assign_state(
        &self,
        region: &mut Region<F>,
        state: &[Option<F>],
        round: usize,
        row: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        state
            .iter()
            .zip(self.config.state.iter())
            .enumerate()
            .map(|(i, (opt, col))| {
                region.assign_advice(
                    || format!("state {} (round {})", i, round),
                    *col,
                    row,
                    || opt.ok_or(Error::Synthesis),
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
        row: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = A::to_usize() + 1;

        self.config.s_full.enable(region, row)?;
        self.assign_round_constants(region, consts, &self.config.rc_a, round, row)?;

        let round_consts = consts
            .round_constants
            .iter()
            .skip(round * width)
            .take(width);

        // Add a round constant to each state elememt, then apply the sbox to each sum.
        let sbox_out: Vec<Option<F>> = state
            .iter()
            .zip(round_consts)
            .map(|(assigned, c)| {
                assigned
                    .value()
                    .map(|elem| (*elem + c).pow_vartime([ALPHA]))
            })
            .collect();

        let next_state: Vec<Option<F>> = if sbox_out.iter().any(|opt| opt.is_none()) {
            vec![None; width]
        } else {
            // Multiply the sbox outputs by the MDS matrix.
            let mut next_state = Vec::with_capacity(width);
            for i in 0..width {
                let mds_col = consts.mds_matrices.m.iter().map(|row| row[i]);
                let mut dot_prod = F::zero();
                for (opt, m) in sbox_out.iter().zip(mds_col) {
                    dot_prod += opt.unwrap() * m;
                }
                next_state.push(Some(dot_prod));
            }
            next_state
        };

        self.assign_state(region, &next_state, round + 1, row + 1)
    }

    // Perform 2 partial rounds (A and B) on `state` and assign the output state in the next row.
    fn partial_rounds(
        &self,
        region: &mut Region<F>,
        consts: &PoseidonConstants<F, A>,
        state: &[AssignedCell<F, F>],
        round: usize,
        row: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let width = A::to_usize() + 1;

        let round_a = round;
        let round_b = round + 1;

        self.config.s_partial.enable(region, row)?;

        // Assign the first and second rounds' round constants in the `rc_a` and `rc_b` columns
        // respectively.
        self.assign_round_constants(region, consts, &self.config.rc_a, round_a, row)?;
        self.assign_round_constants(region, consts, &self.config.rc_b, round_b, row)?;

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

        // Add a round constant to each state elememt, then apply the sbox to the first sum.
        let sbox_out_a: Vec<Option<F>> = state
            .iter()
            .zip(round_consts_a)
            .enumerate()
            .map(|(i, (assigned, c))| {
                if i == 0 {
                    assigned
                        .value()
                        .map(|elem| (*elem + c).pow_vartime([ALPHA]))
                } else {
                    assigned.value().map(|elem| *elem + c)
                }
            })
            .collect();

        // Assign the first state element's sbox output in the `partial_sbox` column.
        region.assign_advice(
            || format!("partial sbox output (round {})", round_a),
            self.config.partial_sbox,
            row,
            || sbox_out_a[0].ok_or(Error::Synthesis),
        )?;

        let input_state_b: Vec<Option<F>> = if sbox_out_a.iter().any(|opt| opt.is_none()) {
            vec![None; width]
        } else {
            // Multiply the sbox outputs by the MDS matrix.
            (0..width)
                .map(|i| {
                    let mds_col = consts.mds_matrices.m.iter().map(|row| row[i]);
                    let mut dot_prod = F::zero();
                    for (opt, m) in sbox_out_a.iter().zip(mds_col) {
                        dot_prod += opt.unwrap() * m;
                    }
                    Some(dot_prod)
                })
                .collect()
        };

        // Add the associated round constant to each state elememt, then apply the sbox to the first
        // element.
        let sbox_out_b: Vec<Option<F>> = input_state_b
            .iter()
            .zip(round_consts_b)
            .enumerate()
            .map(|(i, (opt, c))| {
                if i == 0 {
                    opt.map(|elem| (elem + c).pow_vartime([ALPHA]))
                } else {
                    opt.map(|elem| elem + c)
                }
            })
            .collect();

        // Multiply the sbox outputs by the MDS matrix.
        let output_state_b: Vec<Option<F>> = if sbox_out_b.iter().any(|opt| opt.is_none()) {
            vec![None; width]
        } else {
            (0..width)
                .map(|i| {
                    let mds_col = consts.mds_matrices.m.iter().map(|row| row[i]);
                    let mut dot_prod = F::zero();
                    for (opt, m) in sbox_out_b.iter().zip(mds_col) {
                        dot_prod += opt.unwrap() * m;
                    }
                    Some(dot_prod)
                })
                .collect()
        };

        self.assign_state(region, &output_state_b, round_b + 1, row + 1)
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

        // This circuit does not support padding the preimage with zeros, i.e. if the hash-type is
        // `ConstantLength`, the constant length must be equal to the width.
        assert!(consts.hash_type.is_supported());
        if let HashType::ConstantLength(const_len) = consts.hash_type {
            assert_eq!(const_len, width);
        };

        // This gadget requires that both the number of full and partial rounds are even.
        assert_eq!(consts.full_rounds & 1, 0);
        assert_eq!(consts.partial_rounds & 1, 0);

        layouter.assign_region(
            || "poseidon",
            |mut region| {
                let mut round = 0;
                let mut row = 0;

                let mut state = self.assign_initial_state(&mut region, consts, preimage, row)?;

                for _ in 0..consts.half_full_rounds {
                    state = self.full_round(&mut region, consts, &state, round, row)?;
                    round += 1;
                    row += 1;
                }

                for _ in 0..consts.partial_rounds / 2 {
                    state = self.partial_rounds(&mut region, consts, &state, round, row)?;
                    round += 2;
                    row += 1;
                }

                for _ in 0..consts.half_full_rounds {
                    state = self.full_round(&mut region, consts, &state, round, row)?;
                    round += 1;
                    row += 1;
                }

                Ok(state[1].clone())
            },
        )
    }
}

pub fn poseidon_hash<F, A>(
    layouter: impl Layouter<F>,
    config: PoseidonConfig<F, A>,
    preimage: &[AssignedCell<F, F>],
    consts: &PoseidonConstants<F, A>,
) -> Result<AssignedCell<F, F>, Error>
where
    F: FieldExt,
    A: Arity<F>,
{
    PoseidonChip::construct(config).hash(layouter, preimage, consts)
}

#[cfg(test)]
mod tests {
    use super::*;

    use generic_array::typenum::{Unsigned, U11, U2, U4, U8};
    use halo2_proofs::{
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        pasta::{Fp, Fq},
        plonk::{Circuit, Instance},
    };

    use crate::{round_numbers::round_numbers_halo, Poseidon, Strength};

    #[derive(Clone, Debug)]
    struct MyConfig<F, A>
    where
        F: FieldExt,
        A: Arity<F>,
    {
        poseidon: PoseidonConfig<F, A>,
        digest_pi: Column<Instance>,
    }

    struct MyCircuit<F, A>
    where
        F: FieldExt,
        A: Arity<F>,
    {
        preimage: Vec<Option<F>>,
        _a: PhantomData<A>,
    }

    impl<F, A> Circuit<F> for MyCircuit<F, A>
    where
        F: FieldExt,
        A: Arity<F>,
    {
        type Config = MyConfig<F, A>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MyCircuit {
                preimage: vec![None; A::to_usize()],
                _a: PhantomData,
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let width = A::to_usize() + 1;

            let io = (0..width).map(|_| meta.advice_column()).collect();
            let extra = meta.advice_column();
            let fixed = (0..2 * width).map(|_| meta.fixed_column()).collect();
            let poseidon = PoseidonChip::configure(meta, io, extra, fixed);

            let digest_pi = meta.instance_column();
            meta.enable_equality(digest_pi);

            MyConfig {
                poseidon,
                digest_pi,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let arity = A::to_usize();

            let row = 0;

            let preimage: Vec<AssignedCell<F, F>> = layouter.assign_region(
                || "preimage",
                |mut region| {
                    self.preimage
                        .iter()
                        .zip(config.poseidon.state[..arity].iter())
                        .enumerate()
                        .map(|(i, (opt, col))| {
                            region.assign_advice(
                                || format!("preimage {}", i),
                                *col,
                                row,
                                || opt.ok_or(Error::Synthesis),
                            )
                        })
                        .collect()
                },
            )?;

            let digest = poseidon_hash(
                layouter.namespace(|| "poseidon"),
                config.poseidon.clone(),
                &preimage,
                &PoseidonConstants::<F, A>::new_with_strength(Strength::Halo),
            )?;

            layouter.constrain_instance(digest.cell(), config.digest_pi, row)?;

            Ok(())
        }
    }

    impl<F, A> MyCircuit<F, A>
    where
        F: FieldExt,
        A: Arity<F>,
    {
        fn with_witness(preimage: &[F]) -> Self {
            MyCircuit {
                preimage: preimage.iter().map(|elem| Some(*elem)).collect(),
                _a: PhantomData,
            }
        }

        fn generate_public_inputs(digest: F) -> Vec<Vec<F>> {
            vec![vec![digest]]
        }

        // `k = ceil(log2(num_circuit_rows))`
        fn k() -> u32 {
            let arity = A::to_usize();
            let (rf, rp) = round_numbers_halo(arity);
            let poseidon_rows = rf + rp / 2;
            // Add one row for preimage allocation.
            let num_rows = (poseidon_rows + 1) as f32;
            num_rows.log2().ceil() as u32
        }
    }

    fn test_halo2_circuit_inner<F, A>()
    where
        F: FieldExt,
        A: Arity<F>,
    {
        let arity = A::to_usize();
        let preimage: Vec<F> = (0..arity).map(|i| F::from(i as u64)).collect();
        let consts = PoseidonConstants::<F, A>::new_with_strength(Strength::Halo);
        let digest = Poseidon::new_with_preimage(&preimage, &consts).hash();
        let circ = MyCircuit::<F, A>::with_witness(&preimage);
        let pub_inputs = MyCircuit::<F, A>::generate_public_inputs(digest);
        let k = MyCircuit::<F, A>::k();
        let prover = MockProver::run(k, &circ, pub_inputs).unwrap();
        assert!(prover.verify().is_ok());
    }

    #[test]
    fn test_halo2_circuit() {
        test_halo2_circuit_inner::<Fp, U2>();
        test_halo2_circuit_inner::<Fp, U4>();
        test_halo2_circuit_inner::<Fp, U8>();
        test_halo2_circuit_inner::<Fp, U11>();

        test_halo2_circuit_inner::<Fq, U2>();
        test_halo2_circuit_inner::<Fq, U4>();
        test_halo2_circuit_inner::<Fq, U8>();
        test_halo2_circuit_inner::<Fq, U11>();
    }
}
