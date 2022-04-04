use ff::Field;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    pasta::Fp,
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, SingleVerifier,
    },
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};

use halo2_gadgets::{
    poseidon::{Hash, Pow5Chip, Pow5Config},
    primitives::poseidon::{self, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
};
use std::convert::TryInto;
use std::marker::PhantomData;

use rand::rngs::OsRng;
use generic_array::{self, arr, ArrayLength, typenum::{self, Unsigned, U2}, GenericArray};

#[derive(Debug, Clone, Copy)]
struct MySpec<const WIDTH: usize, const RATE: usize>;

impl Spec<Fp, 3, 2> for MySpec<3, 2> {
    fn full_rounds() -> usize {
        8
    }

    fn partial_rounds() -> usize {
        56
    }

    fn sbox(val: Fp) -> Fp {
        val.pow_vartime(&[5])
    }

    fn secure_mds() -> usize {
        0
    }
}


const K: u32 = 6;

#[derive(Clone, Copy)]
struct HashCircuit<S, const WIDTH: usize, const RATE: usize, L: ArrayLength<Fp>>
where
    S: Spec<Fp, WIDTH, RATE> + Clone + Copy,
    L::ArrayType: Copy,
{
    message: Option<GenericArray<Fp, L>>,
    // For the purpose of this test, witness the result.
    // TODO: Move this into an instance column.
    output: Option<Fp>,
    _spec: PhantomData<S>,
}

#[derive(Debug, Clone)]
struct MyConfig<const WIDTH: usize, const RATE: usize, L: ArrayLength<Column<Advice>>> {
    input: GenericArray<Column<Advice>, L>,
    poseidon_config: Pow5Config<Fp, WIDTH, RATE>,
}

impl<S, const WIDTH: usize, const RATE: usize, L: Unsigned> Circuit<Fp>
    for HashCircuit<S, WIDTH, RATE, L>
where
    S: Spec<Fp, WIDTH, RATE> + Copy + Clone,
    L: ArrayLength<Fp> + ArrayLength<Column<Advice>>, <L as ArrayLength<Fp>>::ArrayType: Copy,
{
    type Config = MyConfig<WIDTH, RATE, L>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            message: None,
            output: None,
            _spec: PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let gen_state = (0..WIDTH).map(|_| meta.advice_column()).collect::<GenericArray<_, _>>();
        let partial_sbox = meta.advice_column();

        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

        meta.enable_constant(rc_b[0]);

        let mut state = Vec::new();
        for i in 0..gen_state.len() {
            state.push(gen_state[i]);
        }

        // TODO: fix ugly things, use idiomatic way to convert when it is working
        let mut state_array: [Column<Advice>; WIDTH] = [gen_state[0]; WIDTH];

        state_array[0..state.len()].copy_from_slice(&state);

        Self::Config {
            input: gen_state,
            poseidon_config: Pow5Chip::configure::<S>(
                meta,
                state_array,
                partial_sbox,
                rc_a.try_into().unwrap(),
                rc_b.try_into().unwrap(),
            ),
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = Pow5Chip::construct(config.poseidon_config.clone());

        let message = layouter.assign_region(
            || "load message",
            |mut region| {
                let message_word = |i: usize| {
                    let value = self.message.map(|message_vals| message_vals[i]);
                    region.assign_advice(
                        || format!("load message_{}", i),
                        config.input[i],
                        0,
                        || value.ok_or(Error::Synthesis),
                    )
                };

                let message: Result<Vec<_>, Error> = (0..L::to_usize()).map(message_word).collect();
                Ok(message?.try_into().unwrap())
            },
        )?;

        let hasher = Hash::<_, _, S, ConstantLength<2>, WIDTH, RATE>::init(
            chip,
            layouter.namespace(|| "init"),
        )?;
        let output = hasher.hash(layouter.namespace(|| "hash"), message)?;

        layouter.assign_region(
            || "constrain output",
            |mut region| {
                let expected_var = region.assign_advice(
                    || "load output",
                    config.input[0],
                    0,
                    || self.output.ok_or(Error::Synthesis),
                )?;
                region.constrain_equal(output.cell(), expected_var.cell())
            },
        )
    }
}


#[test]
fn poseidon_halo2_gadget_test() {
    run_poseidon_test::<MySpec<3, 2>, 3, 2, U2>();
}

fn run_poseidon_test<S, const WIDTH: usize, const RATE: usize, L: ArrayLength<Fp>>()
where
    S: Spec<Fp, WIDTH, RATE> + Copy + Clone, <L as ArrayLength<Fp>>::ArrayType: Copy, L: ArrayLength<Column<Advice>>, GenericArray<Fp, L>: From<[Fp; 2]>
{
    let params: Params<vesta::Affine> = Params::new(K);

    let empty_circuit = HashCircuit::<S, WIDTH, RATE, L> {
        message: None,
        output: None,
        _spec: PhantomData,
    };

    // Initialize the proving key
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");

    let mut rng = OsRng;
    let message = (0..L::to_usize())
        .map(|_| pallas::Base::random(rng))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let output = poseidon::Hash::<_, S, ConstantLength<2>, WIDTH, RATE>::init().hash(message);

    let circuit = HashCircuit::<S, WIDTH, RATE, L> {
        message: Some(message.into()),
        output: Some(output),
        _spec: PhantomData,
    };

    // Create a proof
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
        .expect("proof generation should not fail");
    let proof = transcript.finalize();

    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).is_ok());
}
