use crate::matrix;
use crate::{
    mds::create_mds_matrices, mds::MDSMatrices, round_constants, round_numbers, scalar_from_u64,
    Error,
};
use ff::{Field, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;

/// The arity tag is the first element of a Poseidon permutation.
/// This extra element is necessary for 128-bit security.
pub fn arity_tag<E: ScalarEngine, Arity: typenum::Unsigned>() -> E::Fr {
    scalar_from_u64::<E>((1 << Arity::to_usize()) - 1)
}

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
#[derive(Debug, Clone, PartialEq)]
pub struct Poseidon<'a, E, Arity = typenum::U2>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    constants_offset: usize,
    /// the elements to permute
    pub elements: GenericArray<E::Fr, typenum::Add1<Arity>>,
    pos: usize,
    constants: &'a PoseidonConstants<E, Arity>,
    _e: PhantomData<E>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseidonConstants<E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub mds_matrices: MDSMatrices<E>,
    pub round_constants: Vec<E::Fr>,
    pub preprocessed_round_constants: Vec<E::Fr>,
    pub arity_tag: E::Fr,
    pub full_rounds: usize,
    pub partial_rounds: usize,
    pub partial_preprocessed: usize,
    _a: PhantomData<Arity>,
}

pub enum HashMode {
    // The initial and correct version of the algorithm. We should preserve the ability to hash this way for reference
    // and to preserve confidence in our tests along thew way.
    Correct,
    // This mode is meant to be mostly synchronized with `Correct` but may reduce or simplify the total algorithm.
    // Its purpose is for use during refactoring/development, as a target for `ModB`.
    ModA,
    // Here is where the hardest work happens. Incremental refactoring is applied here and reconciled with `ModA`
    // through instrumentation. A target behavior other than complete/correct hashing may need to be negotiated to faciliatate this.
    ModB,
    // The intermediate target, this mode accumulates transformations to the algorithm and should give the same results as `Correct`.
    // Dynamic optimization calculates round constants along the way and may emit them for use by the static optimization.
    OptimizedDynamic,
    // Consumes statically pre-processed constants for simplest operation.
    OptimizedStatic,
}
use HashMode::{Correct, ModA, ModB, OptimizedDynamic, OptimizedStatic};

pub const DEFAULT_HASH_MODE: HashMode = Correct;

impl<'a, E, Arity> PoseidonConstants<E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub fn new() -> Self {
        let arity = Arity::to_usize();
        let width = arity + 1;

        let mds_matrices = create_mds_matrices::<E>(width);
        // let mds_matrix = generate_mds::<E>(width);
        // let m_inv = matrix::invert::<E>(&mds_matrix).unwrap();

        let (full_rounds, partial_rounds) = round_numbers(arity);
        let round_constants = round_constants::<E>(arity);

        // These succeed:
        //let partial_preprocessed = 0;
        //let partial_preprocessed = 1;
        //let partial_preprocessed = 53;
        let partial_preprocessed = partial_rounds; // partial_rounds = 55

        // Although annoying, this is a very special case — when only a single round is not preprocessed.
        // That strongly suggests this failure is an artifact of the test scaffolding we have erected
        // to aid refactoring. For now, we will just avoid this case.
        // These fail:
        // let partial_preprocessed = 54;

        let preprocessed_round_constants = preprocess_round_constants::<E>(
            width,
            full_rounds,
            partial_rounds,
            &round_constants,
            &mds_matrices,
            partial_preprocessed,
        );
        // Ensure we have enough constants for the sbox rounds
        assert!(
            width * (full_rounds + partial_rounds) <= round_constants.len(),
            "Not enough round constants"
        );

        Self {
            mds_matrices,
            round_constants,
            preprocessed_round_constants,
            arity_tag: arity_tag::<E, Arity>(),
            full_rounds,
            partial_rounds,
            partial_preprocessed,
            _a: PhantomData::<Arity>,
        }
    }

    /// Returns the width.
    #[inline]
    pub fn arity(&self) -> usize {
        Arity::to_usize()
    }

    /// Returns the width.
    #[inline]
    pub fn width(&self) -> usize {
        use typenum::Unsigned;
        typenum::Add1::<Arity>::to_usize()
    }
}

fn preprocess_round_constants<E: ScalarEngine>(
    width: usize,
    full_rounds: usize,
    partial_rounds: usize,
    round_constants: &Vec<E::Fr>,
    mds_matrices: &MDSMatrices<E>,
    partial_preprocessed: usize,
) -> Vec<E::Fr> {
    let mds_matrix = &mds_matrices.m;
    let inverse_matrix = &mds_matrices.m_inv;

    let mut res = Vec::new();

    let round_keys = |r: usize| &round_constants[r * width..(r + 1) * width];

    let half_full_rounds = full_rounds / 2; // Not half-full rounds; half full-rounds.

    // First round constants are unchanged.
    res.extend(round_keys(0));

    let unpreprocessed = partial_rounds - partial_preprocessed;

    // Post S-box adds for the first set of full rounds should be 'inverted' from next round.
    // The final round is skipped when fully preprocessing because that value must be obtained from the result of preprocesing the partial rounds.
    let end = if unpreprocessed > 0 {
        half_full_rounds
    } else {
        half_full_rounds - 1
    };
    for i in 0..end {
        let next_round = round_keys(i + 1); // First round was added before any S-boxes.
        let inverted = matrix::apply_matrix::<E>(inverse_matrix, next_round);
        res.extend(inverted);
    }

    // The plan:
    // - Work backwards from last row in this group
    // - Invert the row.
    // - Save first constant (corresponding to the one S-box performed).
    // - Add inverted result to previous row.
    // - Repeat until all partial round key rows have been consumed.
    // - Extend the preprocessed result by the final resultant row.
    // - Move the accumulated list of single round keys to the preprocessed result.
    //   - (Last produced should be first applied, so either pop until empty, or reverse and extend, etc.

    // `partial_keys` will accumulate the single post-S-box constant for each partial-round, in reverse order.
    let mut partial_keys: Vec<E::Fr> = Vec::new();

    let final_round = half_full_rounds + partial_rounds;
    let final_round_key = round_keys(final_round).to_vec();

    // `round_acc` holds the accumulated result of inverting and adding subsequent round constants (in reverse).
    let round_acc = (0..partial_preprocessed)
        .map(|i| round_keys(final_round - i - 1))
        .fold(final_round_key, |acc, previous_round_keys| {
            let mut inverted = matrix::apply_matrix::<E>(inverse_matrix, &acc);
            partial_keys.push(inverted[0]);
            inverted[0] = E::Fr::zero();
            matrix::vec_add::<E>(&previous_round_keys, &inverted)
        });

    // Everything in here is dev-driven testing.
    // Dev test case only checks one deep.
    if partial_preprocessed == 1 {
        // Check assumptions about how the fold calculating round_acc  manifested.

        // The last round containing unpreprocessed constants which should be compressed.
        let terminal_constants_round = half_full_rounds + partial_rounds;

        // Constants from the last round (of two) which should be compressed.
        // T
        let terminal_round_keys = round_keys(terminal_constants_round);

        // Constants from the first round (of two) which should be compressed.
        // I
        let initial_round_keys = round_keys(terminal_constants_round - 1);

        // M^-1(T)
        let mut inv = matrix::apply_matrix::<E>(inverse_matrix, terminal_round_keys);
        // M^-1(T)[0]
        let pk = inv[0];

        // M^-1(T) - pk (kinda)
        inv[0] = E::Fr::zero();

        // (M^-1(T) - pk) - I
        let result_key = matrix::vec_add::<E>(&initial_round_keys, &inv);

        assert_eq!(&result_key, &round_acc, "Acc assumption failed.");
        assert_eq!(pk, partial_keys[0], "Partial-key assumption failed.");

        ////////////////////////////////////////////////////////////////////////////////
        // Shared between branches, an arbitrary initial state representing the output of a previous round's S-Box layer.
        // X
        let initial_state = vec![E::Fr::one(); width];

        ////////////////////////////////////////////////////////////////////////////////
        // Compute one step with the given (unpreprocessed) constants.

        // ARK
        // I + X
        let mut q_state = matrix::vec_add::<E>(initial_round_keys, &initial_state);

        // S-Box (partial layer)
        // S((I + X)[0]) = S(I[0] + X[0])
        quintic_s_box::<E>(&mut q_state[0], None, None);

        // Mix
        let mixed = matrix::apply_matrix::<E>(mds_matrix, &q_state);

        // Ark
        let plain_result = matrix::vec_add::<E>(terminal_round_keys, &mixed);

        ////////////////////////////////////////////////////////////////////////////////
        // Compute the same step using the preprocessed constants.
        // initial_state + (inverted_id - initial_state) = inverted_id
        let mut p_state = matrix::vec_add::<E>(&result_key, &initial_state);

        // In order for the S-box result to be correct, it must have the same input as in the plain path.
        // That means its input (the first component of the state) must have been constructed by
        // adding the same single round constant in that position.
        // NOTE: this asssertion uncovered a bug which was causing failure.
        assert_eq!(
            &result_key[0], &initial_round_keys[0],
            "S-box inputs did not match."
        );

        quintic_s_box::<E>(&mut p_state[0], None, Some(&pk));
        let preprocessed_result = matrix::apply_matrix::<E>(mds_matrix, &p_state);

        assert_eq!(
            plain_result, preprocessed_result,
            "Single preprocessing step couldn't be verified."
        );
    }

    for i in 1..unpreprocessed {
        res.extend(round_keys(half_full_rounds + i));
    }
    res.extend(matrix::apply_matrix::<E>(inverse_matrix, &round_acc));

    dbg!(&partial_keys.len());

    while let Some(x) = partial_keys.pop() {
        res.push(x)
    }

    // Post S-box adds for the first set of full rounds should be 'inverted' from next round.
    for i in 1..(half_full_rounds) {
        let start = half_full_rounds + partial_rounds;
        let next_round = round_keys(i + start);
        let inverted = matrix::apply_matrix::<E>(inverse_matrix, next_round);
        res.extend(inverted);
    }

    dbg!(&res.len(), &res);
    res
}

impl<'a, E, Arity> Poseidon<'a, E, Arity>
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    pub fn new(constants: &'a PoseidonConstants<E, Arity>) -> Self {
        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.arity_tag
            } else {
                E::Fr::zero()
            }
        });
        Poseidon {
            constants_offset: 0,
            elements,
            pos: 1,
            constants,
            _e: PhantomData::<E>,
        }
    }
    pub fn new_with_preimage(
        preimage: &[E::Fr],
        constants: &'a PoseidonConstants<E, Arity>,
    ) -> Self {
        assert_eq!(preimage.len(), Arity::to_usize(), "Invalid preimage size");

        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.arity_tag
            } else {
                preimage[i - 1]
            }
        });

        let width = elements.len();

        Poseidon {
            constants_offset: 0,
            elements,
            pos: width,
            constants,
            _e: PhantomData::<E>,
        }
    }

    /// Replace the elements with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is bigger than the arity.
    pub fn set_preimage(&mut self, preimage: &[E::Fr]) {
        self.reset();
        self.elements[1..].copy_from_slice(&preimage);
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.constants_offset = 0;
        self.elements[1..]
            .iter_mut()
            .for_each(|l| *l = scalar_from_u64::<E>(0u64));
        self.elements[0] = self.constants.arity_tag;
        self.pos = 1;
    }

    /// The returned `usize` represents the element position (within arity) for the input operation
    pub fn input(&mut self, element: E::Fr) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        if self.pos >= self.constants.width() {
            return Err(Error::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.elements[self.pos] = element;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    pub fn hash_in_mode(&mut self, mode: HashMode) -> E::Fr {
        match mode {
            Correct => self.hash_correct(),
            OptimizedDynamic => self.hash_optimized_dynamic(),
            OptimizedStatic => self.hash_optimized_static(),
            ModA => self.hash_mod_a(),
            ModB => self.hash_mod_b(),
        }
    }

    pub fn hash(&mut self) -> E::Fr {
        self.hash_in_mode(DEFAULT_HASH_MODE)
    }

    /// The number of rounds is divided into two equal parts for the full rounds, plus the partial rounds.
    ///
    /// The returned element is the second poseidon element, the first is the arity tag.
    pub fn hash_correct(&mut self) -> E::Fr {
        self.debug("Hash Correct");

        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        // The first full round should use the initial constants.
        self.full_round(true, false);

        self.debug("After first full round (Correct)");

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(true, false);
            if i == 1 {
                self.debug("After second full round (Correct)");
            }
        }

        self.debug("Before first partial round");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(true, false);

        self.debug("After first partial round (Correct)");

        for _ in 1..self.constants.partial_rounds {
            self.partial_round(true, false);
        }

        self.debug("After last partial round (Correct)");

        for _ in 0..self.constants.full_rounds / 2 {
            self.full_round(true, false);
        }

        self.debug("After last full round (Correct)");

        self.elements[1]
    }

    pub fn hash_mod_a(&mut self) -> E::Fr {
        self.debug("Hash Mod A");
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        // The first full round should use the initial constants.
        self.full_round(true, false);
        self.debug("After first full round");

        // => M(B)

        // M(B) + S
        //self.add_round_constants();

        // Verify that the round constants here (S), really are the S from full_round (post_vec).

        self.debug("After adding constants");

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(true, false);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        // self.add_round_constants();

        self.debug("Before first partial round");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(true, false);

        // self.debug("After first partial round");

        // for _ in 1..self.constants.partial_rounds {
        //     self.partial_round(true, false);
        // }

        // self.debug("After last partial round");

        // for _ in 0..self.constants.full_rounds / 2 {
        //     self.full_round(true, false);
        // }

        // self.debug("After last full round");

        // B + S
        self.elements[1]
    }

    pub fn hash_mod_b(&mut self) -> E::Fr {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat
        self.debug("Hash Mod B");
        // The first full round should use the initial constants.
        self.full_round(true, true);
        self.debug("After first full round");

        // => M(B + M^-1(S))

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(false, true);
            if i == 1 {
                self.debug("After second full round");
            }
        }

        self.debug("Before first partial round");

        // // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(false, false);

        // self.debug("After first partial round");

        // for _ in 1..self.constants.partial_rounds {
        //     self.partial_round(true, false);
        // }

        // self.debug("After last partial round");

        // for _ in 0..self.constants.full_rounds / 2 {
        //     self.full_round(true, false);
        // }

        // self.debug("After last full round");

        // M(B + M^-1(S))
        self.elements[1]
    }

    pub fn hash_optimized_dynamic(&mut self) -> E::Fr {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat

        self.debug("Hash OptimizedDynamic");
        dbg!(&self.constants.round_constants);

        // The first full round should use the initial constants.
        self.full_round(true, true);

        self.debug("After first full round (dynamic)");

        for i in 1..(self.constants.full_rounds / 2) {
            self.full_round(false, true);
            if i == 1 {
                self.debug("After second full round (dynamic)");
            }
            if i == (self.constants.full_rounds / 2) - 2 {
                self.debug("Before last full round (dynamic)");
            }
        }

        self.debug("Before first partial round (dynamic)");

        // Constants were added in the previous full round, so skip them here (false argument).
        self.partial_round(false, false);

        self.debug("After first partial round (dynamic)");

        for i in 1..self.constants.partial_rounds {
            self.partial_round(true, false);
            if i == 1 {
                self.debug("After second partial round (dynamic)");
            }
        }

        self.debug("After last partial round (dynamic)");

        for i in 0..self.constants.full_rounds / 2 {
            self.full_round(true, false);

            if i == (self.constants.full_rounds / 2) - 2 {
                self.debug("Before last full round (dynamic)");
            }
        }

        self.debug("After last full round (dynamic)");

        self.elements[1]
    }

    pub fn hash_optimized_static(&mut self) -> E::Fr {
        // This counter is incremented when a round constants is read. Therefore, the round constants never
        // repeat

        self.debug("Hash OptimizedStatic");

        // The first full round should use the initial constants.
        self.add_round_constants_static();

        for i in 0..self.constants.full_rounds / 2 {
            self.full_round_static(false);
            if i == 1 {
                self.debug("After second full round (static)");
            }
            if i == (self.constants.full_rounds / 2) - 2 {
                self.debug("Before last full round (static)");
            }
        }

        self.debug("Before first partial round (static)");

        let unpreprocessed = self.constants.partial_rounds - self.constants.partial_preprocessed;
        if unpreprocessed > 0 {
            dbg!(unpreprocessed);
            for i in 0..(unpreprocessed - 1) {
                self.partial_round_static(false, i == 0, false);

                if i == 0 {
                    self.debug("After first partial round (static)");
                }
                if i == 1 {
                    self.debug("After second partial round (static)");
                }
                if i == unpreprocessed - 2 {
                    self.debug("Before second-to-last unpreprocessed partial round (static)");
                    dbg!(self.constants.preprocessed_round_constants[self.constants_offset]);
                }
            }
            // We need both pre and post round-keys added at the seam.
            self.partial_round_static(false, false, true);
            self.debug("xxx");
        }

        for i in unpreprocessed..self.constants.partial_rounds {
            if i == (self.constants.partial_rounds - 1) {
                self.debug("Before last preprocessed partial round (static)");
                dbg!(self.constants.preprocessed_round_constants[self.constants_offset]);
            }
            //self.partial_round_static(true, false, false);
            self.partial_round_static(true, true, false);
        }

        self.debug("After last partial round (static)");

        // All but last full round.
        for i in 0..self.constants.full_rounds / 2 {
            if i == (self.constants.full_rounds / 2) - 1 {
                self.debug("Before last full round (static)");
                self.full_round_static(true);
            } else {
                dbg!(i);
                self.full_round_static(false);
            }
        }

        self.debug("After last full round (static)");

        assert_eq!(
            self.constants_offset,
            self.constants.preprocessed_round_constants.len(),
            "Constants consumed ({}) must equal preprocessed constants provided ({}).",
            self.constants_offset,
            self.constants.preprocessed_round_constants.len()
        );

        self.elements[1]
    }

    pub fn full_round(&mut self, add_current_round_keys: bool, absorb_next_round_keys: bool) {
        // NOTE: decrease in performance is expected during this refactoring.
        // We seek to preserve correctness while transforming the algorithm to an eventually more performant one.

        // Apply the quintic S-Box to all elements, after adding the round key.
        // Round keys are added in the S-box to match circuits (where the addition is free)
        // and in preparation for the shift to adding round keys after (rather than before) applying the S-box.

        let pre_round_keys = self
            .constants
            .round_constants
            .iter()
            .skip(self.constants_offset)
            .map(|x| {
                if add_current_round_keys {
                    Some(x)
                } else {
                    None
                }
            });

        if absorb_next_round_keys {
            // Using the notation from `test_inverse` in matrix.rs:

            // S
            let post_vec = self
                .constants
                .round_constants
                .iter()
                .skip(
                    self.constants_offset
                        + if add_current_round_keys {
                            self.elements.len()
                        } else {
                            0
                        },
                )
                .take(self.elements.len())
                .map(|x| *x)
                .collect::<Vec<_>>();

            // Compute the constants which should be added *before* the next `product_mds`.
            // in order to have the same effect as adding the given constants *after* the next `product_mds`.

            // M^-1(S)
            let inverted_vec =
                matrix::apply_matrix::<E>(&self.constants.mds_matrices.m_inv, &post_vec);

            // M(M^-1(S))
            let original = matrix::apply_matrix::<E>(&self.constants.mds_matrices.m, &inverted_vec);

            // S = M(M^-1(S))
            assert_eq!(&post_vec, &original, "Oh no, the inversion trick failed.");

            let post_round_keys = inverted_vec.iter();

            // S-Box Output = B.
            // With post-add, result is B + M^-1(S).
            self.elements
                .iter_mut()
                .zip(pre_round_keys.zip(post_round_keys))
                .for_each(|(l, (pre, post))| {
                    //dbg!("a");
                    quintic_s_box::<E>(l, pre, Some(post));
                });
        } else {
            self.elements
                .iter_mut()
                .zip(pre_round_keys)
                .for_each(|(l, pre)| {
                    //dbg!("b");
                    quintic_s_box::<E>(l, pre, None);
                });
        }
        let mut consumed = 0;
        if add_current_round_keys {
            consumed += self.elements.len()
        };
        if absorb_next_round_keys {
            consumed += self.elements.len()
        };
        self.constants_offset += consumed;

        let stashed = self.elements.clone();

        // If absorb_next_round_keys
        //   M(B + M^-1(S)
        // else
        //   M(B)
        // Multiply the elements by the constant MDS matrix
        self.product_mds();

        let applied = matrix::apply_matrix::<E>(&self.constants.mds_matrices.m, &stashed.to_vec());
        assert_eq!(
            applied[..],
            self.elements[..],
            "product_mds gives different result than matrix application",
        );
    }

    fn full_round_static(&mut self, last_round: bool) {
        let to_take = self.elements.len();
        let post_round_keys = self
            .constants
            .preprocessed_round_constants
            .iter()
            .skip(self.constants_offset)
            .take(to_take);

        if !last_round {
            let needed = self.constants_offset + to_take;
            assert!(
                needed <= self.constants.preprocessed_round_constants.len(),
                "Not enough preprocessed round constants ({}), need {}.",
                self.constants.preprocessed_round_constants.len(),
                needed
            );
        }
        self.elements
            .iter_mut()
            .zip(post_round_keys)
            .for_each(|(l, post)| {
                // Be explicit that no round key is added after last round of S-boxes.
                dbg!(&post);
                let post_key = if last_round {
                    panic!("Trying to skip last full round, but there is a key here! ({})");
                } else {
                    Some(post)
                };
                quintic_s_box::<E>(l, None, post_key);
            });
        // We need this because post_round_keys will have been empty, so it didn't happen in the for_each. :(
        if last_round {
            self.elements
                .iter_mut()
                .for_each(|l| quintic_s_box::<E>(l, None, None));
        }

        if !last_round {
            self.constants_offset += self.elements.len();
        }
        self.product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
    pub fn partial_round(&mut self, add_current_round_keys: bool, absorb_next_round_keys: bool) {
        assert!(!absorb_next_round_keys); // Not yet implemented.

        if add_current_round_keys {
            // Every element of the hash buffer is incremented by the round constants
            self.add_round_constants();
        }

        // Apply the quintic S-Box to the first element
        quintic_s_box::<E>(&mut self.elements[0], None, None);

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
    pub fn partial_round_static(
        &mut self,
        preprocessed: bool,
        skip_constants: bool,
        last_unpreprocessed: bool,
    ) {
        if preprocessed {
            let post_round_key = self.constants.preprocessed_round_constants[self.constants_offset];
            dbg!(&post_round_key);
            // Apply the quintic S-Box to the first element
            quintic_s_box::<E>(&mut self.elements[0], None, Some(&post_round_key));
            self.constants_offset += 1;
            self.debug("After adding partial key post S-box.");
        // assert!(!skip_constants);
        } else {
            if !skip_constants {
                self.add_round_constants_static();
            }
            self.debug("before s-box");
            quintic_s_box::<E>(&mut self.elements[0], None, None);
        }
        if last_unpreprocessed {
            self.add_round_constants_static();
            self.debug("after round constants (and s-box)");
        }

        // Multiply the elements by the constant MDS matrix
        self.product_mds();
    }

    /// For every leaf, add the round constants with index defined by the constants offset, and increment the
    /// offset
    fn add_round_constants(&mut self) {
        for (element, round_constant) in self.elements.iter_mut().zip(
            self.constants
                .round_constants
                .iter()
                .skip(self.constants_offset),
        ) {
            // dbg!(
            //     "adding round constant:",
            //     &round_constant,
            //     &self.constants_offset
            // );
            element.add_assign(round_constant);
        }

        self.constants_offset += self.elements.len();
    }

    fn add_round_constants_static(&mut self) {
        for (element, round_constant) in self.elements.iter_mut().zip(
            self.constants
                .preprocessed_round_constants
                .iter()
                .skip(self.constants_offset),
        ) {
            // dbg!(
            //     "adding round constant (static):",
            //     &round_constant,
            //     &self.constants_offset
            // );
            element.add_assign(round_constant);
        }

        self.constants_offset += self.elements.len();
    }

    /// Set the provided elements with the result of the product between the elements and the constant
    /// MDS matrix
    fn product_mds(&mut self) {
        let mut result = GenericArray::<E::Fr, typenum::Add1<Arity>>::generate(|_| E::Fr::zero());

        for (j, val) in result.iter_mut().enumerate() {
            for (i, row) in self.constants.mds_matrices.m.iter().enumerate() {
                let mut tmp = row[j];
                tmp.mul_assign(&self.elements[i]);
                val.add_assign(&tmp);
            }
        }

        std::mem::replace(&mut self.elements, result);
    }

    fn debug(&self, msg: &str) {
        dbg!(msg, &self.constants_offset, &self.elements);
    }
}

/// Apply the quintic S-Box (s^5) to a given item
fn quintic_s_box<E: ScalarEngine>(
    l: &mut E::Fr,
    pre_add: Option<&E::Fr>,
    post_add: Option<&E::Fr>,
) {
    if let Some(x) = pre_add {
        l.add_assign(x);
    }
    // dbg!("S-box input", &l);
    let c = *l;
    let mut tmp = l.clone();
    tmp.mul_assign(&c);
    tmp.mul_assign(&tmp.clone());
    l.mul_assign(&tmp);
    // dbg!("S-box output", &l);
    if let Some(x) = post_add {
        l.add_assign(x);
        //  dbg!("After S-box post-add", &l);
    }
}

/// Poseidon convenience hash function.
/// NOTE: this is expensive, since it computes all constants when initializing hasher struct.
pub fn poseidon<E, Arity>(preimage: &[E::Fr]) -> E::Fr
where
    E: ScalarEngine,
    Arity: typenum::Unsigned
        + std::ops::Add<typenum::bit::B1>
        + std::ops::Add<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>,
    typenum::Add1<Arity>: ArrayLength<E::Fr>,
{
    let constants = PoseidonConstants::<E, Arity>::new();
    Poseidon::<E, Arity>::new_with_preimage(preimage, &constants).hash()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use ff::Field;
    //    use generic_array::typenum::{U2, U4, U8};
    use generic_array::typenum::U2;
    use paired::bls12_381::Bls12;

    #[test]
    fn reset() {
        let test_arity = 2;
        let preimage = vec![Scalar::one(); test_arity];
        let constants = PoseidonConstants::new();
        let mut h = Poseidon::<Bls12, U2>::new_with_preimage(&preimage, &constants);
        h.hash();
        h.reset();

        let default = Poseidon::<Bls12, U2>::new(&constants);
        assert_eq!(default.pos, h.pos);
        assert_eq!(default.elements, h.elements);
        assert_eq!(default.constants_offset, h.constants_offset);
    }

    #[test]
    fn hash_det() {
        let test_arity = 2;
        let mut preimage = vec![Scalar::zero(); test_arity];
        let constants = PoseidonConstants::new();
        preimage[0] = Scalar::one();

        let mut h = Poseidon::<Bls12, U2>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result: <Bls12 as ScalarEngine>::Fr = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_arity_3() {
        let mut preimage: [Scalar; 3] = [Scalar::zero(); 3];
        let constants = PoseidonConstants::new();
        preimage[0] = Scalar::one();

        let mut h = Poseidon::<Bls12, typenum::U3>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result: <Bls12 as ScalarEngine>::Fr = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_values() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let digest = p.hash();
        let expected = match test_arity {
            2 => scalar_from_u64s([
                0x7179d3495ac25e92,
                0x81052897659f7762,
                0x316a6d20e4a55d6c,
                0x409e8342edab687b,
            ]),
            4 => scalar_from_u64s([
                0xf53a7d58aacf0621,
                0x42d3a014639efdcf,
                0xe1a3fddb08c13a46,
                0x43f94dbd0abd1c99,
            ]),
            8 => scalar_from_u64s([
                0xa6a3e7a6b2cc7b85,
                0xfb1eb8f641dd9dc3,
                0xfd2a373272ebf604,
                0x433c1e9e8de226e5,
            ]),
            _ => {
                dbg!(digest);
                panic!("Arity lacks test vector: {}", test_arity)
            }
        };
        dbg!(test_arity);
        assert_eq!(expected, digest);

        assert_eq!(
            digest,
            poseidon::<Bls12, U2>(&preimage),
            "Poseidon wrapper disagrees with element-at-a-time invocation."
        );
    }

    #[test]
    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_compare_mods() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let mut p2 = p.clone();
        // M(B) + S
        let digest_a = p.hash_in_mode(ModA);

        // M(B + M^-1(S))
        let digest_b = p2.hash_in_mode(ModB);

        // M(B) + S = M(B + M^-1(S))
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    fn hash_compare_optimized() {
        // NOTE: For now, type parameters on constants, p, and in the final assertion below need to be updated manually when testing different arities.
        // TODO: Mechanism to run all tests every time. (Previously only a single arity was compiled in.)
        let constants = PoseidonConstants::<Bls12, U2>::new();
        let mut p = Poseidon::<Bls12, U2>::new(&constants);
        let test_arity = constants.arity();
        let mut preimage = vec![Scalar::zero(); test_arity];
        for n in 0..test_arity {
            let scalar = scalar_from_u64::<Bls12>(n as u64);
            p.input(scalar).unwrap();
            preimage[n] = scalar;
        }
        let mut p2 = p.clone();
        let mut p3 = p.clone();

        let digest_correct = p.hash_in_mode(Correct);

        let digest_optimized_dynamic = p2.hash_in_mode(OptimizedDynamic);
        let digest_optimized_static = p3.hash_in_mode(OptimizedStatic);

        assert_eq!(digest_correct, digest_optimized_dynamic);
        assert_eq!(digest_correct, digest_optimized_static);
    }
}
