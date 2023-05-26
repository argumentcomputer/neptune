use crate::hash_type::HashType;
use crate::poseidon::{Arity, Poseidon, PoseidonConstants};
use crate::sponge::api::{Hasher, IOPattern, InnerSpongeAPI, SpongeOp};
use crate::{Error, Strength};
use ff::PrimeField;
use std::collections::VecDeque;

// General information on sponge construction: https://keccak.team/files/CSF-0.1.pdf

/*
A sponge can be instantiated in either simplex or duplex mode. Once instantiated, a sponge's mode never changes.

At any time, a sponge is operating in one of two directions: squeezing or absorbing. All sponges are initialized in the
absorbing direction. The number of absorbed field elements is incremented each time an element is absorbed and
decremented each time an element is squeezed. In duplex mode, the count of currently absorbed elements can never
decrease below zero, so only as many elements as have been absorbed can be squeezed at any time. In simplex mode, there
is no limit on the number of elements that can be squeezed, once absorption is complete.

In simplex mode, absorbing and squeezing cannot be interleaved. First all elements are absorbed, then all needed
elements are squeezed. At most the number of elements which were absorbed can be squeezed. Elements must be absorbed in
chunks of R (rate). After every R chunks have been absorbed, the state is permuted. After the final element has been
absorbed, any needed padding is added, and the final permutation (or two -- if required by padding) is performed. Then
groups of R field elements are squeezed, and the state is permuted after each group of R elements has been squeezed.
After squeezing is complete, a simplex sponge is exhausted, and no further absorption is possible.

In duplex mode, absorbing and squeezing can be interleaved. The state is permuted after every R elements have been
absorbed. This makes R elements available to be squeezed. If elements remain to be squeezed when the state is permuted,
remaining unsqueezed elements are queued. Otherwise they would be lost when permuting.

*/

pub enum SpongeMode {
    SimplexAbsorb,
    SimplexSqueeze,
    DuplexAbsorb,
    DuplexSqueeze,
}

#[derive(Clone, Copy)]
pub enum Mode {
    Simplex,
    Duplex,
}

#[derive(Clone, Copy)]
pub enum Direction {
    Absorbing,
    Squeezing,
}

pub struct Sponge<'a, F: PrimeField, A: Arity<F>> {
    absorbed: usize,
    squeezed: usize,
    pub state: Poseidon<'a, F, A>,
    mode: Mode,
    direction: Direction,
    squeeze_pos: usize,
    queue: VecDeque<F>,
    pattern: IOPattern,
    io_count: usize,
}

pub trait SpongeTrait<'a, F: PrimeField, A: Arity<F>>
where
    Self: Sized,
{
    type Acc;
    type Elt;
    type Error;

    fn new_with_constants(constants: &'a PoseidonConstants<F, A>, mode: Mode) -> Self;

    fn simplex_constants(size: usize) -> PoseidonConstants<F, A> {
        PoseidonConstants::new_constant_length(size)
    }

    fn duplex_constants() -> PoseidonConstants<F, A> {
        PoseidonConstants::new_constant_length(0)
    }

    fn api_constants(strength: Strength) -> PoseidonConstants<F, A> {
        PoseidonConstants::new_with_strength_and_type(strength, HashType::Sponge)
    }

    fn mode(&self) -> Mode;
    fn direction(&self) -> Direction;
    fn set_direction(&mut self, direction: Direction);
    fn absorbed(&self) -> usize;
    fn set_absorbed(&mut self, absorbed: usize);
    fn squeezed(&self) -> usize;
    fn set_squeezed(&mut self, squeezed: usize);
    fn squeeze_pos(&self) -> usize;
    fn set_squeeze_pos(&mut self, squeeze_pos: usize);
    fn absorb_pos(&self) -> usize;
    fn set_absorb_pos(&mut self, pos: usize);

    fn element(&self, index: usize) -> Self::Elt;
    fn set_element(&mut self, index: usize, elt: Self::Elt);

    /// `make_elt` is deprecated and will be removed. Do not use.
    fn make_elt(&self, val: F, acc: &mut Self::Acc) -> Self::Elt;

    fn is_simplex(&self) -> bool {
        match self.mode() {
            Mode::Simplex => true,
            Mode::Duplex => false,
        }
    }
    fn is_duplex(&self) -> bool {
        match self.mode() {
            Mode::Duplex => true,
            Mode::Simplex => false,
        }
    }

    fn is_absorbing(&self) -> bool {
        match self.direction() {
            Direction::Absorbing => true,
            Direction::Squeezing => false,
        }
    }

    fn is_squeezing(&self) -> bool {
        match self.direction() {
            Direction::Squeezing => true,
            Direction::Absorbing => false,
        }
    }

    fn available(&self) -> usize {
        self.absorbed() - self.squeezed()
    }

    fn is_immediately_squeezable(&self) -> bool {
        self.squeeze_pos() < self.absorb_pos()
    }

    fn rate(&self) -> usize;

    fn capacity(&self) -> usize;

    fn size(&self) -> usize;

    fn total_size(&self) -> usize {
        assert!(self.is_simplex());
        match self.constants().hash_type {
            HashType::ConstantLength(l) => l,
            HashType::VariableLength => unimplemented!(),
            _ => A::to_usize(),
        }
    }

    fn constants(&self) -> &PoseidonConstants<F, A>;

    fn can_squeeze_without_permuting(&self) -> bool {
        self.squeeze_pos() < self.size() - self.capacity()
    }

    fn is_exhausted(&self) -> bool {
        // Exhaustion only applies to simplex.
        self.is_simplex() && self.squeezed() >= self.total_size()
    }

    fn ensure_absorbing(&mut self) {
        match self.direction() {
            Direction::Absorbing => (),
            Direction::Squeezing => {
                if self.is_simplex() {
                    panic!("Simplex sponge cannot absorb after squeezing.");
                } else {
                    self.set_direction(Direction::Absorbing);
                }
            }
        }
    }

    fn permute(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error> {
        // NOTE: this will apply any needed padding in the partially-absorbed case.
        // However, padding should only be applied when no more elements will be absorbed.
        // A duplex sponge should never apply padding implicitly, and a simplex sponge should only do so when it is
        // about to apply its final permutation.
        let unpermuted = self.absorb_pos();
        let needs_padding = self.is_absorbing() && unpermuted < self.rate();

        if needs_padding {
            match self.mode() {
                Mode::Duplex => {
                    panic!("Duplex sponge must permute exactly `rate` absorbed elements.")
                }
                Mode::Simplex => {
                    let final_permutation = self.squeezed() % self.total_size() <= self.rate();
                    assert!(
                        final_permutation,
                        "Simplex sponge may only pad before final permutation"
                    );
                    self.pad();
                }
            }
        }

        self.permute_state(acc)?;
        self.set_absorb_pos(0);
        self.set_squeeze_pos(0);
        Ok(())
    }

    fn pad(&mut self);

    fn permute_state(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error>;

    fn ensure_squeezing(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error> {
        match self.direction() {
            Direction::Squeezing => (),
            Direction::Absorbing => {
                match self.mode() {
                    Mode::Simplex => {
                        let done_squeezing_previous = self.squeeze_pos() >= self.rate();
                        let partially_absorbed = self.absorb_pos() > 0;

                        if done_squeezing_previous || partially_absorbed {
                            self.permute(acc)?;
                        }
                    }
                    Mode::Duplex => (),
                }
                self.set_direction(Direction::Squeezing);
            }
        }
        Ok(())
    }

    fn squeeze_aux(&mut self) -> Self::Elt;

    fn absorb_aux(&mut self, elt: &Self::Elt) -> Self::Elt;

    /// Absorb one field element
    fn absorb(&mut self, elt: &Self::Elt, acc: &mut Self::Acc) -> Result<(), Self::Error> {
        self.ensure_absorbing();

        // Add input element to state and advance absorption position.
        let tmp = self.absorb_aux(elt);
        self.set_element(self.absorb_pos() + self.capacity(), tmp);
        self.set_absorb_pos(self.absorb_pos() + 1);

        // When position equals size, we need to permute.
        if self.absorb_pos() >= self.rate() {
            if self.is_duplex() {
                // When we permute, existing unsqueezed elements will be lost. Enqueue them.
                while self.is_immediately_squeezable() {
                    let elt = self.squeeze_aux();
                    self.enqueue(elt);
                }
            }

            self.permute(acc)?;
        }

        self.set_absorbed(self.absorbed() + 1);
        Ok(())
    }

    fn squeeze(&mut self, acc: &mut Self::Acc) -> Result<Option<Self::Elt>, Self::Error> {
        self.ensure_squeezing(acc)?;

        if self.is_duplex() && self.available() == 0 {
            // What has not yet been absorbed cannot be squeezed.
            return Ok(None);
        };

        self.set_squeezed(self.squeezed() + 1);

        if let Some(queued) = self.dequeue() {
            return Ok(Some(queued));
        }

        if !self.can_squeeze_without_permuting() && self.is_simplex() {
            self.permute(acc)?;
        }

        let squeezed = self.squeeze_aux();

        Ok(Some(squeezed))
    }

    fn enqueue(&mut self, elt: Self::Elt);
    fn dequeue(&mut self) -> Option<Self::Elt>;

    fn absorb_elements(
        &mut self,
        elts: &[Self::Elt],
        acc: &mut Self::Acc,
    ) -> Result<(), Self::Error> {
        for elt in elts {
            self.absorb(elt, acc)?;
        }
        Ok(())
    }

    fn squeeze_elements(&mut self, count: usize, acc: &mut Self::Acc) -> Vec<Self::Elt>;
}

impl<'a, F: PrimeField, A: Arity<F>> SpongeTrait<'a, F, A> for Sponge<'a, F, A> {
    type Acc = ();
    type Elt = F;
    type Error = Error;

    fn new_with_constants(constants: &'a PoseidonConstants<F, A>, mode: Mode) -> Self {
        let poseidon = Poseidon::new(constants);

        Self {
            mode,
            direction: Direction::Absorbing,
            state: poseidon,
            absorbed: 0,
            squeezed: 0,
            squeeze_pos: 0,
            queue: VecDeque::with_capacity(A::to_usize()),
            pattern: IOPattern(Vec::new()),
            io_count: 0,
        }
    }

    fn mode(&self) -> Mode {
        self.mode
    }
    fn direction(&self) -> Direction {
        self.direction
    }
    fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }
    fn absorbed(&self) -> usize {
        self.absorbed
    }
    fn set_absorbed(&mut self, absorbed: usize) {
        self.absorbed = absorbed;
    }
    fn squeezed(&self) -> usize {
        self.squeezed
    }
    fn set_squeezed(&mut self, squeezed: usize) {
        self.squeezed = squeezed;
    }
    fn squeeze_pos(&self) -> usize {
        self.squeeze_pos
    }
    fn set_squeeze_pos(&mut self, squeeze_pos: usize) {
        self.squeeze_pos = squeeze_pos;
    }
    fn absorb_pos(&self) -> usize {
        self.state.pos - 1
    }
    fn set_absorb_pos(&mut self, pos: usize) {
        self.state.pos = pos + 1;
    }

    fn element(&self, index: usize) -> Self::Elt {
        self.state.elements[index]
    }
    fn set_element(&mut self, index: usize, elt: Self::Elt) {
        self.state.elements[index] = elt;
    }

    fn make_elt(&self, val: F, _acc: &mut Self::Acc) -> Self::Elt {
        val
    }

    fn rate(&self) -> usize {
        A::to_usize()
    }

    fn capacity(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.state.constants.width()
    }

    fn constants(&self) -> &PoseidonConstants<F, A> {
        self.state.constants
    }

    fn pad(&mut self) {
        self.state.apply_padding();
    }

    fn permute_state(&mut self, _acc: &mut Self::Acc) -> Result<(), Self::Error> {
        self.state.hash();
        Ok(())
    }

    fn enqueue(&mut self, elt: Self::Elt) {
        self.queue.push_back(elt);
    }
    fn dequeue(&mut self) -> Option<Self::Elt> {
        self.queue.pop_front()
    }

    fn squeeze_aux(&mut self) -> Self::Elt {
        let squeezed = self.element(SpongeTrait::squeeze_pos(self) + SpongeTrait::capacity(self));
        SpongeTrait::set_squeeze_pos(self, SpongeTrait::squeeze_pos(self) + 1);

        squeezed
    }

    fn absorb_aux(&mut self, elt: &Self::Elt) -> Self::Elt {
        self.element(SpongeTrait::absorb_pos(self) + SpongeTrait::capacity(self)) + elt
    }

    fn absorb_elements(&mut self, elts: &[F], acc: &mut Self::Acc) -> Result<(), Self::Error> {
        for elt in elts {
            self.absorb(elt, acc)?;
        }
        Ok(())
    }

    fn squeeze_elements(&mut self, count: usize, _acc: &mut ()) -> Vec<Self::Elt> {
        self.take(count).collect()
    }
}

impl<F: PrimeField, A: Arity<F>> Iterator for Sponge<'_, F, A> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
        self.squeeze(&mut ()).unwrap_or(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.mode {
            Mode::Duplex => (self.available(), None),
            Mode::Simplex => (0, None),
        }
    }
}

impl<F: PrimeField, A: Arity<F>> InnerSpongeAPI<F, A> for Sponge<'_, F, A> {
    type Acc = ();
    type Value = F;

    fn initialize_capacity(&mut self, tag: u128, _: &mut ()) {
        let mut repr = F::Repr::default();
        repr.as_mut()[..16].copy_from_slice(&tag.to_le_bytes());

        let f = F::from_repr(repr).unwrap();
        self.set_element(0, f);
    }

    fn read_rate_element(&mut self, offset: usize) -> F {
        self.element(offset + SpongeTrait::capacity(self))
    }
    fn add_rate_element(&mut self, offset: usize, x: &F) {
        self.set_element(offset + SpongeTrait::capacity(self), *x);
    }
    fn permute(&mut self, acc: &mut ()) {
        SpongeTrait::permute(self, acc).unwrap();
    }

    // Supplemental methods needed for a generic implementation.

    fn zero() -> F {
        F::ZERO
    }

    fn rate(&self) -> usize {
        SpongeTrait::rate(self)
    }
    fn absorb_pos(&self) -> usize {
        SpongeTrait::absorb_pos(self)
    }
    fn squeeze_pos(&self) -> usize {
        SpongeTrait::squeeze_pos(self)
    }
    fn set_absorb_pos(&mut self, pos: usize) {
        SpongeTrait::set_absorb_pos(self, pos);
    }
    fn set_squeeze_pos(&mut self, pos: usize) {
        SpongeTrait::set_squeeze_pos(self, pos);
    }
    fn add(a: F, b: &F) -> F {
        a + b
    }

    fn pattern(&self) -> &IOPattern {
        &self.pattern
    }

    fn set_pattern(&mut self, pattern: IOPattern) {
        self.pattern = pattern
    }

    fn increment_io_count(&mut self) -> usize {
        let old_count = self.io_count;
        self.io_count += 1;
        old_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use blstrs::Scalar as Fr;
    use generic_array::typenum;
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;
    use std::collections::HashSet;

    #[test]
    fn test_simplex() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        // Exercise simplex sponges with eventual size less, equal to, and greater to rate.
        for size in 2..10 {
            test_simplex_aux::<Fr, typenum::U4, _>(&mut rng, size);
        }
    }

    fn test_simplex_aux<F: PrimeField, A: Arity<F>, R: Rng>(rng: &mut R, n: usize) {
        let c = Sponge::<F, A>::simplex_constants(n);
        let mut sponge = Sponge::new_with_constants(&c, Mode::Simplex);

        let mut elements: Vec<F> = Vec::with_capacity(n);
        for _ in 0..n {
            elements.push(F::random(&mut *rng));
        }

        let acc = &mut ();

        // Reminder: a duplex sponge should encode its length as a prefix.
        sponge.absorb_elements(&elements, acc).unwrap();

        let result = sponge.squeeze_elements(n, acc);

        // Simple sanity check that are all non-zero and distinct.
        for (i, elt) in result.iter().enumerate() {
            assert!(*elt != F::ZERO);
            // This is expensive (n^2), but it's hard to put field element into a set since we can't hash or compare (except equality).
            for (j, elt2) in result.iter().enumerate() {
                if i != j {
                    assert!(elt != elt2);
                }
            }
        }

        assert_eq!(n, elements.len());
        assert_eq!(n, result.len());
    }

    #[test]
    fn test_duplex_consistency() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        // Exercise duplex sponges with eventual size less, equal to, and greater to rate.
        for size in 4..10 {
            test_duplex_consistency_aux::<Fr, typenum::U8, _>(&mut rng, size, 20);
        }

        // Exercise duplex sponges with eventual size less, equal to, and greater than multiples of rate.
        for _ in 0..10 {
            let size = rng.gen_range(15..25);
            test_duplex_consistency_aux::<Fr, typenum::U4, _>(&mut rng, size, 100);
        }

        // Use very small rate to ensure exercising edge cases.
        for _ in 0..10 {
            let size = rng.gen_range(15..25);
            test_duplex_consistency_aux::<Fr, typenum::U2, _>(&mut rng, size, 100);
        }
    }

    fn test_duplex_consistency_aux<F: PrimeField, A: Arity<F>, R: Rng>(
        rng: &mut R,
        n: usize,
        trials: usize,
    ) {
        let mut output = None;
        let mut signatures = HashSet::new();

        for _ in 0..trials {
            let (o, sig) = test_duplex_consistency_inner::<F, A, R>(rng, n);
            signatures.insert(sig);
            if let Some(output) = output {
                assert_eq!(output, o);
            };
            output = Some(o);
        }
        // Make sure many different paths were taken.
        assert!(trials as f64 > 0.9 * signatures.len() as f64);
    }

    fn test_duplex_consistency_inner<F: PrimeField, A: Arity<F>, R: Rng>(
        rng: &mut R,
        n: usize,
    ) -> (Vec<F>, Vec<bool>) {
        let c = Sponge::<F, A>::duplex_constants();
        let mut sponge = Sponge::new_with_constants(&c, Mode::Duplex);
        let acc = &mut ();

        // Reminder: a duplex sponge should encode its length as a prefix.
        sponge.absorb(&F::from(n as u64), acc).unwrap();

        let mut output = Vec::with_capacity(n);
        let mut signature = Vec::with_capacity(n);
        while output.len() < n {
            let try_to_squeeze: bool = rng.gen();
            signature.push(try_to_squeeze);

            if try_to_squeeze {
                if let Ok(Some(squeezed)) = sponge.squeeze(acc) {
                    output.push(squeezed);
                }
            } else {
                sponge
                    .absorb(&F::from(sponge.absorbed as u64), acc)
                    .unwrap();
            }
        }

        assert_eq!(n, output.len());

        (output, signature)
    }

    #[test]
    fn test_sponge_api_simple() {
        use crate::sponge::api::SpongeAPI;

        let parameter = IOPattern(vec![
            SpongeOp::Absorb(1),
            SpongeOp::Absorb(5),
            SpongeOp::Squeeze(3),
        ]);

        {
            let p = Sponge::<Fr, typenum::U5>::api_constants(Strength::Standard);
            let mut sponge = Sponge::new_with_constants(&p, Mode::Simplex);
            let acc = &mut ();

            sponge.start(parameter, None, acc);
            SpongeAPI::absorb(&mut sponge, 1, &[Fr::from(123)], acc);
            SpongeAPI::absorb(
                &mut sponge,
                5,
                &[
                    Fr::from(123),
                    Fr::from(123),
                    Fr::from(123),
                    Fr::from(123),
                    Fr::from(123),
                ],
                acc,
            );

            let output = SpongeAPI::squeeze(&mut sponge, 3, acc);

            sponge.finish(acc).unwrap();
            assert_eq!(
                vec![
                    scalar_from_u64s([
                        0xd891815983f3ea1e,
                        0xa1f7c82951d37ba6,
                        0xfe4d3c5fa63ed71c,
                        0x0ca887769c6aa1ae
                    ]),
                    scalar_from_u64s([
                        0xc7909f76adede2c9,
                        0x635c7b88ed65384e,
                        0x87de07b469968b55,
                        0x44d46d6e6c5955a1
                    ]),
                    scalar_from_u64s([
                        0x17867c2f5d82207b,
                        0xa809e7e861a2580c,
                        0xb022568089e9d532,
                        0x65536a37b5eef0f2
                    ])
                ],
                output
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_sponge_api_failure() {
        use crate::sponge::api::SpongeAPI;

        let parameter = IOPattern(vec![
            SpongeOp::Absorb(1),
            SpongeOp::Absorb(5),
            SpongeOp::Squeeze(3),
        ]);

        {
            let p = Sponge::<Fr, typenum::U5>::api_constants(Strength::Standard);
            let mut sponge = Sponge::new_with_constants(&p, Mode::Simplex);
            let acc = &mut ();

            sponge.start(parameter, None, acc);
            SpongeAPI::absorb(&mut sponge, 1, &[Fr::from(123)], acc);
            SpongeAPI::absorb(
                &mut sponge,
                4,
                &[Fr::from(123), Fr::from(123), Fr::from(123), Fr::from(123)],
                acc,
            );

            let _ = SpongeAPI::squeeze(&mut sponge, 3, acc);
        }
    }
}
