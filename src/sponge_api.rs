use crate::poseidon::{Arity, Poseidon, PoseidonConstants};
use ff::PrimeField;

pub enum Error {
    ParameterUsageMismatch,
}

// Secure Sponge API for Field Elements
// https://hackmd.io/bHgsH6mMStCVibM_wYvb2w

#[derive(Clone, Copy, Debug)]
pub enum SpongeOp {
    Absorb(u32),
    Squeeze(u32),
}

// A large 128-bit prime, per // https://primes.utm.edu/lists/2small/100bit.html.
const HASHER_BASE: u128 = (0 - 159) as u128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Hasher {
    x: u128,
    x_i: u128,
    state: u128,
    current_op: SpongeOp,
}

impl Default for Hasher {
    fn default() -> Self {
        Self {
            x: HASHER_BASE,
            x_i: 1,
            state: 0,
            current_op: SpongeOp::Absorb(0),
        }
    }
}

impl Hasher {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_with_dst(dst: u32) -> Self {
        let mut h = Self {
            x: HASHER_BASE,
            x_i: 1,
            state: 0,
            current_op: SpongeOp::Absorb(0),
        };

        h.update(dst);
        h
    }

    pub fn update_op(&mut self, op: SpongeOp) {
        if self.current_op.matches(op) {
            self.current_op = self.current_op.combine(op)
        } else {
            self.finish_op();
        }
    }

    pub fn update(&mut self, a: u32) {
        self.x_i *= self.x;
        self.state += self.x_i * a as u128;
    }

    fn finish_op(&mut self) {
        let op_value = self.current_op.value();
        self.update(op_value);
        self.current_op = self.current_op.reset();
    }

    pub fn finalize(&mut self) -> u128 {
        self.finish_op();
        self.state
    }
}

impl SpongeOp {
    pub fn reset(&self) -> Self {
        match self {
            Self::Absorb(_) => Self::Squeeze(0),
            Self::Squeeze(_) => Self::Absorb(0),
        }
    }

    pub fn count(&self) -> u32 {
        match self {
            Self::Absorb(n) => *n,
            Self::Squeeze(n) => *n,
        }
    }

    pub fn is_absorb(&self) -> bool {
        match self {
            Self::Absorb(_) => true,
            Self::Squeeze(_) => false,
        }
    }

    pub fn is_squeeze(&self) -> bool {
        !self.is_absorb()
    }

    pub fn combine(&self, other: Self) -> Self {
        assert!(self.matches(other));
        let other_count = other.count();

        match self {
            Self::Absorb(n) => Self::Absorb(n + other_count),
            Self::Squeeze(n) => Self::Squeeze(n + other_count),
        }
    }

    pub fn matches(&self, other: Self) -> bool {
        self.is_absorb() == other.is_absorb()
    }

    pub fn value(&self) -> u32 {
        match self {
            Self::Absorb(n) => {
                assert_eq!(0, n >> 31);
                n + (1 << 31)
            }
            Self::Squeeze(n) => {
                assert_eq!(0, n >> 31);
                *n
            }
        }
    }
}

#[derive(Clone)]
pub enum SpongeParameter {
    OpSequence(Vec<SpongeOp>),
}

impl SpongeParameter {
    pub fn value(&self) -> u128 {
        match self {
            Self::OpSequence(ops) => {
                let mut hasher = Hasher::new();

                for op in ops {
                    hasher.update(op.value());
                }
                hasher.finalize()
            }
        }
    }
}

pub trait SpongeAPI<F: PrimeField, A: Arity<F>> {
    fn start(&mut self, p: SpongeParameter);
    fn absorb(&mut self, length: usize, elements: &[F]);
    fn squeeze(&mut self, length: usize);
    fn finish(&mut self) -> Result<(), Error>;
}

pub trait InnerSpongeAPI<F: PrimeField, A: Arity<F>> {
    fn initialize_capacity(&mut self, tag: u128);
    fn read_rate_element(&mut self, offset: usize) -> F;
    fn write_rate_element(&mut self, offset: usize, x: F);
    fn permute(&mut self);

    // Supplemental methods needed for a generic implementation.
    fn capacity(&self) -> usize;
    fn rate(&self) -> usize;
    fn absorb_pos(&self) -> usize;
    fn squeeze_pos(&self) -> usize;
    fn set_absorb_pos(&mut self, pos: usize);
    fn set_squeeze_pos(&mut self, pos: usize);

    fn initialize_hasher(&mut self);
    fn update_hasher(&mut self, op: SpongeOp);
    fn finalize_hasher(&mut self) -> u128;

    fn set_parameter(&mut self, p: SpongeParameter);
    fn get_parameter(&mut self) -> SpongeParameter;
}

impl<F: PrimeField, A: Arity<F>, S: InnerSpongeAPI<F, A>> SpongeAPI<F, A> for S {
    fn start(&mut self, p: SpongeParameter) {
        self.initialize_capacity(p.value());
        self.set_parameter(p);
        for i in 0..self.rate() {
            self.write_rate_element(i, F::zero());
        }
        self.set_absorb_pos(0);
        self.set_squeeze_pos(0);
        self.initialize_hasher();
    }

    fn absorb(&mut self, length: usize, elements: &[F]) {
        assert_eq!(length as usize, elements.len());
        let rate = self.rate();

        for element in elements.iter() {
            let absorb_pos = self.absorb_pos();
            if absorb_pos == rate {
                self.permute();
                self.set_absorb_pos(0);
            }
            self.write_rate_element(absorb_pos, *element);
            self.set_absorb_pos(absorb_pos + 1);
        }
        self.set_squeeze_pos(rate);
        self.update_hasher(SpongeOp::Absorb(length as u32));
    }

    fn squeeze(&mut self, length: usize) {
        let rate = self.rate();
        assert!(length <= rate);

        let mut out = Vec::with_capacity(length);

        for i in 0..length {
            let squeeze_pos = self.squeeze_pos();
            if squeeze_pos == rate {
                self.permute();
                self.set_squeeze_pos(0);
                self.set_absorb_pos(0);
            }
            out[i] = self.read_rate_element(squeeze_pos);
            self.set_squeeze_pos(squeeze_pos + 1);
        }
        self.update_hasher(SpongeOp::Squeeze(length as u32));
    }

    fn finish(&mut self) -> Result<(), Error> {
        let result = self.finalize_hasher();

        if result == self.get_parameter().value() {
            Ok(())
        } else {
            Err(Error::ParameterUsageMismatch)
        }
    }
}
