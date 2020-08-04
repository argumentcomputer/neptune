use crate::{scalar_from_u64, Arity, Strength};
use ff::{Field, PrimeField, ScalarEngine};

#[derive(Clone, Debug, PartialEq)]
pub enum HashType<Fr: PrimeField, A: Arity<Fr>> {
    MerkleTree,
    MerkleTreeSparse(u64),
    VariableLength,
    ConstantLength(usize),
    Encryption,
    Custom(CType<Fr, A>),
}

impl<Fr: PrimeField, A: Arity<Fr>> HashType<Fr, A> {
    pub fn domain_tag(&self, strength: &Strength) -> Fr {
        let pow2 = |n| pow2::<Fr, A>(n);
        let x_pow2 = |coeff, n| x_pow2::<Fr, A>(coeff, n);
        let with_strength = |x: Fr| {
            let mut tmp = x;
            tmp.add_assign(&Self::strength_tag_component(strength));
            tmp
        };

        match self {
            // 2^arity - 1
            HashType::MerkleTree => with_strength(A::tag()),
            // bitmask
            HashType::MerkleTreeSparse(bitmask) => with_strength(scalar_from_u64(*bitmask)),
            // 2^64
            HashType::VariableLength => with_strength(pow2(64)),
            // length * 2^64
            HashType::ConstantLength(length) => {
                assert!(*length as usize <= A::to_usize());
                with_strength(x_pow2(*length as u64, 64))
            }
            // 2^32
            HashType::Encryption => with_strength(pow2(32)),
            // identifier * 2^32
            HashType::Custom(ref ctype) => ctype.domain_tag(&strength),
        }
    }

    fn strength_tag_component(strength: &Strength) -> Fr {
        let id = match strength {
            // Standard strength doesn't affect the base tag.
            Strength::Standard => 0,
            Strength::Strengthened => 1,
        };

        x_pow2::<Fr, A>(id, 32)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CType<Fr: PrimeField, A: Arity<Fr>> {
    Arbitrary(u64),
    _Phantom((Fr, A)),
}

impl<Fr: PrimeField, A: Arity<Fr>> CType<Fr, A> {
    fn identifier(&self) -> u64 {
        match self {
            CType::Arbitrary(id) => *id,
            CType::_Phantom(_) => panic!("_Phantom is not a real custom tag type."),
        }
    }

    fn domain_tag(&self, _strength: &Strength) -> Fr {
        x_pow2::<Fr, A>(self.identifier(), 32)
    }
}

/// pow2(n) = 2^n
fn pow2<Fr: PrimeField, A: Arity<Fr>>(n: i32) -> Fr {
    let two: Fr = scalar_from_u64(2);
    two.pow([n as u64, 0, 0, 0])
}

/// x_pow2(x, n) = x * 2^n
fn x_pow2<Fr: PrimeField, A: Arity<Fr>>(coeff: u64, n: i32) -> Fr {
    let mut tmp: Fr = pow2::<Fr, A>(n);
    tmp.mul_assign(&scalar_from_u64(coeff));
    tmp
}
