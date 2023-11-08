// This should only run with the abomonation feature, see https://github.com/rust-lang/cargo/issues/4663
#[cfg(feature = "abomonation")]
use abomonation::{decode, encode};
#[cfg(feature = "abomonation")]
use blstrs::Scalar as Fr;
#[cfg(feature = "abomonation")]
use generic_array::typenum::U24;
#[cfg(feature = "abomonation")]
use neptune::{poseidon::PoseidonConstants, Strength};

fn main() {
    #[cfg(feature = "abomonation")]
    {
        let mut bytes = Vec::new();
        unsafe {
            let constants = PoseidonConstants::<Fr, U24>::new_with_strength(Strength::Standard);
            encode(&constants, &mut bytes).unwrap()
        };
        println!("Encoded!");
        println!("Read size: {}", bytes.len());

        if let Some((result, remaining)) =
            unsafe { decode::<PoseidonConstants<Fr, U24>>(&mut bytes) }
        {
            let constants = PoseidonConstants::<Fr, U24>::new_with_strength(Strength::Standard);
            assert!(result.clone() == constants, "not equal!");
            assert!(remaining.is_empty());
        } else {
            println!("Something terrible happened");
        }
    }
}
