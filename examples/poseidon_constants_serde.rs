use abomonation::{decode, encode};
use blstrs::Scalar as Fr;
use generic_array::typenum::U2;
use neptune::{poseidon::PoseidonConstants, Strength};

fn main() {

    let mut bytes = Vec::new();
    unsafe { 
        let constants = PoseidonConstants::<Fr, U2>::new_with_strength(Strength::Standard);
        encode(&constants, &mut bytes).unwrap()
    };
    println!("Encoded!");
    println!("Read size: {}", bytes.len());

    if let Some((result, remaining)) = unsafe { decode::<PoseidonConstants<Fr, U2>>(&mut bytes) } {
        let constants = PoseidonConstants::<Fr, U2>::new_with_strength(Strength::Standard);
        assert!(result.clone() == constants.clone(), "not equal!");
        assert!(remaining.len() == 0);
    } else {
        println!("Something terrible happened");
    }
}
