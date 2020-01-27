pub use crate::poseidon::Poseidon;
pub use crate::Error;
use ff::{PrimeField, PrimeFieldDecodingError, PrimeFieldRepr, ScalarEngine};
use paired::bls12_381::Bls12;
pub use paired::bls12_381::Fr as Scalar;
use paired::Engine;

/// Following https://extgit.iaik.tugraz.at/krypto/hadeshash/blob/master/code/scripts/create_rcs_grain.sage
pub fn generate_constants(field: u8, sbox: u8, n: u16, t: u16, r_f: u16, r_p: u16) -> Vec<Scalar> {
    let num_constants = (r_f + r_p) * t;
    let bit_list_field: u128 = (field & 0b11).into(); // Bits 0-1
    let bit_list_sbox: u128 = ((sbox & 0b1111) << 2).into(); // Bits 2-5
    let bit_list_n: u128 = ((n & 0b111111111111) << 6).into(); // Bits 6-17
    let bit_list_t: u128 = ((t as u128 & 0b111111111111) << 18).into(); // Bits 18-29
    let bit_list_r_f: u128 = ((r_f as u128 & 0b1111111111) << 30).into(); // Bits 30-39
    let bit_list_r_p: u128 = ((r_p as u128 & 0b1111111111) << 40).into(); // Bits 40-49
    let bit_list_1: u128 = 0b111111111111111111111111111111u128 << 50; // Bits 50-79
    let init_sequence = bit_list_field
        + bit_list_sbox
        + bit_list_n
        + bit_list_t
        + bit_list_r_f
        + bit_list_r_p
        + bit_list_1;

    let mut grain = Grain::new(init_sequence);
    let mut round_constants: Vec<Scalar> = Vec::new();
    match field {
        1 => {
            for _ in 0..num_constants {
                while {
                    let mut bytes = [0u8; 32];
                    grain.get_next_bytes(&mut bytes);
                    if let Ok(f) = bytes_into_fr::<Bls12>(&mut bytes) {
                        round_constants.push(f);
                        false
                    } else {
                        true
                    }
                } {}
            }
        }
        _ => {
            panic!("Only prime fields are supported.");
        }
    }
    return round_constants;
}
struct Grain {
    state: Vec<bool>,
}

impl Grain {
    fn new(init_sequence: u128) -> Self {
        let mut init = init_sequence;
        let mut state: Vec<bool> = Vec::new();
        for _ in 0..80 {
            state.push(init & 1 == 1);
            init >>= 1;
        }
        let mut g = Grain { state };
        assert_eq!(0, init);
        for _ in 0..160 {
            g.generate_new_bit();
        }
        assert!(g.state.len() == 80);
        g
    }

    fn generate_new_bit(&mut self) -> bool {
        let new_bit =
            self.bit(62) ^ self.bit(51) ^ self.bit(38) ^ self.bit(23) ^ self.bit(13) ^ self.bit(0);
        self.state.remove(0);
        self.state.push(new_bit);
        new_bit
    }

    fn bit(&self, index: usize) -> bool {
        self.state[index]
    }

    fn next_byte(&mut self) -> u8 {
        let mut acc: u8 = 0;
        self.take(8).for_each(|bit| {
            acc <<= 1;
            if bit {
                acc += 1;
            }
        });

        acc
    }
    fn get_next_bytes(&mut self, result: &mut [u8; 32]) {
        for i in 0..32 {
            result[i] = self.next_byte();
        }
    }
}

impl Iterator for Grain {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let mut new_bit = self.generate_new_bit();
        while !new_bit {
            let _new_bit = self.generate_new_bit();
            new_bit = self.generate_new_bit();
        }
        new_bit = self.generate_new_bit();
        Some(new_bit)
    }
}

#[allow(dead_code)]
#[inline]
fn bool_to_u8(bit: bool, offset: usize) -> u8 {
    if bit {
        1u8 << offset
    } else {
        0u8
    }
}

/// Converts a slice of bools into their byte representation, in little endian.
#[allow(dead_code)]
pub fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|bits| {
            bool_to_u8(bits[7], 7)
                | bool_to_u8(bits[6], 6)
                | bool_to_u8(bits[5], 5)
                | bool_to_u8(bits[4], 4)
                | bool_to_u8(bits[3], 3)
                | bool_to_u8(bits[2], 2)
                | bool_to_u8(bits[1], 1)
                | bool_to_u8(bits[0], 0)
        })
        .collect()
}

// Takes a slice of bytes and returns an Fr if byte slice is exactly 32 bytes and does not overflow.
// Otherwise, returns a BadFrBytesError.
fn bytes_into_fr<E: Engine>(bytes: &[u8]) -> Result<E::Fr, PrimeFieldDecodingError> {
    assert_eq!(bytes.len(), 32);

    let mut fr_repr = <<<E as ScalarEngine>::Fr as PrimeField>::Repr as Default>::default();
    fr_repr
        .read_le(bytes)
        .map_err(|e| PrimeFieldDecodingError::NotInField(e.to_string()))?;

    E::Fr::from_repr(fr_repr)
}
