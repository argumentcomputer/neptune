use crate::hash_type::HashType;
use crate::poseidon::{Arity, Poseidon, PoseidonConstants};
use crate::Error;
use ff::PrimeField;

/// Encryption
/// https://link.springer.com/content/pdf/10.1007%2F978-3-642-28496-0_19.pdf
impl<'a, F, A> Poseidon<'a, F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    fn set_encryption_domain_tag(&mut self, key_length: usize, message_length: usize) {
        self.elements[0] = self
            .constants
            .hash_type
            .encryption_domain_tag(key_length, message_length);
    }

    /// Initialize state with key, shared by encryption and decryption.
    fn shared_initialize(&mut self, key: &[F], message_length: usize) -> Result<(), Error> {
        self.reset();
        self.set_encryption_domain_tag(key.len(), message_length);

        for chunk in key.chunks(self.constants.arity()) {
            self.duplex(chunk)?;
        }

        Ok(())
    }

    /// Incorporate input (by element-wise field addition), then permute the state.
    /// Output is left in state (self.elements).
    pub fn duplex(&mut self, input: &[F]) -> Result<(), Error> {
        assert!(input.len() <= self.constants.arity());
        self.partial_reset();

        for elt in input.iter() {
            self.elements[self.pos] += elt;
            self.pos += 1;
        }

        self.hash();

        Ok(())
    }

    pub fn encrypt(&mut self, key: &[F], plaintext: &[F]) -> Result<(Vec<F>, F), Error> {
        // https://link.springer.com/content/pdf/10.1007%2F978-3-642-28496-0_19.pdf
        let arity = A::to_usize();
        assert!(!key.is_empty());

        self.shared_initialize(key, plaintext.len())?;

        let mut ciphertext = Vec::with_capacity(plaintext.len());

        for plaintext_chunk in plaintext.chunks(arity) {
            for (elt, resp) in plaintext_chunk.iter().zip(self.elements.iter().skip(1)) {
                ciphertext.push(*elt + *resp);
            }
            self.duplex(plaintext_chunk)?;
        }

        let tag = self.extract_output();
        Ok((ciphertext, tag))
    }

    pub fn decrypt(&mut self, key: &[F], ciphertext: &[F], tag: &F) -> Result<Vec<F>, Error> {
        let arity = A::to_usize();
        assert!(!key.is_empty());

        self.shared_initialize(key, ciphertext.len())?;
        let mut plaintext = Vec::with_capacity(ciphertext.len());

        let mut last_chunk_start = 0;
        for ciphertext_chunk in ciphertext.chunks(arity) {
            for (elt, resp) in ciphertext_chunk.iter().zip(self.elements.iter().skip(1)) {
                plaintext.push(*elt - *resp);
            }
            let plaintext_chunk = &plaintext[last_chunk_start..];
            self.duplex(plaintext_chunk)?;

            last_chunk_start += arity;
        }

        let computed_tag = self.extract_output();

        if *tag != computed_tag {
            return Err(Error::TagMismatch);
        };

        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use blstrs::Scalar as Fr;
    use ff::Field;
    use generic_array::typenum::U2;

    #[test]
    fn encrypt_decrypt() {
        let constants = PoseidonConstants::<Fr, U2>::new_with_strength_and_type(
            Strength::Standard,
            HashType::Encryption,
        );
        let mut p = Poseidon::<Fr, U2>::new(&constants);

        let plaintext = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            .iter()
            .map(|n| Fr::from(*n as u64))
            .collect::<Vec<_>>();

        let key = [987, 234]
            .iter()
            .map(|n| Fr::from(*n as u64))
            .collect::<Vec<_>>();

        let (ciphertext, tag) = p.encrypt(&key, &plaintext).unwrap();
        let decrypted = p.decrypt(&key, &ciphertext, &tag).unwrap();

        assert_eq!(plaintext, decrypted);
        assert_eq!(
            ciphertext,
            [
                scalar_from_u64s([
                    0xaec64216978527ac,
                    0xdf5f10f7a1a9a8b7,
                    0xe34ddf5197d75feb,
                    0x1be88365866ae3d6
                ]),
                scalar_from_u64s([
                    0xda0d5ec9eff654da,
                    0x7017055a0a081c34,
                    0x1bce42bb6937ab48,
                    0x35a2e74eeaa97f6c
                ]),
                scalar_from_u64s([
                    0x298936f51cf3aa12,
                    0x906cf40d00e4411c,
                    0xc195c1ed48a6c223,
                    0x4598c18291315dbc
                ]),
                scalar_from_u64s([
                    0xd59a3a87f0dec416,
                    0x0d9fd7b5282925d8,
                    0x0ea1b98d0b00d561,
                    0x023704693c4abf1b
                ]),
                scalar_from_u64s([
                    0x211b61f66285bd55,
                    0xbf26070055e78d4a,
                    0x3682aa0ce38835cf,
                    0x4e6a9d5424f77ac5
                ]),
                scalar_from_u64s([
                    0xa1b8442758bec43b,
                    0xaf3248c718643bf9,
                    0x66ad9b69d73bc44a,
                    0x243e604b5138226a
                ]),
                scalar_from_u64s([
                    0xf92fd3ed19af0733,
                    0x6b96bc196f6c2d5b,
                    0xefe6d3b5c1dc730a,
                    0x0dabad8c3dbd4147
                ]),
                scalar_from_u64s([
                    0x556595727f046c2a,
                    0xaecc434fb16c8631,
                    0xd5da55ffc78a420f,
                    0x081a166a1909cbed
                ]),
                scalar_from_u64s([
                    0x40d5a2d5052cb583,
                    0x5c0b5265c006a5cb,
                    0xfd936f0a297114f8,
                    0x1191f085dc4d2286
                ]),
            ]
        )
    }
}
