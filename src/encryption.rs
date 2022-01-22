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
        self.elements[0] = self.constants.hash_type.encryption_domain_tag(
            &self.constants.strength,
            key_length,
            message_length,
        );
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
                    0x5ff0cfcb4e85930b,
                    0x414ba007bd03ebf4,
                    0xd6f74ff3f8c551d2,
                    0x559e8d0119147a58
                ]),
                scalar_from_u64s([
                    0x27b444621fb66c53,
                    0xfd93155d4bfd6390,
                    0x057278df25ed0755,
                    0x6aba53c18d1e6e19
                ]),
                scalar_from_u64s([
                    0xcf372c291876cbb7,
                    0x11c0fd7f08e2cd3c,
                    0x3fcda5c47584f1bf,
                    0x1ace845c63a281c5
                ]),
                scalar_from_u64s([
                    0x1c6ece3946a87525,
                    0x5a5e7a53bfa88a06,
                    0x80b1b7fc3057c99f,
                    0x384830fff68907ae
                ]),
                scalar_from_u64s([
                    0x9cf97e49da0764d9,
                    0xb22f72ce3848eec4,
                    0x44b456ea90ff3eb2,
                    0x6e4bd3b52abc3da1
                ]),
                scalar_from_u64s([
                    0xb5bd052b5cd4950e,
                    0x2310f368ebaa3a7e,
                    0x28e774fd3c3f65f1,
                    0x22270e75c580e1a9
                ]),
                scalar_from_u64s([
                    0xf28d6c4beca28050,
                    0x356f6048a3db8b1d,
                    0x48ede4c915994a31,
                    0x317816a5fb29b815
                ]),
                scalar_from_u64s([
                    0xec253d996f303928,
                    0xba901e485ba84221,
                    0xf31af5cb8dfd1c03,
                    0x05b026c0904e93ed
                ]),
                scalar_from_u64s([
                    0x612315aa5c696bfd,
                    0x65feed64c6aa4f02,
                    0x75f5bef81e7e5043,
                    0x38e55933719f8a21
                ]),
            ]
        )
    }
}
