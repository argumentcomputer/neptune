use crate::hash_type::HashType;
use crate::poseidon::{Arity, Poseidon, PoseidonConstants};
use crate::Error;
use ff::PrimeField;
use generic_array::{
    sequence::GenericSequence, typenum::operator_aliases::Sub1, ArrayLength, GenericArray,
};

/// https://link.springer.com/content/pdf/10.1007%2F978-3-642-28496-0_19.pdf

pub struct Crypt<'a, F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    poseidon: Poseidon<'a, F, A>,
    direction: EncDec,
    is_streaming: bool,
}

impl<'a, F, A> Crypt<'a, F, A>
where
    F: PrimeField,
    A: Arity<F>,
{
    fn require_enc(&self) -> Result<(), Error> {
        if self.direction.is_enc() {
            Ok(())
        } else {
            Err(Error::UnsupportedCryptOp)
        }
    }

    fn require_dec(&self) -> Result<(), Error> {
        if self.direction.is_dec() {
            Ok(())
        } else {
            Err(Error::UnsupportedCryptOp)
        }
    }

    fn require_streaming(&self) -> Result<(), Error> {
        if self.is_streaming {
            Ok(())
        } else {
            Err(Error::UnsupportedCryptOp)
        }
    }

    fn require_fixed(&self) -> Result<(), Error> {
        if !self.is_streaming {
            Ok(())
        } else {
            Err(Error::UnsupportedCryptOp)
        }
    }
}

pub enum EncDec {
    Enc,
    Dec,
}

impl EncDec {
    fn is_enc(&self) -> bool {
        match self {
            Self::Enc => true,
            Self::Dec => false,
        }
    }
    fn is_dec(&self) -> bool {
        match self {
            Self::Enc => false,
            Self::Dec => true,
        }
    }
}

impl<'a, F, A> Crypt<'a, F, A>
where
    F: PrimeField + Sized,
    A: Arity<F>,
{
    // NOTE: If an empty key is provided, tags still provide authentication,
    // but decryption is trivial.
    pub fn init_stream(&mut self, key: &[F]) -> Result<(), Error> {
        self.require_streaming()?;

        self.poseidon.shared_initialize(key, 0)?;
        self.poseidon.reset_offsets();
        Ok(())
    }

    pub fn init_fixed(&mut self, key: &[F], message_length: usize) -> Result<(), Error> {
        self.require_fixed()?;

        self.poseidon.shared_initialize(key, message_length)?;
        self.poseidon.reset_offsets();
        Ok(())
    }

    // Returns array containing transient tag as first element and ciphertext chunk as remainder.
    // Transient tags may have application value, since they summarize the state of the stream.
    // However, they must not be leaked, since discovery of a transient tag will allow decryption
    // of the remainder of the stream.
    fn encrypt_chunk_internal(
        &mut self,
        plaintext_chunk: &[F],
    ) -> Result<GenericArray<F, A::ConstantsSize>, Error> {
        let mut ciphertext_chunk = GenericArray::generate(|i| {
            if i > 0 && i <= plaintext_chunk.len() {
                plaintext_chunk[i - 1] + self.poseidon.elements[i]
            } else {
                F::zero()
            }
        });
        self.poseidon.duplex(plaintext_chunk)?;

        let tag = self.poseidon.extract_output();
        ciphertext_chunk[0] = tag;

        Ok(ciphertext_chunk)
    }

    pub fn encrypt(
        poseidon: Poseidon<'a, F, A>,
        key: &[F],
        plaintext: &[F],
    ) -> Result<(Vec<F>, F), Error> {
        let arity = A::to_usize();
        assert!(!key.is_empty());

        let mut enc = Crypt {
            poseidon,
            is_streaming: false,
            direction: EncDec::Enc,
        };

        let effective_length = plaintext.len();
        assert!(effective_length > 0);

        enc.init_fixed(key, effective_length)?;

        let mut ciphertext = Vec::with_capacity(effective_length);

        let mut tag = enc.poseidon.extract_output();

        for plaintext_chunk in plaintext.chunks(arity) {
            let ciphertext_chunk = enc.encrypt_chunk_internal(plaintext_chunk)?;
            ciphertext_chunk
                .iter()
                .skip(1)
                .zip(plaintext_chunk)
                .for_each(|(x, _)| ciphertext.push(*x));
            tag = ciphertext_chunk[0];
        }

        Ok((ciphertext.to_vec(), tag))
    }

    pub fn new_encryption_stream(poseidon: Poseidon<'a, F, A>, key: &[F]) -> Result<Self, Error> {
        assert!(!key.is_empty());

        let mut enc = Crypt {
            poseidon,
            direction: EncDec::Enc,
            is_streaming: true,
        };
        enc.init_stream(key)?;
        Ok(enc)
    }

    pub fn new_decryption_stream(poseidon: Poseidon<'a, F, A>, key: &[F]) -> Result<Self, Error> {
        assert!(!key.is_empty());

        let mut enc = Crypt {
            poseidon,
            direction: EncDec::Dec,
            is_streaming: true,
        };
        enc.init_stream(key)?;
        Ok(enc)
    }

    pub fn start_header(&mut self, body_length: usize, nonce: Option<F>) -> Result<(), Error> {
        self.require_streaming()?;

        self.add_header_element(F::from(body_length as u64))?;
        self.add_header_element(nonce.unwrap_or_else(|| F::zero()))?;

        Ok(())
    }

    pub fn add_header_element(&mut self, elt: F) -> Result<(), Error> {
        self.require_streaming()?;

        self.poseidon.duplex1(&elt)?;

        Ok(())
    }

    pub fn encrypt_element(&mut self, elt: F) -> Result<F, Error> {
        self.require_enc()?;

        let encrypted = self.poseidon.elements[self.poseidon.pos] + elt;

        self.poseidon.duplex1(&elt)?;

        Ok(encrypted)
    }

    pub fn decrypt_element(&mut self, elt: F) -> Result<F, Error> {
        self.require_dec()?;

        let decrypted = elt - self.poseidon.elements[self.poseidon.pos];

        self.poseidon.duplex1(&decrypted)?;

        Ok(decrypted)
    }

    pub fn finalize_part(&mut self) -> Result<(), Error> {
        self.require_streaming()?;

        if self.poseidon.pos != self.poseidon.constants.width() {
            self.poseidon.hash();
            self.poseidon.reset_offsets();
        };

        Ok(())
    }

    pub fn finalize_header(&mut self) -> Result<(), Error> {
        self.finalize_part()?;
        Ok(())
    }

    pub fn finalize_body(&mut self) -> Result<(), Error> {
        self.finalize_part()?;
        Ok(())
    }

    pub fn finalize_message(&mut self) -> Result<F, Error> {
        self.require_streaming()?;

        if self.poseidon.pos != self.poseidon.constants.width() {
            self.poseidon.hash();
            self.poseidon.reset_offsets();
        };

        let tag = self.poseidon.extract_output();

        Ok(tag)
    }

    pub fn finalize(&mut self, decrypted: &mut Vec<F>) -> Result<(Vec<F>, F), Error> {
        self.require_dec()?;
        self.require_streaming()?;
        let tag = self.poseidon.extract_output();

        loop {
            if let Some(last) = decrypted.last() {
                if *last == F::zero() {
                    decrypted.pop();
                    continue;
                } else {
                    break;
                }
            } else {
                return Ok((decrypted.to_vec(), tag));
            }
        }

        if let Some(last) = decrypted.last() {
            if *last == F::one() {
                decrypted.pop();
            }
        }

        Ok((decrypted.to_vec(), tag))
    }
}

/// Encryption
///
/// References:
/// - https://link.springer.com/content/pdf/10.1007%2F978-3-642-28496-0_19.pdf
/// - https://drive.google.com/file/d/1EVrP3DzoGbmzkRmYnyEDcIQcXVU7GlOd/edit
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
    fn duplex(&mut self, input: &[F]) -> Result<(), Error> {
        assert!(input.len() <= self.constants.arity());

        for elt in input.iter() {
            self.duplex1(elt)?;
        }

        Ok(())
    }

    fn duplex1(&mut self, input_elt: &F) -> Result<F, Error> {
        let res = self.elements[self.pos] + input_elt;

        self.elements[self.pos] = res;
        self.pos += 1;

        if self.pos >= self.constants.width() {
            self.hash();
            self.reset_offsets();
        }

        Ok(res)
    }

    pub fn encrypt(&mut self, key: &[F], plaintext: &[F]) -> Result<(Vec<F>, F), Error> {
        // https://link.springer.com/content/pdf/10.1007%2F978-3-642-28496-0_19.pdf
        let arity = A::to_usize();
        // assert!(!key.is_empty());

        let effective_length = plaintext.len();
        assert!(effective_length > 0);

        self.shared_initialize(key, effective_length)?;

        let mut ciphertext = Vec::with_capacity(effective_length);

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
        // assert!(!key.is_empty());

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
    use generic_array::typenum::{U2, U5};
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

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
        let (ciphertext2, tag2) = Crypt::encrypt(p.clone(), &key, &plaintext).unwrap();

        let decrypted = p.decrypt(&key, &ciphertext, &tag).unwrap();
        let decrypted2 = p.decrypt(&key, &ciphertext2, &tag2).unwrap();

        assert_eq!(plaintext, decrypted);
        assert_eq!(plaintext, decrypted2);
        assert_eq!(ciphertext2, ciphertext2);
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

    #[test]
    fn streaming_encrypt_decrypt() {
        // TODO: other arities.
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let constants = PoseidonConstants::<Fr, U5>::new_with_strength_and_type(
            Strength::Standard,
            HashType::Encryption,
        );
        let p = Poseidon::<Fr, U5>::new(&constants);

        let trials = 20;
        let messages = 10;
        let max_key_length = 7;
        let max_header_length = 7;
        let max_message_length = 20;

        for _ in 0..trials {
            // Create random key
            let mut key = Vec::new();

            for _ in 0..rng.gen_range(1..max_key_length) {
                key.push(Fr::random(&mut rng))
            }

            let mut e = Crypt::new_encryption_stream(p.clone(), &key).unwrap();
            let mut d = Crypt::new_decryption_stream(p.clone(), &key).unwrap();

            let mut message_data = Vec::with_capacity(messages);

            for _ in 0..messages {
                // Create nonce
                let nonce = Some(Fr::random(&mut rng));

                // Create random header
                let mut header = Vec::new();

                for _ in 0..rng.gen_range(1..max_header_length) {
                    header.push(Fr::random(&mut rng))
                }

                let message_length = rng.gen_range(0..max_message_length);
                // Create random plaintext
                let mut plaintext = Vec::with_capacity(message_length);

                for _ in 0..message_length {
                    let elt = Fr::random(&mut rng);
                    plaintext.push(elt);
                }

                ////////////////////////////////////////
                // Encrypt plaintext
                let mut ciphertext = Vec::with_capacity(plaintext.len());

                // Add header
                e.start_header(plaintext.len(), nonce).unwrap();

                for elt in header.iter() {
                    e.add_header_element(elt.clone()).unwrap();
                }
                e.finalize_header().unwrap();

                // Encrypt body
                for elt in plaintext.iter() {
                    let encrypted = e.encrypt_element(*elt).unwrap();

                    ciphertext.push(encrypted);
                }

                // Finalize message
                let tag = e.finalize_message().unwrap();

                message_data.push((header, nonce, ciphertext, tag));
            }

            for (header, nonce, ciphertext, tag) in message_data.iter() {
                ////////////////////////////////////////
                // Decrypt ciphertext
                let mut decrypted = Vec::with_capacity(ciphertext.len());

                // Add header
                d.start_header(ciphertext.len(), *nonce).unwrap();

                for elt in header.iter() {
                    d.add_header_element(*elt).unwrap();
                }
                d.finalize_header().unwrap();

                // Decrypt body
                for elt in ciphertext.iter() {
                    let decrypted_elt = d.decrypt_element(*elt).unwrap();

                    decrypted.push(decrypted_elt);
                }

                // Finalize message
                let computed_tag = d.finalize_message().unwrap();

                assert_eq!(tag, &computed_tag);
                assert_eq!(decrypted, decrypted);
            }
        }
    }
}
