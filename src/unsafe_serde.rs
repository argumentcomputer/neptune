#![allow(non_snake_case)]

use ff::PrimeField;
use std::concat;
use std::{
    io::Write,
    mem::{size_of, transmute},
};

use crate::mds::SparseMatrix;

/// Unspeakable horrors
pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

// This method currently enables undefined behavior, by exposing padding bytes.
#[inline]
pub unsafe fn typed_to_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * size_of::<T>())
}

macro_rules! encode_decode {
    ($entomb:ident, $exhume:ident, $encode:ident, $decode:ident, $type:ty) => {
        #[inline]
        pub unsafe fn $encode<F: PrimeField, W: Write>(
            typed: &$type,
            write: &mut W,
        ) -> std::io::Result<()> {
            let slice = std::slice::from_raw_parts(transmute(typed), size_of::<$type>());
            write.write_all(slice)?;
            $entomb(typed, write)?;
            Ok(())
        }

        #[inline]
        pub unsafe fn $decode<F: PrimeField>(bytes: &mut [u8]) -> Option<(&$type, &mut [u8])> {
            if bytes.len() < size_of::<$type>() {
                None
            } else {
                let (split1, split2) = bytes.split_at_mut(size_of::<$type>());
                let result: &mut $type = transmute(split1.get_unchecked_mut(0));
                if let Some(remaining) = $exhume(result, split2) {
                    Some((result, remaining))
                } else {
                    None
                }
            }
        }
    };
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn entomb_F<F: PrimeField, W: Write>(_f: &F, _bytes: &mut W) -> std::io::Result<()> {
    Ok(())
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn exhume_F<'a, 'b, F: PrimeField>(
    _f: &mut F,
    bytes: &'a mut [u8],
) -> Option<&'a mut [u8]> {
    Some(bytes)
}

#[inline]
pub fn extent_F<F: PrimeField>(_this: &F) -> usize {
    0
}

encode_decode!(entomb_F, exhume_F, encode_F, decode_F, F);

macro_rules! vec_abomonate {
    ($entomb_name:ident, $exhume_name:ident, $extent_name:ident, $entomb_inner_name:ident, $exhume_inner_name:ident, $extent_inner_name:ident, $inner_type:ty) => {
        #[inline]
        pub unsafe fn $entomb_name<F: PrimeField, W: Write>(
            this: &Vec<$inner_type>,
            write: &mut W,
        ) -> std::io::Result<()> {
            write.write_all(typed_to_bytes(&this[..]))?;
            for element in this.iter() {
                $entomb_inner_name(element, write)?;
            }
            Ok(())
        }

        #[inline]
        pub unsafe fn $exhume_name<'a, 'b, F: PrimeField>(
            this: &'a mut Vec<$inner_type>,
            bytes: &'b mut [u8],
        ) -> Option<&'b mut [u8]> {
            // extract memory from bytes to back our vector
            let binary_len = this.len() * size_of::<$inner_type>();
            if binary_len > bytes.len() {
                None
            } else {
                let (mine, mut rest) = bytes.split_at_mut(binary_len);
                let slice = std::slice::from_raw_parts_mut(
                    mine.as_mut_ptr() as *mut $inner_type,
                    this.len(),
                );
                std::ptr::write(
                    this,
                    Vec::from_raw_parts(slice.as_mut_ptr(), this.len(), this.len()),
                );
                for element in this.iter_mut() {
                    let temp = rest; // temp variable explains lifetimes (mysterious!)
                    rest = $exhume_inner_name(element, temp)?;
                }
                Some(rest)
            }
        }

        #[inline]
        pub fn $extent_name<F: PrimeField>(this: &Vec<$inner_type>) -> usize {
            let mut sum = size_of::<$inner_type>() * this.len();
            for element in this.iter() {
                sum += $extent_inner_name(element);
            }
            sum
        }
    };
}

#[inline]
pub unsafe fn entomb_vec_F<F: PrimeField, W: Write>(
    this: &Vec<F>,
    write: &mut W,
) -> std::io::Result<()> {
    write.write_all(typed_to_bytes(&this[..]))?;
    for element in this.iter() {
        entomb_F(element, write)?;
    }
    Ok(())
}
#[inline]
pub unsafe fn exhume_vec_F<'a, 'b, F: PrimeField>(
    this: &'a mut Vec<F>,
    bytes: &'b mut [u8],
) -> Option<&'b mut [u8]> {
    let binary_len = this.len() * size_of::<F>();
    if binary_len > bytes.len() {
        None
    } else {
        let (mine, mut rest) = bytes.split_at_mut(binary_len);
        let slice = std::slice::from_raw_parts_mut(mine.as_mut_ptr() as *mut F, this.len());
        std::ptr::write(
            this,
            Vec::from_raw_parts(slice.as_mut_ptr(), this.len(), this.len()),
        );
        for element in this.iter_mut() {
            let temp = rest;
            rest = exhume_F(element, temp)?;
        }
        Some(rest)
    }
}
#[inline]
pub fn extent_vec_F<F: PrimeField>(this: &Vec<F>) -> usize {
    let mut sum = size_of::<F>() * this.len();
    for element in this.iter() {
        sum += extent_F(element);
    }
    sum
}

encode_decode!(
    entomb_vec_F,
    exhume_vec_F,
    encode_vec_F,
    decode_vec_F,
    Vec<F>
);

#[inline]
pub unsafe fn entomb_vec_vec_F<F: PrimeField, W: Write>(
    this: &Vec<Vec<F>>,
    write: &mut W,
) -> std::io::Result<()> {
    write.write_all(typed_to_bytes(&this[..]))?;
    for element in this.iter() {
        entomb_vec_F(element, write)?;
    }
    Ok(())
}
#[inline]
pub unsafe fn exhume_vec_vec_F<'a, 'b, F: PrimeField>(
    this: &'a mut Vec<Vec<F>>,
    bytes: &'b mut [u8],
) -> Option<&'b mut [u8]> {
    let binary_len = this.len() * size_of::<Vec<F>>();
    if binary_len > bytes.len() {
        None
    } else {
        let (mine, mut rest) = bytes.split_at_mut(binary_len);
        let slice = std::slice::from_raw_parts_mut(mine.as_mut_ptr() as *mut Vec<F>, this.len());
        std::ptr::write(
            this,
            Vec::from_raw_parts(slice.as_mut_ptr(), this.len(), this.len()),
        );
        for element in this.iter_mut() {
            let temp = rest;
            rest = exhume_vec_F(element, temp)?;
        }
        Some(rest)
    }
}
#[inline]
pub fn extent_vec_vec_F<F: PrimeField>(this: &Vec<Vec<F>>) -> usize {
    let mut sum = size_of::<Vec<F>>() * this.len();
    for element in this.iter() {
        sum += extent_vec_F(element);
    }
    sum
}

encode_decode!(
    entomb_vec_vec_F,
    exhume_vec_vec_F,
    encode_vec_vec_F,
    decode_vec_vec_F,
    Vec<Vec<F>>
);

#[inline]
pub unsafe fn entomb_option_vec_F<F: PrimeField, W: Write>(
    this: &Option<Vec<F>>,
    bytes: &mut W,
) -> std::io::Result<()> {
    if let &Some(ref inner) = this {
        entomb_vec_F(inner, bytes)?;
    }
    Ok(())
}

#[inline]
pub unsafe fn exhume_option_vec_F<'a, 'b, F: PrimeField>(
    this: &'a mut Option<Vec<F>>,
    mut bytes: &'b mut [u8],
) -> Option<&'b mut [u8]> {
    if let &mut Some(ref mut inner) = this {
        bytes = exhume_vec_F(inner, bytes)?;
    }
    Some(bytes)
}

#[inline]
pub fn extent_option_vec_F<F: PrimeField>(this: &Option<Vec<F>>) -> usize {
    this.as_ref().map(|inner| extent_vec_F(inner)).unwrap_or(0)
}

#[inline]
pub unsafe fn entomb_sparse_matrix_F<F: PrimeField, W: Write>(
    this: &SparseMatrix<F>,
    bytes: &mut W,
) -> std::io::Result<()> {
    entomb_vec_F(&this.w_hat, bytes)?;
    entomb_vec_F(&this.v_rest, bytes)?;
    Ok(())
}

#[inline]
pub unsafe fn exhume_sparse_matrix_F<'a, 'b, F: PrimeField>(
    this: &'a mut SparseMatrix<F>,
    mut bytes: &'b mut [u8],
) -> Option<&'b mut [u8]> {
    let temp = bytes;
    bytes = exhume_vec_F(&mut this.w_hat, temp)?;
    let temp = bytes;
    bytes = exhume_vec_F(&mut this.v_rest, temp)?;
    Some(bytes)
}

#[inline]
pub fn extent_sparse_matrix_F<F: PrimeField>(this: &SparseMatrix<F>) -> usize {
    let mut size = 0;
    size += extent_vec_F(&this.w_hat);
    size += extent_vec_F(&this.v_rest);
    size
}

vec_abomonate!(
    entomb_vec_sparse_matrix_F,
    exhume_vec_sparse_matrix_F,
    extent_vec_sparse_matrix_F,
    entomb_sparse_matrix_F,
    exhume_sparse_matrix_F,
    extent_sparse_matrix_F,
    SparseMatrix<F>
);

encode_decode!(
    entomb_vec_sparse_matrix_F,
    exhume_vec_sparse_matrix_F,
    encode_vec_sparse_matrix_F,
    decode_vec_sparse_matrix_F,
    Vec<SparseMatrix<F>>
);
