// Copyright 2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The purpose of this module is to allow us to write the core part of Teddy's algorithm in a way
//! that is generic over the SIMD width. Unfortunately, the `simd` crate is not organized to
//! facilitate this: the traits it defines are based on which instruction set the required
//! operations belong to, not on what the operations actually do. This module therefore does a
//! minimal version of this sort of logical grouping. It may be worth expanding this module into an
//! alternative to the `simd` crate.

// Because this file conditionally compiles many things, some of the imports might only be used in
// certain configurations.
#![allow(unused_imports)]

use simd::{bool8ix16, u8x16};
use simd::x86::sse2::{u64x2, Sse2Bool8ix16};
use simd::x86::ssse3::Ssse3U8x16;
use std::fmt::Debug;
use std::mem::transmute;
use std::ops::{BitAnd, Shr};
use std::ptr;

#[cfg(target_feature="avx2")]
use simd::x86::avx::{bool8ix32, i8x32, u8x32, u64x4};

// Here are some operations that we need but are not (currently) exposed by `simd`.
extern "platform-intrinsic" {
    fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
}

#[cfg(target_feature="avx2")]
extern "platform-intrinsic" {
    #[cfg(not(feature="asm"))]
    fn simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U;
    fn x86_mm256_shuffle_epi8(x: i8x32, y: i8x32) -> i8x32;
    fn x86_mm256_movemask_epi8(x: i8x32) -> i32;
}

/// This trait contains all the SIMD operations necessary for implementing the Teddy algorithm.
pub trait TeddySIMD: BitAnd<Output=Self> + Clone + Copy + Debug + Shr<u8, Output=Self> + Sized {
    /// How many bytes fit into this SIMD type?
    ///
    /// This should probably be an associated const, but those aren't stable.
    fn block_size() -> usize;

    fn splat(x: u8) -> Self;
    fn extract(self, idx: u32) -> u8;
    fn replace(self, idx: u32, elem: u8) -> Self;
    fn shuffle_bytes(self, indices: Self) -> Self;
    fn is_nonzero(self) -> bool;

    /// Puts `left` on the left, `right` on the right, then shifts the whole thing by one byte to
    /// the right and returns the right half (so that the right-most byte of `left` will become the
    /// left-most byte of the answer).
    fn right_shift_1(left: Self, right: Self) -> Self;

    /// Same as `right_shift_1`, but shifts by 2 bytes.
    fn right_shift_2(left: Self, right: Self) -> Self;

    /// Returns a bitfield indicating which bytes in this vector are non-zero.
    fn nonzero_bytes(self) -> u32;

    /// Creates a new SIMD vector from the elements in `slice` starting at `offset`. `slice` must
    /// have at least the number of elements required to fill a SIMD vector.
    unsafe fn load_unchecked(slice: &[u8], offset: usize) -> Self;
}

impl TeddySIMD for u8x16 {
    #[inline]
    fn block_size() -> usize { 16 }
    #[inline]
    fn splat(x: u8) -> Self { u8x16::splat(x) }
    #[inline]
    fn extract(self, idx: u32) -> u8 { u8x16::extract(self, idx) }
    #[inline]
    fn replace(self, idx: u32, elem: u8) -> Self { u8x16::replace(self, idx, elem) }
    #[inline]
    fn is_nonzero(self) -> bool { bool8ix16::any(u8x16::ne(self, u8x16::splat(0))) }
    #[inline]
    fn nonzero_bytes(self) -> u32 { Sse2Bool8ix16::move_mask(u8x16::ne(self, u8x16::splat(0))) }

    #[inline]
    fn right_shift_1(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle16(left, right, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]) }
    }

    #[inline]
    fn right_shift_2(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle16(left, right, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) }
    }

    #[inline]
    fn shuffle_bytes(self, indices: Self) -> Self {
        Ssse3U8x16::shuffle_bytes(self, indices)
    }

    #[inline]
    unsafe fn load_unchecked(slice: &[u8], offset: usize) -> u8x16 {
        // I'm not sure if this is the best way to write an unaligned load, but it does seem to
        // compile down to a single `movdqu` instruction.
        let mut x = u8x16::splat(0);
        ptr::copy_nonoverlapping(
            slice.get_unchecked(offset),
            &mut x as *mut u8x16 as *mut u8,
            16);
        x
    }
}

#[cfg(target_feature="avx2")]
impl TeddySIMD for u8x32 {
    #[inline]
    fn block_size() -> usize { 32 }
    #[inline]
    fn splat(x: u8) -> Self { u8x32::splat(x) }
    #[inline]
    fn extract(self, idx: u32) -> u8 { u8x32::extract(self, idx) }
    #[inline]
    fn replace(self, idx: u32, elem: u8) -> Self { u8x32::replace(self, idx, elem) }
    #[inline]
    fn is_nonzero(self) -> bool { bool8ix32::any(u8x32::ne(self, u8x32::splat(0))) }
    #[inline]
    fn nonzero_bytes(self) -> u32 {
        let nonzero = u8x32::ne(self, u8x32::splat(0));
        unsafe { transmute(x86_mm256_movemask_epi8(transmute(nonzero))) }
    }

    #[cfg(not(feature="asm"))]
    #[inline]
    fn right_shift_1(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle32(left, right, [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58, 59, 60, 61, 62]) }
    }

    #[cfg(not(feature="asm"))]
    #[inline]
    fn right_shift_2(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle32(left, right, [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58, 59, 60, 61]) }
    }

    #[cfg(feature="asm")]
    #[inline]
    fn right_shift_1(left: Self, right: Self) -> Self {
        // It would be nicer just to use an intrinsic for this shuffle, but LLVM generates four
        // instructions for the shuffle we need, whereas the optimal sequence only takes two.
        unsafe {
            let ret: Self;
            // 33 = 0x21, so the `vperm2i128` instruction concatenates the least significant 16
            // bytes of `right` with the most significant 16 bytes of `left`. That is,
            //
            // left:  31L .. 16L | 15L .. 0L
            // right: 31R .. 16R | 15R .. 0R
            //
            // gives
            //
            // ret: 15R .. 0R | 31L .. 16L
            //
            // Then we shift both lanes of `ret` to the right by 15, while shifting in `right`.
            // This gives
            //
            // ret: 30R ... 16R 15R | 14R .. 0R 31L
            //
            // Overall, we managed to shift `right` one byte to the left, while shifting in `left`
            // from the right. But then why is this method called `right_shift_1`?! It's because
            // the pictures above are with the most significant bytes on the left, but in most of
            // this crate (for example, in the crate documentation describing the Teddy algorithm)
            // we think of text as running from left to right. Since x86 is little endian, the left
            // shift above is a right shift from the text perspective.
            asm!("vperm2i128 $$33, $2, $1, $0; vpalignr $$15, $0, $2, $0"
                 // The output of the code above. That is, `$0` above refers to the variable `ret`.
                 // The '=' means that we write to it, and the '&' means that we also use it as a
                 // temporary value. The 'x' means it's a SIMD register.
                 : "=&x"(ret)
                 // The inputs (which we do not write to). That is, `$1` above means `left`, while
                 // `$2` means `right`.
                 : "x"(left), "x"(right)
            );
            ret
        }
    }

    #[cfg(feature="asm")]
    #[inline]
    fn right_shift_2(left: Self, right: Self) -> Self {
        unsafe {
            let ret: Self;
            asm!("vperm2i128 $$33, $2, $1, $0; vpalignr $$14, $0, $2, $0"
                 : "=&x"(ret)
                 : "x"(left), "x"(right)
            );
            ret
        }
    }

    #[inline]
    fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe { transmute(x86_mm256_shuffle_epi8(transmute(self), transmute(indices))) }
    }

    #[inline]
    unsafe fn load_unchecked(slice: &[u8], offset: usize) -> u8x32 {
        let mut x = u8x32::splat(0);
        ptr::copy_nonoverlapping(
            slice.get_unchecked(offset),
            &mut x as *mut u8x32 as *mut u8,
            32);
        x
    }
}

#[cfg(test)]
mod tests {
    use simd::u8x16;
    #[cfg(target_feature="avx2")]
    use simd::x86::avx::u8x32;
    use super::TeddySIMD;

    #[test]
    fn right_shifts_128() {
        let left: u8x16 = unsafe { TeddySIMD::load_unchecked(b"0123456789ABCDEF", 0) };
        let right: u8x16 = unsafe { TeddySIMD::load_unchecked(b"0123456789abcdef", 0) };
        let result = TeddySIMD::right_shift_1(left, right);
        let expected = unsafe { TeddySIMD::load_unchecked(b"F0123456789abcde", 0) };
        assert!(result.eq(expected).all());

        let result = TeddySIMD::right_shift_2(left, right);
        let expected = unsafe { TeddySIMD::load_unchecked(b"EF0123456789abcd", 0) };
        assert!(result.eq(expected).all());
    }

    #[cfg(target_feature="avx2")]
    #[test]
    fn right_shifts_256() {
        let left: u8x32 = unsafe { TeddySIMD::load_unchecked(b"0123456789ABCDEFqwertyuiopasdfgh", 0) };
        let right: u8x32 = unsafe { TeddySIMD::load_unchecked(b"0123456789abcdefQWERTYUIOPASDFGH", 0) };
        let result = TeddySIMD::right_shift_1(left, right);
        let expected = unsafe { TeddySIMD::load_unchecked(b"h0123456789abcdefQWERTYUIOPASDFG", 0) };
        assert!(result.eq(expected).all());

        let result = TeddySIMD::right_shift_2(left, right);
        let expected = unsafe { TeddySIMD::load_unchecked(b"gh0123456789abcdefQWERTYUIOPASDF", 0) };
        assert!(result.eq(expected).all());
    }
}

