// Copyright 2016 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use simd::{bool8ix16, u8x16};
use simd::x86::sse2::u64x2;
use simd::x86::ssse3::Ssse3U8x16;
use std::mem::transmute;
use std::ops::{BitAnd, Shr};
use std::ptr;

#[cfg(target_feature="avx2")]
use simd::x86::avx::u8x32;

extern "platform-intrinsic" {
    fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
}

pub trait TeddySIMDBool: Clone + Copy + Sized {
    fn any(self) -> bool;
}

/// This trait contains all the SIMD operations necessary for implementing the Teddy algorithm.
pub trait TeddySIMD: BitAnd<Output=Self> + Clone + Copy + Shr<u8, Output=Self> + Sized {
    const BLOCK_SIZE: usize;

    /// The boolean version of this vector.
    type Bool: TeddySIMDBool;

    fn ne(self, other: Self) -> Self::Bool;
    fn splat(x: u8) -> Self;
    fn extract(self, idx: u32) -> u8;
    fn replace(self, idx: u32, elem: u8) -> Self;
    fn shuffle_bytes(self, indices: Self) -> Self;

    /// Puts `left` on the left, `right` on the right, then shifts the whole thing by one byte to
    /// the right and returns the right half (so that the right-most byte of `left` will become the
    /// left-most byte of the answer).
    fn right_shift_1(left: Self, right: Self) -> Self;

    /// Same as `right_shift_1`, but shifts by 2 bytes.
    fn right_shift_2(left: Self, right: Self) -> Self;

    /// Applies the function `f` to each of the `u64` values in this vector (beginning with the
    /// least significant). Returns the first non-`None` value that `f` returned.
    fn first_u64<T, F>(self, f: F) -> Option<T> where F: Fn(u64, usize) -> Option<T>;

    /// Creates a new SIMD vector from the elements in `slice` starting at `offset`. `slice` must
    /// have at least the number of elements required to fill a SIMD vector.
    unsafe fn load_unchecked(slice: &[u8], offset: usize) -> Self;
}

impl TeddySIMD for u8x16 {
    const BLOCK_SIZE: usize = 16;
    type Bool = bool8ix16;

    fn ne(self, other: Self) -> Self::Bool { u8x16::ne(self, other) }
    fn splat(x: u8) -> Self { u8x16::splat(x) }
    fn extract(self, idx: u32) -> u8 { u8x16::extract(self, idx) }
    fn replace(self, idx: u32, elem: u8) -> Self { u8x16::replace(self, idx, elem) }

    fn first_u64<T, F>(self, f: F) -> Option<T> where F: Fn(u64, usize) -> Option<T> {
        let res64: u64x2 = unsafe { transmute(self) };

        if let Some(m) = f(res64.extract(0), 0) {
            Some(m)
        } else if let Some(m) = f(res64.extract(1), 8) {
            Some(m)
        } else {
            None
        }
    }

    fn right_shift_1(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle16(left, right, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]) }
    }

    fn right_shift_2(left: Self, right: Self) -> Self {
        unsafe { simd_shuffle16(left, right, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) }
    }

    fn shuffle_bytes(self, indices: Self) -> Self {
        Ssse3U8x16::shuffle_bytes(self, indices)
    }

    unsafe fn load_unchecked(slice: &[u8], offset: usize) -> u8x16 {
        // TODO: Can we just do pointer casting here? I don't think so, since
        // this could be an unaligned load? Help me.
        let mut x = u8x16::splat(0);
        ptr::copy_nonoverlapping(
            slice.get_unchecked(offset),
            &mut x as *mut u8x16 as *mut u8,
            16);
        x
    }
}

impl TeddySIMDBool for bool8ix16 {
    fn any(self) -> bool { bool8ix16::any(self) }
}


