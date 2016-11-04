//! This crate contains two types: `Teddy` is the main one. You create one by passing in a set of
//! patterns. It does some preprocessing, and then you can use it to quickly search for those
//! patterns in some text. Searching returns a `Match`, which will tell you which of the patterns
//! matched and where.
//!
//! ```
//! use teddy::{Match, Teddy};
//!
//! let patterns = vec![b"cat", b"dog", b"fox"];
//! let ted = Teddy::new(patterns.iter().map(|s| &s[..])).unwrap();
//! assert_eq!(
//!     Some(Match { pat: 2, start: 16, end: 19 }),
//!     ted.find(b"The quick brown fox jumped over the laxy dog.")
//! );
//! ```

#![cfg_attr(feature="simd-accel", feature(asm, associated_consts, cfg_target_feature, platform_intrinsics))]

extern crate aho_corasick;
#[cfg(feature="simd-accel")]
extern crate simd;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(all(feature="simd-accel", any(target_feature="sse3", target_feature="avx2")))]
mod x86;
#[cfg(all(feature="simd-accel", any(target_feature="sse3", target_feature="avx2")))]
pub use x86::Teddy;

#[cfg(not(all(feature="simd-accel", any(target_feature="sse3", target_feature="avx2"))))]
mod fallback;
#[cfg(not(all(feature="simd-accel", any(target_feature="sse3", target_feature="avx2"))))]
pub use fallback::Teddy;

/// Match reports match information.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The index of the pattern that matched. The index is in correspondence
    /// with the order of the patterns given at construction.
    pub pat: usize,
    /// The start byte offset of the match.
    pub start: usize,
    /// The end byte offset of the match. This is always `start + pat.len()`.
    pub end: usize,
}

