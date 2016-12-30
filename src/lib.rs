//! This is a crate for SIMD-accelerated multi-substring matching. You may find it useful if:
//!
//! - you have lots of text to search through, and
//! - you're looking for a fairly small number of fairly short patterns, and
//! - your program will be running on a CPU that supports at least SSSE3.
//!
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
//!     ted.find(b"The quick brown fox jumped over the lazy dog.")
//! );
//! ```
//!
//! # Warning
//!
//! In order to get SIMD acceleration, you need to build this crate with the appropriate CPU
//! features turned on. You also need a nightly rust compiler. Please see the README for more
//! details.

#![deny(missing_docs)]
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

/// All the details for the match that Teddy found.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The index of the pattern that matched.
    ///
    /// The index is in correspondence with the order of the patterns given at construction. If
    /// you've already forgotten which order that was, don't panic! You can use `pat` as an index
    /// into the result of `Teddy::patterns()`.
    ///
    /// ```
    /// use teddy::{Match, Teddy};
    ///
    /// let patterns = vec![b"cat", b"dog", b"fox"];
    /// let ted = Teddy::new(patterns.iter().map(|s| &s[..])).unwrap();
    /// let pat = ted.find(b"The quick brown fox").unwrap().pat;
    /// assert_eq!(&ted.patterns()[pat], b"fox");
    /// ```
    pub pat: usize,
    /// The start byte offset of the match.
    ///
    /// This is an index into the search string, and it is inclusive.
    pub start: usize,
    /// The end byte offset of the match.
    ///
    /// This is an index into the search string, and it is exclusive. That is, if `m` is the
    /// `Match` struct that we got after searching through the string `haystack`, then you can
    /// retrieve the matched text using `haystack[m.start..m.end]`.
    pub end: usize,
}

