mod core;
mod mask;
mod teddy_simd;

use self::core::Teddy as TeddyInner;
use simd;
use Match;

#[cfg(target_feature="avx2")]
#[derive(Clone, Debug)]
pub struct Teddy(TeddyInner<simd::x86::avx::u8x32>);
#[cfg(all(not(target_feature="avx2"), target_feature="ssse3"))]
#[derive(Clone, Debug)]
pub struct Teddy(TeddyInner<simd::u8x16>);

impl Teddy {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (i.e., `pats` is empty or has
    /// an empty substring), then `None` is returned.
    pub fn new(pats: &[Vec<u8>]) -> Option<Teddy> {
        TeddyInner::new(pats).map(|t| Teddy(t))
    }

    /// Returns all of the substrings matched by this `Teddy`.
    pub fn patterns(&self) -> &[Vec<u8>] {
        self.0.patterns()
    }

    /// Returns the approximate size on the heap used by this matcher.
    pub fn approximate_size(&self) -> usize {
        self.0.approximate_size()
    }

    /// Searches `haystack` for the substrings in this `Teddy`. If a match was
    /// found, then it is returned. Otherwise, `None` is returned.
    pub fn find(&self, haystack: &[u8]) -> Option<Match> {
        self.0.find(haystack)
    }

    /// Were we compiled with SIMD support?
    pub fn is_accelerated() -> bool {
        true
    }

}
