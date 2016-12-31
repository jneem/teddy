mod core;
mod mask;
mod teddy_simd;

use self::core::Teddy as TeddyInner;
use simd;
use Match;

#[cfg(target_feature="avx2")]
#[derive(Clone, Debug)]
/// A SIMD accelerated multi substring searcher.
pub struct Teddy(TeddyInner<simd::x86::avx::u8x32>);
#[cfg(all(not(target_feature="avx2"), target_feature="ssse3"))]
#[derive(Clone, Debug)]
/// A SIMD accelerated multi substring searcher.
pub struct Teddy(TeddyInner<simd::u8x16>);

impl Teddy {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (e.g., `pats` is empty or contains an empty
    /// pattern), then `None` is returned.
    ///
    /// # Warning
    ///
    /// If `teddy` was built without SIMD support, then this method will *always* return `None`.
    /// See the README for more information on how to compile `teddy` with SIMD support.
    pub fn new<'a, I>(pats: I) -> Option<Teddy> where I: IntoIterator<Item=&'a [u8]> {
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

    /// Searches `haystack` for the substrings in this `Teddy`. Returns the first match if one
    /// exists, and otherwise `None`.
    pub fn find(&self, haystack: &[u8]) -> Option<Match> {
        self.0.find(haystack)
    }

    /// Were we compiled with SIMD support? If not, `Teddy::new` will always just return `None`.
    ///
    /// See the README for more information on how to compile `teddy` with SIMD support.
    pub fn is_accelerated() -> bool {
        true
    }
}
