// If SIMD instructions are not available, this file provides a dummy implementation of Teddy that
// will compile, but do nothing at runtime.

use Match;

#[derive(Clone, Debug)]
pub struct Teddy;

impl Teddy {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (i.e., `pats` is empty or has
    /// an empty substring), then `None` is returned.
    ///
    /// # Warning
    ///
    /// If `teddy` was built without SIMD support, then this method will *always* return `None`.
    /// See the README for more information on how to compile `teddy` with SIMD support.
    pub fn new<'a, I>(_pats: I) -> Option<Teddy> where I: IntoIterator<Item=&'a [u8]> {
        None
    }

    /// Returns all of the substrings matched by this `Teddy`.
    pub fn patterns(&self) -> &[Vec<u8>] {
        unimplemented!();
    }

    /// Returns the approximate size on the heap used by this matcher.
    pub fn approximate_size(&self) -> usize {
        unimplemented!();
    }

    /// Searches `haystack` for the substrings in this `Teddy`. Returns the first match if one
    /// exists, and otherwise `None`.
    pub fn find(&self, _haystack: &[u8]) -> Option<Match> {
        unimplemented!();
    }

    /// Were we compiled with SIMD support? If not, `Teddy::new` will always just return `None`.
    ///
    /// See the README for more information on how to compile `teddy` with SIMD support.
    pub fn is_accelerated() -> bool {
        false
    }
}

