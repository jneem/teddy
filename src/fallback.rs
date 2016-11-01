// If SIMD instructions are not available, this file provides a dummy implementation of Teddy that
// will compile, but do nothing at runtime.

use Match;

pub struct Teddy;

impl Teddy {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (i.e., `pats` is empty or has
    /// an empty substring), then `None` is returned.
    pub fn new(_pats: &[Vec<u8>]) -> Option<Teddy> {
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

    /// Searches `haystack` for the substrings in this `Teddy`. If a match was
    /// found, then it is returned. Otherwise, `None` is returned.
    pub fn find(&self, _haystack: &[u8]) -> Option<Match> {
        unimplemented!();
    }
}

