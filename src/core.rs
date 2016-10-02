// Copyright 2016 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use aho_corasick::{Automaton, AcAutomaton, FullAcAutomaton};
use mask::Masks;
use Match;
use teddy_simd::{TeddySIMD, TeddySIMDBool};

/// A SIMD accelerated multi substring searcher.
#[derive(Debug, Clone)]
pub struct Teddy<T: TeddySIMD> {
    /// A list of substrings to match.
    pats: Vec<Vec<u8>>,
    /// A set of 8 buckets. Each bucket corresponds to a single member of a
    /// bitset. A bucket contains zero or more substrings. This is useful
    /// when the number of substrings exceeds 8, since our bitsets cannot have
    /// more than 8 members.
    buckets: Vec<Vec<usize>>,
    /// Our set of masks. There's one mask for each byte in the fingerprint.
    masks: Masks<T>,
    /// An Aho-Corasick automaton, which we use for quickly testing for a match
    /// after we've found a fingerprint.
    ac: FullAcAutomaton<Vec<u8>>,
}

// This is the content of the inner loop for the case when there is only 1 mask. This is the easy
// case and is pretty much as described in the module documentation.
//
// The main reason that we separate this out as a macro is to reuse the same code for both aligned
// and unaligned loads. (Also, it helps us to manually unroll a loop.)
//
// `$slf` is an instance of `Teddy<T>`.
// `$load` is an expression for loading the next bunch of bytes from the haystack.
// `$prev0` and `$prev1` are unused here, but see find2_step.
macro_rules! find1_step {
    ($slf:expr, $load:expr, $prev0:expr, $prev1:expr) => {
        {
            let h = unsafe { $load };
            $slf.masks.members1(h)
        }
    };
}

// This is the inner loop for the case when the fingerprint is 2 bytes long.
//
// `$prev0` is the previous value of `C` (from the crate documentation) for the
// *first* byte in the fingerprint. On subsequent iterations, we take
// the last bitset from the previous `C` and insert it into the first
// position of the current `C`, shifting all other bitsets to the right
// one lane. This causes `C` for the first byte to line up with `C` for
// the second byte, so that they can be `AND`'d together.
//
// `$prev1` is similar, except that it is used in addition to `prev1` in the
// case of a 3-byte fingerprint. In this case, `prev1` is the previous value
// of `C` for the second byte in the fingerprint.
macro_rules! find2_step {
    ($slf:expr, $load:expr, $prev0:expr, $prev1:expr) => {
        {
            let h = unsafe { $load };
            let (res0, res1) = $slf.masks.members2(h);
            let res0prev0 = T::right_shift_1($prev0, res0);
            $prev0 = res0;
            res1 & res0prev0
        }
    };
}

// This is the inner loop for the case when the fingerprint is 3 bytes long.
//
// This is a straight-forward extrapolation of the two-byte case. The only
// difference is that we need to keep track of two previous values of `C`,
// since we now need to align for three bytes.
macro_rules! find3_step {
    ($slf:expr, $load:expr, $prev0:expr, $prev1:expr) => {
        {
            let h = unsafe { $load };
            let (res0, res1, res2) = $slf.masks.members3(h);
            let res0prev0 = T::right_shift_2($prev0, res0);
            let res1prev1 = T::right_shift_1($prev1, res1);
            $prev0 = res0;
            $prev1 = res1;

            res2 & res1prev1 & res0prev0
        }
    };
}

impl<T: TeddySIMD> Teddy<T> {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (i.e., `pats` is empty or has
    /// an empty substring), then `None` is returned.
    pub fn new(pats: &[Vec<u8>]) -> Option<Teddy<T>> {
        if pats.is_empty() || pats.iter().any(|p| p.is_empty()) {
            None
        } else {
            let (buckets, masks) = Masks::buckets_and_masks(pats);
            Some(Teddy {
                pats: pats.to_vec(),
                buckets: buckets,
                masks: masks,
                ac: FullAcAutomaton::new((AcAutomaton::new(pats.to_vec()))),
            })
        }
    }

    /// Returns all of the substrings matched by this `Teddy`.
    pub fn patterns(&self) -> &[Vec<u8>] {
        &self.pats
    }

    /// Returns the number of substrings in this matcher.
    pub fn len(&self) -> usize {
        self.pats.len()
    }

    /// Returns the approximate size on the heap used by this matcher.
    pub fn approximate_size(&self) -> usize {
        self.pats.iter().fold(0, |a, b| a + b.len())
    }

    /// Searches `haystack` for the substrings in this `Teddy`. If a match was
    /// found, then it is returned. Otherwise, `None` is returned.
    // This function uses macros to expand out three different cases. Not all of the declared
    // variables are really used in all the cases, and so we allow unused assignments in order to
    // squelch those warnings.
    #[allow(unused_assignments)]
    pub fn find(&self, haystack: &[u8]) -> Option<Match> {
        // If our haystack is too small, fall back to Aho-Corasick.
        // TODO: the threshold here should be informed by benchmarks.
        if haystack.is_empty() || haystack.len() < 2 * T::BLOCK_SIZE {
            return self.slow(haystack, 0);
        }

        // With a multi-byte fingerprint, we need to include results from previous iterations. To
        // avoid special casing at the beginning of the input, it's easiest to start a byte or two
        // after the beginning.
        let mut pos = self.masks.len() - 1;
        let zero = T::splat(0);
        let len = haystack.len();

        let mut prev0 = T::splat(0xFF);
        let mut prev1 = T::splat(0xFF);

        macro_rules! find_loop {
            ($step:ident) => {
                {
                    // Do the first, unaligned, iteration.
                    let mut res = $step!(self, T::load_unchecked(haystack, pos), prev0, prev1);

                    // Only do expensive verification if there are any non-zero bits.
                    let bitfield = res.ne(zero).move_mask();
                    if bitfield != 0 {
                        let pos = pos - (self.masks.len() - 1);
                        if let Some(m) = self.verify(haystack, pos, res, bitfield) {
                            return Some(m);
                        }
                    }

                    // Increment pos by up to BLOCK_SIZE, but only as far as the next alignment boundary.
                    let pos_align = (haystack.as_ptr() as usize + pos) % T::BLOCK_SIZE;
                    pos = pos + T::BLOCK_SIZE - pos_align;

                    // Since we shifted by an amount not necessarily equal to BLOCK_SIZE, prev0 and
                    // prev1 are not correct. We could fix it by shifting them, but that isn't
                    // terribly easy since the shift is not known at compile-time (which is what
                    // SSE prefers). It seems easier to just conservatively set prev0 and prev1 to
                    // all 1's. This allows some false positives in the fingerprint matching step,
                    // but we're not in the inner loop yet, so that's ok.
                    prev0 = T::splat(0xFF);
                    prev1 = T::splat(0xFF);

                    // The main loop (in which the loads are all aligned). The control flow here is
                    // a bit funky. Logically, this is the same as
                    //    while ... {
                    //        step!(...);
                    //        if we should verify {
                    //            verify();
                    //        }
                    //    }
                    // However, this weird double-loop version is faster when verifying is rare
                    // (and if it isn't rare then you should be using Teddy anyway). Also,
                    // unrolling the inner loop gives a nice little boost.
                    //
                    // Duplicating the conditionals is a little unfortunate, but the only way I
                    // found to avoid it was using labelled loops. That works, but spits out a huge
                    // number of (un-shut-up-able) warnings because labels aren't hygienic in
                    // macros.
                    let end = len.saturating_sub(2 * T::BLOCK_SIZE - 1);
                    while pos < end {
                        let mut bitfield = 0;

                        // TODO: it may be worthwhile writing this inner loop in assembly. Among
                        // other things, the code generated by LLVM spills a lot of YMM registers
                        // that probably don't need to be spilled.
                        while pos < end {
                            res = $step!(self, *(haystack.get_unchecked(pos) as *const u8 as *const T), prev0, prev1);
                            bitfield = res.ne(zero).move_mask();
                            if bitfield != 0 {
                                break;
                            }
                            pos += T::BLOCK_SIZE;

                            res = $step!(self, *(haystack.get_unchecked(pos) as *const u8 as *const T), prev0, prev1);
                            bitfield = res.ne(zero).move_mask();
                            if bitfield != 0 {
                                break;
                            }
                            pos += T::BLOCK_SIZE;
                        }

                        if bitfield != 0 {
                            let start_pos = pos - (self.masks.len() - 1);
                            if let Some(m) = self.verify(haystack, start_pos, res, bitfield) {
                                return Some(m);
                            }
                            pos += T::BLOCK_SIZE;
                        }
                    }
                }
            };
        }

        match self.masks.len() {
            1 => { find_loop!(find1_step) },
            2 => { find_loop!(find2_step) },
            3 => { find_loop!(find3_step) },
            _ => unreachable!(),
        }

        // Do a slow search through the last part of the haystack, which was not big enough to do
        // SIMD on.  Because of the windowing involved in looking for a multi-byte fingerprint, the
        // code above doesn't check the last `self.masks.len() - 1` bytes in the last window, so
        // start the slow search that many bytes earlier to compensate.
        self.slow(haystack, pos - (self.masks.len() - 1))
    }

    /// Runs the verification procedure on `res` (i.e., `C` from the module
    /// documentation), where the haystack block starts at `pos` in
    /// `haystack`.
    ///
    /// If a match exists, it returns the first one.
    // TODO: it may be worth optimizing a special case: if the fingerprints have no false positives
    // and the patterns are all the same length as the fingerprints then we don't have to verify
    // anything at all. This sounds like a very special case, but it seems to actually happen (for
    // example, when the patterns are a smallish number of short, case-insensitive strings).
    #[inline(always)]
    fn verify(&self, haystack: &[u8], pos: usize, res: T, mut bitfield: u32) -> Option<Match> {
        while bitfield != 0 {
            // The next offset, relative to pos, where some fingerprint matched.
            let byte_pos = bitfield.trailing_zeros();
            bitfield &= !(1 << byte_pos);

            // Offset relative to the beginning of the haystack.
            let start = pos + byte_pos as usize;

            // The bitfield telling us which patterns had fingerprints that match at this starting
            // position.
            let mut patterns = res.extract(byte_pos);
            while patterns != 0 {
                let bucket = patterns.trailing_zeros() as usize;
                patterns &= !(1 << bucket);

                // Actual substring search verification.
                if let Some(m) = self.verify_bucket(haystack, bucket, start) {
                    return Some(m);
                }
            }
        }

        None
    }

    /// Verifies whether any substring in the given bucket matches in haystack
    /// at the given starting position.
    #[inline(always)]
    fn verify_bucket(
        &self,
        haystack: &[u8],
        bucket: usize,
        start: usize,
    ) -> Option<Match> {
        // This cycles through the patterns in the bucket in the order that
        // the patterns were given. Therefore, we guarantee leftmost-first
        // semantics.
        for &pati in &self.buckets[bucket] {
            let pat = &*self.pats[pati];
            if start + pat.len() > haystack.len() {
                continue;
            }
            if pat == &haystack[start..start + pat.len()] {
                return Some(Match {
                    pat: pati,
                    start: start,
                    end: start + pat.len(),
                });
            }
        }
        None
    }

    /// Slow substring search through all patterns in this matcher.
    ///
    /// This is used when we don't have enough bytes in the haystack for our
    /// block based approach.
    fn slow(&self, haystack: &[u8], pos: usize) -> Option<Match> {
        self.ac.find(&haystack[pos..]).next().map(|m| {
            Match {
                pat: m.pati,
                start: m.start + pos,
                end: m.end + pos,
            }
        })
    }
}

/*
#[cfg(target_feature="avx2")]
impl Finder for Teddy<u8x32> {
    fn do_find(&self, haystack: &[u8]) -> Option<Match> {
        if haystack.is_empty() || haystack.len() < (u8x32::BLOCK_SIZE + 2) {
            return self.slow(haystack, 0);
        }

        let mut pos = self.masks.len() - 1;
        let zero = u8x32::splat(0);
        let len = haystack.len();
        type T = u8x32;

        let mut prev0 = u8x32::splat(0xFF);
        let mut prev1 = u8x32::splat(0xFF);

        match self.masks.len() {
            0 => { return None; },
            1 => { return None; },
            2 => {
                find2_step!(self, u8x32::load_unchecked(haystack, pos), haystack, pos, prev0, prev1, zero);

                let pos_align = (haystack.as_ptr() as usize + pos) % u8x32::BLOCK_SIZE;
                pos = pos + u8x32::BLOCK_SIZE - pos_align;

                prev0 = u8x32::splat(0xFF);
                prev1 = u8x32::splat(0xFF);

                while pos <= len - u8x32::BLOCK_SIZE {
                    find2_step!(self, *(haystack.get_unchecked(pos) as *const u8 as *const T), haystack, pos, prev0, prev1, zero);
                    /*
                    let h = unsafe { *(haystack.get_unchecked(pos) as *const u8 as *const u8x32) };
                    let (part0, part1) = self.masks.members2(h);

                    let prev0part0 = u8x32::right_shift_1(prev0, part0);
                    let res = prev0part0 & part1;
                    prev0 = part0;

                    let bitfield = res.ne(zero).move_mask();
                    if bitfield != 0 {
                        let pos = pos.checked_sub(1).unwrap();
                        if let Some(m) = self.verify(haystack, pos, res, bitfield) {
                            return Some(m);
                        }
                    }
                    */

                    pos += u8x32::BLOCK_SIZE;
                }
            },
            3 => { return None; },
            _ => unreachable!(),
        }

        self.slow(haystack, pos - (self.masks.len() - 1))
    }
}
*/

#[cfg(test)]
mod tests {
    use core::Teddy;
    use simd::u8x16;
    //use quickcheck::TestResult;

    #[test]
    fn one_pattern() {
        let pats = vec![b"abc".to_vec()];
        let ted = Teddy::<u8x16>::new(&pats).unwrap();
        assert_eq!(ted.find(b"123abcxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").unwrap().start, 3);
        assert_eq!(ted.find(b"xxxxxxxxxxxxxxabc123xxxxxxxxxxxxxxxx").unwrap().start, 14);
        assert_eq!(ted.find(b"xxxxxxxxxxxxxxxabc123xxxxxxxxxxxxxxx").unwrap().start, 15);
        assert_eq!(ted.find(b"xxxxxxxxxxxxxxxxabc123xxxxxxxxxxxxxx").unwrap().start, 16);
        assert_eq!(ted.find(b"xxxxxxxxxxxxxxxxxabc123xxxxxxxxxxxxx").unwrap().start, 17);
        assert_eq!(ted.find(b"xxxxxxxxxxxxxxxxxxabc123xxxxxxxxxxxx").unwrap().start, 18);
        assert_eq!(ted.find(b"abcabcxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").unwrap().start, 0);
        assert_eq!(ted.find(b"789xyzxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"), None);
        // TODO: test different alignments.
    }

    // TODO: these tests don't really end up testing anything. We need better choices for arbitrary
    // patterns and haystacks.
    /*
    quickcheck! {
        fn fast_equal_slow_128(pats: Vec<Vec<u8>>, haystack: Vec<u8>) -> TestResult {
            if let Some(ted) = Teddy::<u8x16>::new(&pats) {
                if let Some(res) = ted.slow(&haystack, 0) {
                    return TestResult::from_bool(ted.find(&haystack) == Some(res));
                }
            }
            TestResult::discard()
        }
    }



    #[cfg(target_feature="avx2")]
    quickcheck! {
        fn fast_equal_slow_256(pats: Vec<Vec<u8>>, haystack: Vec<u8>) -> TestResult {
            if let Some(ted) = Teddy::<u8x32>::new(&pats) {
                if let Some(res) = ted.slow(&haystack, 0) {
                    return TestResult::from_bool(ted.find(&haystack) == Some(res));
                }
            }
            TestResult::discard()
        }
    }
    */
}

