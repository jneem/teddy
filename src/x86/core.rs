// Copyright 2016 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use aho_corasick::{Automaton, AcAutomaton, FullAcAutomaton};
use x86::mask::{Mask, Masks};
use Match;
use x86::teddy_simd::{TeddySIMD, TeddySIMDBool};

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
    /// An Aho-Corasick automaton, which we use as a fall-back.
    ac: FullAcAutomaton<Vec<u8>>,
}

/// `State<T>` represents the state that we need to maintain in the Teddy inner loop.
///
/// The main job of the inner loop is to keep calling `update` with new input until `needs_verify`
/// returns true. Therefore, these should be the most optimized operations.
trait State<T> {
    /// Forget all the existing state.
    fn reset(&mut self);

    /// Update the state to reflect reading in a new block of input.
    fn update(&mut self, input: T);

    /// After reading in the most recent block of input, did a fingerprint match? If so, we need to
    /// verify whether there was a full match.
    fn needs_verify(&self) -> bool;

    /// The SIMD vector showing all the matched fingerprints. In the crate documentation, this is
    /// `C`.
    fn result(&self) -> T;

    /// If this state represents a multi-byte fingerprint, the matches it finds are offset from the
    /// block of input that we got in `update`. This returns the size of that offset: if we got a
    /// block of input starting at index `pos` then we are looking for fingerprints starting from
    /// index `pos - self.offset()`.
    fn offset(&self) -> usize;
}

// Turns a block of input into two: the first contains only the high nybbles and the second
// contains only the low ones.
#[inline(always)]
fn nybble_input<T: TeddySIMD>(haystack_block: T) -> (T, T) {
    let masklo = T::splat(0xF);
    let hlo = haystack_block & masklo;
    let hhi = (haystack_block >> 4) & masklo;

    (hhi, hlo)
}

struct State1<T: TeddySIMD> {
    mask: Mask<T>,
    /// The result of shuffling the high nybbles (i.e. `C1` in the crate documentation).
    res_hi: T,
    /// The result of shuffling the low nybbles (i.e. `C0` in the crate documentation).  According
    /// to the crate documentation, it would seem more sensible to just store the AND of `res_hi`
    /// and `res_lo`. The reason we store them separately is an optimization: the `vptest`
    /// instruction means that we can avoid explicitly ANDing these together in the fast path.
    res_lo: T,
}

struct State2<T: TeddySIMD> {
    /// The mask for the first byte of the fingerprint.
    mask0: Mask<T>,
    /// The mask for the second byte of the fingerprint.
    mask1: Mask<T>,
    /// The result (i.e. `C` in the crate documentation) of searching for the first byte of the
    /// fingerprint, but for the old block of input.
    prev0: T,
    /// We compute `C` for the first byte of the fingerprint, then we shift it one byte to the left
    /// and shift in the last byte of `prev0`. The answer goes here.
    res0: T,
    /// This is `C` for the second byte of the fingerprint. Thanks to the shifting that went into
    /// defining `res0`, these two line up.
    res1: T,
}

struct State3<T: TeddySIMD> {
    mask0: Mask<T>,
    mask1: Mask<T>,
    mask2: Mask<T>,
    prev0: T,
    prev1: T,
    /// As in `State2`, but shifted by two bytes instead of one.
    res0: T,
    /// Pretend that we had `res1` and `res2`, which are analogous to `res0` but with different
    /// amounts of shift. Then this is `res1 & res2`.
    res12: T,
}

impl<T: TeddySIMD> State1<T> {
    fn new(masks: &Masks<T>) -> State1<T> {
        State1 {
            mask: masks.0[0],
            res_hi: T::splat(0),
            res_lo: T::splat(0),
        }
    }
}

impl<T: TeddySIMD> State<T> for State1<T> {
    fn reset(&mut self) {}
    fn offset(&self) -> usize { 0 }

    #[inline(always)]
    fn update(&mut self, input: T) {
        let (hi, lo) = nybble_input(input);
        self.res_hi = self.mask.hi.shuffle_bytes(hi);
        self.res_lo = self.mask.lo.shuffle_bytes(lo);
    }

    #[inline(always)]
    fn needs_verify(&self) -> bool {
        !self.res_hi.test_zero(self.res_lo)
    }

    #[inline(always)]
    fn result(&self) -> T {
        self.res_hi & self.res_lo
    }
}

impl<T: TeddySIMD> State2<T> {
    fn new(masks: &Masks<T>) -> State2<T> {
        State2 {
            mask0: masks.0[0],
            mask1: masks.0[1],
            prev0: T::splat(0xFF),
            res0: T::splat(0),
            res1: T::splat(0),
        }
    }
}

impl<T: TeddySIMD> State<T> for State2<T> {
    fn reset(&mut self) {
        self.prev0 = T::splat(0xFF);
    }
    fn offset(&self) -> usize { 1 }

    #[inline(always)]
    fn update(&mut self, input: T) {
        let (hi, lo) = nybble_input(input);
        let shuf0 = self.mask0.hi.shuffle_bytes(hi) & self.mask0.lo.shuffle_bytes(lo);

        self.res0 = T::right_shift_1(self.prev0, shuf0);
        self.res1 = self.mask1.hi.shuffle_bytes(hi) & self.mask1.lo.shuffle_bytes(lo);
        self.prev0 = shuf0;
    }

    #[inline(always)]
    fn needs_verify(&self) -> bool {
        !self.res0.test_zero(self.res1)
    }

    #[inline(always)]
    fn result(&self) -> T {
        self.res0 & self.res1
    }
}

impl<T: TeddySIMD> State3<T> {
    fn new(masks: &Masks<T>) -> State3<T> {
        State3 {
            mask0: masks.0[0],
            mask1: masks.0[1],
            mask2: masks.0[2],
            prev0: T::splat(0xFF),
            prev1: T::splat(0xFF),
            res0: T::splat(0),
            res12: T::splat(0),
        }
    }
}

impl<T: TeddySIMD> State<T> for State3<T> {
    fn reset(&mut self) {
        self.prev0 = T::splat(0xFF);
        self.prev1 = T::splat(0xFF);
    }
    fn offset(&self) -> usize { 2 }

    #[inline(always)]
    fn update(&mut self, input: T) {
        let (hi, lo) = nybble_input(input);
        let shuf0 = self.mask0.hi.shuffle_bytes(hi) & self.mask0.lo.shuffle_bytes(lo);
        let shuf1 = self.mask1.hi.shuffle_bytes(hi) & self.mask1.lo.shuffle_bytes(lo);
        let res2 = self.mask2.hi.shuffle_bytes(hi) & self.mask2.lo.shuffle_bytes(lo);
        let res1 = T::right_shift_1(self.prev1, shuf1);

        self.res0 = T::right_shift_2(self.prev0, shuf0);
        self.prev0 = shuf0;
        self.prev1 = shuf1;
        self.res12 = res1 & res2;
    }

    #[inline(always)]
    fn needs_verify(&self) -> bool {
        !self.res0.test_zero(self.res12)
    }

    #[inline(always)]
    fn result(&self) -> T {
        self.res0 & self.res12
    }
}

impl<T: TeddySIMD> Teddy<T> {
    /// Create a new `Teddy` multi substring matcher.
    ///
    /// If a `Teddy` matcher could not be created (i.e., `pats` is empty or has
    /// an empty substring), then `None` is returned.
    pub fn new<'a, I>(pats: I) -> Option<Teddy<T>> where I: IntoIterator<Item=&'a [u8]> {
        let pats: Vec<Vec<u8>> = pats.into_iter().map(|p| p.to_vec()).collect();
        if pats.is_empty() || pats.iter().any(|p| p.is_empty()) {
            None
        } else {
            let (buckets, masks) = Masks::buckets_and_masks(&pats);
            let ac = FullAcAutomaton::new(AcAutomaton::new(pats.clone()));
            Some(Teddy {
                pats: pats,
                buckets: buckets,
                masks: masks,
                ac: ac,
            })
        }
    }

    /// Returns all of the substrings matched by this `Teddy`.
    pub fn patterns(&self) -> &[Vec<u8>] {
        &self.pats
    }

    /// Returns the approximate size on the heap used by this matcher.
    pub fn approximate_size(&self) -> usize {
        self.pats.iter().fold(0, |a, b| a + b.len()) + self.ac.heap_bytes()
    }

    fn find_loop<S: State<T>>(&self, haystack: &[u8], mut state: S) -> Option<Match> {
        // With a multi-byte fingerprint, we need to include results from previous iterations. To
        // avoid special casing at the beginning of the input, it's easiest to start a byte or two
        // after the beginning.
        let mut pos = state.offset();
        let len = haystack.len();

        // Do the first iteration of the loop, with an unaligned load.
        let hay = unsafe { T::load_unchecked(haystack, pos) };
        state.update(hay);
        if state.needs_verify() {
            let pos = pos - state.offset();
            if let Some(m) = self.verify(haystack, pos, state.result()) {
                return Some(m);
            }
        }

        // Increment pos by up to BLOCK_SIZE, but only as far as the next alignment boundary.
        let pos_align = (haystack.as_ptr() as usize + pos) % T::BLOCK_SIZE;
        pos = pos + T::BLOCK_SIZE - pos_align;

        // Since we shifted by an amount not necessarily equal to BLOCK_SIZE, the state preserved
        // in `state` cannot necessarily be used for the next iteration. Here, we reset the state.
        // This allows some false positives in the fingerprint matching step, but only on the first
        // iteration through the inner loop that follows.
        state.reset();

        // The main loop (in which the loads are all aligned). The control flow here is a bit
        // funky. Logically, we want:
        //    while pos < end {
        //        // do something
        //        if cond {
        //            // do something else
        //        }
        //    }
        // Instead, we write:
        //  'outer: loop {
        //      loop {
        //         if pos >= end { break 'outer; }
        //         // do something
        //         if cond { break; }
        //      }
        //      // do something else
        //  }
        // This weird double-loop version is faster when `cond` is usually false (and if it isn't
        // usually false then you shouldn't be using Teddy anyway). Also, we can unroll the inner
        // loop for another little boost.
        let end = len.saturating_sub(2 * T::BLOCK_SIZE - 1);
        'outer: loop {
            'inner: loop {
                if pos >= end { break 'outer; }

                let hay = unsafe { *(haystack.get_unchecked(pos) as *const u8 as *const T) };
                state.update(hay);
                if state.needs_verify() {
                    break 'inner;
                }
                pos += T::BLOCK_SIZE;

                let hay = unsafe { *(haystack.get_unchecked(pos) as *const u8 as *const T) };
                state.update(hay);
                if state.needs_verify() {
                    break 'inner;
                }
                pos += T::BLOCK_SIZE;
            }

            // If we got here, it means that a fingerprint matched and we need to verify it.
            let start_pos = pos - state.offset();
            if let Some(m) = self.verify(haystack, start_pos, state.result()) {
                return Some(m);
            }
            pos += T::BLOCK_SIZE;
        }

        // Do a slow search through the last part of the haystack, which was not big enough to do
        // SIMD on.
        self.slow(haystack, pos - state.offset())
    }

    /// Searches `haystack` for the substrings in this `Teddy`. If a match was
    /// found, then it is returned. Otherwise, `None` is returned.
    pub fn find(&self, haystack: &[u8]) -> Option<Match> {
        // If our haystack is too small, fall back to Aho-Corasick.
        if haystack.is_empty() || haystack.len() < 2 * T::BLOCK_SIZE {
            return self.slow(haystack, 0);
        }

        match self.masks.len() {
            1 => { self.find_loop(haystack, State1::new(&self.masks)) },
            2 => { self.find_loop(haystack, State2::new(&self.masks)) },
            3 => { self.find_loop(haystack, State3::new(&self.masks)) },
            _ => unreachable!(),
        }
    }

    /// Runs the verification procedure on `res` (i.e., `C` from the module
    /// documentation), where the haystack block starts at `pos` in
    /// `haystack`.
    ///
    /// If a match exists, it returns the first one.
    #[inline(always)]
    fn verify(&self, haystack: &[u8], pos: usize, res: T) -> Option<Match> {
        let mut bitfield = res.ne(T::splat(0)).move_mask();

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
        // Aho-Corasick finds matches in order of the end position, but we want them in order of
        // the start position. Going through all matches and finding the one that starts earliest
        // is not the most efficient way to solve this, but it isn't too wasteful since we only do
        // it to small haystacks.
        self.ac.find_overlapping(&haystack[pos..])
            .min_by_key(|m| m.start)
            .map(|m| Match {
                pat: m.pati,
                start: m.start + pos,
                end: m.end + pos,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::Teddy;
    use x86::teddy_simd::TeddySIMD;
    use simd::u8x16;
    use std::iter::repeat;
    use quickcheck::TestResult;

    #[cfg(target_feature="avx2")]
    use simd::x86::avx::u8x32;

    fn one_pattern_inner<T: TeddySIMD>(needle: &str) {
        let len = T::BLOCK_SIZE * 4;
        let ted: Teddy<T> = Teddy::new(vec![needle.as_bytes()]).unwrap();

        // Allocate a string just once. This ensures that its allocation has the same alignment
        // throughout these tests.
        let mut hay = Vec::with_capacity(T::BLOCK_SIZE * 4);
        for needle_pos in 0..(len - needle.len() + 1) {
            hay.clear();

            // Embed the needle at offset `needle_pos` in a string of x's.
            hay.extend(repeat('x' as u8).take(needle_pos));
            hay.extend(needle.as_bytes().iter().cloned());
            let len_left = len - hay.len();
            hay.extend(repeat('x' as u8).take(len_left));

            // Try starting from different offsets in the string. Since the fingerprint matching
            // depends on memory alignment, this tests out different code paths.
            for offset in 0..T::BLOCK_SIZE {
                assert_eq!(ted.find(&hay[offset..]), ted.slow(&hay[offset..], 0));
            }
        }
    }

    #[test]
    fn one_pattern_128() {
        one_pattern_inner::<u8x16>("abc");
        one_pattern_inner::<u8x16>("ab");
        one_pattern_inner::<u8x16>("a");
    }

    #[cfg(target_feature="avx2")]
    #[test]
    fn one_pattern_256() {
        one_pattern_inner::<u8x32>("abc");
        one_pattern_inner::<u8x32>("ab");
        one_pattern_inner::<u8x32>("a");
    }

    fn fast_equal_slow_inner<T: TeddySIMD>(
        pats: Vec<Vec<u8>>,
        haystack_prefix: Vec<u8>,
        haystack_suffix: Vec<u8>)
    -> TestResult {
        if pats.is_empty() {
            return TestResult::discard();
        }

        // Hide one of the patterns in the haystack, so there is something to find.
        let mut haystack = haystack_prefix;
        haystack.extend_from_slice(&pats[0]);
        haystack.extend_from_slice(&haystack_suffix);

        if let Some(ted) = Teddy::<T>::new(pats.iter().map(|x| &x[..])) {
            let fast = ted.find(&haystack).unwrap();
            let slow = ted.slow(&haystack, 0).unwrap();

            TestResult::from_bool(fast.start == slow.start)
        } else {
            TestResult::discard()
        }
    }

    quickcheck! {
        fn fast_equal_slow_128(pats: Vec<Vec<u8>>, hay_pref: Vec<u8>, hay_suf: Vec<u8>) -> TestResult {
            fast_equal_slow_inner::<u8x16>(pats, hay_pref, hay_suf)
        }
    }

    #[cfg(target_feature="avx2")]
    quickcheck! {
        fn fast_equal_slow_256(pats: Vec<Vec<u8>>, hay_pref: Vec<u8>, hay_suf: Vec<u8>) -> TestResult {
            fast_equal_slow_inner::<u8x32>(pats, hay_pref, hay_suf)
        }
    }
}

