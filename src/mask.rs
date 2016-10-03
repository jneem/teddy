use std::cmp;
use std::collections::BTreeMap;
use std::fmt::{Debug, Error as FmtError, Formatter};
use std::usize;
use teddy_simd::TeddySIMD;

/// A list of masks. This has length equal to the length of the fingerprint.
/// The length of the fingerprint is always `max(3, len(smallest_substring))`.
#[derive(Debug, Clone)]
pub struct Masks<T: TeddySIMD>(Vec<Mask<T>>);

/// A single mask.
#[derive(Debug, Clone, Copy)]
struct Mask<T: TeddySIMD> {
    /// Bitsets for the low nybbles in a fingerprint.
    lo: T,
    /// Bitsets for the high nybbles in a fingerprint.
    hi: T,
}

/// A bitset representing a set of nybbles.
#[derive(Clone, Copy)]
struct NybbleSet(u16);

impl NybbleSet {
    /// The number of nybbles in this set.
    fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// The union between this set and `other`.
    fn union(self, other: NybbleSet) -> NybbleSet {
        NybbleSet(self.0 | other.0)
    }

    /// The intersection between `self` and `other`.
    fn intersection(self, other: NybbleSet) -> NybbleSet {
        NybbleSet(self.0 & other.0)
    }
}

impl Debug for NybbleSet {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{:016b}", self.0)
    }
}

/// A set of bytes.
///
/// This is not just any set of bytes however; it must be a product set: a set of the form `{b:
/// hi(b) in S and lo(b) in T}`, where `S` and `T` are sets of nybbles, and `hi(b)` and `lo(b)` are
/// the high and low nybbles of `b`. The reason for considering byte sets of this form is that
/// these are exactly the sorts of bytesets that can be efficiently searched for using the `PSHUFB`
/// instruction.
#[derive(Clone, Copy)]
struct ByteSet {
    hi: NybbleSet,
    lo: NybbleSet,
}

impl ByteSet {
    /// Creates a new, empty, `ByteSet`.
    fn new() -> ByteSet {
        ByteSet {
            hi: NybbleSet(0),
            lo: NybbleSet(0),
        }
    }

    /// Adds a single byte to this `ByteSet`.
    fn add_byte(&mut self, b: u8) {
        let hi = b >> 4;
        let lo = b & 0x0F;

        self.hi.0 |= 1 << hi;
        self.lo.0 |= 1 << lo;
    }

    /// The number of bytes in this set.
    fn len(self) -> usize {
        self.hi.len() * self.lo.len()
    }

    /// The smallest `ByteSet` that contains both `self` and `other`. Because `ByteSet` can only
    /// represent product sets, this may be larger than the union of `self` and `other`.
    fn cover(self, other: ByteSet) -> ByteSet {
        ByteSet {
            hi: self.hi.union(other.hi),
            lo: self.lo.union(other.lo),
        }
    }

    /// The intersection between `self` and `other`.
    fn intersection(self, other: ByteSet) -> ByteSet {
        ByteSet {
            hi: self.hi.intersection(other.hi),
            lo: self.lo.intersection(other.lo),
        }
    }
}

impl Debug for ByteSet {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{:?}/{:?}", self.hi, self.lo)
    }
}

/// A fingerprint is a sequence of 1 to 3 `ByteSets`. A string of bytes matches a fingerprint if
/// its first byte is contained in the first `ByteSet`, its second byte is contained in the second
/// `ByteSet`, and so on.
#[derive(Debug, Clone)]
struct Fingerprint(Vec<ByteSet>);

impl Fingerprint {
    /// Creates an empty fingerprint of length `n`, which must be between 1 and 3.
    fn new(n: usize) -> Fingerprint {
        debug_assert!(1 <= n && n <= 3);
        Fingerprint(vec![ByteSet::new(); n])
    }

    /// Adds a string to this fingerprint. The length of the string must be at least the length of
    /// this fingerprint.
    fn add_string(&mut self, pat: &[u8]) {
        debug_assert!(pat.len() >= self.0.len());
        for (i, b) in self.0.iter_mut().enumerate() {
            b.add_byte(pat[i]);
        }
    }

    /// The number of distinct strings of length `self.0.len()` that match this fingerprint.
    fn len(&self) -> usize {
        self.0.iter().map(|set| set.len()).product()
    }

    /// The number of sequences that belong to both `self` and `other`.
    ///
    /// If we had implemented `Fingerprint::intersection`, this would be equivalent to (but faster
    /// than) `self.intersection(other).len()`.
    fn intersection_size(&self, other: &Fingerprint) -> usize {
        debug_assert!(self.0.len() == other.0.len());
        self.0.iter()
            .zip(other.0.iter())
            .map(|(bs1, bs2)| bs1.intersection(*bs2).len())
            .product()
    }

    /// The number of sequences that belong to the smallest `Fingerprint` containing both
    /// `self` and `other`.
    ///
    /// This is equivalent to `self.include(other); self.len()`, but it doesn't modify `self`.
    fn cover_size(&self, other: &Fingerprint) -> usize {
        debug_assert!(self.0.len() == other.0.len());
        self.0.iter()
            .zip(other.0.iter())
            .map(|(bs1, bs2)| bs1.cover(*bs2).len())
            .product()
    }

    /// Modifies this `Fingerprint` in place so that it contains `other`. The result will be the
    /// smallest fingerprint that contains both `other` and (the old value of) `self`.
    fn include(&mut self, other: &Fingerprint) {
        debug_assert!(self.0.len() == other.0.len());
        for (bs1, bs2) in self.0.iter_mut().zip(other.0.iter()) {
            *bs1 = bs1.cover(*bs2);
        }
    }
}

/// A `Bucket` is a collection of strings along with a fingerprint that matches all of them.
#[derive(Debug, Clone)]
struct Bucket {
    /// The collection of strings in this bucket. We store them as indices into some external
    /// collection of strings.
    pats: Vec<usize>,
    /// The fingerprint matching all of the strings.
    fing: Fingerprint,
}

impl Bucket {
    fn new(fing_len: usize) -> Bucket {
        Bucket {
            pats: Vec::new(),
            fing: Fingerprint::new(fing_len),
        }

    }

    /// Adds a single string to this bucket.
    ///
    /// `pati` is the index of the string (in the external collection containing all of the
    /// strings).
    /// `bytes` is the string.
    fn add_string(&mut self, pati: usize, bytes: &[u8]) {
        self.pats.push(pati);
        self.fing.add_string(bytes);
    }

    /// Calculate a penalty based on how much we don't want to merge these two buckets. The most
    /// important field in the penalty is how many un-needed strings are contained in the new
    /// bucket. Given a tie on that score, we break it by preferring buckets that match a smaller
    /// total number of strings.
    fn merge_penalty(&self, other: &Bucket) -> (usize, usize) {
        let old_size = self.fing.len() + other.fing.len() - self.fing.intersection_size(&other.fing);
        let new_size = self.fing.cover_size(&other.fing);
        (new_size.checked_sub(old_size).unwrap(), new_size)
    }

    /// Adds all strings in `other` to this bucket.
    fn merge(&mut self, other: Bucket) {
        self.pats.extend(&other.pats);
        self.fing.include(&other.fing);
    }
}

/// Compares all pairs of buckets and merges the pair that results in the smallest increase in
/// fingerprint size. If there is a tie, break it by merging the pair that will result in the
/// smallest total fingerprint size.
///
/// Note that this function is O(N^2), where N is the number of buckets. Since this gets called
/// O(N) times, the total time is O(N^3). Fortunately, N is usually not very large. Nevertheless,
/// it might be nice to make it faster...
fn merge_one_bucket(buckets: &mut Vec<Bucket>) {
    let mut best_pair = (usize::MAX, usize::MAX);
    let mut best_penalty = (usize::MAX, usize::MAX);

    for (i, b1) in buckets.iter().enumerate() {
        for (j, b2) in buckets[(i+1)..].iter().enumerate() {
            let penalty = b1.merge_penalty(b2);
            if penalty <= best_penalty {
                best_penalty = penalty;
                best_pair = (i, i + 1 + j);
            }
        }
    }

    let (i, j) = best_pair;
    debug_assert!(i < j && j < buckets.len());
    let b2 = buckets.swap_remove(j);
    buckets[i].merge(b2);
}

/// Given a set of strings and a fingerprint length (between 1 and 3), divide the strings into up
/// to 8 buckets in such a way that the fingerprints for the buckets are as small as possible.
fn gather_buckets(pats: &[Vec<u8>], fing_len: usize) -> Vec<Vec<usize>> {
    // Start by putting all the patterns with the exact same fingerprint into a single bucket.
    let mut buckets = BTreeMap::new();
    for (pati, pat) in pats.iter().enumerate() {
        buckets.entry(&pat[0..fing_len])
            .or_insert_with(|| Bucket::new(fing_len))
            .add_string(pati, pat);
    }

    // Now continue merging the buckets as best we can until there are at most 8.
    let mut buckets: Vec<Bucket> = buckets.into_iter().map(|(_, bucket)| bucket).collect();
    while buckets.len() > 8 {
        merge_one_bucket(&mut buckets);
    }

    buckets.into_iter()
        .map(|bucket| bucket.pats)
        .collect()
}

impl<T: TeddySIMD> Masks<T> {
    /// Given a set of strings, returns a set of buckets and a corresponding set of masks.
    ///
    /// # Panics
    /// if the set of strings is empty, or if it contains any empty strings.
    pub fn buckets_and_masks(pats: &[Vec<u8>]) -> (Vec<Vec<usize>>, Masks<T>) {
        let min_len = pats.iter().map(|p| p.len()).min().unwrap_or(0);
        // Don't allow any empty patterns and require that we have at
        // least one pattern.
        debug_assert!(min_len >= 1);
        // Pick the largest mask possible, but no larger than 3.
        let nmasks = cmp::min(3, min_len);
        let buckets = gather_buckets(pats, nmasks);
        debug_assert!(buckets.len() <= cmp::min(8, pats.len()));

        let mut masks = Masks(vec![Mask::new(); nmasks]);
        for (bucki, bucket) in buckets.iter().enumerate() {
            for &pati in bucket {
                // The cast is ok because bucki is at most 7.
                masks.add(bucki as u8, &pats[pati]);
            }
        }

        (buckets, masks)
    }

    /// Returns the number of masks.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Adds the given pattern to the given bucket. The bucket should be a
    /// power of `2 <= 2^7`.
    pub fn add(&mut self, bucket: u8, pat: &[u8]) {
        for (i, mask) in self.0.iter_mut().enumerate() {
            mask.add(bucket, pat[i]);
        }
    }

    /// Finds the fingerprints that are in the given haystack block. i.e., this
    /// returns `C` as described in the module documentation.
    ///
    /// More specifically, `for i in 0..BLOCK_SIZE` and `j in 0..8, C[i][j] == 1` if and
    /// only if `haystack_block[i]` corresponds to a fingerprint that is part
    /// of a pattern in bucket `j`.
    #[inline(always)]
    pub fn members1(&self, haystack_block: T) -> T {
        let masklo = T::splat(0xF);
        let hlo = haystack_block & masklo;
        let hhi = (haystack_block >> 4) & masklo;

        self.0[0].lo.shuffle_bytes(hlo) & self.0[0].hi.shuffle_bytes(hhi)
    }

    /// Like members1, but computes C for the first and second bytes in the
    /// fingerprint.
    #[inline(always)]
    pub fn members2(&self, haystack_block: T) -> (T, T) {
        let masklo = T::splat(0xF);
        let hlo = haystack_block & masklo;
        let hhi = (haystack_block >> 4) & masklo;

        let res0 = self.0[0].lo.shuffle_bytes(hlo)
                   & self.0[0].hi.shuffle_bytes(hhi);
        let res1 = self.0[1].lo.shuffle_bytes(hlo)
                   & self.0[1].hi.shuffle_bytes(hhi);
        (res0, res1)
    }

    /// Like `members1`, but computes `C` for the first, second and third bytes
    /// in the fingerprint.
    #[inline(always)]
    pub fn members3(&self, haystack_block: T) -> (T, T, T) {
        let masklo = T::splat(0xF);
        let hlo = haystack_block & masklo;
        let hhi = (haystack_block >> 4) & masklo;

        let res0 = self.0[0].lo.shuffle_bytes(hlo)
                   & self.0[0].hi.shuffle_bytes(hhi);
        let res1 = self.0[1].lo.shuffle_bytes(hlo)
                   & self.0[1].hi.shuffle_bytes(hhi);
        let res2 = self.0[2].lo.shuffle_bytes(hlo)
                   & self.0[2].hi.shuffle_bytes(hhi);
        (res0, res1, res2)
    }
}

impl<T: TeddySIMD> Mask<T> {
    /// Create a new mask with no members.
    fn new() -> Mask<T> {
        Mask {
            lo: T::splat(0),
            hi: T::splat(0),
        }
    }

    /// Adds the given byte to the given bucket.
    fn add(&mut self, bucket: u8, byte: u8) {
        // Split our byte into two nybbles, and add each nybble to our
        // mask.
        let byte_lo = (byte & 0xF) as u32;
        let byte_hi = (byte >> 4) as u32;

        // Our mask is repeated across 16 byte lanes. (TODO: explain why)
        let lo = self.lo.extract(byte_lo) | ((1 << bucket) as u8);
        let hi = self.hi.extract(byte_hi) | ((1 << bucket) as u8);

        for lane in 0..(T::BLOCK_SIZE as u32 / 16) {
            self.lo = self.lo.replace(byte_lo + 16 * lane, lo);
            self.hi = self.hi.replace(byte_hi + 16 * lane, hi);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::gather_buckets;
    use std::collections::BTreeSet;

    macro_rules! merge_number {
        ($name:ident, $strs:expr, $fing_len:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let pats: Vec<Vec<u8>> = $strs.iter().map(|s| s.as_bytes().to_vec()).collect();
                let buckets = gather_buckets(&pats, $fing_len);
                assert_eq!(buckets.len(), $expected);
            }
        }
    }

    macro_rules! merge {
        ($name:ident, $strs:expr, $fing_len:expr, $merged:expr) => {
            #[test]
            fn $name() {
                let pats: Vec<Vec<u8>> = $strs.iter().map(|s| s.as_bytes().to_vec()).collect();
                let buckets = gather_buckets(&pats, $fing_len);
                let merged: BTreeSet<BTreeSet<usize>> =
                    $merged.into_iter().map(|set| set.iter().cloned().collect()).collect();
                for b in &buckets {
                    if b.len() > 1 {
                        println!("bucket {:?}", b);
                        assert!(merged.contains(&b.iter().cloned().collect()));
                    }
                }
            }
        }
    }

    // Test the bit where we merge together strings with identical prefixes into the same bucket.
    merge_number!(merge_equal_fings_1a, ["abc", "ae", "az"], 1, 1);
    merge_number!(merge_equal_fings_1b, ["ba", "abc", "ae", "az", "ba", "bc"], 1, 2);
    merge_number!(merge_equal_fings_2a, ["ba", "abc", "ab", "az", "ba", "bc"], 2, 4);
    merge_number!(merge_equal_fings_2b, ["abc", "abdef", "abcde"], 2, 1);
    merge_number!(merge_equal_fings_3a, ["abc", "abdef", "abcde"], 3, 2);
    merge_number!(merge_equal_fings_3b, ["abc", "abczyx", "abcde"], 3, 1);

    merge!(merge_lossless_a, ["she", "She", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[0, 1]]);
    merge!(merge_lossless_b, ["she", "the", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[0, 1]]);

    // We prefer to merge 4 fingerprints into one rather than add any false positives.
    merge!(merge_lossless_c,
           ["she", "She", "sHe", "SHe", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"],
           3,
           [&[0, 1, 2, 3]]);

    // Given several choices, we will prefer the one that keeps the max bucket size small.  (The
    // fact that we merge [0, 2] and [1, 3] instead of [0, 1] and [2, 3] is not particularly
    // intentional.)
    merge!(merge_lossless_d,
           ["she", "She", "sHe", "SHe", "bla", "Bla", "ghi", "jkl", "mno", "pqr", "stu"],
           3,
           [&[0, 2], &[1, 3], &[4, 5]]);

    // Merging "she" and "tho" introduces "the" and "sho" as false positives, but that's better
    // than the alternatives.
    merge!(merge_lossy_a, ["she", "tho", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[0, 1]]);
    merge!(merge_lossy_b, ["she", "SHe", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[0, 1]]);
    // Merging "she" and "The" introduces "the" and "She" as false positives.
    merge!(merge_lossy_c, ["she", "The", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[0, 1]]);
    merge!(merge_lossy_d, ["she", "THe", "abc", "ABc", "ghi", "jkl", "mno", "pqr", "stu"], 3, [&[2, 3]]);
}

