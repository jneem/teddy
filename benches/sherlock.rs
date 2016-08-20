#![feature(cfg_target_feature, test)]

extern crate simd;
extern crate test;
extern crate teddy;

use simd::u8x16;
use test::Bencher;
use teddy::Teddy;

#[cfg(target_feature="avx2")]
use simd::x86::avx::u8x32;

static HAYSTACK: &'static str = include_str!("sherlock.txt");

macro_rules! sherlock {
    ($name:ident, $simd_type: ident, $pats:expr, $count:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let pats: Vec<Vec<u8>> = $pats.into_iter().map(|s| s.as_bytes().to_vec()).collect();
            let ted = Teddy::<$simd_type>::new(&pats).unwrap();
            b.bytes = HAYSTACK.len() as u64;

            b.iter(|| {
                let mut hay = HAYSTACK.as_bytes();
                let mut count = 0;
                while let Some(mat) = ted.find(hay) {
                    count += 1;
                    hay = &hay[(mat.start + 1)..];
                }
                assert_eq!(count, $count);
            });
        }
    }
}

sherlock!(names_128, u8x16, &["Sherlock", "Holmes"], 558);
#[cfg(target_feature="avx2")]
sherlock!(names_256, u8x32, &["Sherlock", "Holmes"], 558);

// This one doesn't have any matches, but the fingerprints should match a log.
sherlock!(names_lower_128, u8x16, &["sherlock", "holmes"], 0);
#[cfg(target_feature="avx2")]
sherlock!(names_lower_256, u8x32, &["sherlock", "holmes"], 0);

sherlock!(words_long_128, u8x16, &["pull", "cabby", "three", "side"], 348);
#[cfg(target_feature="avx2")]
sherlock!(words_long_256, u8x32, &["pull", "cabby", "three", "side"], 348);

sherlock!(words_short_128, u8x16, &["pu", "ca", "th", "si"], 15202);
#[cfg(target_feature="avx2")]
sherlock!(words_short_256, u8x32, &["pu", "ca", "th", "si"], 15202);

sherlock!(chars_128, u8x16, &["S", "H"], 2115);
#[cfg(target_feature="avx2")]
sherlock!(chars_256, u8x32, &["S", "H"], 2115);

sherlock!(chars_rare_128, u8x16, &["Z", "X"], 12);
#[cfg(target_feature="avx2")]
sherlock!(chars_rare_256, u8x32, &["Z", "X"], 12);

// The fingerprints here shouldn't match anything.
sherlock!(rare_128, u8x16, &["xyzxyz"], 0);
#[cfg(target_feature="avx2")]
sherlock!(rare_256, u8x32, &["xyzxyz"], 0);
