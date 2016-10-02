#![feature(cfg_target_feature, test)]

extern crate regex_syntax;
extern crate simd;
extern crate test;
extern crate teddy;

use regex_syntax::{CharClass, ClassRange};
use simd::u8x16;
use std::char;
use std::mem;
use test::Bencher;
use teddy::Teddy;

#[cfg(target_feature="avx2")]
use simd::x86::avx::u8x32;

static HAYSTACK: &'static str = include_str!("data/sherlock.txt");

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

fn casei(s: &str) -> Vec<String> {
    fn casei_char(c: char) -> Vec<char> {
        let c_class = CharClass::new(vec![ClassRange { start: c, end: c }]);
        c_class.case_fold()
            .into_iter()
            .flat_map(|range| ((range.start as u32)..(range.end as u32 + 1)).into_iter())
            .filter_map(char::from_u32)
            .collect()
    }

    let mut ret = vec![String::new()];
    let mut next = Vec::new();

    for c in s.chars() {
        for c_case in casei_char(c)  {
            for partial_s in &ret {
                let mut new_s = partial_s.clone();
                new_s.push(c_case);
                next.push(new_s);
            }
        }
        mem::swap(&mut ret, &mut next);
        next.clear();
    }
    ret
}

fn casei_multi(s: &[&str]) -> Vec<String> {
    s.iter()
        .flat_map(|s| casei(s).into_iter())
        .collect()
}

sherlock!(sherlock_names_128, u8x16, &["Sherlock", "Holmes"], 558);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_names_256, u8x32, &["Sherlock", "Holmes"], 558);

sherlock!(sherlock_names_casei_128, u8x16, casei_multi(&["Sherlock", "Holmes"]), 569);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_names_casei_256, u8x32, casei_multi(&["Sherlock", "Holmes"]), 569);

sherlock!(sherlock_names_casei_short_128, u8x16, casei_multi(&["She", "Hol"]), 1307);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_names_casei_short_256, u8x32, casei_multi(&["She", "Hol"]), 1307);

sherlock!(sherlock_names_more_casei_short_128, u8x16, casei_multi(&["She", "Hol", "Joh", "Wat", "Ire", "Adl"]), 1720);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_names_more_casei_short_256, u8x32, casei_multi(&["She", "Hol", "Joh", "Wat", "Ire", "Adl"]), 1720);

// This one doesn't have any matches, but the fingerprints should match a lot.
sherlock!(sherlock_names_lower_128, u8x16, &["sherlock", "holmes"], 0);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_names_lower_256, u8x32, &["sherlock", "holmes"], 0);

sherlock!(sherlock_words_long_128, u8x16, &["pull", "cabby", "three", "side"], 348);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_words_long_256, u8x32, &["pull", "cabby", "three", "side"], 348);

sherlock!(sherlock_words_short_128, u8x16, &["pu", "ca", "th", "si"], 15202);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_words_short_256, u8x32, &["pu", "ca", "th", "si"], 15202);

sherlock!(sherlock_chars_128, u8x16, &["S", "H"], 2115);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_chars_256, u8x32, &["S", "H"], 2115);

sherlock!(sherlock_chars_rare_128, u8x16, &["Z", "X"], 12);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_chars_rare_256, u8x32, &["Z", "X"], 12);

// The fingerprints here shouldn't match anything.
sherlock!(sherlock_rare_128, u8x16, &["xyzxyz"], 0);
#[cfg(target_feature="avx2")]
sherlock!(sherlock_rare_256, u8x32, &["xyzxyz"], 0);
