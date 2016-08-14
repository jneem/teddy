#![feature(test)]

extern crate test;
extern crate teddy;

use test::Bencher;
use teddy::Teddy;

static HAYSTACK: &'static str = include_str!("sherlock.txt");

macro_rules! sherlock {
    ($name:ident, $pats:expr, $count:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let pats: Vec<Vec<u8>> = $pats.into_iter().map(|s| s.as_bytes().to_vec()).collect();
            let ted = Teddy::new(&pats).unwrap();
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

sherlock!(names, &["Sherlock", "Holmes"], 558);

// This one doesn't have any matches, but the fingerprints should match a log.
sherlock!(names_lower, &["sherlock", "holmes"], 0);

sherlock!(words_long, &["pull", "cabby", "three", "side"], 348);
sherlock!(words_short, &["pu", "ca", "th", "si"], 15202);

// The fingerprints here shouldn't match anything.
sherlock!(rare, &["xyzxyz"], 0);
