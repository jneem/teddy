// This file was generated by text_gen.py.
#![feature(cfg_target_feature, test)]

extern crate simd;
extern crate test;
extern crate teddy;

use simd::u8x16;
use test::Bencher;
use teddy::Teddy;

#[cfg(target_feature="avx2")]
use simd::x86::avx::u8x32;

macro_rules! bench {
    ($name:ident, $simd_type: ident, $pats:expr, $haystack:expr, $count:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let pats: Vec<Vec<u8>> = $pats.into_iter().map(|s| s.as_bytes().to_vec()).collect();
            let ted = Teddy::<$simd_type>::new(&pats).unwrap();
            b.bytes = $haystack.len() as u64;

            b.iter(|| {
                let mut hay = $haystack.as_bytes();
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

static RARITY_8_1: [&'static str; 8] = ["k", "H", "F", "'", "+", "K", "\\", "4"];

static RARITY_8_1_8196: &'static str = include_str!("data/rarity_8_1_8196.txt");
bench!(rarity_8_1_8196_65536_sse, u8x16, RARITY_8_1, RARITY_8_1_8196[0..65536], 8);
#[cfg(target_feature="avx2")]
bench!(rarity_8_1_8196_65536_avx2, u8x32, RARITY_8_1, RARITY_8_1_8196[0..65536], 8);

static RARITY_8_1_1024: &'static str = include_str!("data/rarity_8_1_1024.txt");
bench!(rarity_8_1_1024_65536_sse, u8x16, RARITY_8_1, RARITY_8_1_1024[0..65536], 60);
#[cfg(target_feature="avx2")]
bench!(rarity_8_1_1024_65536_avx2, u8x32, RARITY_8_1, RARITY_8_1_1024[0..65536], 60);

static RARITY_8_1_128: &'static str = include_str!("data/rarity_8_1_128.txt");
bench!(rarity_8_1_128_65536_sse, u8x16, RARITY_8_1, RARITY_8_1_128[0..65536], 557);
#[cfg(target_feature="avx2")]
bench!(rarity_8_1_128_65536_avx2, u8x32, RARITY_8_1, RARITY_8_1_128[0..65536], 557);

static RARITY_8_1_16: &'static str = include_str!("data/rarity_8_1_16.txt");
bench!(rarity_8_1_16_65536_sse, u8x16, RARITY_8_1, RARITY_8_1_16[0..65536], 4508);
#[cfg(target_feature="avx2")]
bench!(rarity_8_1_16_65536_avx2, u8x32, RARITY_8_1, RARITY_8_1_16[0..65536], 4508);
static RARITY_8_2: [&'static str; 8] = ["Sz", "aW", ",L", "CI", "]^", " h", "'R", "v>"];

static RARITY_8_2_8196: &'static str = include_str!("data/rarity_8_2_8196.txt");
bench!(rarity_8_2_8196_65536_sse, u8x16, RARITY_8_2, RARITY_8_2_8196[0..65536], 18);
#[cfg(target_feature="avx2")]
bench!(rarity_8_2_8196_65536_avx2, u8x32, RARITY_8_2, RARITY_8_2_8196[0..65536], 18);

static RARITY_8_2_1024: &'static str = include_str!("data/rarity_8_2_1024.txt");
bench!(rarity_8_2_1024_65536_sse, u8x16, RARITY_8_2, RARITY_8_2_1024[0..65536], 58);
#[cfg(target_feature="avx2")]
bench!(rarity_8_2_1024_65536_avx2, u8x32, RARITY_8_2, RARITY_8_2_1024[0..65536], 58);

static RARITY_8_2_128: &'static str = include_str!("data/rarity_8_2_128.txt");
bench!(rarity_8_2_128_65536_sse, u8x16, RARITY_8_2, RARITY_8_2_128[0..65536], 518);
#[cfg(target_feature="avx2")]
bench!(rarity_8_2_128_65536_avx2, u8x32, RARITY_8_2, RARITY_8_2_128[0..65536], 518);

static RARITY_8_2_16: &'static str = include_str!("data/rarity_8_2_16.txt");
bench!(rarity_8_2_16_65536_sse, u8x16, RARITY_8_2, RARITY_8_2_16[0..65536], 3893);
#[cfg(target_feature="avx2")]
bench!(rarity_8_2_16_65536_avx2, u8x32, RARITY_8_2, RARITY_8_2_16[0..65536], 3893);
static RARITY_8_3: [&'static str; 8] = ["bfU", "_PP", "XQF", "DU$", " ?^", "IaP", "9PA", "pf|"];

static RARITY_8_3_8196: &'static str = include_str!("data/rarity_8_3_8196.txt");
bench!(rarity_8_3_8196_65536_sse, u8x16, RARITY_8_3, RARITY_8_3_8196[0..65536], 10);
#[cfg(target_feature="avx2")]
bench!(rarity_8_3_8196_65536_avx2, u8x32, RARITY_8_3, RARITY_8_3_8196[0..65536], 10);

static RARITY_8_3_1024: &'static str = include_str!("data/rarity_8_3_1024.txt");
bench!(rarity_8_3_1024_65536_sse, u8x16, RARITY_8_3, RARITY_8_3_1024[0..65536], 64);
#[cfg(target_feature="avx2")]
bench!(rarity_8_3_1024_65536_avx2, u8x32, RARITY_8_3, RARITY_8_3_1024[0..65536], 64);

static RARITY_8_3_128: &'static str = include_str!("data/rarity_8_3_128.txt");
bench!(rarity_8_3_128_65536_sse, u8x16, RARITY_8_3, RARITY_8_3_128[0..65536], 480);
#[cfg(target_feature="avx2")]
bench!(rarity_8_3_128_65536_avx2, u8x32, RARITY_8_3, RARITY_8_3_128[0..65536], 480);

static RARITY_8_3_16: &'static str = include_str!("data/rarity_8_3_16.txt");
bench!(rarity_8_3_16_65536_sse, u8x16, RARITY_8_3, RARITY_8_3_16[0..65536], 3622);
#[cfg(target_feature="avx2")]
bench!(rarity_8_3_16_65536_avx2, u8x32, RARITY_8_3, RARITY_8_3_16[0..65536], 3622);
static NEEDLE_NUM_8_3: [&'static str; 8] = ["A,z", "6Ab", "4x{", "/f[", "Gyo", "w>T", "#i%", "jeX"];

static NEEDLE_NUM_8_3_1024: &'static str = include_str!("data/needle_num_8_3_1024.txt");
bench!(needle_num_8_3_1024_65536_sse, u8x16, NEEDLE_NUM_8_3, NEEDLE_NUM_8_3_1024[0..65536], 61);
#[cfg(target_feature="avx2")]
bench!(needle_num_8_3_1024_65536_avx2, u8x32, NEEDLE_NUM_8_3, NEEDLE_NUM_8_3_1024[0..65536], 61);
static NEEDLE_NUM_16_3: [&'static str; 16] = ["4mA", "N(P", "lZ]", "&@&", " ^\"", "4Xa", "d)X", "wN\"", "v\t>", "_|p", "WYQ", "nU7", "Z( ", "|&~", "IG7", "\nf-"];

static NEEDLE_NUM_16_3_1024: &'static str = include_str!("data/needle_num_16_3_1024.txt");
bench!(needle_num_16_3_1024_65536_sse, u8x16, NEEDLE_NUM_16_3, NEEDLE_NUM_16_3_1024[0..65536], 65);
#[cfg(target_feature="avx2")]
bench!(needle_num_16_3_1024_65536_avx2, u8x32, NEEDLE_NUM_16_3, NEEDLE_NUM_16_3_1024[0..65536], 65);
static NEEDLE_NUM_32_3: [&'static str; 32] = ["0]\n", "`4d", "G18", "f<-", " gZ", "yH7", "'R5", "1F{", "puj", "g5m", "Wfs", "M=\"", "xcM", "<8i", "!H`", "wc2", "Hc]", "z0-", "m<1", "Ml,", "i6J", "^y6", "5IZ", "nxj", "W\nB", "&,5", "&d#", "p$N", "a[e", "h%l", "H=_", "*v1"];

static NEEDLE_NUM_32_3_1024: &'static str = include_str!("data/needle_num_32_3_1024.txt");
bench!(needle_num_32_3_1024_65536_sse, u8x16, NEEDLE_NUM_32_3, NEEDLE_NUM_32_3_1024[0..65536], 65);
#[cfg(target_feature="avx2")]
bench!(needle_num_32_3_1024_65536_avx2, u8x32, NEEDLE_NUM_32_3, NEEDLE_NUM_32_3_1024[0..65536], 65);
static TEXT_LEN_8_1: [&'static str; 8] = ["Z", "o", "Q", "Y", "}", "\"", "A", "m"];

static TEXT_LEN_8_1_65536: &'static str = include_str!("data/text_len_8_1_65536.txt");
bench!(text_len_8_1_65536_16_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..16], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_16_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..16], 0);
bench!(text_len_8_1_65536_32_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..32], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_32_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..32], 0);
bench!(text_len_8_1_65536_64_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..64], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_64_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..64], 0);
bench!(text_len_8_1_65536_128_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..128], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_128_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..128], 0);
bench!(text_len_8_1_65536_256_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..256], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_256_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..256], 0);
bench!(text_len_8_1_65536_512_sse, u8x16, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..512], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_1_65536_512_avx2, u8x32, TEXT_LEN_8_1, TEXT_LEN_8_1_65536[0..512], 0);
static TEXT_LEN_8_2: [&'static str; 8] = ["CH", "Ne", ",2", "ta", "Bd", ")7", "6h", "a\n"];

static TEXT_LEN_8_2_65536: &'static str = include_str!("data/text_len_8_2_65536.txt");
bench!(text_len_8_2_65536_16_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..16], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_16_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..16], 0);
bench!(text_len_8_2_65536_32_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..32], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_32_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..32], 0);
bench!(text_len_8_2_65536_64_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..64], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_64_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..64], 0);
bench!(text_len_8_2_65536_128_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..128], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_128_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..128], 0);
bench!(text_len_8_2_65536_256_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..256], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_256_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..256], 0);
bench!(text_len_8_2_65536_512_sse, u8x16, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..512], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_2_65536_512_avx2, u8x32, TEXT_LEN_8_2, TEXT_LEN_8_2_65536[0..512], 0);
static TEXT_LEN_8_3: [&'static str; 8] = ["\"W8", ":L6", "L1m", "tx%", "R>W", "RHL", "FH(", "Dfv"];

static TEXT_LEN_8_3_65536: &'static str = include_str!("data/text_len_8_3_65536.txt");
bench!(text_len_8_3_65536_16_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..16], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_16_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..16], 0);
bench!(text_len_8_3_65536_32_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..32], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_32_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..32], 0);
bench!(text_len_8_3_65536_64_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..64], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_64_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..64], 0);
bench!(text_len_8_3_65536_128_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..128], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_128_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..128], 0);
bench!(text_len_8_3_65536_256_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..256], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_256_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..256], 0);
bench!(text_len_8_3_65536_512_sse, u8x16, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..512], 0);
#[cfg(target_feature="avx2")]
bench!(text_len_8_3_65536_512_avx2, u8x32, TEXT_LEN_8_3, TEXT_LEN_8_3_65536[0..512], 0);
