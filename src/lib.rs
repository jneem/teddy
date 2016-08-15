#![feature(associated_consts, cfg_target_feature, platform_intrinsics)]

extern crate aho_corasick;
extern crate simd;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

mod teddy_simd;
mod teddy128;

pub use teddy128::Teddy;
