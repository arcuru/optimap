#![feature(portable_simd)]
#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

mod raw;
mod map;
mod set;

pub use map::UnorderedFlatMap;
pub use set::UnorderedFlatSet;
