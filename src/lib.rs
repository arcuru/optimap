#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

mod raw;
mod map;
mod set;
pub mod split_overflow;

pub use map::UnorderedFlatMap;
pub use set::UnorderedFlatSet;
pub use raw::hash::IsAvalanching;
