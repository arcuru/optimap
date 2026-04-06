#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

mod raw;
mod map;
mod set;
pub mod split_overflow;
pub mod in_place_overflow;
pub mod gaps;

pub use map::UnorderedFlatMap;
pub use set::UnorderedFlatSet;
pub use raw::hash::IsAvalanching;
pub use split_overflow::Splitsies;
pub use in_place_overflow::InPlaceOverflow;
pub use gaps::Gaps;
