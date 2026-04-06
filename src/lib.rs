#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

mod raw;
mod map;
mod set;
mod traits;
pub mod split_overflow;
pub mod in_place_overflow;
pub mod ipo64;
pub mod gaps;

pub use map::UnorderedFlatMap;
pub use set::UnorderedFlatSet;
pub use raw::hash::IsAvalanching;
pub use traits::Map;
pub use split_overflow::Splitsies;
pub use in_place_overflow::InPlaceOverflow;
pub use ipo64::IPO64;
pub use gaps::Gaps;
