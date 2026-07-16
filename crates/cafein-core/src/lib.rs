//! Transit data model and routing algorithms for cafein.
//!
//! This crate holds the timetable data structures and the routing
//! implementations. It compiles and tests without Python.

pub mod ch;
pub mod exhaustive;
pub mod fares;
pub mod geometry;
pub mod journey;
pub mod mcraptor;
pub mod mctbtr;
pub mod mcultra;
mod path_key;
pub mod router;
pub mod routers;
pub use routers::raptor;
pub mod streets;
pub mod tbtr;
pub mod timetable;
pub mod transfers;
pub mod ultra;
