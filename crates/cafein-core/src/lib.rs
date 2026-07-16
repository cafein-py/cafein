//! Transit data model and routing algorithms for cafein.
//!
//! This crate holds the timetable data structures and the routing
//! implementations. It compiles and tests without Python.

pub mod ch;
pub mod exhaustive;
pub mod fares;
pub mod geometry;
pub mod journey;
pub mod mcultra;
mod path_key;
pub mod router;
pub mod routers;
pub use routers::mcraptor;
pub use routers::mctbtr;
pub use routers::raptor;
pub use routers::tbtr;
pub mod streets;
pub mod timetable;
pub mod transfers;
pub mod ultra;
