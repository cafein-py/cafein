//! RAPTOR: earliest-arrival routing for a single departure time, and its
//! range (rRAPTOR) extension for a departure window.
//!
//! Round-based: round `k` finds the earliest arrivals reachable with
//! exactly `k` rides. Within a pattern, the earliest catchable trip at a
//! stop position is found by binary search over departures, which is valid
//! at every position because the timetable's patterns are FIFO chains.
//!
//! The range query runs one pass per candidate departure time, in
//! decreasing order, on shared state: arrivals found for a later departure
//! stay feasible for every earlier one, so each pass explores only what
//! its departure improves, and journeys dominated by a later departure are
//! never emitted.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::fares::{FareLeg, FareTables};
use crate::geometry::{wkb_multi_line_string, LegGeometry, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::path_key::{challenger_wins, PathToken};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// The RAPTOR router.
pub struct Raptor;

mod costs;
mod matrices;
mod search;

pub(crate) use costs::fold_better;
use costs::*;
pub use costs::{CostInputs, CostRow, Objective};
pub(crate) use matrices::{
    access_floor, departure_candidates, nearest_rank, propagate_point_percentiles, travel_time,
};
pub(crate) use search::Search;
use search::*;

#[cfg(test)]
mod tests;
