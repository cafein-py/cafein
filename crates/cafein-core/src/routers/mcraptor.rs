//! McRAPTOR: multicriteria RAPTOR over (arrival, emissions).
//!
//! Round-based label bags per stop over (arrival, grams CO₂e). Arrivals
//! compare exactly; grams compare at a configurable bucket width, so
//! labels within one bucket count as equal and bag sizes — and the
//! search — stay bounded. Each bag insertion may substitute a
//! same-bucket representative arriving no later, so a reported
//! journey's emissions sit within one bucket of a true frontier value
//! per insertion its labels survived — a worst case of
//! `(2 × rides + 1) × bucket`, in practice well under one bucket. A
//! vanishing bucket (one microgram, matching the label quantization)
//! reproduces the exhaustive oracle's exact frontier.
//!
//! The emissions firewall holds: a label's grams update is one
//! cumulative-distance subtraction per alight, nothing per-leg beyond
//! that enters the search. Trips without a resolved emission factor are
//! skipped — journeys riding them can never sit on an emissions
//! frontier. Boarding considers, besides the earliest boardable trip,
//! the later trips of the line whose factor strictly improves on every
//! earlier boardable one: waiting for a cleaner vehicle can hold a true
//! Pareto point. On lines with uniform factors — the common case — this
//! collapses to the classic earliest-trip rule.

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::{Exclusions, Request};
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

mod bag;
mod products;
mod search;
mod stats;

pub(crate) use super::mc_bounds::resolved_bounds;
use bag::*;
pub(crate) use bag::{Bag, InsertProbes};
pub use products::{
    frontier_matrix, least_emissions_matrix, route, route_range, route_with_policy,
};
use search::*;
pub(crate) use stats::McRaptorStats;
use stats::*;

#[cfg(test)]
mod tests;

/// Street-policy label sets for the multicriteria search (design §7.2):
/// per-stop access seeds and egress offsets carrying the street grams of
/// the Pareto reduction's frontier points. `None` keeps the walking
/// behaviour — the request's offsets at zero grams. `(stop, seconds)`
/// identifies a frontier point at reconstruction: within one frontier,
/// equal seconds at different grams cannot coexist.
#[derive(Debug, Clone, Default)]
pub struct PolicyLabels {
    /// ``(stop, seconds, grams, sealed)`` — sealed when the point's
    /// composition rode a rental-bearing merged transfer edge, so its
    /// bag entry must not shadow a transit arrival's walk extensions.
    pub access: Vec<(StopIdx, u32, f64, bool)>,
    pub egress: Vec<(StopIdx, u32, f64)>,
}
