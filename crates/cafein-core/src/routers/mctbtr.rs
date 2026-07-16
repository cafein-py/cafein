//! McTBTR: multicriteria trip-based routing over (arrival, emissions).
//!
//! This module holds the multicriteria transfer set — the precompute
//! stage. Witt's transfer reduction is unsound under a second
//! criterion (a covering transfer may ride a dirtier vehicle), so the
//! set is built by **global candidate/witness enumeration** instead
//! (Baum et al. 2023's integrated preprocessing, over (arrival,
//! grams)): from every source stop and departure, ride the
//! earliest-or-strictly-cleaner first trips, transfer same-stop onto
//! second trips under the same boarding frontier, and keep a transfer only
//! when a two-trip journey using it survives every witness in the
//! per-stop Pareto bags (strict dominance always evicts; an exact tie
//! resolves by a context-independent order — witness over candidate,
//! then lower label identity — so every enumeration context elects
//! the same canonical journey). Grams anchor at the real
//! boarding, strictly stronger than a trip-local reduction: a
//! transfer survives only where a genuine origin context needs it,
//! and the set is deliberately **not** a superset of the time set.
//! The same-line skip applies only to siblings whose factor is no
//! cleaner; U-turn drops stay (the alight-one-stop-earlier
//! alternative rides strictly less distance on the current trip and
//! the identical distance on the boarded one). Trips without a
//! resolved factor are never boarded and get no transfers — journeys
//! riding them are excluded from emissions frontiers by contract.
//! The set is footpath-blind: walked transfers are the query's job
//! (it boards over installed footpaths from every scanned alight), so
//! one set serves every footpath choice. Dominance is exact (no
//! bucketing): the set must stay complete for every query bucket.

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::mcraptor::{Bag, InsertProbes};
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::Request;
use crate::tbtr::{
    earliest_boardable, DayView, TransferSet, TransferSetBuild, TripTransfer, ViewTrip,
};
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

mod engine;
mod matrices;
mod scan;
mod set;
mod stats;

pub use engine::McTbtrEngine;
use scan::*;
pub use set::transfer_set;
use set::*;
use stats::*;

#[cfg(test)]
mod tests;
