//! The reverse RAPTOR: latest-departure routing for an arrival deadline.
//!
//! Round-based like the forward engine — round `k` holds journeys with
//! exactly `k` rides — but labels grow from the destination toward the
//! origins on the same absolute clock. Per stop and round the engine
//! keeps the nondominated frontier over (latest feasible departure,
//! achieved arrival): a single lexicographic best is unsafe under
//! prefixing, since a shared upstream wait can equalize two
//! continuations' origin departures and the discarded earlier-arrival
//! continuation would be unrecoverable. The complete-journey order —
//! applied wherever continuations compose with prefixes — is (latest
//! departure, fewest rides, earliest achieved arrival).
//!
//! A label's achieved arrival is fixed the moment its deadline-side
//! transit leg is chosen: initialization labels carry only their
//! egress walk, and the boarding that consumes one sets the achieved
//! arrival to the trip's actual arrival at the egress stop plus that
//! walk — unused slack against the deadline is excluded before any
//! comparison, never corrected at reconstruction.
//!
//! The labels order candidates only; journeys are materialized by
//! replaying the forward engine at each elected departure, so an
//! arrive-by answer is leg-identical to the depart-at answer for the
//! departure it discovers.

use crate::journey::Journey;
use crate::router::Request;
use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::ReversedTransfers;

mod search;

pub use search::{reverse_one_to_all, reverse_one_to_all_fold, reverse_route, ReverseState};

#[cfg(test)]
mod tests;
