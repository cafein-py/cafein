//! The routing request and the router interface.

use crate::journey::Journey;
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// A single-departure routing request.
///
/// The request describes one service day: times are seconds past that day's
/// start, and `active_services` says which services run on it. The previous
/// day's over-midnight trips are also considered, shifted back one day, so a
/// query early on the service day still catches trips whose stored times run
/// past `24:00:00` on the day before (`active_services_previous`). Access and
/// egress lists come from the street-side search and must already cover
/// everything reachable on foot from the origin and to the destination.
#[derive(Debug, Clone)]
pub struct Request {
    /// Departure time at the origin.
    pub departure: u32,
    /// Stops reachable from the origin: `(stop, seconds from the origin)`.
    pub access: Vec<(StopIdx, u32)>,
    /// Stops the destination is reachable from: `(stop, seconds to the
    /// destination)`.
    pub egress: Vec<(StopIdx, u32)>,
    /// One flag per service identifier carried on timetable trips; trips
    /// whose service index is out of range never run.
    pub active_services: Vec<bool>,
    /// The same, one service day earlier: trips of these services run with
    /// their stored times shifted back a day, so `25:30:00` on the previous
    /// day is reachable at `01:30:00` on this one.
    pub active_services_previous: Vec<bool>,
    /// Journeys may use at most `max_transfers + 1` transit legs.
    pub max_transfers: u8,
}

/// A public-transport routing algorithm.
///
/// Returns the Pareto set over (arrival time, number of rides): one journey
/// per ride count that arrives strictly earlier than every journey with
/// fewer rides, ordered by increasing ride count. Transit legs carry the
/// trip and its board/alight positions, so per-leg distance, geometry, and
/// emissions annotation attaches after routing without router involvement.
pub trait TransitRouter: Sync {
    fn route(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Journey>;
}
