//! The routing request and the router interface.

use crate::journey::Journey;
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// Query-time supply exclusions: stops, trips, and routes the journey
/// must not use. Empty vectors mean the family has no exclusions; an
/// absent or out-of-range bit reads as not excluded. A vehicle may
/// still ride through an excluded stop — the stop only refuses
/// boarding, alighting, transfers, and access/egress. Shared across a
/// matrix's per-origin requests through the `Arc`.
#[derive(Debug)]
pub struct Exclusions {
    stops: Vec<bool>,
    trips: Vec<bool>,
    routes: Vec<bool>,
}

impl Exclusions {
    pub fn new(stops: Vec<bool>, trips: Vec<bool>, routes: Vec<bool>) -> Exclusions {
        Exclusions {
            stops,
            trips,
            routes,
        }
    }

    #[inline]
    pub fn excludes_stop(&self, stop: StopIdx) -> bool {
        self.stops.get(stop.0 as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn excludes_trip(&self, trip: TripIdx) -> bool {
        self.trips.get(trip.0 as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn excludes_route(&self, route: u32) -> bool {
        self.routes.get(route as usize).copied().unwrap_or(false)
    }
}

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
    /// Query-time supply exclusions; `None` is the unrestricted query.
    pub exclusions: Option<std::sync::Arc<Exclusions>>,
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

/// Whether a `router="auto"` time-criterion query runs on the trip-based engine.
///
/// Only when a cached time transfer set was precomputed for the query's
/// service date: the trip-based engine's advantage is riding a precomputed
/// set, and an ad-hoc per-call build would make one-shot queries pay for it.
pub fn auto_time_tbtr(cached_date: Option<&str>, date: &str, needs_raptor: bool) -> bool {
    !needs_raptor && cached_date == Some(date)
}

/// Whether two per-trip emission-factor vectors are the same configuration.
///
/// Bitwise equality per element: the vectors are NaN-padded for trips without
/// a factor, so float `==` would never match two identical configurations.
/// This exact comparison — not a hash — is the equality proof binding a
/// cached multicriteria set to the factors it was built with.
pub fn same_factors(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A deterministic, NaN-safe fingerprint of a per-trip emission-factor
/// vector, for inspection only — cache equality is proven by `same_factors`,
/// never by this hash, whose collisions would silently reuse a set built for
/// other factors. Not a cryptographic digest.
pub fn factor_fingerprint(per_trip: &[f64]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    let mut hash = 0xcbf29ce484222325u64;
    for &factor in per_trip {
        hash = (hash ^ factor.to_bits()).wrapping_mul(PRIME);
    }
    (hash ^ per_trip.len() as u64).wrapping_mul(PRIME)
}

/// Whether a `router="auto"` multicriteria query runs on the trip-based
/// engine.
///
/// Only when a cached multicriteria transfer set was precomputed for the
/// query's service date **and** exactly the resolved per-trip factor vector
/// (`same_factors`), and the query asks nothing the trip-based engine cannot
/// answer (`needs_raptor`).
/// The boundary is a contract, not a gap: the persisted set is reduced
/// under strict unpenalized dominance at build time, so positive slack
/// and route bans or penalties (the relaxed and diverse candidates) can
/// invalidate transfers discarded against build-time witnesses and stay
/// on McRAPTOR, as does a door-to-door upgrade only the RAPTOR path
/// has. `max_slower` restricts the strict search and runs on either
/// engine.
pub fn auto_mc_tbtr(
    cached: Option<(&str, &[f64])>,
    date: &str,
    per_trip: &[f64],
    needs_raptor: bool,
) -> bool {
    !needs_raptor
        && cached.is_some_and(|(cached_date, factors)| {
            cached_date == date && same_factors(factors, per_trip)
        })
}

#[cfg(test)]
mod tests;
