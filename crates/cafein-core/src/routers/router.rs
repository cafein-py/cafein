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
mod tests {
    use super::{auto_mc_tbtr, auto_time_tbtr, factor_fingerprint, same_factors};

    #[test]
    fn auto_time_requires_matching_cached_date() {
        assert!(auto_time_tbtr(Some("2022-02-22"), "2022-02-22", false));
        assert!(!auto_time_tbtr(Some("2022-02-21"), "2022-02-22", false));
        assert!(!auto_time_tbtr(None, "2022-02-22", false));
        // Exclusions force RAPTOR even over a matching cache.
        assert!(!auto_time_tbtr(Some("2022-02-22"), "2022-02-22", true));
    }

    #[test]
    fn auto_mc_requires_matching_cache_and_supported_query() {
        let factors = [f64::NAN, 74.0, 101.0];
        let cached = Some(("2022-02-22", &factors[..]));
        assert!(auto_mc_tbtr(cached, "2022-02-22", &factors, false));
        assert!(!auto_mc_tbtr(cached, "2022-02-23", &factors, false));
        let other = [f64::NAN, 74.0, 88.0];
        assert!(!auto_mc_tbtr(cached, "2022-02-22", &other, false));
        assert!(!auto_mc_tbtr(cached, "2022-02-22", &factors, true));
        assert!(!auto_mc_tbtr(None, "2022-02-22", &factors, false));
    }

    #[test]
    fn factor_equality_is_bitwise_and_exact() {
        // NaN-padded identical vectors are the same configuration (float `==`
        // would say no), and a prefix is not.
        assert!(same_factors(&[f64::NAN, 74.0], &[f64::NAN, 74.0]));
        assert!(!same_factors(&[f64::NAN, 74.0], &[f64::NAN]));

        // A crafted FNV collision: solve the second element with the modular
        // inverse of the FNV prime so both vectors hash identically. Under
        // fingerprint equality a set built for `a` would serve a query
        // resolving to `b`; the exact comparison refuses it.
        const PRIME: u64 = 0x100000001b3;
        let mut inverse = PRIME;
        for _ in 0..6 {
            // Newton's iteration doubles the correct low bits each round.
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(PRIME.wrapping_mul(inverse)));
        }
        assert_eq!(PRIME.wrapping_mul(inverse), 1);
        let offset = 0xcbf29ce484222325u64;
        let a = [74.0f64, 101.0];
        let after_first = (offset ^ a[0].to_bits()).wrapping_mul(PRIME);
        let target = (after_first ^ a[1].to_bits()).wrapping_mul(PRIME);
        let b_first = 75.0f64;
        let after_b_first = (offset ^ b_first.to_bits()).wrapping_mul(PRIME);
        let b = [
            b_first,
            f64::from_bits(target.wrapping_mul(inverse) ^ after_b_first),
        ];
        assert_eq!(factor_fingerprint(&a), factor_fingerprint(&b));
        assert!(!same_factors(&a, &b));
        assert!(!auto_mc_tbtr(
            Some(("2022-02-22", &a[..])),
            "2022-02-22",
            &b,
            false
        ));
    }
}
