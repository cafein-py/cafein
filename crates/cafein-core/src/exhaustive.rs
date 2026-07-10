//! The exact time × emissions Pareto oracle.
//!
//! A brute-force enumerator of the true Pareto set over (arrival,
//! grams) for one departure: round-based label bags with
//! microgram-quantized grams, every boardable trip considered (emission factors may differ
//! between trips of one line), and single-hop footpaths, mirroring the
//! routing contract. Journeys riding a trip without a resolved factor
//! carry undefined emissions and can never sit on an emissions
//! frontier, so such trips are skipped outright.
//!
//! This is an oracle, not a router: it trades every speed technique
//! for evident correctness, and its cost grows with the bag sizes the
//! data produces. Use it to verify multicriteria engines and to
//! inspect true frontiers for sampled origin–destination pairs.

use crate::geometry::TripGeometry;
use crate::tbtr::{DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// One point of the true frontier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParetoPoint {
    /// Arrival at the destination, in seconds past the service day's
    /// start.
    pub arrival: u32,
    /// Grams CO₂e over the ridden legs.
    pub grams: f64,
    /// Transit legs ridden.
    pub rides: u32,
}

/// A label bag: the (arrival, grams) points no other label dominates.
///
/// Grams are quantized to a microgram at insertion: leg emissions sum
/// float noise in journey-dependent orders, and an oracle must not
/// split one true Pareto point into near-duplicates over it.
#[derive(Debug, Clone, Default)]
struct Bag {
    labels: Vec<(u32, f64)>,
}

pub(crate) fn quantized(grams: f64) -> f64 {
    (grams * 1e6).round() / 1e6
}

impl Bag {
    /// Inserts unless dominated; evicts what the newcomer dominates.
    /// Returns whether the bag changed.
    fn insert(&mut self, arrival: u32, grams: f64) -> bool {
        let grams = quantized(grams);
        for &(at, g) in &self.labels {
            if at <= arrival && g <= grams {
                return false;
            }
        }
        self.labels
            .retain(|&(at, g)| !(arrival <= at && grams <= g));
        self.labels.push((arrival, grams));
        true
    }
}

/// The exact Pareto set over (arrival, grams) for one departure, with
/// at most `max_transfers` transfers; `rides` reports the fewest legs
/// achieving each point. Same journey rules as the routers: seed the
/// access stops, ride, walk at most one footpath between rides, join
/// the egress list (with one incoming footpath hop covered by the
/// closed transfer contract).
#[allow(clippy::too_many_arguments)]
pub fn pareto_oracle(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    departure: u32,
    access: &[(StopIdx, u32)],
    egress: &[(StopIdx, u32)],
    max_transfers: u8,
) -> Vec<ParetoPoint> {
    let stop_count = timetable.stop_count() as usize;
    let rounds = max_transfers as usize + 1;
    // Per-round bags drive the next round's boardings; the cumulative
    // per-stop bags (seeded with the access labels) enforce the label
    // semantics the routers promise: a label dominated by anything
    // that ever reached the stop — a ride looping back to the origin,
    // say — walks and joins nowhere. Dominated labels also cannot
    // carry destination Pareto points, so nothing true is lost.
    let mut current: Vec<Bag> = vec![Bag::default(); stop_count];
    let mut ever: Vec<Bag> = vec![Bag::default(); stop_count];
    for &(stop, seconds) in access {
        current[stop.0 as usize].insert(departure.saturating_add(seconds), 0.0);
        ever[stop.0 as usize].insert(departure.saturating_add(seconds), 0.0);
    }
    let mut destination = Bag::default();
    let mut best_rides: Vec<(u32, f64, u32)> = Vec::new();

    for round in 1..=rounds {
        let mut next: Vec<Bag> = vec![Bag::default(); stop_count];
        // Ride every boardable trip from every label of the previous
        // round's bags.
        for (stop, bag) in current.iter().enumerate() {
            if bag.labels.is_empty() {
                continue;
            }
            for served in timetable.patterns_at_stop(StopIdx(stop as u32)) {
                let positions = timetable.pattern_stops(served.pattern).len();
                if served.position as usize + 1 >= positions {
                    continue;
                }
                for line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                    let offset = view.line_day_offset(line);
                    for trip in view.line_trips(line).map(ViewTrip) {
                        let backing = view.backing(trip);
                        let factor = factors[backing.0 as usize];
                        if !factor.is_finite() {
                            continue;
                        }
                        let times = view.stored_times(timetable, trip);
                        let stored_departure = times[served.position as usize].departure;
                        if stored_departure < offset {
                            continue;
                        }
                        let trip_departure = stored_departure - offset;
                        let stops = timetable.pattern_stops(served.pattern);
                        for &(ready, grams) in &bag.labels {
                            if trip_departure < ready {
                                continue;
                            }
                            for alight in served.position as usize + 1..positions {
                                let arrival = times[alight].arrival - offset;
                                let meters =
                                    geometry.leg_distance(backing, served.position, alight as u16)
                                        as f64;
                                let total = grams + meters / 1000.0 * factor;
                                next[stops[alight].0 as usize].insert(arrival, total);
                            }
                        }
                    }
                }
            }
        }
        // Keep only the labels nothing earlier dominates; walk one
        // footpath hop from those, gated the same way.
        let mut surviving: Vec<Bag> = vec![Bag::default(); stop_count];
        for (stop, bag) in next.iter().enumerate() {
            for &(arrival, grams) in &bag.labels {
                if ever[stop].insert(arrival, grams) {
                    surviving[stop].insert(arrival, grams);
                }
            }
        }
        let walked: Vec<(usize, u32, f64)> = (0..stop_count)
            .flat_map(|stop| {
                let footpaths = footpaths.from_stop(StopIdx(stop as u32));
                surviving[stop]
                    .labels
                    .iter()
                    .flat_map(|&(arrival, grams)| {
                        footpaths.iter().map(move |footpath| {
                            (
                                footpath.to.0 as usize,
                                arrival.saturating_add(footpath.duration),
                                grams,
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for (stop, arrival, grams) in walked {
            if ever[stop].insert(arrival, grams) {
                surviving[stop].insert(arrival, grams);
            }
        }
        let next = surviving;
        // Join the egress list.
        for &(stop, seconds) in egress {
            for &(arrival, grams) in &next[stop.0 as usize].labels {
                let joined = arrival.saturating_add(seconds);
                let grams = quantized(grams);
                if destination.insert(joined, grams) {
                    best_rides.retain(|&(at, g, _)| !(joined <= at && grams <= g));
                    best_rides.push((joined, grams, round as u32));
                }
            }
        }
        current = next;
    }

    let mut points: Vec<ParetoPoint> = destination
        .labels
        .iter()
        .map(|&(arrival, grams)| {
            let rides = best_rides
                .iter()
                .filter(|&&(at, g, _)| at == arrival && g == grams)
                .map(|&(_, _, rides)| rides)
                .min()
                .expect("every frontier point was recorded with its rides");
            ParetoPoint {
                arrival,
                grams,
                rides,
            }
        })
        .collect();
    points.sort_by(|a, b| {
        a.arrival
            .cmp(&b.arrival)
            .then(a.grams.partial_cmp(&b.grams).expect("grams are finite"))
    });
    points
}

#[cfg(test)]
#[path = "exhaustive_tests.rs"]
mod tests;
