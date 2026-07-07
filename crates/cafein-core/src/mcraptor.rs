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

use crate::exhaustive::quantized;
use crate::geometry::TripGeometry;
use crate::journey::{Journey, Leg};
use crate::raptor::departure_candidates;
use crate::router::Request;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// How a label reached its stop; parents index the label arena.
#[derive(Debug, Clone, Copy)]
enum Origin {
    Access,
    Ride {
        parent: u32,
        trip: ViewTrip,
        board: u16,
        alight: u16,
    },
    Walk {
        parent: u32,
        duration: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct Label {
    arrival: u32,
    grams: f64,
    stop: StopIdx,
    origin: Origin,
}

/// One bag entry; `key` is the grams bucket, `grams` the exact
/// (microgram-quantized) value behind it.
#[derive(Debug, Clone, Copy)]
struct Entry {
    arrival: u32,
    key: i64,
    grams: f64,
}

/// A per-stop label bag under bucketed dominance, cumulative across
/// rounds and profile passes.
#[derive(Debug, Clone, Default)]
struct Bag {
    entries: Vec<Entry>,
}

impl Bag {
    /// Inserts unless an entry arriving no later sits in the same or a
    /// cleaner bucket; evicts what the newcomer covers. An entry equal
    /// on both axes but strictly dirtier in exact grams is refined
    /// (replaced), keeping the bucket's representative as clean as the
    /// search has seen.
    fn insert(&mut self, arrival: u32, grams: f64, key: i64) -> bool {
        for entry in &self.entries {
            if entry.arrival <= arrival
                && entry.key <= key
                && !(entry.arrival == arrival && entry.key == key && grams < entry.grams)
            {
                return false;
            }
        }
        self.entries
            .retain(|entry| !(arrival <= entry.arrival && key <= entry.key));
        self.entries.push(Entry {
            arrival,
            key,
            grams,
        });
        true
    }
}

/// A destination frontier entry; `departure` is the profile pass that
/// produced it.
#[derive(Debug, Clone, Copy)]
struct Arrived {
    departure: u32,
    arrival: u32,
    key: i64,
    grams: f64,
    label: u32,
}

/// The destination bag: Pareto over (departure descending, arrival,
/// grams bucket). Passes run at descending departures, so entries from
/// later departures are never evicted by earlier ones.
#[derive(Debug, Default)]
struct DestinationBag {
    entries: Vec<Arrived>,
}

impl DestinationBag {
    fn insert(&mut self, candidate: Arrived) -> bool {
        for entry in &self.entries {
            if entry.departure >= candidate.departure
                && entry.arrival <= candidate.arrival
                && entry.key <= candidate.key
                && !(entry.departure == candidate.departure
                    && entry.arrival == candidate.arrival
                    && entry.key == candidate.key
                    && candidate.grams < entry.grams)
            {
                return false;
            }
        }
        self.entries.retain(|entry| {
            !(candidate.departure >= entry.departure
                && candidate.arrival <= entry.arrival
                && candidate.key <= entry.key)
        });
        self.entries.push(candidate);
        true
    }
}

/// A label riding a trip during one line scan. `kappa` folds the grams
/// at boarding and the boarding position's cumulative distance into one
/// value comparable between same-trip riders: at every future alight
/// both share the trip's arrival, so the lower `kappa` alights with
/// fewer grams everywhere.
#[derive(Debug, Clone, Copy)]
struct Rider {
    trip: ViewTrip,
    board: u16,
    kappa: f64,
    grams: f64,
    factor: f64,
    parent: u32,
}

struct Search<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    footpaths: &'a Transfers,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    bucket: f64,
    arena: Vec<Label>,
    bags: Vec<Bag>,
    destination: DestinationBag,
    /// Per line: the (position, label) boardings queued this round.
    queue: Vec<Vec<(u16, u32)>>,
    touched: Vec<u32>,
}

/// The multicriteria journeys for a single departure: the Pareto set
/// over (arrival, emissions bucket), as full journeys.
pub fn route(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    bucket: f64,
) -> Vec<Journey> {
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &[request.departure],
        bucket,
    )
}

/// The multicriteria departure-window profile: the Pareto set over
/// (departure, arrival, emissions bucket), each journey's departure
/// being the latest time the origin can be left to catch it.
#[allow(clippy::too_many_arguments)]
pub fn route_range(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    window: u32,
    bucket: f64,
) -> Vec<Journey> {
    let departures = departure_candidates(timetable, request, window);
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &departures,
        bucket,
    )
}

#[allow(clippy::too_many_arguments)]
fn profile(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
    bucket: f64,
) -> Vec<Journey> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    let mut search = Search {
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        bucket,
        arena: Vec::new(),
        bags: vec![Bag::default(); timetable.stop_count() as usize],
        destination: DestinationBag::default(),
        queue: vec![Vec::new(); view.line_count() as usize],
        touched: Vec::new(),
    };
    for &departure in departures {
        search.pass(request, departure);
    }
    let mut journeys: Vec<Journey> = search
        .destination
        .entries
        .iter()
        .map(|arrived| search.assemble(arrived))
        .collect();
    journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
    journeys
}

impl Search<'_> {
    fn key(&self, grams: f64) -> i64 {
        (grams / self.bucket).floor() as i64
    }

    /// One profile pass: seed the access labels at `departure`, then
    /// ride/walk/join for each round. Bags persist across passes —
    /// labels of later departures suppress what they dominate, exactly
    /// the range-RAPTOR reuse.
    fn pass(&mut self, request: &Request, departure: u32) {
        let mut fresh: Vec<u32> = Vec::new();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            let label = self.arena.len() as u32;
            let key = self.key(0.0);
            if self.bags[stop.0 as usize].insert(arrival, 0.0, key) {
                self.arena.push(Label {
                    arrival,
                    grams: 0.0,
                    stop,
                    origin: Origin::Access,
                });
                fresh.push(label);
            }
        }
        for _round in 1..=request.max_transfers as u32 + 1 {
            if fresh.is_empty() {
                break;
            }
            for &label in &fresh {
                let stop = self.arena[label as usize].stop;
                for served in self.timetable.patterns_at_stop(stop) {
                    let positions = self.timetable.pattern_stops(served.pattern).len();
                    if served.position as usize + 1 >= positions {
                        continue;
                    }
                    for line in self
                        .view
                        .lines_of_pattern(served.pattern)
                        .into_iter()
                        .flatten()
                    {
                        if self.queue[line as usize].is_empty() {
                            self.touched.push(line);
                        }
                        self.queue[line as usize].push((served.position, label));
                    }
                }
            }
            let mut rode: Vec<u32> = Vec::new();
            let touched = std::mem::take(&mut self.touched);
            for &line in &touched {
                self.scan_line(line, &mut rode);
            }
            // One footpath hop from the improving ride labels; the
            // closed transfer contract makes chains redundant.
            let mut next = rode.clone();
            for &label in &rode {
                let from = self.arena[label as usize];
                let key = self.key(from.grams);
                for footpath in self.footpaths.from_stop(from.stop) {
                    let arrival = from.arrival.saturating_add(footpath.duration);
                    let walked = self.arena.len() as u32;
                    if self.bags[footpath.to.0 as usize].insert(arrival, from.grams, key) {
                        self.arena.push(Label {
                            arrival,
                            grams: from.grams,
                            stop: footpath.to,
                            origin: Origin::Walk {
                                parent: label,
                                duration: footpath.duration,
                            },
                        });
                        next.push(walked);
                    }
                }
            }
            for &label in &next {
                let reached = self.arena[label as usize];
                for &(stop, seconds) in &request.egress {
                    if stop == reached.stop {
                        let arrival = reached.arrival.saturating_add(seconds);
                        self.destination.insert(Arrived {
                            departure,
                            arrival,
                            key: self.key(reached.grams),
                            grams: reached.grams,
                            label,
                        });
                    }
                }
            }
            fresh = next;
        }
    }

    /// Scans one line: boarding labels enter the rider bag at their
    /// queued positions, riders alight at every later position. Besides
    /// the earliest boardable trip, later trips join while their factor
    /// strictly improves; same-trip riders reduce to the lowest
    /// `kappa`.
    fn scan_line(&mut self, line: u32, rode: &mut Vec<u32>) {
        let mut entries = std::mem::take(&mut self.queue[line as usize]);
        entries.sort_unstable_by_key(|&(position, _)| position);
        let pattern = self.view.line_pattern(line);
        let stops = self.timetable.pattern_stops(pattern);
        let offset = self.view.line_day_offset(line);
        let mut riders: Vec<Rider> = Vec::new();
        let mut queued = 0;
        for position in entries[0].0 as usize..stops.len() {
            for rider in &riders {
                if (rider.board as usize) >= position {
                    continue;
                }
                let times = self.view.stored_times(self.timetable, rider.trip);
                let arrival = times[position].arrival - offset;
                let meters = self.geometry.leg_distance(
                    self.view.backing(rider.trip),
                    rider.board,
                    position as u16,
                ) as f64;
                let grams = quantized(rider.grams + meters / 1000.0 * rider.factor);
                let label = self.arena.len() as u32;
                let key = self.key(grams);
                if self.bags[stops[position].0 as usize].insert(arrival, grams, key) {
                    self.arena.push(Label {
                        arrival,
                        grams,
                        stop: stops[position],
                        origin: Origin::Ride {
                            parent: rider.parent,
                            trip: rider.trip,
                            board: rider.board,
                            alight: position as u16,
                        },
                    });
                    rode.push(label);
                }
            }
            while queued < entries.len() && entries[queued].0 as usize == position {
                let (_, label) = entries[queued];
                queued += 1;
                let boarding = self.arena[label as usize];
                let Some(first) = earliest_boardable(
                    self.view,
                    self.timetable,
                    line,
                    position as u16,
                    boarding.arrival,
                ) else {
                    continue;
                };
                let mut cleanest = f64::INFINITY;
                for rank in first.0..self.view.line_trips(line).end {
                    let trip = ViewTrip(rank);
                    let factor = self.factors[self.view.backing(trip).0 as usize];
                    if !factor.is_finite() || factor >= cleanest {
                        continue;
                    }
                    cleanest = factor;
                    let travelled =
                        self.geometry
                            .leg_distance(self.view.backing(trip), 0, position as u16)
                            as f64;
                    let kappa = boarding.grams - travelled / 1000.0 * factor;
                    if riders
                        .iter()
                        .any(|rider| rider.trip == trip && rider.kappa <= kappa)
                    {
                        continue;
                    }
                    riders.retain(|rider| !(rider.trip == trip && kappa < rider.kappa));
                    riders.push(Rider {
                        trip,
                        board: position as u16,
                        kappa,
                        grams: boarding.grams,
                        factor,
                        parent: label,
                    });
                }
            }
        }
        entries.clear();
        self.queue[line as usize] = entries;
    }

    /// Walks a destination entry's label chain back into the journey
    /// contract.
    fn assemble(&self, arrived: &Arrived) -> Journey {
        let last = &self.arena[arrived.label as usize];
        let mut legs = vec![Leg::Egress {
            from_stop: last.stop,
            departure: last.arrival,
            arrival: arrived.arrival,
        }];
        let mut at = arrived.label;
        loop {
            let label = &self.arena[at as usize];
            match label.origin {
                Origin::Access => {
                    legs.push(Leg::Access {
                        to_stop: label.stop,
                        departure: arrived.departure,
                        arrival: label.arrival,
                    });
                    break;
                }
                Origin::Ride {
                    parent,
                    trip,
                    board,
                    alight,
                } => {
                    let line = self.view.line_of(trip);
                    let offset = self.view.line_day_offset(line);
                    let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                    let times = self.view.stored_times(self.timetable, trip);
                    legs.push(Leg::Transit {
                        trip: self.view.backing(trip),
                        board_stop: stops[board as usize],
                        alight_stop: stops[alight as usize],
                        board_position: board,
                        alight_position: alight,
                        board_time: times[board as usize].departure - offset,
                        alight_time: times[alight as usize].arrival - offset,
                    });
                    at = parent;
                }
                Origin::Walk { parent, duration } => {
                    let from = &self.arena[parent as usize];
                    legs.push(Leg::Transfer {
                        from_stop: from.stop,
                        to_stop: label.stop,
                        departure: from.arrival,
                        arrival: from.arrival.saturating_add(duration),
                    });
                    at = parent;
                }
            }
        }
        legs.reverse();
        Journey {
            departure: arrived.departure,
            arrival: arrived.arrival,
            legs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exhaustive::pareto_oracle;
    use crate::geometry::DistanceProvenance;
    use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    fn request(access: StopIdx, egress: StopIdx, max_transfers: u8) -> Request {
        Request {
            departure: 0,
            access: vec![(access, 0)],
            egress: vec![(egress, 0)],
            active_services: Vec::new(),
            active_services_previous: Vec::new(),
            max_transfers,
        }
    }

    fn grams_of(journey: &Journey, geometry: &TripGeometry, factors: &[f64]) -> f64 {
        quantized(
            journey
                .legs
                .iter()
                .map(|leg| match leg {
                    Leg::Transit {
                        trip,
                        board_position,
                        alight_position,
                        ..
                    } => {
                        geometry.leg_distance(*trip, *board_position, *alight_position) as f64
                            / 1000.0
                            * factors[trip.0 as usize]
                    }
                    _ => 0.0,
                })
                .sum(),
        )
    }

    fn triples(
        journeys: &[Journey],
        geometry: &TripGeometry,
        factors: &[f64],
    ) -> Vec<(u32, f64, u32)> {
        let mut triples: Vec<(u32, f64, u32)> = journeys
            .iter()
            .map(|journey| {
                (
                    journey.arrival,
                    grams_of(journey, geometry, factors),
                    journey.rides() as u32,
                )
            })
            .collect();
        triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        triples
    }

    fn oracle_triples(points: &[crate::exhaustive::ParetoPoint]) -> Vec<(u32, f64, u32)> {
        points
            .iter()
            .map(|point| (point.arrival, point.grams, point.rides))
            .collect()
    }

    /// The exhaustive oracle's frontier fixture: a fast dirty direct
    /// line, a slow clean one, and a cleaner-but-slower combination
    /// over a transfer.
    fn frontier_fixture() -> (Timetable, TripGeometry, [f64; 4]) {
        let mut builder = TimetableBuilder::new(4);
        let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 0).unwrap();
        let clean = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 1).unwrap();
        let combo_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
        let combo_b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 3).unwrap();
        builder
            .add_trip(dirty, vec![time(100), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(clean, vec![time(100), time(900)], 1, 0)
            .unwrap();
        builder
            .add_trip(combo_a, vec![time(100), time(300)], 2, 0)
            .unwrap();
        builder
            .add_trip(combo_b, vec![time(400), time(700)], 3, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 500.0], DistanceProvenance::CrowFly),
                (TripIdx(3), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry, [100.0, 10.0, 100.0, 10.0])
    }

    #[test]
    fn matches_the_oracle_with_a_vanishing_bucket() {
        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            3,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(
            oracle_triples(&points),
            vec![(500, 100.0, 1), (700, 55.0, 2), (900, 10.0, 1)]
        );
    }

    #[test]
    fn a_wide_bucket_collapses_to_the_fastest_journey() {
        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e9,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            vec![(500, 100.0, 1)]
        );
    }

    #[test]
    fn loop_backs_walk_nowhere() {
        // The oracle's regression shape: ride out and back, then walk
        // to the destination — cleaner and earlier than the direct
        // line, but suppressed by the access label's dominance.
        let mut builder = TimetableBuilder::new(3);
        let out = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let back = builder.add_pattern(&[StopIdx(1), StopIdx(0)], 1).unwrap();
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(out, vec![time(10), time(50)], 0, 0)
            .unwrap();
        builder
            .add_trip(back, vec![time(60), time(100)], 1, 0)
            .unwrap();
        builder
            .add_trip(direct, vec![time(20), time(200)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 10.0, 100.0];
        let footpaths = Transfers::from_edges(
            3,
            &[
                (StopIdx(0), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(0), 30, 30.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(2), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            vec![(200, 80.0, 1)]
        );
    }

    #[test]
    fn waits_for_the_cleaner_trip_on_a_mixed_factor_line() {
        // One line, a dirty early trip and a clean later one: the true
        // frontier holds both, so boarding must look past the earliest
        // boardable trip when a later factor strictly improves.
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(300), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [100.0, 10.0];
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 1);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            1,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(
            oracle_triples(&points),
            vec![(200, 100.0, 1), (400, 10.0, 1)]
        );
    }

    #[test]
    fn transfers_over_a_footpath_match_the_oracle() {
        // Ride, walk a footpath, ride again — the walked hop must
        // carry its grams unchanged and reconstruct as a transfer leg.
        let mut builder = TimetableBuilder::new(4);
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let second = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 20.0];
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            3,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(oracle_triples(&points), vec![(300, 50.0, 2)]);
        let legs: Vec<&str> = journeys[0]
            .legs
            .iter()
            .map(|leg| match leg {
                Leg::Access { .. } => "access",
                Leg::Transit { .. } => "transit",
                Leg::Transfer { .. } => "transfer",
                Leg::Egress { .. } => "egress",
            })
            .collect();
        assert_eq!(
            legs,
            vec!["access", "transit", "transfer", "transit", "egress"]
        );
    }

    #[test]
    fn skips_unresolved_factors() {
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(0), time(100)], 0, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![(TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly)],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(1), 1);
        let journeys = route(
            &view,
            &timetable,
            &Transfers::empty(2),
            &geometry,
            &[f64::NAN],
            &request,
            1e-6,
        );
        assert!(journeys.is_empty());
    }

    #[test]
    fn profiles_the_departure_window() {
        // Two departures of one line inside the window: the profile
        // keeps both journeys, each at the latest departure catching
        // it.
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(200), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [50.0, 50.0];
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = Request {
            departure: 50,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 1,
        };
        let journeys = route_range(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 200, 1e-6,
        );
        let profile: Vec<(u32, u32)> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival))
            .collect();
        assert_eq!(profile, vec![(100, 300), (200, 400)]);
    }
}
