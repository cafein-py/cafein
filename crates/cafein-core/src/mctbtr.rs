//! McTBTR: multicriteria trip-based routing over (arrival, emissions).
//!
//! This module holds the multicriteria transfer set — the precompute
//! stage. Witt's transfer reduction is unsound under a second
//! criterion (a covering transfer may ride a dirtier vehicle), so both
//! generation and reduction become dominance-aware:
//!
//! - Generation boards, per (line, position), the earliest catchable
//!   trip **plus** the strictly-decreasing-factor suffix of later
//!   trips — transferring to a later-but-cleaner vehicle can hold a
//!   true Pareto point. The same-line skip applies only to siblings
//!   whose factor is no cleaner; U-turn drops stay (the
//!   alight-one-stop-earlier alternative rides strictly less distance
//!   on the current trip and the identical distance on the boarded
//!   one). Trips without a resolved factor are never boarded and get
//!   no transfers — journeys riding them are excluded from emissions
//!   frontiers by contract.
//! - Reduction keeps a transfer iff some onward stop (directly or over
//!   a footpath) accepts its (arrival, grams) into a per-stop Pareto
//!   bag. Grams sit on the trip-absolute scale — the cumulative
//!   distance along the current trip times its factor — so every
//!   compared journey shares the "riding this trip" prefix and
//!   dominance is invariant to the actual boarding position, the same
//!   argument as McRAPTOR's same-trip rider reduction. Dominance is
//!   exact (no bucketing): the reduced set must stay complete for
//!   every query bucket.
//!
//! Every transfer the time reduction keeps improves an arrival, so it
//! also enters the bags: with every factor resolved, the multicriteria
//! set is a superset of the time set (unresolved-factor trips are the
//! deliberate exception — the time set boards them, this set never
//! does).

use rayon::prelude::*;

use crate::geometry::TripGeometry;
use crate::tbtr::{
    earliest_boardable, u_turn, DayView, TransferSet, TransferSetBuild, TripTransfer, ViewTrip,
};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// Builds the multicriteria transfer set of a day view, fanned out
/// over virtual trips with rayon. `factors` are grams CO₂e per
/// passenger-kilometre per backing trip, NaN where unresolved.
pub fn transfer_set(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
) -> TransferSetBuild {
    let per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..view.trip_count())
        .into_par_iter()
        .map_init(
            || Bags::new(timetable.stop_count()),
            |bags, trip| {
                let trip = ViewTrip(trip);
                let mut generated = generate(view, timetable, footpaths, factors, trip);
                let count = generated.iter().map(Vec::len).sum();
                reduce(
                    view,
                    timetable,
                    footpaths,
                    geometry,
                    factors,
                    trip,
                    bags,
                    &mut generated,
                );
                (generated, count)
            },
        )
        .collect();
    TransferSet::assemble(per_trip)
}

/// Grams CO₂e accumulated along `trip` from its first position to
/// `position` — the trip-absolute scale reduction compares on.
fn absolute_grams(
    geometry: &TripGeometry,
    view: &DayView,
    factors: &[f64],
    trip: ViewTrip,
    position: u16,
) -> f64 {
    let backing = view.backing(trip);
    geometry.leg_distance(backing, 0, position) as f64 / 1000.0 * factors[backing.0 as usize]
}

/// The feasible multicriteria transfers of one virtual trip, per alight
/// position.
fn generate(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    factors: &[f64],
    trip: ViewTrip,
) -> Vec<Vec<TripTransfer>> {
    let line = view.line_of(trip);
    let pattern = view.line_pattern(line);
    let offset = view.line_day_offset(line);
    let stops = timetable.pattern_stops(pattern);
    let times = view.stored_times(timetable, trip);
    let mut per_position: Vec<Vec<TripTransfer>> = vec![Vec::new(); stops.len()];
    let factor = factors[view.backing(trip).0 as usize];
    if !factor.is_finite() {
        return per_position;
    }
    let alight_from = view.first_boardable(trip) as usize + 1;
    for (alight, kept) in per_position.iter_mut().enumerate().skip(alight_from) {
        let arrival = times[alight].arrival - offset;
        let stop = stops[alight];
        let mut board_from = |at: StopIdx, ready: u32| {
            for served in timetable.patterns_at_stop(at) {
                for candidate_line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                    let candidate =
                        earliest_boardable(view, timetable, candidate_line, served.position, ready);
                    let Some(first) = candidate else { continue };
                    // The earliest catchable trip plus the later trips
                    // whose factor strictly improves on every earlier
                    // boardable one.
                    let mut cleanest = f64::INFINITY;
                    for rank in first.0..view.line_trips(candidate_line).end {
                        let boarded = ViewTrip(rank);
                        let boarded_factor = factors[view.backing(boarded).0 as usize];
                        if !boarded_factor.is_finite() || boarded_factor >= cleanest {
                            continue;
                        }
                        cleanest = boarded_factor;
                        if candidate_line == line
                            && boarded.0 >= trip.0
                            && served.position as usize >= alight
                            && boarded_factor >= factor
                        {
                            continue;
                        }
                        if u_turn(
                            view,
                            timetable,
                            stops,
                            times,
                            offset,
                            alight,
                            boarded,
                            served.position,
                        ) {
                            continue;
                        }
                        kept.push(TripTransfer {
                            trip: boarded,
                            position: served.position,
                        });
                    }
                }
            }
        };
        board_from(stop, arrival);
        for footpath in footpaths.from_stop(stop) {
            board_from(footpath.to, arrival.saturating_add(footpath.duration));
        }
    }
    per_position
}

/// The dominance-aware reduction for one virtual trip: back-to-front
/// over the alight positions, a transfer survives iff riding the
/// boarded trip onward lands a (arrival, grams) point no kept
/// alternative dominates, at some stop — directly or over a footpath.
#[allow(clippy::too_many_arguments)]
fn reduce(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    trip: ViewTrip,
    bags: &mut Bags,
    per_position: &mut [Vec<TripTransfer>],
) {
    bags.clear();
    let offset = view.line_day_offset(view.line_of(trip));
    let stops = timetable.pattern_stops(view.line_pattern(view.line_of(trip)));
    let times = view.stored_times(timetable, trip);
    let alight_from = view.first_boardable(trip) as usize + 1;
    for alight in (alight_from..stops.len()).rev() {
        let arrival = times[alight].arrival - offset;
        let staying = absolute_grams(geometry, view, factors, trip, alight as u16);
        bags.improve(stops[alight], arrival, staying);
        for footpath in footpaths.from_stop(stops[alight]) {
            bags.improve(
                footpath.to,
                arrival.saturating_add(footpath.duration),
                staying,
            );
        }
        let alighted = staying;
        per_position[alight].retain(|transfer| {
            let boarded = transfer.trip;
            let boarded_offset = view.day_offset(boarded);
            let boarded_factor = factors[view.backing(boarded).0 as usize];
            let boarded_stops = timetable.pattern_stops(view.line_pattern(view.line_of(boarded)));
            let boarded_times = view.stored_times(timetable, boarded);
            let boarded_from = absolute_grams(geometry, view, factors, boarded, transfer.position);
            let mut keeps = false;
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival - boarded_offset;
                let ridden =
                    absolute_grams(geometry, view, factors, boarded, k as u16) - boarded_from;
                debug_assert!(boarded_factor.is_finite());
                let grams = alighted + ridden;
                if bags.improve(boarded_stops[k], reached, grams) {
                    keeps = true;
                }
                for footpath in footpaths.from_stop(boarded_stops[k]) {
                    if bags.improve(
                        footpath.to,
                        reached.saturating_add(footpath.duration),
                        grams,
                    ) {
                        keeps = true;
                    }
                }
            }
            keeps
        });
    }
}

/// Per-stop (arrival, grams) Pareto bags with cheap reuse: only the
/// touched stops reset between trips.
struct Bags {
    labels: Vec<Vec<(u32, f64)>>,
    touched: Vec<u32>,
}

impl Bags {
    fn new(stop_count: u32) -> Bags {
        Bags {
            labels: vec![Vec::new(); stop_count as usize],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &stop in &self.touched {
            self.labels[stop as usize].clear();
        }
        self.touched.clear();
    }

    /// Pareto insert; returns whether the point entered the bag.
    fn improve(&mut self, stop: StopIdx, arrival: u32, grams: f64) -> bool {
        let bag = &mut self.labels[stop.0 as usize];
        for &(at, g) in bag.iter() {
            if at <= arrival && g <= grams {
                return false;
            }
        }
        if bag.is_empty() {
            self.touched.push(stop.0);
        }
        bag.retain(|&(at, g)| !(arrival <= at && grams <= g));
        bag.push((arrival, grams));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DistanceProvenance;
    use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Line A rides 0→1→2; a fast dirty line and a slow clean line both
    /// ride 1→3.
    fn forked() -> (Timetable, TripGeometry) {
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let fast = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        let slow = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(fast, vec![time(120), time(400)], 1, 0)
            .unwrap();
        builder
            .add_trip(slow, vec![time(150), time(600)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry)
    }

    fn boarded(set: &TransferSet, trip: ViewTrip, position: u16) -> Vec<(u32, u16)> {
        let mut list: Vec<(u32, u16)> = set
            .from_trip_position(trip, position)
            .iter()
            .map(|transfer| (transfer.trip.0, transfer.position))
            .collect();
        list.sort_unstable();
        list
    }

    #[test]
    fn keeps_the_cleaner_slower_transfer() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        // The time reduction sees the slow line arriving later
        // everywhere and drops it.
        let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
        assert_eq!(boarded(&time_set, ViewTrip(0), 1), vec![(1, 0)]);
        // Clean-but-slow (factor 10 vs 100): a true Pareto move, kept.
        let factors = [50.0, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
        // With the slow line just as dirty, the mc reduction drops it
        // too — and matches the time set exactly here.
        let uniform = [50.0, 100.0, 100.0];
        let same = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&same, ViewTrip(0), 1), vec![(1, 0)]);
    }

    #[test]
    fn the_mc_set_covers_the_time_set() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
        for factors in [[50.0, 100.0, 10.0], [50.0, 50.0, 50.0], [1.0, 2.0, 3.0]] {
            let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
            for trip in 0..view.trip_count() {
                let stops = timetable
                    .pattern_stops(view.line_pattern(view.line_of(ViewTrip(trip))))
                    .len();
                for position in 0..stops as u16 {
                    for kept in time_set.from_trip_position(ViewTrip(trip), position) {
                        assert!(
                            mc.from_trip_position(ViewTrip(trip), position)
                                .contains(kept),
                            "time-kept transfer missing under factors {factors:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_factor_lines_board_the_cleaner_later_trip() {
        // One connecting line with a dirty early trip and a clean later
        // one: generation must board both.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(120), time(400)], 1, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(200), time(500)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let mixed = [50.0, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
        // Uniform factors collapse to the earliest-trip rule.
        let uniform = [50.0, 100.0, 100.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0)]);
    }

    #[test]
    fn same_line_reboarding_survives_only_toward_cleaner_siblings() {
        // One line, dirty trip then clean trip: alighting the dirty
        // trip mid-pattern may re-board the cleaner sibling at the same
        // position; with uniform factors the sibling stays skipped.
        let mut builder = TimetableBuilder::new(3);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(50), time(200), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(3);
        let mixed = [100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 1)]);
        let uniform = [100.0, 100.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
    }

    #[test]
    fn unresolved_factors_are_excluded() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        // The fast line's factor is unresolved: never boarded.
        let factors = [50.0, f64::NAN, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(2, 0)]);
        // An unresolved source trip gets no transfers at all.
        let factors = [f64::NAN, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
    }

    #[test]
    fn u_turns_stay_dropped() {
        // Alight A at stop 2, walk back to stop 1, and re-ride the
        // 1→2 segment on B although B was catchable by alighting A at
        // stop 1 directly: the classic U-turn, dropped even though B
        // is cleaner — the earlier alight saves grams on A and rides
        // the identical distance on B. Boarding B at stop 2 itself
        // stays a plain, kept transfer.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder
            .add_pattern(&[StopIdx(1), StopIdx(2), StopIdx(3)], 1)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(250), time(350), time(450)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
            ],
        )
        .unwrap();
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(1), 30, 30.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let factors = [100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        // From (A, position 2): the walk-back boarding of B at
        // position 0 is the U-turn and is gone; boarding B at stop 2
        // (position 1) survives.
        assert_eq!(boarded(&mc, ViewTrip(0), 2), vec![(1, 1)]);
    }
}
