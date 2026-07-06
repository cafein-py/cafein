//! The TBTR day views, trip-to-trip transfer sets, and query engine
//! over the Helsinki region GTFS feed shared with r5py
//! (r5py.sampledata.helsinki v1.1.1).

mod common;

use std::sync::OnceLock;

use cafein_core::raptor::Raptor;
use cafein_core::router::{Request, TransitRouter};
use cafein_core::tbtr::{DayView, TbtrEngine, TransferSet, TransferSetBuild, ViewTrip};
use cafein_core::timetable::{StopIdx, Timetable};
use cafein_core::transfers::Transfers;
use cafein_gtfs::{build_timetable, Feed, ServiceCalendar};
use chrono::NaiveDate;

fn helsinki() -> Option<&'static (Timetable, ServiceCalendar, TransferSetBuild)> {
    static DATA: OnceLock<Option<(Timetable, ServiceCalendar, TransferSetBuild)>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = common::helsinki_gtfs_path()?;
        let feed = Feed::from_path(path).unwrap();
        let build = build_timetable(&feed).unwrap();
        // Same-stop transfers only: footpaths come from OSM streets,
        // which the Rust integration layer does not build.
        let transfers = TransferSet::build(
            &build.timetable,
            &Transfers::empty(build.timetable.stop_count()),
        );
        Some((build.timetable, build.services, transfers))
    })
    .as_ref()
}

#[test]
fn reduction_keeps_a_fraction_of_the_feasible_transfers() {
    let Some((_, _, build)) = helsinki() else {
        return;
    };
    assert!(!build.transfers.is_empty());
    assert!(build.transfers.len() < build.generated);
    // Pinned counts: deterministic for the pinned fixture. The
    // reduction keeps ~12 % of the feasible transfers, in line with
    // the reductions Witt reports.
    assert_eq!(build.generated, 42_937_748);
    assert_eq!(build.transfers.len(), 5_064_961);
}

#[test]
fn a_date_view_shrinks_the_universe_and_the_set() {
    let Some((timetable, services, universal)) = helsinki() else {
        return;
    };
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let view = DayView::for_date(
        timetable,
        &services.active_on(date),
        &services.active_on(date.pred_opt().unwrap()),
    );
    assert!(view.trip_count() < timetable.trip_count());
    assert!(view.trip_count() > 0);
    let build = TransferSet::for_view(&view, timetable, &Transfers::empty(timetable.stop_count()));
    assert!(!build.transfers.is_empty());
    assert!(build.transfers.len() < universal.transfers.len());
    // Pinned for the fixture: the Tuesday universe holds 24 280 of the
    // 195 351 trips (the feed spans weeks of service days).
    assert_eq!(view.trip_count(), 24_280);
    assert_eq!(build.transfers.len(), 576_932);
}

#[test]
fn queries_match_raptor_across_sampled_pairs() {
    let Some((timetable, services, _)) = helsinki() else {
        return;
    };
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let active = services.active_on(date);
    let active_previous = services.active_on(date.pred_opt().unwrap());
    let footpaths = Transfers::empty(timetable.stop_count());
    let engine = TbtrEngine::for_date(timetable, &footpaths, &active, &active_previous);
    let pareto = |journeys: &[cafein_core::journey::Journey]| -> Vec<(u32, usize)> {
        journeys
            .iter()
            .map(|journey| (journey.arrival, journey.rides()))
            .collect()
    };
    // Deterministic strides across the stop space, morning and evening
    // departures: the (arrival, rides) Pareto sets must agree pair for
    // pair with RAPTOR's.
    let mut compared = 0;
    for origin in (0..timetable.stop_count()).step_by(611).map(StopIdx) {
        for destination in (37..timetable.stop_count()).step_by(1259).map(StopIdx) {
            for departure in [8 * 3600 + 30 * 60, 17 * 3600] {
                let request = Request {
                    departure,
                    access: vec![(origin, 0)],
                    egress: vec![(destination, 0)],
                    active_services: active.clone(),
                    active_services_previous: active_previous.clone(),
                    max_transfers: 4,
                };
                let raptor = Raptor.route(timetable, &footpaths, &request);
                let tbtr = engine.query(
                    departure,
                    &request.access,
                    &request.egress,
                    request.max_transfers,
                );
                assert_eq!(
                    pareto(&tbtr),
                    pareto(&raptor),
                    "pareto sets diverge for {origin:?}->{destination:?} at {departure}"
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 150);
}

#[test]
fn range_profiles_and_one_to_all_match_raptor() {
    let Some((timetable, services, _)) = helsinki() else {
        return;
    };
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let active = services.active_on(date);
    let active_previous = services.active_on(date.pred_opt().unwrap());
    let footpaths = Transfers::empty(timetable.stop_count());
    let engine = TbtrEngine::for_date(timetable, &footpaths, &active, &active_previous);
    let triples = |journeys: &[cafein_core::journey::Journey]| -> Vec<(u32, u32, usize)> {
        journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect()
    };
    let mut window_pairs = 0;
    for origin in (13..timetable.stop_count()).step_by(1103).map(StopIdx) {
        for destination in (271..timetable.stop_count()).step_by(2417).map(StopIdx) {
            for departure in [8 * 3600 + 30 * 60, 16 * 3600 + 30 * 60] {
                let request = Request {
                    departure,
                    access: vec![(origin, 0)],
                    egress: vec![(destination, 0)],
                    active_services: active.clone(),
                    active_services_previous: active_previous.clone(),
                    max_transfers: 4,
                };
                let raptor = Raptor.route_range(timetable, &footpaths, &request, 1800);
                let tbtr = engine.route_range(&request, 1800);
                assert_eq!(
                    triples(&tbtr),
                    triples(&raptor),
                    "window profiles diverge for {origin:?}->{destination:?} at {departure}"
                );
                window_pairs += 1;
            }
        }
    }
    assert!(window_pairs >= 40);
    // One-to-all: full per-stop arrival arrays, origin by origin.
    for origin in (401..timetable.stop_count()).step_by(1667).map(StopIdx) {
        let request = Request {
            departure: 8 * 3600 + 30 * 60,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: active.clone(),
            active_services_previous: active_previous.clone(),
            max_transfers: 4,
        };
        let raptor = Raptor.one_to_all(timetable, &footpaths, &request);
        let tbtr = engine.one_to_all(request.departure, &request.access, request.max_transfers);
        let mismatches: Vec<_> = tbtr
            .iter()
            .zip(raptor.iter())
            .enumerate()
            .filter(|(_, (ours, theirs))| ours != theirs)
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "one-to-all diverges from {origin:?} at (stop, (tbtr, raptor)): {mismatches:?}"
        );
    }
}

#[test]
fn kept_transfers_are_feasible_and_earliest() {
    let Some((timetable, _, build)) = helsinki() else {
        return;
    };
    let view = DayView::universal(timetable);
    let set = &build.transfers;
    // Every kept transfer boards a catchable trip that still has track
    // ahead, and no earlier trip of the same line position was
    // catchable — sampled across the trip space. In the universal view
    // virtual trips coincide with timetable trips.
    for trip in (0..view.trip_count()).step_by(997).map(ViewTrip) {
        let stops = timetable.pattern_stops(view.line_pattern(view.line_of(trip)));
        let times = view.stored_times(timetable, trip);
        for alight in 1..stops.len() {
            for transfer in set.from_trip_position(trip, alight as u16) {
                let line = view.line_of(transfer.trip);
                let boarded_stops = timetable.pattern_stops(view.line_pattern(line));
                let position = transfer.position as usize;
                assert!(position + 1 < boarded_stops.len());
                assert_eq!(boarded_stops[position], stops[alight]);
                let departure = view.stored_times(timetable, transfer.trip)[position].departure;
                assert!(departure >= times[alight].arrival);
                // Earliest boardable of its line position: the
                // previous trip of the line departs too early.
                let range = view.line_trips(line);
                if transfer.trip.0 > range.start {
                    let previous = ViewTrip(transfer.trip.0 - 1);
                    assert!(
                        view.stored_times(timetable, previous)[position].departure
                            < times[alight].arrival
                    );
                }
            }
        }
    }
}
