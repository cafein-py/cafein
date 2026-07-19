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
    // tie-complete reduction keeps ~34 % of the feasible transfers
    // (Witt's strict-improvement rule kept ~12 %; the difference is
    // same-ride equal-arrival competitors, retained so the cost
    // matrices can reconstruct the journey RAPTOR's tie-break elects —
    // an equal transfer fails the scan's strict horizon test, so query
    // walls are unchanged).
    assert_eq!(build.generated, 42_937_748);
    assert_eq!(build.transfers.len(), 14_527_982);
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
    // 195 351 trips (the feed spans weeks of service days). The set
    // count includes the tie-complete reduction's retained equal
    // competitors.
    assert_eq!(view.trip_count(), 24_280);
    assert_eq!(build.transfers.len(), 1_580_419);
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
                    exclusions: None,
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
                    exclusions: None,
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
            exclusions: None,
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
fn footpath_queries_match_raptor() {
    let Some((timetable, services, _)) = helsinki() else {
        return;
    };
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let active = services.active_on(date);
    let active_previous = services.active_on(date.pred_opt().unwrap());
    // A synthetic footpath set over a patch of stops, transitively
    // closed: cafein's transfer contract (see cafein.streets) is a
    // closed set — single-hop routers are incomplete on non-closed
    // sets, RAPTOR included — so the oracle must honour it. Chains of
    // varying strides plus a dense clique, closed by Floyd–Warshall,
    // exercise the query-time relaxation without OSM data.
    let count = timetable.stop_count();
    const PATCH_START: u32 = 3_300;
    const PATCH: usize = 300;
    let mut walk = vec![[u32::MAX; PATCH]; PATCH];
    let mut connect = |a: usize, b: usize, seconds: u32| {
        walk[a][b] = walk[a][b].min(seconds);
        walk[b][a] = walk[b][a].min(seconds);
    };
    for local in 0..PATCH as u32 {
        for (stride, base) in [(1u32, 90u32), (7, 240), (131, 420)] {
            let to = local + stride;
            if (to as usize) < PATCH {
                connect(local as usize, to as usize, base + (local * 37) % 180);
            }
        }
    }
    for a in 40..110usize {
        for b in 40..110usize {
            if a != b {
                connect(a, b, 60 + (a + b) as u32 % 240);
            }
        }
    }
    for via in 0..PATCH {
        for from in 0..PATCH {
            if walk[from][via] == u32::MAX {
                continue;
            }
            for to in 0..PATCH {
                if walk[via][to] != u32::MAX {
                    let chained = walk[from][via] + walk[via][to];
                    if chained < walk[from][to] {
                        walk[from][to] = chained;
                    }
                }
            }
        }
    }
    let mut edges = Vec::new();
    for (from, row) in walk.iter().enumerate() {
        for (to, &seconds) in row.iter().enumerate() {
            if from != to && seconds != u32::MAX {
                edges.push((
                    StopIdx(PATCH_START + from as u32),
                    StopIdx(PATCH_START + to as u32),
                    seconds,
                    seconds as f64,
                ));
            }
        }
    }
    let footpaths = Transfers::from_edges(count, &edges).unwrap();
    let engine = TbtrEngine::for_date(timetable, &footpaths, &active, &active_previous);
    let pareto = |journeys: &[cafein_core::journey::Journey]| -> Vec<(u32, usize)> {
        journeys
            .iter()
            .map(|journey| (journey.arrival, journey.rides()))
            .collect()
    };
    for origin in (5..count).step_by(1471).map(StopIdx) {
        for destination in (900..count).step_by(2903).map(StopIdx) {
            let request = Request {
                departure: 8 * 3600 + 30 * 60,
                access: vec![(origin, 0)],
                egress: vec![(destination, 0)],
                active_services: active.clone(),
                active_services_previous: active_previous.clone(),
                max_transfers: 4,
                exclusions: None,
            };
            let raptor = Raptor.route(timetable, &footpaths, &request);
            let tbtr = engine.query(
                request.departure,
                &request.access,
                &request.egress,
                request.max_transfers,
            );
            assert_eq!(
                pareto(&tbtr),
                pareto(&raptor),
                "footpath pareto sets diverge for {origin:?}->{destination:?}"
            );
            let window_raptor = Raptor.route_range(timetable, &footpaths, &request, 1200);
            let window_tbtr = engine.route_range(&request, 1200);
            assert_eq!(
                pareto(&window_tbtr),
                pareto(&window_raptor),
                "footpath window profiles diverge for {origin:?}->{destination:?}"
            );
        }
    }
    // One-to-all with footpaths, full arrays.
    for origin in (777..count).step_by(3251).map(StopIdx) {
        let request = Request {
            departure: 8 * 3600 + 30 * 60,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: active.clone(),
            active_services_previous: active_previous.clone(),
            max_transfers: 4,
            exclusions: None,
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
            "footpath one-to-all diverges from {origin:?}: {mismatches:?}"
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

#[test]
fn cost_rows_match_raptor_across_sampled_origins() {
    use cafein_core::geometry::{DistanceProvenance, TripGeometry};
    use cafein_core::raptor::{CostInputs, Objective};

    let Some((timetable, services, _)) = helsinki() else {
        return;
    };
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let active = services.active_on(date);
    let active_previous = services.active_on(date.pred_opt().unwrap());
    let footpaths = Transfers::empty(timetable.stop_count());
    // Synthetic per-trip distances and factors, deterministic and
    // distinct, so any election divergence between the engines shows
    // in the aggregates even without the Python-side distance ladder.
    let trips: Vec<_> = (0..timetable.trip_count())
        .map(cafein_core::timetable::TripIdx)
        .map(|trip| {
            let stops = timetable.pattern_stops(timetable.trip_pattern(trip)).len();
            let step = 250.0 + (trip.0 % 13) as f32 * 17.0;
            (
                trip,
                (0..stops).map(|k| k as f32 * step).collect::<Vec<f32>>(),
                DistanceProvenance::CrowFly,
            )
        })
        .collect();
    let geometry = TripGeometry::from_trips(timetable, trips).unwrap();
    let factors: Vec<f64> = (0..timetable.trip_count())
        .map(|trip| 20.0 + (trip % 11) as f64 * 7.0)
        .collect();
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
    };
    let destinations: Vec<StopIdx> = (0..timetable.stop_count())
        .step_by(97)
        .map(StopIdx)
        .collect();
    let requests: Vec<Request> = (5..timetable.stop_count())
        .step_by(1481)
        .map(|origin| Request {
            departure: 8 * 3600 + 30 * 60,
            access: vec![(StopIdx(origin), 0)],
            egress: Vec::new(),
            active_services: active.clone(),
            active_services_previous: active_previous.clone(),
            max_transfers: 4,
            exclusions: None,
        })
        .collect();
    assert!(requests.len() >= 5);
    let engine = TbtrEngine::for_date(timetable, &footpaths, &active, &active_previous);
    let compare = |tbtr: &[Vec<cafein_core::raptor::CostRow>],
                   raptor: &[Vec<cafein_core::raptor::CostRow>]| {
        assert_eq!(tbtr.len(), raptor.len());
        for (origin, (t_rows, r_rows)) in tbtr.iter().zip(raptor).enumerate() {
            assert_eq!(t_rows.len(), r_rows.len(), "rows for origin {origin}");
            for (t, r) in t_rows.iter().zip(r_rows) {
                assert_eq!(
                    (t.to, t.seconds, t.rides),
                    (r.to, r.seconds, r.rides),
                    "origin {origin}"
                );
                assert_eq!(
                    t.transit_meters.to_bits(),
                    r.transit_meters.to_bits(),
                    "origin {origin} -> {}: transit meters {} vs {}",
                    r.to,
                    t.transit_meters,
                    r.transit_meters
                );
                assert_eq!(
                    t.walk_meters.to_bits(),
                    r.walk_meters.to_bits(),
                    "origin {origin} -> {}: walk meters",
                    r.to
                );
                assert_eq!(
                    t.emission_grams.to_bits(),
                    r.emission_grams.to_bits(),
                    "origin {origin} -> {}: grams",
                    r.to
                );
            }
        }
    };
    let raptor = Raptor.cost_matrix(timetable, &footpaths, &inputs, &requests, &destinations);
    let tbtr = engine.cost_matrix(&inputs, &requests, &destinations);
    assert!(raptor.iter().any(|rows| !rows.is_empty()));
    compare(&tbtr, &raptor);
    let raptor = Raptor.least_cost_matrix(
        timetable,
        &footpaths,
        &inputs,
        &requests,
        &destinations,
        1800,
        Some(3600),
        Objective::Emissions,
    );
    let tbtr = engine.least_cost_matrix(
        &inputs,
        &requests,
        &destinations,
        1800,
        Some(3600),
        Objective::Emissions,
    );
    compare(&tbtr, &raptor);
}
