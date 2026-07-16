use super::*;
use crate::timetable::TimetableBuilder;

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

/// Line A rides 0→1→2; line B rides 1→3 at 90, 120, and 200. B's
/// trips carry services 0, 1, and 2.
fn crossing() -> Timetable {
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(90), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(120), time(500)], 2, 1)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(600)], 3, 2)
        .unwrap();
    builder.finish()
}

#[test]
fn boards_the_earliest_catchable_trip_only() {
    let timetable = crossing();
    let build = TransferSet::build(&timetable, &Transfers::empty(4));
    let set = &build.transfers;
    // Alighting A at stop 1 (arrival 100) catches B's 120 trip —
    // not the missed 90 nor the later 200.
    assert_eq!(
        set.from_trip_position(ViewTrip(0), 1),
        &[TripTransfer {
            trip: ViewTrip(2),
            position: 0,
        }]
    );
    // Nothing rides on from A's or B's last stop.
    assert!(set.from_trip_position(ViewTrip(0), 2).is_empty());
    assert!(set.from_trip_position(ViewTrip(2), 1).is_empty());
    assert_eq!(build.generated, set.len());
}

#[test]
fn date_views_board_the_earliest_running_trip() {
    let timetable = crossing();
    // B's 120 trip (service 1) does not run: the 200 trip is the
    // day's earliest catchable — judged against the date, not the
    // whole timetable.
    let active = vec![true, false, true];
    let view = DayView::for_date(&timetable, &active, &[false; 3]);
    assert_eq!(view.trip_count(), 3);
    let build = TransferSet::for_view(&view, &timetable, &Transfers::empty(4));
    let set = &build.transfers;
    // Virtual trips: 0 = A's trip, 1 = B's 90, 2 = B's 200.
    assert_eq!(view.backing(ViewTrip(2)), TripIdx(3));
    assert_eq!(
        set.from_trip_position(ViewTrip(0), 1),
        &[TripTransfer {
            trip: ViewTrip(2),
            position: 0,
        }]
    );
}

#[test]
fn previous_day_tails_join_as_shifted_lines() {
    // A today trip arrives stop 1 at 01:40; the connecting line's
    // only run today left at 00:01:40 stored (missed yesterday's
    // clock is irrelevant), but yesterday's 25:00 tail — 01:00
    // shifted — no: 25:33:20 stored = 01:33:20 shifted misses too;
    // 26:06:40 stored = 02:06:40 shifted connects.
    let mut builder = TimetableBuilder::new(4);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(5_000), time(6_000)], 0, 0)
        .unwrap();
    // Stored times: one trip wholly before midnight (never joins),
    // one over-midnight tail alighting too early, one connecting.
    builder
        .add_trip(b, vec![time(80_000), time(81_000)], 1, 1)
        .unwrap();
    builder
        .add_trip(b, vec![time(86_400 + 5_600), time(86_400 + 9_000)], 2, 1)
        .unwrap();
    builder
        .add_trip(b, vec![time(86_400 + 7_600), time(86_400 + 11_000)], 3, 1)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::for_date(&timetable, &[true, false], &[false, true]);
    // The wholly-pre-midnight trip is unboardable and stays out.
    assert_eq!(view.trip_count(), 3);
    assert_eq!(view.line_count(), 2);
    assert_eq!(view.day_offset(ViewTrip(1)), DAY_SECONDS);
    let build = TransferSet::for_view(&view, &timetable, &Transfers::empty(4));
    // A arrives stop 1 at 6 000: yesterday's 5 600-shifted tail has
    // left; the 7 600-shifted one boards.
    assert_eq!(
        build.transfers.from_trip_position(ViewTrip(0), 1),
        &[TripTransfer {
            trip: ViewTrip(2),
            position: 0,
        }]
    );
}

#[test]
fn reduction_drops_transfers_that_improve_nothing() {
    // Line C parallels A from stop 1 to stop 2, arriving later than
    // staying on A: feasible, but improves no arrival.
    let mut builder = TimetableBuilder::new(3);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let c = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(c, vec![time(150), time(400)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let build = TransferSet::build(&timetable, &Transfers::empty(3));
    assert_eq!(build.generated, 1);
    assert!(build.transfers.is_empty());
}

#[test]
fn same_pattern_transfers_cannot_help_under_fifo() {
    // Two trips of one line: the earlier trip never "transfers" to
    // the later one at the same or a later position.
    let mut builder = TimetableBuilder::new(3);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(50), time(150), time(350)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let build = TransferSet::build(&timetable, &Transfers::empty(3));
    assert_eq!(build.generated, 0);
    assert!(build.transfers.is_empty());
}

fn request(services: usize, origin: StopIdx, destination: StopIdx, departure: u32) -> Request {
    Request {
        departure,
        access: vec![(origin, 0)],
        egress: vec![(destination, 0)],
        active_services: vec![true; services],
        active_services_previous: vec![false; services],
        max_transfers: 7,
    }
}

fn pareto(journeys: &[Journey]) -> Vec<(u32, usize)> {
    journeys
        .iter()
        .map(|journey| (journey.arrival, journey.rides()))
        .collect()
}

#[test]
fn equal_arrival_same_ride_competitors_both_survive_reduction() {
    use crate::raptor::Raptor;

    // Sol's tie counterexample: a source trip rides O→X→Y; trip B leaves
    // X, trip C leaves Y, and both reach D at the same second with the
    // same ride count. They are different journeys (different boarded
    // trips and transfer stops), so the tie-complete reduction must keep
    // both transfers — the old strict-improvement rule kept only C
    // (backward walk visits Y first) and made RAPTOR's elected chain
    // unreconstructible.
    let mut builder = TimetableBuilder::new(4);
    let source = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let c = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(source, vec![time(0), time(20), time(40)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(50), time(100)], 1, 0)
        .unwrap();
    builder
        .add_trip(c, vec![time(60), time(100)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::empty(4);
    let build = TransferSet::build(&timetable, &footpaths);
    let kept = build.transfers;
    // Both equal competitors survive: X→B off position 1, Y→C off
    // position 2.
    assert_eq!(
        kept.from_trip_position(ViewTrip(0), 1),
        &[TripTransfer {
            trip: ViewTrip(1),
            position: 0,
        }]
    );
    assert_eq!(
        kept.from_trip_position(ViewTrip(0), 2),
        &[TripTransfer {
            trip: ViewTrip(2),
            position: 0,
        }]
    );
    // With the deterministic walked scratch, the reconstructed journey
    // is RAPTOR's, leg for leg — a pin on this fixture (the engine-wide
    // election-alignment proof is the cost-matrix stage's referee).
    let request = request(3, StopIdx(0), StopIdx(3), 0);
    let raptor = Raptor.route(&timetable, &footpaths, &request);
    let tbtr = Tbtr.route(&timetable, &footpaths, &request);
    assert_eq!(pareto(&raptor), vec![(100, 2)]);
    assert_eq!(tbtr, raptor);
}

#[test]
fn a_same_trip_tie_is_pruned_even_when_another_trip_holds_the_label() {
    use crate::raptor::Raptor;

    // Three equal-arrival candidates off one alight, encountered in the
    // order A, B-at-its-later-position, B-at-its-earlier-position: after
    // A claims the label, both B boardings tie it cross-trip, but only
    // B's earliest boarding is electable (RAPTOR boards a trip at its
    // earliest catchable position). The later B boarding must be pruned
    // however the tie list is ordered.
    let mut builder = TimetableBuilder::new(6);
    let source = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let a = builder.add_pattern(&[StopIdx(2), StopIdx(5)], 1).unwrap();
    let b = builder
        .add_pattern(&[StopIdx(4), StopIdx(3), StopIdx(5)], 2)
        .unwrap();
    builder
        .add_trip(source, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(200), time(1000)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(300), time(1000)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(1), StopIdx(3), 60, 50.0),
            (StopIdx(1), StopIdx(4), 60, 50.0),
        ],
    )
    .unwrap();
    let build = TransferSet::build(&timetable, &footpaths);
    let kept = build.transfers.from_trip_position(ViewTrip(0), 1);
    assert_eq!(kept.len(), 2);
    assert!(kept.contains(&TripTransfer {
        trip: ViewTrip(1),
        position: 0,
    }));
    // B survives only at its earliest boarding; the later boarding died
    // despite tying the label while trip A held it.
    assert!(kept.contains(&TripTransfer {
        trip: ViewTrip(2),
        position: 0,
    }));
    // As above: with deterministic walked boarding the election is
    // RAPTOR's, leg for leg, pinned on this fixture.
    let request = request(3, StopIdx(0), StopIdx(5), 0);
    let raptor = Raptor.route(&timetable, &footpaths, &request);
    let tbtr = Tbtr.route(&timetable, &footpaths, &request);
    assert_eq!(pareto(&raptor), vec![(1000, 2)]);
    assert_eq!(tbtr, raptor);
}

#[test]
fn a_tie_against_staying_on_the_trip_still_prunes() {
    // The source trip itself reaches D; a transfer at X to trip B ties
    // its arrival with one more ride. RAPTOR's round-ascending tie-break
    // elects the stayed (fewer-rides) journey, so the transfer stays
    // prunable — tie-completeness must not weaken the Stay side.
    let mut builder = TimetableBuilder::new(3);
    let source = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(source, vec![time(0), time(20), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(50), time(100)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let build = TransferSet::build(&timetable, &Transfers::empty(3));
    assert_eq!(build.generated, 1);
    assert!(build.transfers.is_empty());
}

#[test]
fn query_matches_raptor_on_the_crossing() {
    use crate::raptor::Raptor;

    let timetable = crossing();
    let footpaths = Transfers::empty(4);
    let request = request(3, StopIdx(0), StopIdx(3), 0);
    let raptor = Raptor.route(&timetable, &footpaths, &request);
    let tbtr = Tbtr.route(&timetable, &footpaths, &request);
    assert!(!raptor.is_empty());
    assert_eq!(pareto(&tbtr), pareto(&raptor));
    // The winning chain: A boarded at the origin, B caught at 120.
    assert_eq!(tbtr[0].rides(), 2);
    assert_eq!(tbtr[0].arrival, 500);
}

#[test]
fn footpath_egress_joins_like_raptors_transfer_relaxation() {
    use crate::raptor::Raptor;

    // The u-turn fixture: a destination 30 s from stop 2, best
    // reached by alighting at stop 1 and walking the footpath —
    // the one-hop closure behind the egress stop.
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(2), StopIdx(1), 60, 50.0),
        ],
    )
    .unwrap();
    let mut request = request(1, StopIdx(0), StopIdx(2), 0);
    request.egress = vec![(StopIdx(2), 30)];
    let raptor = Raptor.route(&timetable, &footpaths, &request);
    let tbtr = Tbtr.route(&timetable, &footpaths, &request);
    assert_eq!(pareto(&tbtr), pareto(&raptor));
    assert_eq!(tbtr[0].arrival, 190);
    // Egress leaves from the footpath's far end.
    assert!(matches!(
        tbtr[0].legs.last(),
        Some(Leg::Egress { from_stop, .. }) if *from_stop == StopIdx(2)
    ));
}

#[test]
fn query_matches_raptor_over_midnight() {
    use crate::raptor::Raptor;

    // The shifted-tails fixture: the connection rides yesterday's
    // 26:06:40 tail, 02:06:40 on the query day's clock.
    let mut builder = TimetableBuilder::new(4);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(5_000), time(6_000)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(80_000), time(81_000)], 1, 1)
        .unwrap();
    builder
        .add_trip(b, vec![time(86_400 + 5_600), time(86_400 + 9_000)], 2, 1)
        .unwrap();
    builder
        .add_trip(b, vec![time(86_400 + 7_600), time(86_400 + 11_000)], 3, 1)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::empty(4);
    let mut request = request(2, StopIdx(0), StopIdx(3), 5_000);
    request.active_services = vec![true, false];
    request.active_services_previous = vec![false, true];
    let raptor = Raptor.route(&timetable, &footpaths, &request);
    let tbtr = Tbtr.route(&timetable, &footpaths, &request);
    assert_eq!(pareto(&tbtr), pareto(&raptor));
    assert_eq!(tbtr[0].arrival, 11_000);
}

#[test]
fn range_profiles_match_raptor() {
    use crate::raptor::Raptor;

    // The crossing plus a later A trip and a later B trip: the
    // window offers two distinct departures with different
    // connections, and the descending passes must suppress
    // journeys dominated by the later departure.
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(150), time(250), time(450)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(120), time(500)], 2, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(300), time(650)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::empty(4);
    let request = request(1, StopIdx(0), StopIdx(3), 0);
    let raptor = Raptor.route_range(&timetable, &footpaths, &request, 400);
    let engine = TbtrEngine::for_date(
        &timetable,
        &footpaths,
        &request.active_services,
        &request.active_services_previous,
    );
    let tbtr = engine.route_range(&request, 400);
    let triples = |journeys: &[Journey]| -> Vec<(u32, u32, usize)> {
        journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect()
    };
    assert_eq!(triples(&tbtr), triples(&raptor));
    assert!(tbtr.len() >= 2);
}

#[test]
fn one_to_all_matches_raptor() {
    use crate::raptor::Raptor;

    let timetable = crossing();
    let footpaths = Transfers::from_edges(4, &[(StopIdx(2), StopIdx(3), 45, 40.0)]).unwrap();
    let request = request(3, StopIdx(0), StopIdx(3), 0);
    let raptor = Raptor.one_to_all(&timetable, &footpaths, &request);
    let engine = TbtrEngine::for_date(
        &timetable,
        &footpaths,
        &request.active_services,
        &request.active_services_previous,
    );
    let tbtr = engine.one_to_all(request.departure, &request.access, request.max_transfers);
    assert_eq!(tbtr, raptor);
    // The footpath tail is reachable: stop 3 over stop 2's walk.
    assert!(tbtr[3].is_some());
}

// The crossing over a window with two departures plus a footpath tail:
// exercises the descending passes and the last-round walk relaxation.
fn windowed_crossing() -> (Timetable, Transfers) {
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(150), time(250), time(450)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(120), time(500)], 2, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(300), time(650)], 3, 0)
        .unwrap();
    let footpaths = Transfers::from_edges(4, &[(StopIdx(2), StopIdx(3), 45, 40.0)]).unwrap();
    (builder.finish(), footpaths)
}

#[test]
fn window_samples_match_one_to_all() {
    let (timetable, footpaths) = windowed_crossing();
    let request = request(1, StopIdx(0), StopIdx(3), 0);
    let engine = TbtrEngine::for_date(
        &timetable,
        &footpaths,
        &request.active_services,
        &request.active_services_previous,
    );
    let samples = engine.window_samples(&request, 400);
    assert!(samples.len() >= 2);
    // Each mark's travel times match a one_to_all launched there. The
    // access floor (0 at the origin) reconciles the access stop, whose
    // windowed label is the next boardable departure rather than the mark
    // itself — the same correction RAPTOR's sampler relies on.
    let floor = crate::raptor::access_floor(4, &request);
    for (mark, snapshot) in &samples {
        let direct = engine.one_to_all(*mark, &request.access, request.max_transfers);
        for stop in 0..4 {
            let windowed = crate::raptor::travel_time(snapshot[stop], *mark, floor[stop]);
            let expected =
                crate::raptor::travel_time(direct[stop].unwrap_or(u32::MAX), *mark, floor[stop]);
            assert_eq!(windowed, expected, "mark {mark} stop {stop}");
        }
    }
}

#[test]
fn percentile_matrix_matches_raptor() {
    use crate::raptor::Raptor;

    let (timetable, footpaths) = windowed_crossing();
    let requests = vec![request(1, StopIdx(0), StopIdx(3), 0)];
    let percentiles = [0.0, 25.0, 50.0, 75.0, 100.0];
    let raptor = Raptor.percentile_matrix(&timetable, &footpaths, &requests, 400, &percentiles);
    let engine = TbtrEngine::for_date(
        &timetable,
        &footpaths,
        &requests[0].active_services,
        &requests[0].active_services_previous,
    );
    let tbtr = engine.percentile_matrix(&requests, 400, &percentiles);
    assert_eq!(tbtr, raptor);
    assert!(raptor[0].iter().any(|&value| value != u32::MAX));
}

#[test]
fn u_turns_are_dropped_at_generation() {
    // Line A rides 0→1→2; line B rides 1→2→(3); footpaths join
    // stops 1 and 2 both ways. Alighting A at stop 2 and walking
    // back to reboard B over the same 1→2 segment is a U-turn: B
    // was already catchable at stop 1.
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
        .add_trip(b, vec![time(400), time(500), time(600)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(2), StopIdx(1), 60, 50.0),
        ],
    )
    .unwrap();
    let build = TransferSet::build(&timetable, &footpaths);
    let set = &build.transfers;
    // Walking from stop 2 back to stop 1 to re-ride the 1→2 segment
    // is dropped at generation: only the three genuine boardings of
    // B are generated (from stop 1 at either end of its footpath,
    // and at stop 2 directly).
    assert_eq!(build.generated, 3);
    // The tie-complete reduction still prunes same-trip redundancy: all
    // three boardings ride the same B trip to the same arrivals, and
    // RAPTOR boards a trip at its earliest catchable position, so the
    // footpath boarding at the later position can never be elected and
    // is dropped. The earliest-position boarding survives off alight 1;
    // the direct boarding kept first off alight 2 remains as retained
    // slack (the backward walk had already committed it).
    assert_eq!(
        set.from_trip_position(ViewTrip(0), 1),
        &[TripTransfer {
            trip: ViewTrip(1),
            position: 0,
        }]
    );
    assert_eq!(
        set.from_trip_position(ViewTrip(0), 2),
        &[TripTransfer {
            trip: ViewTrip(1),
            position: 1,
        }]
    );
}

/// Full-row referee: both engines' cost matrices agree bit-for-bit —
/// integers exactly, floats by `to_bits` (NaN included).
fn cost_rows_agree(
    timetable: &Timetable,
    geometry: &crate::geometry::TripGeometry,
    footpaths: &Transfers,
    factors: &[f64],
    services: usize,
    max_transfers: u8,
) {
    use crate::raptor::{CostInputs, Raptor};

    let inputs = CostInputs {
        geometry,
        factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
    };
    let stop_count = timetable.stop_count();
    let destinations: Vec<StopIdx> = (0..stop_count).map(StopIdx).collect();
    let requests: Vec<Request> = (0..stop_count)
        .map(|origin| Request {
            departure: 0,
            access: vec![(StopIdx(origin), 0)],
            egress: Vec::new(),
            active_services: vec![true; services],
            active_services_previous: vec![false; services],
            max_transfers,
        })
        .collect();
    let raptor = Raptor.cost_matrix(timetable, footpaths, &inputs, &requests, &destinations);
    let engine = TbtrEngine::for_date(
        timetable,
        footpaths,
        &vec![true; services],
        &vec![false; services],
    );
    let tbtr = engine.cost_matrix(&inputs, &requests, &destinations);
    assert_eq!(tbtr.len(), raptor.len());
    for (origin, (t_rows, r_rows)) in tbtr.iter().zip(&raptor).enumerate() {
        assert_eq!(t_rows.len(), r_rows.len(), "row count for origin {origin}");
        for (t, r) in t_rows.iter().zip(r_rows) {
            let cell = format!("origin {origin} -> stop {}", r.to);
            assert_eq!(t.to, r.to, "{cell}: to");
            assert_eq!(t.seconds, r.seconds, "{cell}: seconds");
            assert_eq!(t.rides, r.rides, "{cell}: rides");
            assert_eq!(
                t.transit_meters.to_bits(),
                r.transit_meters.to_bits(),
                "{cell}: transit meters {} vs {}",
                t.transit_meters,
                r.transit_meters
            );
            assert_eq!(
                t.walk_meters.to_bits(),
                r.walk_meters.to_bits(),
                "{cell}: walk meters {} vs {}",
                t.walk_meters,
                r.walk_meters
            );
            assert_eq!(
                t.emission_grams.to_bits(),
                r.emission_grams.to_bits(),
                "{cell}: grams {} vs {}",
                t.emission_grams,
                r.emission_grams
            );
        }
    }
}

#[test]
fn cost_rows_match_raptor_on_the_tie_fixtures() {
    use crate::geometry::{DistanceProvenance, TripGeometry};

    // The three-way tie fixture: distinct distances and factors per trip
    // make any election divergence visible in the row aggregates.
    let mut builder = TimetableBuilder::new(6);
    let source = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let a = builder.add_pattern(&[StopIdx(2), StopIdx(5)], 1).unwrap();
    let b = builder
        .add_pattern(&[StopIdx(4), StopIdx(3), StopIdx(5)], 2)
        .unwrap();
    builder
        .add_trip(source, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(200), time(1000)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(300), time(1000)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 700.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 900.0], DistanceProvenance::CrowFly),
            (
                TripIdx(2),
                vec![0.0, 400.0, 1300.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(1), StopIdx(3), 60, 55.0),
            (StopIdx(1), StopIdx(4), 60, 65.0),
        ],
    )
    .unwrap();
    cost_rows_agree(&timetable, &geometry, &footpaths, &[40.0, 90.0, 25.0], 3, 4);

    // Sol's cross-trip counterexample, same treatment.
    let mut builder = TimetableBuilder::new(4);
    let source = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let c = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(source, vec![time(0), time(20), time(40)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(50), time(100)], 1, 0)
        .unwrap();
    builder
        .add_trip(c, vec![time(60), time(100)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 300.0, 800.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(1), vec![0.0, 1100.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 600.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    cost_rows_agree(
        &timetable,
        &geometry,
        &Transfers::empty(4),
        &[40.0, 90.0, 25.0],
        3,
        4,
    );
}

#[test]
fn least_cost_rows_match_raptor_on_the_tie_fixtures() {
    use crate::geometry::{DistanceProvenance, TripGeometry};
    use crate::raptor::{CostInputs, Objective, Raptor};

    // The three-way tie fixture again, now over a departure window with
    // both objectives and budgets that pass, bind exactly, and reject.
    let mut builder = TimetableBuilder::new(6);
    let source = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let a = builder.add_pattern(&[StopIdx(2), StopIdx(5)], 1).unwrap();
    let b = builder
        .add_pattern(&[StopIdx(4), StopIdx(3), StopIdx(5)], 2)
        .unwrap();
    builder
        .add_trip(source, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(source, vec![time(300), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(200), time(1000)], 2, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(500), time(1200)], 3, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(300), time(1000)], 4, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(500), time(600), time(1400)], 5, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 700.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 700.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 900.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 950.0], DistanceProvenance::CrowFly),
            (
                TripIdx(4),
                vec![0.0, 400.0, 1300.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(5),
                vec![0.0, 420.0, 1350.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(1), StopIdx(3), 60, 55.0),
            (StopIdx(1), StopIdx(4), 60, 65.0),
        ],
    )
    .unwrap();
    // NaN factor on one trip exercises unresolved-emission candidates.
    let factors = [40.0, 40.0, 90.0, f64::NAN, 25.0, 25.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
    };
    let stop_count = timetable.stop_count();
    let destinations: Vec<StopIdx> = (0..stop_count).map(StopIdx).collect();
    let requests: Vec<Request> = (0..stop_count)
        .map(|origin| Request {
            departure: 0,
            access: vec![(StopIdx(origin), 0)],
            egress: Vec::new(),
            active_services: vec![true; 6],
            active_services_previous: vec![false; 6],
            max_transfers: 4,
        })
        .collect();
    let engine = TbtrEngine::for_date(&timetable, &footpaths, &[true; 6], &[false; 6]);
    for objective in [Objective::Emissions, Objective::Fare] {
        for budget in [None, Some(1000), Some(999), Some(10)] {
            let raptor = Raptor.least_cost_matrix(
                &timetable,
                &footpaths,
                &inputs,
                &requests,
                &destinations,
                600,
                budget,
                objective,
            );
            let tbtr =
                engine.least_cost_matrix(&inputs, &requests, &destinations, 600, budget, objective);
            assert_eq!(tbtr.len(), raptor.len());
            for (origin, (t_rows, r_rows)) in tbtr.iter().zip(&raptor).enumerate() {
                assert_eq!(
                    t_rows.len(),
                    r_rows.len(),
                    "rows for origin {origin} ({objective:?}, {budget:?})"
                );
                for (t, r) in t_rows.iter().zip(r_rows) {
                    let cell = format!("origin {origin} -> {} ({objective:?}, {budget:?})", r.to);
                    assert_eq!(t.to, r.to, "{cell}: to");
                    assert_eq!(t.seconds, r.seconds, "{cell}: seconds");
                    assert_eq!(t.rides, r.rides, "{cell}: rides");
                    assert_eq!(
                        t.transit_meters.to_bits(),
                        r.transit_meters.to_bits(),
                        "{cell}: transit meters"
                    );
                    assert_eq!(
                        t.walk_meters.to_bits(),
                        r.walk_meters.to_bits(),
                        "{cell}: walk meters"
                    );
                    assert_eq!(
                        t.emission_grams.to_bits(),
                        r.emission_grams.to_bits(),
                        "{cell}: grams"
                    );
                }
            }
        }
    }
}

#[test]
fn point_and_fare_cost_rows_match_raptor() {
    use std::collections::HashMap;

    use crate::fares::{FareTables, RuleFares};
    use crate::geometry::{DistanceProvenance, TripGeometry};
    use crate::raptor::{CostInputs, Objective, Raptor};

    // The windowed tie fixture with egress link tables, access walks,
    // and rule-based fares: the point join (first equal link wins) and
    // the fare column ride the same referee. Geometry WKB follows leg
    // identity — both engines feed identical (trip, board, alight)
    // lists to the same builder — so the aggregate fields are the
    // election witness here.
    let mut builder = TimetableBuilder::new(6);
    let source = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let a = builder.add_pattern(&[StopIdx(2), StopIdx(5)], 1).unwrap();
    let b = builder
        .add_pattern(&[StopIdx(4), StopIdx(3), StopIdx(5)], 2)
        .unwrap();
    builder
        .add_trip(source, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(source, vec![time(300), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(200), time(1000)], 2, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(500), time(1200)], 3, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(300), time(1000)], 4, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(500), time(600), time(1400)], 5, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 700.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 700.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 900.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 950.0], DistanceProvenance::CrowFly),
            (
                TripIdx(4),
                vec![0.0, 400.0, 1300.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(5),
                vec![0.0, 420.0, 1350.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 50.0),
            (StopIdx(1), StopIdx(3), 60, 55.0),
            (StopIdx(1), StopIdx(4), 60, 65.0),
        ],
    )
    .unwrap();
    let fares = FareTables::RuleBased(RuleFares {
        route_type: vec![0, 0, 0],
        route_fare: vec![2.0, 3.0, 4.0],
        unlimited_transfers: vec![false],
        allow_same_route: vec![false],
        pair_fare: vec![4.5],
        max_discounted_transfers: 1,
        transfer_allowance: 600.0,
        fare_cap: f64::INFINITY,
    });
    let factors = [40.0, 40.0, 90.0, 90.0, 25.0, 25.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: Some(&fares),
    };
    let requests: Vec<Request> = (0..2)
        .map(|origin| Request {
            departure: 0,
            access: vec![(StopIdx(origin), 30)],
            egress: Vec::new(),
            active_services: vec![true; 6],
            active_services_previous: vec![false; 6],
            max_transfers: 4,
        })
        .collect();
    let access_meters: Vec<HashMap<StopIdx, f64>> = (0..2)
        .map(|origin| HashMap::from([(StopIdx(origin), 40.0)]))
        .collect();
    // Two destination points: one joined over two links (equal-link
    // election), one over a single link.
    let egress: Vec<Vec<(StopIdx, u32, f64)>> = vec![
        vec![(StopIdx(5), 30, 25.0), (StopIdx(3), 30, 35.0)],
        vec![(StopIdx(2), 45, 20.0)],
    ];
    let compare = |tbtr: &[Vec<crate::raptor::CostRow>], raptor: &[Vec<crate::raptor::CostRow>]| {
        assert_eq!(tbtr.len(), raptor.len());
        for (origin, (t_rows, r_rows)) in tbtr.iter().zip(raptor).enumerate() {
            assert_eq!(t_rows.len(), r_rows.len(), "rows for origin {origin}");
            for (t, r) in t_rows.iter().zip(r_rows) {
                let cell = format!("origin {origin} -> point {}", r.to);
                assert_eq!(t.to, r.to, "{cell}: to");
                assert_eq!(t.seconds, r.seconds, "{cell}: seconds");
                assert_eq!(t.rides, r.rides, "{cell}: rides");
                assert_eq!(
                    t.transit_meters.to_bits(),
                    r.transit_meters.to_bits(),
                    "{cell}: transit meters"
                );
                assert_eq!(
                    t.walk_meters.to_bits(),
                    r.walk_meters.to_bits(),
                    "{cell}: walk meters"
                );
                assert_eq!(
                    t.emission_grams.to_bits(),
                    r.emission_grams.to_bits(),
                    "{cell}: grams"
                );
                assert_eq!(t.fare.to_bits(), r.fare.to_bits(), "{cell}: fare");
            }
        }
    };
    let engine = TbtrEngine::for_date(&timetable, &footpaths, &[true; 6], &[false; 6]);
    let raptor = Raptor.cost_matrix_to_points(
        &timetable,
        &footpaths,
        &inputs,
        &requests,
        &access_meters,
        &egress,
    );
    let tbtr = engine.cost_matrix_to_points(&inputs, &requests, &access_meters, &egress);
    assert!(raptor.iter().any(|rows| !rows.is_empty()));
    compare(&tbtr, &raptor);
    for objective in [Objective::Emissions, Objective::Fare] {
        for budget in [None, Some(1100), Some(10)] {
            let raptor = Raptor.least_cost_matrix_to_points(
                &timetable,
                &footpaths,
                &inputs,
                &requests,
                &access_meters,
                &egress,
                600,
                budget,
                objective,
            );
            let tbtr = engine.least_cost_matrix_to_points(
                &inputs,
                &requests,
                &access_meters,
                &egress,
                600,
                budget,
                objective,
            );
            compare(&tbtr, &raptor);
        }
    }
}
