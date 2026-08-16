//! The reverse engine against the forward candidate-set oracle on the
//! Helsinki feed: for each OD pair and deadline, the latest-departure
//! answer must equal the inversion of the forward profile built from
//! every candidate departure (the origin's own trip times), and where
//! the criteria tie the two engines must elect the same legs.

mod common;

use std::sync::OnceLock;

use cafein_core::raptor::Raptor;
use cafein_core::reverse::reverse_route;
use cafein_core::router::{Request, TransitRouter};
use cafein_core::timetable::StopIdx;
use cafein_core::transfers::{ReversedTransfers, Transfers};
use cafein_gtfs::{build_timetable, Feed, TimetableBuild};
use chrono::NaiveDate;

fn helsinki() -> Option<&'static (Feed, TimetableBuild)> {
    static DATA: OnceLock<Option<(Feed, TimetableBuild)>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = common::helsinki_gtfs_path()?;
        let feed = Feed::from_path(path).unwrap();
        let build = build_timetable(&feed).unwrap();
        Some((feed, build))
    })
    .as_ref()
}

fn stop_index(feed: &Feed, stop_id: &str) -> StopIdx {
    StopIdx(
        feed.stops
            .iter()
            .position(|stop| stop.id == stop_id)
            .unwrap() as u32,
    )
}

fn request(build: &TimetableBuild, from: StopIdx, to: StopIdx, at: u32) -> Request {
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    Request {
        departure: at,
        access: vec![(from, 0)],
        egress: vec![(to, 0)],
        active_services: build.services.active_on(date),
        active_services_previous: build
            .services
            .active_on(date.pred_opt().expect("date has a previous day")),
        max_transfers: 4,
        exclusions: None,
    }
}

/// A symmetric, closed transfer set from the feed's parent stations:
/// every pair of stops sharing a parent connects both ways at 120
/// seconds — small, realistic, and transitively complete within each
/// station cluster. The production walking closure is built by the
/// Python layer, so its equivalence runs in the Python suite once the
/// reverse engine is callable from an installed network.
fn station_transfers(feed: &Feed, stop_count: u32) -> Transfers {
    use std::collections::HashMap;
    let mut clusters: HashMap<&str, Vec<u32>> = HashMap::new();
    for (index, stop) in feed.stops.iter().enumerate() {
        if let Some(parent) = stop.parent_station.as_deref() {
            clusters.entry(parent).or_default().push(index as u32);
        }
    }
    let mut edges = Vec::new();
    for members in clusters.values() {
        for &a in members {
            for &b in members {
                if a != b {
                    edges.push((StopIdx(a), StopIdx(b), 120, 100.0));
                }
            }
        }
    }
    // from_edges sets are closed by construction flagging; each
    // station cluster is transitively complete, and the engines take
    // the exact phase for loaded sets anyway.
    Transfers::from_edges(stop_count, &edges).unwrap()
}

/// Checks one OD pair against the profile-inversion oracle for each
/// deadline, and replays every reverse journey through the forward
/// engine to confirm both elect identical legs. Returns how many
/// (pair, deadline) combinations produced at least one journey.
fn oracle_matches(
    build: &TimetableBuild,
    transfers: &Transfers,
    reversed: &ReversedTransfers,
    from: StopIdx,
    to: StopIdx,
    deadlines: &[u32],
) -> usize {
    // Candidate departures: every distinct departure second of a
    // trip calling at the origin (the classic profile candidate set —
    // journeys only change at these instants), on both service
    // streams: stored query-day times, and previous-day trips shifted
    // a day earlier onto the query clock. No lower bound — the
    // reverse engine has none either.
    let mut candidates: Vec<u32> = Vec::new();
    for pattern_stop in build.timetable.patterns_at_stop(from) {
        for trip in build.timetable.pattern_trips(pattern_stop.pattern) {
            let departure =
                build.timetable.trip_stop_times(trip)[pattern_stop.position as usize].departure;
            candidates.push(departure);
            if let Some(shifted) = departure.checked_sub(86_400) {
                candidates.push(shifted);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let mut covered = 0;
    for &deadline in deadlines {
        let mut profile: Vec<(u32, usize, u32)> = Vec::new();
        for &departure in candidates.iter().filter(|&&at| at <= deadline) {
            let forward = request(build, from, to, departure);
            for journey in Raptor.route(&build.timetable, transfers, &forward) {
                if journey.arrival <= deadline {
                    profile.push((departure, journey.rides(), journey.arrival));
                }
            }
        }
        profile.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut expected: Vec<(u32, usize, u32)> = Vec::new();
        for candidate in profile {
            let dominated = expected.iter().any(|held| {
                held.0 >= candidate.0
                    && held.1 <= candidate.1
                    && (held.0 > candidate.0 || held.1 < candidate.1)
            });
            let duplicate = expected
                .iter()
                .any(|held| held.0 == candidate.0 && held.1 == candidate.1);
            if !dominated && !duplicate {
                expected.push(candidate);
            }
        }
        let journeys = reverse_route(
            &build.timetable,
            transfers,
            reversed,
            &request(build, from, to, deadline),
        );
        let got: Vec<(u32, usize, u32)> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.rides(), journey.arrival))
            .collect();
        // The oracle's candidate set contains the true latest
        // departures (a journey's first boarding is a candidate),
        // so equality is exact.
        assert_eq!(got, expected, "{from:?}→{to:?} by {deadline}");
        if !journeys.is_empty() {
            covered += 1;
        }
        // Canonical identity: the forward engine replayed at the
        // reverse journey's own departure must elect the same legs,
        // not merely the same (departure, rides, arrival) tuple.
        for journey in &journeys {
            let forward = request(build, from, to, journey.departure);
            let replayed = Raptor.route(&build.timetable, transfers, &forward);
            let matched = replayed
                .iter()
                .find(|candidate| candidate.arrival == journey.arrival)
                .unwrap_or_else(|| {
                    panic!("{from:?}→{to:?}: no forward twin at {}", journey.departure)
                });
            assert_eq!(matched.rides(), journey.rides());
            assert_eq!(matched.legs, journey.legs, "{from:?}→{to:?} by {deadline}");
        }
    }
    covered
}

fn named_pairs(feed: &Feed) -> Vec<(StopIdx, StopIdx)> {
    // Korso → Käpylä (the K train), Käpylä → Korso, and a cross-town
    // pair; deadlines around the canonical 08:30 morning.
    [
        ("4810551", "1250551"),
        ("1250551", "4810551"),
        ("1020453", "1070422"),
    ]
    .into_iter()
    .map(|(from, to)| (stop_index(feed, from), stop_index(feed, to)))
    .collect()
}

#[test]
fn reverse_answers_invert_the_forward_profile_on_helsinki() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    let transfers = Transfers::empty(build.timetable.stop_count());
    let reversed = ReversedTransfers::build(&transfers);
    let deadlines = [9 * 3600, 9 * 3600 + 30 * 60, 10 * 3600];
    for (from, to) in named_pairs(feed) {
        oracle_matches(build, &transfers, &reversed, from, to, &deadlines);
    }
}

#[test]
fn the_inversion_holds_over_a_symmetric_transfer_closure() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    let transfers = station_transfers(feed, build.timetable.stop_count());
    let reversed = ReversedTransfers::build(&transfers);
    let deadlines = [9 * 3600, 9 * 3600 + 30 * 60, 10 * 3600];
    for (from, to) in named_pairs(feed) {
        oracle_matches(build, &transfers, &reversed, from, to, &deadlines);
    }
}

#[test]
fn a_seeded_sweep_agrees_with_the_oracle_across_the_feed() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    let transfers = station_transfers(feed, build.timetable.stop_count());
    let reversed = ReversedTransfers::build(&transfers);
    // Every stop with scheduled service is eligible; a fixed-seed LCG
    // draws the OD sample so the sweep is deterministic yet spread
    // across the feed rather than hand-picked.
    let eligible: Vec<StopIdx> = (0..build.timetable.stop_count())
        .map(StopIdx)
        .filter(|&stop| !build.timetable.patterns_at_stop(stop).is_empty())
        .collect();
    let mut state: u64 = 0x5EED_CAFE;
    let mut draw = |bound: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % bound
    };
    let mut pairs: Vec<(StopIdx, StopIdx)> = Vec::new();
    while pairs.len() < 10 {
        let from = eligible[draw(eligible.len())];
        let to = eligible[draw(eligible.len())];
        if from != to && !pairs.contains(&(from, to)) {
            pairs.push((from, to));
        }
    }
    let deadlines = [9 * 3600, 10 * 3600];
    let mut covered = 0;
    for (from, to) in pairs {
        covered += oracle_matches(build, &transfers, &reversed, from, to, &deadlines);
    }
    // The sample must actually exercise journeys, not vacuous
    // empty-vs-empty agreement on unreachable pairs.
    assert!(
        covered >= 5,
        "only {covered}/20 combinations found journeys"
    );
}
