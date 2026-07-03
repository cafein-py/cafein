//! Timetable construction from the Helsinki region GTFS feed shared with
//! r5py (r5py.sampledata.helsinki v1.1.1).

mod common;

use std::sync::OnceLock;

use cafein_core::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use cafein_gtfs::{build_timetable, Feed};

fn helsinki() -> Option<&'static (Feed, Timetable)> {
    static DATA: OnceLock<Option<(Feed, Timetable)>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = common::helsinki_gtfs_path()?;
        let feed = Feed::from_path(path).unwrap();
        let build = build_timetable(&feed).unwrap();
        // The HSL feed has no quarantinable trips.
        assert!(build.quarantined.is_empty());
        Some((feed, build.timetable))
    })
    .as_ref()
}

#[test]
fn extracts_patterns_from_the_full_feed() {
    let Some((_, timetable)) = helsinki() else {
        return;
    };
    assert_eq!(timetable.stop_count(), 8305);
    // 1 011 distinct (route, stop sequence) pairs, split into 1 395 FIFO
    // patterns because HSL trips overtake within a stop sequence.
    assert_eq!(timetable.pattern_count(), 1395);
    assert_eq!(timetable.trip_count(), 195_351);
    let total_stop_times: usize = (0..timetable.trip_count())
        .map(|trip| timetable.trip_stop_times(TripIdx(trip)).len())
        .sum();
    assert_eq!(total_stop_times, 5_353_583);
}

#[test]
fn trips_within_a_pattern_never_overtake() {
    let Some((_, timetable)) = helsinki() else {
        return;
    };
    for pattern in 0..timetable.pattern_count() {
        let trips: Vec<TripIdx> = timetable.pattern_trips(PatternIdx(pattern)).collect();
        for pair in trips.windows(2) {
            let earlier = timetable.trip_stop_times(pair[0]);
            let later = timetable.trip_stop_times(pair[1]);
            assert!(earlier
                .iter()
                .zip(later)
                .all(|(e, l)| { e.arrival <= l.arrival && e.departure <= l.departure }));
        }
    }
}

#[test]
fn night_bus_pattern_matches_the_feed() {
    let Some((feed, timetable)) = helsinki() else {
        return;
    };
    let (trip_index, feed_trip) = feed
        .trips
        .iter()
        .enumerate()
        .find(|(_, trip)| trip.id == "2235N_20220222_La_1_2835")
        .unwrap();
    let trip = (0..timetable.trip_count())
        .map(TripIdx)
        .find(|trip| timetable.trip_source(*trip) == trip_index as u32)
        .unwrap();

    let pattern = timetable.trip_pattern(trip);
    assert_eq!(timetable.pattern_stops(pattern).len(), 72);
    assert_eq!(timetable.pattern_trips(pattern).count(), 55);
    let route = &feed.routes[timetable.pattern_route(pattern) as usize];
    assert_eq!(route.id, "2235N");

    let pattern_stops: Vec<u32> = timetable
        .pattern_stops(pattern)
        .iter()
        .map(|stop| stop.0)
        .collect();
    let trip_stops: Vec<u32> = feed_trip
        .stop_times
        .iter()
        .map(|stop_time| stop_time.stop)
        .collect();
    assert_eq!(pattern_stops, trip_stops);

    let times = timetable.trip_stop_times(trip);
    assert_eq!(
        times[0].departure,
        feed_trip.stop_times[0].departure.unwrap()
    );
    assert_eq!(times[71].arrival, feed_trip.stop_times[71].arrival.unwrap());
}

#[test]
fn stop_pattern_index_is_consistent() {
    let Some((_, timetable)) = helsinki() else {
        return;
    };
    let mut indexed_entries = 0usize;
    for stop in 0..timetable.stop_count() {
        for pattern_stop in timetable.patterns_at_stop(StopIdx(stop)) {
            assert_eq!(
                timetable.pattern_stops(pattern_stop.pattern)[pattern_stop.position as usize],
                StopIdx(stop)
            );
            indexed_entries += 1;
        }
    }
    let pattern_stop_entries: usize = (0..timetable.pattern_count())
        .map(|pattern| timetable.pattern_stops(PatternIdx(pattern)).len())
        .sum();
    assert_eq!(indexed_entries, pattern_stop_entries);
}
