//! Timetable construction from the Helsinki region GTFS feed shared with
//! r5py (r5py.sampledata.helsinki v1.1.1).

use std::path::PathBuf;
use std::sync::OnceLock;

use cafein_core::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use cafein_gtfs::{build_timetable, Feed};

fn helsinki() -> &'static (Feed, Timetable) {
    static DATA: OnceLock<(Feed, Timetable)> = OnceLock::new();
    DATA.get_or_init(|| {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/helsinki_gtfs.zip");
        assert!(
            path.exists(),
            "test data missing at {}; run `python scripts/fetch_test_data.py`",
            path.display()
        );
        let feed = Feed::from_path(path).unwrap();
        let timetable = build_timetable(&feed).unwrap();
        (feed, timetable)
    })
}

#[test]
fn extracts_patterns_from_the_full_feed() {
    let (_, timetable) = helsinki();
    assert_eq!(timetable.stop_count(), 8305);
    assert_eq!(timetable.pattern_count(), 1011);
    assert_eq!(timetable.trip_count(), 195_351);
    let total_stop_times: usize = (0..timetable.trip_count())
        .map(|trip| timetable.trip_stop_times(TripIdx(trip)).len())
        .sum();
    assert_eq!(total_stop_times, 5_353_583);
}

#[test]
fn trips_within_a_pattern_depart_in_order() {
    let (_, timetable) = helsinki();
    for pattern in 0..timetable.pattern_count() {
        let departures: Vec<u32> = timetable
            .pattern_trips(PatternIdx(pattern))
            .map(|trip| timetable.trip_stop_times(trip)[0].departure)
            .collect();
        assert!(departures.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}

#[test]
fn night_bus_pattern_matches_the_feed() {
    let (feed, timetable) = helsinki();
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
    let (_, timetable) = helsinki();
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
