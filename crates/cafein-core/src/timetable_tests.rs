use super::*;

fn time(arrival: u32, departure: u32) -> StopTime {
    StopTime { arrival, departure }
}

fn two_pattern_timetable() -> Timetable {
    let mut builder = TimetableBuilder::new(4);
    let ab = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 10)
        .unwrap();
    let ba = builder.add_pattern(&[StopIdx(2), StopIdx(0)], 11).unwrap();
    // Added out of departure order on purpose.
    builder
        .add_trip(
            ab,
            vec![time(600, 600), time(660, 665), time(720, 720)],
            1,
            21,
        )
        .unwrap();
    builder
        .add_trip(ab, vec![time(0, 0), time(60, 65), time(120, 120)], 0, 20)
        .unwrap();
    builder
        .add_trip(ba, vec![time(30, 30), time(90, 90)], 2, 22)
        .unwrap();
    builder.finish()
}

#[test]
fn builds_csr_layout_with_sorted_trips() {
    let timetable = two_pattern_timetable();
    assert_eq!(timetable.stop_count(), 4);
    assert_eq!(timetable.pattern_count(), 2);
    assert_eq!(timetable.trip_count(), 3);

    let ab = PatternIdx(0);
    assert_eq!(
        timetable.pattern_stops(ab),
        &[StopIdx(0), StopIdx(1), StopIdx(2)]
    );
    let trips: Vec<_> = timetable.pattern_trips(ab).collect();
    assert_eq!(trips, vec![TripIdx(0), TripIdx(1)]);
    // Trips sorted by first-stop departure: source 0 departs first.
    assert_eq!(timetable.trip_source(TripIdx(0)), 0);
    assert_eq!(timetable.trip_source(TripIdx(1)), 1);
    assert_eq!(timetable.trip_stop_times(TripIdx(1))[1], time(660, 665));
    assert_eq!(timetable.trip_service(TripIdx(0)), 20);
    assert_eq!(timetable.trip_service(TripIdx(1)), 21);
    assert_eq!(timetable.pattern_route(ab), 10);
}

#[test]
fn indexes_patterns_by_stop() {
    let timetable = two_pattern_timetable();
    let at_first = timetable.patterns_at_stop(StopIdx(0));
    assert_eq!(at_first.len(), 2);
    assert!(at_first.contains(&PatternStop {
        pattern: PatternIdx(0),
        position: 0
    }));
    assert!(at_first.contains(&PatternStop {
        pattern: PatternIdx(1),
        position: 1
    }));
    assert_eq!(timetable.patterns_at_stop(StopIdx(3)), &[]);
}

#[test]
fn rejects_inconsistent_input() {
    let mut builder = TimetableBuilder::new(2);
    assert_eq!(
        builder.add_pattern(&[], 0),
        Err(TimetableError::EmptyPattern)
    );
    assert_eq!(
        builder.add_pattern(&[StopIdx(2)], 0),
        Err(TimetableError::StopOutOfRange {
            stop: 2,
            stop_count: 2
        })
    );
    let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    assert_eq!(
        builder.add_trip(PatternIdx(9), vec![], 7, 0),
        Err(TimetableError::UnknownPattern { pattern: 9 })
    );
    assert_eq!(
        builder.add_trip(pattern, vec![time(0, 0)], 7, 0),
        Err(TimetableError::StopTimeCountMismatch {
            source: 7,
            stop_times: 1,
            pattern_stops: 2
        })
    );
}

#[test]
fn rejects_backwards_stop_times() {
    let mut builder = TimetableBuilder::new(2);
    let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    // Departure before arrival at the same stop.
    assert_eq!(
        builder.add_trip(pattern, vec![time(10, 5), time(20, 20)], 7, 0),
        Err(TimetableError::NonIncreasingStopTimes {
            source: 7,
            position: 0
        })
    );
    // Arrival before the previous stop's departure.
    assert_eq!(
        builder.add_trip(pattern, vec![time(0, 30), time(20, 40)], 8, 0),
        Err(TimetableError::NonIncreasingStopTimes {
            source: 8,
            position: 1
        })
    );
}

#[test]
fn splits_overtaking_trips_into_fifo_patterns() {
    let mut builder = TimetableBuilder::new(2);
    let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 5).unwrap();
    builder
        .add_trip(pattern, vec![time(0, 0), time(100, 100)], 0, 0)
        .unwrap();
    // Departs later than trip 0 but arrives earlier: overtakes it.
    builder
        .add_trip(pattern, vec![time(10, 10), time(50, 50)], 1, 0)
        .unwrap();
    // Follows trip 0 at both stops.
    builder
        .add_trip(pattern, vec![time(20, 20), time(120, 120)], 2, 0)
        .unwrap();
    let timetable = builder.finish();

    assert_eq!(timetable.pattern_count(), 2);
    let first: Vec<u32> = timetable
        .pattern_trips(PatternIdx(0))
        .map(|trip| timetable.trip_source(trip))
        .collect();
    let second: Vec<u32> = timetable
        .pattern_trips(PatternIdx(1))
        .map(|trip| timetable.trip_source(trip))
        .collect();
    assert_eq!(first, vec![0, 2]);
    assert_eq!(second, vec![1]);
    // The split patterns share stops and route.
    assert_eq!(
        timetable.pattern_stops(PatternIdx(0)),
        timetable.pattern_stops(PatternIdx(1))
    );
    assert_eq!(timetable.pattern_route(PatternIdx(1)), 5);
}
