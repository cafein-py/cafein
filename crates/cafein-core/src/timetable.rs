//! The timetable: stop-sequence patterns, their trips, and stop times, in
//! structure-of-arrays CSR layouts.
//!
//! A *pattern* is a unique ordered stop sequence served by one route (the
//! RAPTOR "route" convention). Trips within a pattern are stored
//! contiguously, sorted by departure time at the pattern's first stop, and
//! no trip overtakes an earlier one at any stop — patterns violating this
//! are split into FIFO chains at build time — so boarding lookups can
//! binary-search and range-RAPTOR can iterate departures in decreasing
//! order at every stop of the pattern.

/// Index of a stop in a [`Timetable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StopIdx(pub u32);

/// Index of a stop-sequence pattern in a [`Timetable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PatternIdx(pub u32);

/// Index of a trip in a [`Timetable`].
///
/// Trips are numbered contiguously per pattern, in departure-time order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TripIdx(pub u32);

/// A pattern serving a stop, and the position of that stop in the pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternStop {
    pub pattern: PatternIdx,
    pub position: u16,
}

/// Scheduled arrival and departure at one position of a trip, in seconds
/// past the start of the service day (over-midnight times exceed 86 400).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopTime {
    pub arrival: u32,
    pub departure: u32,
}

/// An immutable, index-linked timetable ready for routing.
///
/// Built through [`TimetableBuilder`]; all cross-references are vector
/// indices, and all per-pattern data lives in flat arrays addressed through
/// offset tables (CSR).
#[derive(Debug)]
pub struct Timetable {
    stop_count: u32,
    /// CSR offsets into `pattern_stops`, one entry per pattern plus a tail.
    pattern_stops_offsets: Vec<u32>,
    pattern_stops: Vec<StopIdx>,
    /// CSR offsets into the trip index space, one entry per pattern plus a
    /// tail; trips of pattern `p` are `pattern_trips_offsets[p]..[p + 1]`.
    pattern_trips_offsets: Vec<u32>,
    /// Base offset of each pattern's block in `stop_times`.
    pattern_times_offsets: Vec<u32>,
    /// Stop times of all trips: trip `t` of pattern `p` with `n` stops and
    /// rank `r` within the pattern occupies
    /// `pattern_times_offsets[p] + r * n ..` for `n` entries.
    stop_times: Vec<StopTime>,
    /// Pattern of each trip.
    trip_patterns: Vec<PatternIdx>,
    /// Caller-defined identifier carried per trip (e.g. a feed trip index).
    trip_sources: Vec<u32>,
    /// Caller-defined service identifier carried per trip, for filtering
    /// trips by the services active on a query date.
    trip_services: Vec<u32>,
    /// Caller-defined identifier carried per pattern (e.g. a feed route index).
    pattern_routes: Vec<u32>,
    /// CSR offsets into `stop_patterns`, one entry per stop plus a tail.
    stop_patterns_offsets: Vec<u32>,
    stop_patterns: Vec<PatternStop>,
}

impl Timetable {
    pub fn stop_count(&self) -> u32 {
        self.stop_count
    }

    pub fn pattern_count(&self) -> u32 {
        self.pattern_routes.len() as u32
    }

    pub fn trip_count(&self) -> u32 {
        self.trip_patterns.len() as u32
    }

    /// The ordered stops of a pattern.
    pub fn pattern_stops(&self, pattern: PatternIdx) -> &[StopIdx] {
        let start = self.pattern_stops_offsets[pattern.0 as usize] as usize;
        let end = self.pattern_stops_offsets[pattern.0 as usize + 1] as usize;
        &self.pattern_stops[start..end]
    }

    /// The trips of a pattern, in departure-time order at its first stop.
    pub fn pattern_trips(&self, pattern: PatternIdx) -> impl Iterator<Item = TripIdx> {
        self.pattern_trip_range(pattern).map(TripIdx)
    }

    /// The contiguous trip-index range of a pattern.
    pub fn pattern_trip_range(&self, pattern: PatternIdx) -> std::ops::Range<u32> {
        self.pattern_trips_offsets[pattern.0 as usize]
            ..self.pattern_trips_offsets[pattern.0 as usize + 1]
    }

    /// The pattern a trip belongs to.
    pub fn trip_pattern(&self, trip: TripIdx) -> PatternIdx {
        self.trip_patterns[trip.0 as usize]
    }

    /// The stop times of a trip, aligned with its pattern's stops.
    pub fn trip_stop_times(&self, trip: TripIdx) -> &[StopTime] {
        let pattern = self.trip_pattern(trip);
        let stops = self.pattern_stops(pattern).len();
        let rank = (trip.0 - self.pattern_trips_offsets[pattern.0 as usize]) as usize;
        let start = self.pattern_times_offsets[pattern.0 as usize] as usize + rank * stops;
        &self.stop_times[start..start + stops]
    }

    /// The patterns serving a stop, with the stop's position in each.
    pub fn patterns_at_stop(&self, stop: StopIdx) -> &[PatternStop] {
        let start = self.stop_patterns_offsets[stop.0 as usize] as usize;
        let end = self.stop_patterns_offsets[stop.0 as usize + 1] as usize;
        &self.stop_patterns[start..end]
    }

    /// The caller-defined source identifier of a trip.
    pub fn trip_source(&self, trip: TripIdx) -> u32 {
        self.trip_sources[trip.0 as usize]
    }

    /// The caller-defined service identifier of a trip.
    pub fn trip_service(&self, trip: TripIdx) -> u32 {
        self.trip_services[trip.0 as usize]
    }

    /// The caller-defined route identifier of a pattern.
    pub fn pattern_route(&self, pattern: PatternIdx) -> u32 {
        self.pattern_routes[pattern.0 as usize]
    }
}

/// Accumulates patterns and trips, then assembles a [`Timetable`].
///
/// Patterns are registered first with [`add_pattern`](Self::add_pattern);
/// trips are added in any order and are sorted per pattern at
/// [`finish`](Self::finish).
#[derive(Debug, Default)]
pub struct TimetableBuilder {
    stop_count: u32,
    pattern_stops_offsets: Vec<u32>,
    pattern_stops: Vec<StopIdx>,
    pattern_routes: Vec<u32>,
    trips: Vec<BuilderTrip>,
}

#[derive(Debug)]
struct BuilderTrip {
    pattern: PatternIdx,
    source: u32,
    service: u32,
    stop_times: Vec<StopTime>,
}

/// Errors raised while assembling a [`Timetable`].
#[derive(Debug, PartialEq, Eq)]
pub enum TimetableError {
    /// A pattern was registered with no stops.
    EmptyPattern,
    /// A stop index is not below the declared stop count.
    StopOutOfRange { stop: u32, stop_count: u32 },
    /// A trip references a pattern index that was never registered.
    UnknownPattern { pattern: u32 },
    /// A trip's stop-time count differs from its pattern's stop count.
    StopTimeCountMismatch {
        source: u32,
        stop_times: usize,
        pattern_stops: usize,
    },
    /// A trip's stop times go backwards: a departure before its arrival, or
    /// an arrival before the previous stop's departure.
    NonIncreasingStopTimes { source: u32, position: usize },
}

impl std::fmt::Display for TimetableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimetableError::EmptyPattern => write!(f, "pattern has no stops"),
            TimetableError::StopOutOfRange { stop, stop_count } => {
                write!(f, "stop index {stop} is out of range ({stop_count} stops)")
            }
            TimetableError::UnknownPattern { pattern } => {
                write!(f, "unknown pattern index {pattern}")
            }
            TimetableError::StopTimeCountMismatch {
                source,
                stop_times,
                pattern_stops,
            } => write!(
                f,
                "trip (source {source}) has {stop_times} stop times but its pattern has {pattern_stops} stops"
            ),
            TimetableError::NonIncreasingStopTimes { source, position } => write!(
                f,
                "trip (source {source}) has stop times going backwards at position {position}"
            ),
        }
    }
}

impl std::error::Error for TimetableError {}

impl TimetableBuilder {
    /// Creates a builder for a network with `stop_count` stops.
    pub fn new(stop_count: u32) -> Self {
        TimetableBuilder {
            stop_count,
            pattern_stops_offsets: vec![0],
            ..TimetableBuilder::default()
        }
    }

    /// Registers a pattern with its ordered stops and a caller-defined route
    /// identifier, returning the pattern's index.
    pub fn add_pattern(
        &mut self,
        stops: &[StopIdx],
        route: u32,
    ) -> Result<PatternIdx, TimetableError> {
        if stops.is_empty() {
            return Err(TimetableError::EmptyPattern);
        }
        for stop in stops {
            if stop.0 >= self.stop_count {
                return Err(TimetableError::StopOutOfRange {
                    stop: stop.0,
                    stop_count: self.stop_count,
                });
            }
        }
        let pattern = PatternIdx(self.pattern_routes.len() as u32);
        self.pattern_stops.extend_from_slice(stops);
        self.pattern_stops_offsets
            .push(self.pattern_stops.len() as u32);
        self.pattern_routes.push(route);
        Ok(pattern)
    }

    /// Adds a trip to a registered pattern with one stop time per pattern
    /// stop, a caller-defined source identifier, and a caller-defined
    /// service identifier.
    ///
    /// Stop times must move forwards: at every stop `arrival <= departure`,
    /// and every arrival must be at or after the previous stop's departure.
    pub fn add_trip(
        &mut self,
        pattern: PatternIdx,
        stop_times: Vec<StopTime>,
        source: u32,
        service: u32,
    ) -> Result<(), TimetableError> {
        let pattern_index = pattern.0 as usize;
        if pattern_index >= self.pattern_routes.len() {
            return Err(TimetableError::UnknownPattern { pattern: pattern.0 });
        }
        let pattern_stops = (self.pattern_stops_offsets[pattern_index + 1]
            - self.pattern_stops_offsets[pattern_index]) as usize;
        if stop_times.len() != pattern_stops {
            return Err(TimetableError::StopTimeCountMismatch {
                source,
                stop_times: stop_times.len(),
                pattern_stops,
            });
        }
        for (position, stop_time) in stop_times.iter().enumerate() {
            let backwards_dwell = stop_time.departure < stop_time.arrival;
            let backwards_hop =
                position > 0 && stop_time.arrival < stop_times[position - 1].departure;
            if backwards_dwell || backwards_hop {
                return Err(TimetableError::NonIncreasingStopTimes { source, position });
            }
        }
        self.trips.push(BuilderTrip {
            pattern,
            source,
            service,
            stop_times,
        });
        Ok(())
    }

    /// Sorts trips per registered pattern by first-stop departure, splits
    /// each pattern into FIFO chains, and assembles the timetable.
    ///
    /// Within a final pattern no trip overtakes an earlier one at any stop.
    /// Trips of a registered pattern that violate this are moved into
    /// additional patterns sharing the same stops and route (greedy
    /// first-fit in departure order), so the final pattern count can exceed
    /// the number registered. Registered patterns without trips are dropped.
    pub fn finish(self) -> Timetable {
        let registered_count = self.pattern_routes.len();
        let mut trips = self.trips;
        trips.sort_by(|left, right| {
            left.pattern
                .cmp(&right.pattern)
                .then(
                    left.stop_times[0]
                        .departure
                        .cmp(&right.stop_times[0].departure),
                )
                .then_with(|| compare_stop_times(&left.stop_times, &right.stop_times))
        });

        let mut pattern_stops_offsets = vec![0u32];
        let mut pattern_stops: Vec<StopIdx> = Vec::new();
        let mut pattern_routes: Vec<u32> = Vec::new();
        let mut pattern_trips_offsets = vec![0u32];
        let mut sorted_trips: Vec<BuilderTrip> = Vec::with_capacity(trips.len());

        let mut trips = trips.into_iter().peekable();
        for registered in 0..registered_count {
            let mut registered_trips = Vec::new();
            while trips
                .peek()
                .is_some_and(|trip| trip.pattern.0 as usize == registered)
            {
                registered_trips.push(trips.next().unwrap());
            }

            let mut chains: Vec<Vec<BuilderTrip>> = Vec::new();
            'next_trip: for trip in registered_trips {
                for chain in &mut chains {
                    let last = chain.last().unwrap();
                    if follows(&trip.stop_times, &last.stop_times) {
                        chain.push(trip);
                        continue 'next_trip;
                    }
                }
                chains.push(vec![trip]);
            }

            let stops_start = self.pattern_stops_offsets[registered] as usize;
            let stops_end = self.pattern_stops_offsets[registered + 1] as usize;
            for chain in chains {
                pattern_stops.extend_from_slice(&self.pattern_stops[stops_start..stops_end]);
                pattern_stops_offsets.push(pattern_stops.len() as u32);
                pattern_routes.push(self.pattern_routes[registered]);
                pattern_trips_offsets
                    .push(pattern_trips_offsets.last().unwrap() + chain.len() as u32);
                sorted_trips.extend(chain);
            }
        }

        let pattern_count = pattern_routes.len();
        let mut pattern_times_offsets = vec![0u32; pattern_count];
        let mut total_times = 0u32;
        for pattern in 0..pattern_count {
            pattern_times_offsets[pattern] = total_times;
            let stops = pattern_stops_offsets[pattern + 1] - pattern_stops_offsets[pattern];
            let pattern_trip_count =
                pattern_trips_offsets[pattern + 1] - pattern_trips_offsets[pattern];
            total_times += stops * pattern_trip_count;
        }

        let mut stop_times = Vec::with_capacity(total_times as usize);
        let mut trip_patterns = Vec::with_capacity(sorted_trips.len());
        let mut trip_sources = Vec::with_capacity(sorted_trips.len());
        let mut trip_services = Vec::with_capacity(sorted_trips.len());
        for pattern in 0..pattern_count {
            let start = pattern_trips_offsets[pattern];
            let end = pattern_trips_offsets[pattern + 1];
            for trip in &sorted_trips[start as usize..end as usize] {
                stop_times.extend_from_slice(&trip.stop_times);
                trip_patterns.push(PatternIdx(pattern as u32));
                trip_sources.push(trip.source);
                trip_services.push(trip.service);
            }
        }

        let mut stop_patterns_offsets = vec![0u32; self.stop_count as usize + 1];
        for stop in &pattern_stops {
            stop_patterns_offsets[stop.0 as usize + 1] += 1;
        }
        for stop in 0..self.stop_count as usize {
            stop_patterns_offsets[stop + 1] += stop_patterns_offsets[stop];
        }
        let mut stop_patterns = vec![
            PatternStop {
                pattern: PatternIdx(0),
                position: 0
            };
            pattern_stops.len()
        ];
        let mut cursor = stop_patterns_offsets.clone();
        for pattern in 0..pattern_count {
            let start = pattern_stops_offsets[pattern] as usize;
            let end = pattern_stops_offsets[pattern + 1] as usize;
            for (position, stop) in pattern_stops[start..end].iter().enumerate() {
                let slot = cursor[stop.0 as usize] as usize;
                stop_patterns[slot] = PatternStop {
                    pattern: PatternIdx(pattern as u32),
                    position: position as u16,
                };
                cursor[stop.0 as usize] += 1;
            }
        }

        Timetable {
            stop_count: self.stop_count,
            pattern_stops_offsets,
            pattern_stops,
            pattern_trips_offsets,
            pattern_times_offsets,
            stop_times,
            trip_patterns,
            trip_sources,
            trip_services,
            pattern_routes,
            stop_patterns_offsets,
            stop_patterns,
        }
    }
}

/// Orders stop-time sequences lexicographically by `(arrival, departure)`.
fn compare_stop_times(left: &[StopTime], right: &[StopTime]) -> std::cmp::Ordering {
    for (a, b) in left.iter().zip(right) {
        let ordering = (a.arrival, a.departure).cmp(&(b.arrival, b.departure));
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// Whether `later` runs at or after `earlier` at every stop (no overtaking).
fn follows(later: &[StopTime], earlier: &[StopTime]) -> bool {
    later
        .iter()
        .zip(earlier)
        .all(|(l, e)| l.arrival >= e.arrival && l.departure >= e.departure)
}

#[cfg(test)]
mod tests {
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
}
