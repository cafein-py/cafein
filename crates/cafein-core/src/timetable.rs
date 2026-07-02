//! The timetable: stop-sequence patterns, their trips, and stop times, in
//! structure-of-arrays CSR layouts.
//!
//! A *pattern* is a unique ordered stop sequence served by one route (the
//! RAPTOR "route" convention). Trips within a pattern are stored
//! contiguously, sorted by departure time at the pattern's first stop, so
//! boarding lookups can binary-search and range-RAPTOR can iterate
//! departures in decreasing order.

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
        let start = self.pattern_trips_offsets[pattern.0 as usize];
        let end = self.pattern_trips_offsets[pattern.0 as usize + 1];
        (start..end).map(TripIdx)
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
    /// stop and a caller-defined source identifier.
    pub fn add_trip(
        &mut self,
        pattern: PatternIdx,
        stop_times: Vec<StopTime>,
        source: u32,
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
        self.trips.push(BuilderTrip {
            pattern,
            source,
            stop_times,
        });
        Ok(())
    }

    /// Sorts trips per pattern by first-stop departure and assembles the
    /// timetable.
    pub fn finish(self) -> Timetable {
        let pattern_count = self.pattern_routes.len();
        let mut trips = self.trips;
        trips.sort_by_key(|trip| (trip.pattern, trip.stop_times[0].departure));

        let mut pattern_trips_offsets = vec![0u32; pattern_count + 1];
        for trip in &trips {
            pattern_trips_offsets[trip.pattern.0 as usize + 1] += 1;
        }
        for pattern in 0..pattern_count {
            pattern_trips_offsets[pattern + 1] += pattern_trips_offsets[pattern];
        }

        let mut pattern_times_offsets = vec![0u32; pattern_count];
        let mut total_times = 0u32;
        for pattern in 0..pattern_count {
            pattern_times_offsets[pattern] = total_times;
            let stops =
                self.pattern_stops_offsets[pattern + 1] - self.pattern_stops_offsets[pattern];
            let pattern_trip_count =
                pattern_trips_offsets[pattern + 1] - pattern_trips_offsets[pattern];
            total_times += stops * pattern_trip_count;
        }

        let mut stop_times = Vec::with_capacity(total_times as usize);
        let mut trip_patterns = Vec::with_capacity(trips.len());
        let mut trip_sources = Vec::with_capacity(trips.len());
        for trip in &trips {
            stop_times.extend_from_slice(&trip.stop_times);
            trip_patterns.push(trip.pattern);
            trip_sources.push(trip.source);
        }

        let mut stop_patterns_offsets = vec![0u32; self.stop_count as usize + 1];
        for stop in &self.pattern_stops {
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
            self.pattern_stops.len()
        ];
        let mut cursor = stop_patterns_offsets.clone();
        for pattern in 0..pattern_count {
            let start = self.pattern_stops_offsets[pattern] as usize;
            let end = self.pattern_stops_offsets[pattern + 1] as usize;
            for (position, stop) in self.pattern_stops[start..end].iter().enumerate() {
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
            pattern_stops_offsets: self.pattern_stops_offsets,
            pattern_stops: self.pattern_stops,
            pattern_trips_offsets,
            pattern_times_offsets,
            stop_times,
            trip_patterns,
            trip_sources,
            pattern_routes: self.pattern_routes,
            stop_patterns_offsets,
            stop_patterns,
        }
    }
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
            .add_trip(ab, vec![time(600, 600), time(660, 665), time(720, 720)], 1)
            .unwrap();
        builder
            .add_trip(ab, vec![time(0, 0), time(60, 65), time(120, 120)], 0)
            .unwrap();
        builder
            .add_trip(ba, vec![time(30, 30), time(90, 90)], 2)
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
            builder.add_trip(PatternIdx(9), vec![], 7),
            Err(TimetableError::UnknownPattern { pattern: 9 })
        );
        assert_eq!(
            builder.add_trip(pattern, vec![time(0, 0)], 7),
            Err(TimetableError::StopTimeCountMismatch {
                source: 7,
                stop_times: 1,
                pattern_stops: 2
            })
        );
    }
}
