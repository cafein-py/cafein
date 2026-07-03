//! Per-trip cumulative travel distances, CSR by trip.
//!
//! The distances come from the Python-side fallback ladder (GTFS shapes,
//! linear referencing, crow-fly estimation); each trip carries the
//! provenance tier of its estimate so uncertainty stays visible per leg.

use crate::timetable::{Timetable, TripIdx};

/// How a trip's distances were estimated (the fallback-ladder tier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceProvenance {
    /// Valid `shape_dist_traveled` values taken directly from the feed.
    ShapeDist,
    /// Stops linear-referenced onto the feed's shape geometry.
    ShapeLinRef,
    /// Distances cut from a matched OSM route relation.
    OsmRelation,
    /// Distances map-matched onto the mode's network graph.
    MapMatched,
    /// Great-circle distances scaled by a mode detour coefficient.
    CrowFly,
}

/// Cumulative distances in meters at every stop position of every trip.
#[derive(Debug, PartialEq)]
pub struct TripGeometry {
    /// CSR offsets into `distances`, one entry per trip plus a tail.
    offsets: Vec<u32>,
    distances: Vec<f32>,
    provenance: Vec<DistanceProvenance>,
}

/// Errors raised while assembling a [`TripGeometry`].
#[derive(Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// A trip index is not below the timetable's trip count.
    TripOutOfRange { trip: u32, trip_count: u32 },
    /// A trip was given twice.
    DuplicateTrip { trip: u32 },
    /// Not every trip of the timetable was given distances.
    MissingTrips { missing: u32 },
    /// A trip's distance count differs from its pattern's stop count.
    LengthMismatch {
        trip: u32,
        distances: usize,
        stops: usize,
    },
    /// A trip's cumulative distances decrease, or are not finite.
    InvalidDistances { trip: u32, position: usize },
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::TripOutOfRange { trip, trip_count } => {
                write!(f, "trip index {trip} is out of range ({trip_count} trips)")
            }
            GeometryError::DuplicateTrip { trip } => {
                write!(f, "trip index {trip} was given distances twice")
            }
            GeometryError::MissingTrips { missing } => {
                write!(f, "{missing} trip(s) were given no distances")
            }
            GeometryError::LengthMismatch {
                trip,
                distances,
                stops,
            } => write!(
                f,
                "trip index {trip} has {distances} distances but its pattern has {stops} stops"
            ),
            GeometryError::InvalidDistances { trip, position } => write!(
                f,
                "trip index {trip} has a decreasing or non-finite distance at position {position}"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

impl TripGeometry {
    /// Builds the CSR structure from per-trip cumulative distances.
    ///
    /// Every trip of the timetable must appear exactly once, with one
    /// non-decreasing, finite cumulative distance per pattern stop.
    pub fn from_trips(
        timetable: &Timetable,
        trips: Vec<(TripIdx, Vec<f32>, DistanceProvenance)>,
    ) -> Result<TripGeometry, GeometryError> {
        let trip_count = timetable.trip_count();
        let mut per_trip: Vec<Option<(Vec<f32>, DistanceProvenance)>> =
            (0..trip_count).map(|_| None).collect();
        for (trip, distances, provenance) in trips {
            if trip.0 >= trip_count {
                return Err(GeometryError::TripOutOfRange {
                    trip: trip.0,
                    trip_count,
                });
            }
            let stops = timetable.pattern_stops(timetable.trip_pattern(trip)).len();
            if distances.len() != stops {
                return Err(GeometryError::LengthMismatch {
                    trip: trip.0,
                    distances: distances.len(),
                    stops,
                });
            }
            for (position, value) in distances.iter().enumerate() {
                if !value.is_finite() {
                    return Err(GeometryError::InvalidDistances {
                        trip: trip.0,
                        position,
                    });
                }
            }
            for (position, pair) in distances.windows(2).enumerate() {
                if pair[1] < pair[0] {
                    return Err(GeometryError::InvalidDistances {
                        trip: trip.0,
                        position: position + 1,
                    });
                }
            }
            let slot = &mut per_trip[trip.0 as usize];
            if slot.is_some() {
                return Err(GeometryError::DuplicateTrip { trip: trip.0 });
            }
            *slot = Some((distances, provenance));
        }
        let missing = per_trip.iter().filter(|slot| slot.is_none()).count() as u32;
        if missing > 0 {
            return Err(GeometryError::MissingTrips { missing });
        }

        let mut offsets = Vec::with_capacity(trip_count as usize + 1);
        offsets.push(0u32);
        let mut flat = Vec::new();
        let mut provenance = Vec::with_capacity(trip_count as usize);
        for slot in per_trip {
            let (distances, tier) = slot.unwrap();
            flat.extend_from_slice(&distances);
            offsets.push(flat.len() as u32);
            provenance.push(tier);
        }
        Ok(TripGeometry {
            offsets,
            distances: flat,
            provenance,
        })
    }

    /// The distance in meters travelled on `trip` between two positions.
    pub fn leg_distance(&self, trip: TripIdx, board_position: u16, alight_position: u16) -> f32 {
        let start = self.offsets[trip.0 as usize] as usize;
        self.distances[start + alight_position as usize]
            - self.distances[start + board_position as usize]
    }

    /// The provenance tier of a trip's distances.
    pub fn provenance(&self, trip: TripIdx) -> DistanceProvenance {
        self.provenance[trip.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timetable::{StopIdx, StopTime, TimetableBuilder};

    fn timetable() -> Timetable {
        let mut builder = TimetableBuilder::new(3);
        let pattern = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        for departure in [0, 100] {
            builder
                .add_trip(
                    pattern,
                    vec![
                        StopTime {
                            arrival: departure,
                            departure,
                        },
                        StopTime {
                            arrival: departure + 60,
                            departure: departure + 60,
                        },
                        StopTime {
                            arrival: departure + 120,
                            departure: departure + 120,
                        },
                    ],
                    0,
                    0,
                )
                .unwrap();
        }
        builder.finish()
    }

    #[test]
    fn stores_leg_distances_and_provenance() {
        let timetable = timetable();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(1),
                    vec![100.0, 700.0, 1600.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::ShapeDist,
                ),
            ],
        )
        .unwrap();
        assert_eq!(geometry.leg_distance(TripIdx(0), 0, 2), 1200.0);
        assert_eq!(geometry.leg_distance(TripIdx(1), 1, 2), 900.0);
        assert_eq!(
            geometry.provenance(TripIdx(0)),
            DistanceProvenance::ShapeDist
        );
        assert_eq!(geometry.provenance(TripIdx(1)), DistanceProvenance::CrowFly);
    }

    #[test]
    fn rejects_inconsistent_input() {
        let timetable = timetable();
        let full = |trip| (trip, vec![0.0, 1.0, 2.0], DistanceProvenance::CrowFly);
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(0))]),
            Err(GeometryError::MissingTrips { missing: 1 })
        );
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(0)), full(TripIdx(0))]),
            Err(GeometryError::DuplicateTrip { trip: 0 })
        );
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(2))]),
            Err(GeometryError::TripOutOfRange {
                trip: 2,
                trip_count: 2
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (TripIdx(1), vec![0.0, 1.0], DistanceProvenance::CrowFly)
                ]
            ),
            Err(GeometryError::LengthMismatch {
                trip: 1,
                distances: 2,
                stops: 3
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (TripIdx(1), vec![0.0, 2.0, 1.0], DistanceProvenance::CrowFly)
                ]
            ),
            Err(GeometryError::InvalidDistances {
                trip: 1,
                position: 2
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (
                        TripIdx(1),
                        vec![f32::NAN, 1.0, 2.0],
                        DistanceProvenance::CrowFly
                    )
                ]
            ),
            Err(GeometryError::InvalidDistances {
                trip: 1,
                position: 0
            })
        );
    }
}
